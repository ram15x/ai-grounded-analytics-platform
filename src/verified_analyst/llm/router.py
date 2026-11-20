"""The model router (PHASE_1C.md section b, ADR-0003).

Order of checks for every call, all deterministic and all before any network traffic:
  1. resolve the data sensitivity of the payload (unknown dataset = internal);
  2. enforce the per-run token cap (BudgetExceeded);
  3. drop hosted providers for internal data (SensitivityRefusal when none is left);
  4. per provider: circuit breaker, token buckets, then the call with retries.
The same sensitivity rule applies on every fallback step, because the eligible list is built once
and fallbacks only walk that list. Hosted adapters refuse unstamped requests as a second check.
"""

from __future__ import annotations

import dataclasses
import random
import time
from collections.abc import Callable
from typing import Any

from ..config import ConfigError
from .accounting import CallRecord, RunContext
from .cassette import Cassette
from .providers import ADAPTERS, Provider
from .settings import RouterSettings
from .types import (
    INTERNAL,
    PUBLIC,
    RATE_LIMIT,
    RETRYABLE,
    AllProvidersFailed,
    BudgetExceeded,
    LLMRequest,
    LLMResponse,
    ProviderError,
    SensitivityRefusal,
)


class TokenBucket:
    """Classic token bucket on an injectable clock. `capacity` tokens, refilled continuously."""

    def __init__(self, capacity: float, refill_per_s: float, clock: Callable[[], float]):
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.clock = clock
        self.level = capacity
        self.last = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.level = min(self.capacity, self.level + (now - self.last) * self.refill_per_s)
        self.last = now

    def wait_needed(self, amount: float) -> float:
        """Seconds until `amount` could be taken (inf if it can never fit)."""
        if amount > self.capacity:
            return float("inf")
        self._refill()
        if self.level >= amount:
            return 0.0
        return (amount - self.level) / self.refill_per_s

    def take(self, amount: float) -> None:
        self._refill()
        self.level -= amount

    def give_back(self, amount: float) -> None:
        """Return an over-reservation. Never lifts the level above capacity."""
        self._refill()
        self.level = min(self.capacity, self.level + amount)


def estimate_tokens(req: LLMRequest) -> int:
    """Conservative pre-call estimate: prompt characters / 4 plus the completion allowance."""
    return req.prompt_chars() // 4 + req.max_tokens


def build_providers(settings: RouterSettings, **adapter_kwargs: Any) -> dict[str, Provider]:
    """Instantiate the hosted adapters named in the config. `fake` providers are supplied by
    the caller (tests), never built from config."""
    out: dict[str, Provider] = {}
    for name, p in settings.providers.items():
        if p.kind == "fake":
            continue
        out[name] = ADAPTERS[p.kind](name, p.env_key or "", endpoint=p.endpoint, **adapter_kwargs)
    return out


class Router:
    def __init__(
        self,
        settings: RouterSettings,
        providers: dict[str, Provider],
        *,
        cassette: Cassette | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ):
        self.settings = settings
        self.providers = providers
        self.cassette = cassette
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random()
        self._rpm: dict[str, TokenBucket] = {}
        self._tpd: dict[str, TokenBucket] = {}
        self._tpm: dict[str, TokenBucket] = {}
        self._consec_429: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        frac = settings.bucket.fraction
        for name, p in settings.providers.items():
            if p.rpm:
                cap = max(1.0, p.rpm * frac)
                self._rpm[name] = TokenBucket(cap, cap / 60.0, clock)
            if p.tpd:
                cap = p.tpd * frac
                self._tpd[name] = TokenBucket(cap, cap / 86400.0, clock)
            if p.tpm:
                cap = p.tpm * frac
                self._tpm[name] = TokenBucket(cap, cap / 60.0, clock)

    # ---- public ----
    def new_run(self, run_id: str, token_cap: int | None = None) -> RunContext:
        return RunContext(run_id, token_cap or self.settings.budget.per_run_tokens)

    def call(self, req: LLMRequest, run: RunContext) -> LLMResponse:
        sensitivity = self.settings.sensitivity_of(req.datasets)
        req = dataclasses.replace(req, sensitivity=sensitivity)
        est = estimate_tokens(req)
        if run.tokens_used + est > run.token_cap:
            run.add(CallRecord(0, req.task, "-", "-", "budget"))
            raise BudgetExceeded(
                f"run {run.run_id}: {run.tokens_used} tokens used + about {est} for this call "
                f"exceeds the cap of {run.token_cap}"
            )
        order = self.settings.tasks.get(req.task)
        if not order:
            raise ConfigError(f"no providers configured for task {req.task!r}")
        eligible: list[str] = []
        for name in order:
            if self.settings.providers[name].hosted and sensitivity != PUBLIC:
                run.add(CallRecord(0, req.task, name, "-", "refused_sensitivity"))
            else:
                eligible.append(name)
        if not eligible:
            raise SensitivityRefusal(
                f"task {req.task!r} has no non-hosted provider and the payload is "
                f"{INTERNAL} (datasets: {', '.join(req.datasets) or 'none'})"
            )
        if self.cassette is not None and self.cassette.mode == "replay":
            return self._replay(req, run, eligible)
        attempts: list[dict] = []
        for name in eligible:
            preq = self._for_provider(name, req)
            resp = self._try_provider(name, preq, run, attempts)
            if resp is not None:
                if self.cassette is not None:
                    self.cassette.record(name, resp.model, preq, resp)
                return resp
        raise AllProvidersFailed(attempts)

    # ---- replay ----
    def _replay(self, req: LLMRequest, run: RunContext, eligible: list[str]) -> LLMResponse:
        assert self.cassette is not None
        tried = []
        for name in eligible:
            try:
                model = self.settings.providers[name].model_for(req.task)
            except ConfigError:
                continue
            tried.append(name)
            preq = self._for_provider(name, req)
            hit = self.cassette.lookup(name, model, preq)
            if hit is not None:
                run.add(self._record(preq, hit, "cassette", 0, 0.0))
                return hit
        raise self.cassette.miss_error(req, tried)

    # ---- live ----
    def _try_provider(
        self, name: str, req: LLMRequest, run: RunContext, attempts: list[dict]
    ) -> LLMResponse | None:
        cfg = self.settings.providers[name]
        provider = self.providers.get(name)

        def fail(outcome: str, model: str = "-", **extra: Any) -> None:
            attempts.append({"provider": name, "outcome": outcome, **extra})
            run.add(CallRecord(0, req.task, name, model, outcome, **extra))

        if provider is None:
            fail("no_adapter")
            return None
        try:
            model = cfg.model_for(req.task)
        except ConfigError:
            fail("config")
            return None
        now = self.clock()
        if self._open_until.get(name, 0) > now:
            fail("breaker_open", model)
            return None
        retry = self.settings.retry
        wait_ms = 0.0
        reserved = estimate_tokens(req)
        for attempt in range(retry.max_retries + 1):
            wait = self._bucket_wait(name, reserved)
            if wait > self.settings.bucket.max_wait_s:
                fail("throttled", model, attempt=attempt)
                return None
            if wait > 0:
                self.sleep(wait)
                wait_ms += wait * 1000
            self._bucket_take(name, reserved)
            start = self.clock()
            try:
                resp = provider.complete(req, model, cfg.timeout_s)
            except SensitivityRefusal:
                raise
            except ConfigError:
                fail("config", model, attempt=attempt)
                return None
            except ProviderError as exc:
                elapsed = (self.clock() - start) * 1000
                run.add(
                    CallRecord(
                        0,
                        req.task,
                        name,
                        model,
                        exc.kind,
                        attempt=attempt,
                        latency_ms=elapsed,
                        wait_ms=wait_ms,
                        retry_after_s=exc.retry_after,
                    )  # fmt: skip
                )
                attempts.append({"provider": name, "outcome": exc.kind, "attempt": attempt})
                wait_ms = 0.0
                if exc.kind == RATE_LIMIT:
                    n = self._consec_429.get(name, 0) + 1
                    self._consec_429[name] = n
                    if n >= self.settings.breaker.consecutive_429:
                        self._open_breaker(name)
                        return None
                else:
                    self._consec_429[name] = 0
                if exc.kind not in RETRYABLE or attempt == retry.max_retries:
                    return None
                delay = self._delay(exc, attempt)
                if delay > retry.max_retry_after_s:
                    return None
                wait_ms += delay * 1000
                self.sleep(delay)
                continue
            self._consec_429[name] = 0
            rec = self._record(req, resp, "ok", attempt, wait_ms)
            run.add(rec)
            if not rec.tokens_estimated:  # settle the reservation against what the provider said
                self._bucket_settle(name, reserved - (rec.prompt_tokens + rec.completion_tokens))
            return rec_to_response(resp, rec)
        return None

    # ---- helpers ----
    def _for_provider(self, name: str, req: LLMRequest) -> LLMRequest:
        """The request as this provider will see it: its configured reasoning effort, if any,
        becomes part of the request (and so of the cassette key)."""
        effort = self.settings.providers[name].reasoning_effort
        if effort and req.reasoning_effort != effort:
            return dataclasses.replace(req, reasoning_effort=effort)
        return req

    def _record(
        self, req: LLMRequest, resp: LLMResponse, outcome: str, attempt: int, wait_ms: float
    ) -> CallRecord:
        p, c, est = resp.prompt_tokens, resp.completion_tokens, resp.tokens_estimated
        if (p + c) == 0 or est:
            # the provider gave no usable usage block: fall back to a characters/4 estimate
            p = p or req.prompt_chars() // 4
            c = c or len(resp.text) // 4
            est = True
        return CallRecord(
            0, req.task, resp.provider, resp.model, outcome, attempt=attempt,
            prompt_tokens=p, completion_tokens=c, tokens_estimated=est,
            latency_ms=resp.latency_ms, wait_ms=wait_ms, from_cassette=resp.from_cassette,
            reasoning_tokens=resp.reasoning_tokens, finish_reason=resp.finish_reason,
        )  # fmt: skip

    def _delay(self, exc: ProviderError, attempt: int) -> float:
        r = self.settings.retry
        if exc.kind == RATE_LIMIT and exc.retry_after is not None:
            return exc.retry_after
        d = min(r.cap_s, r.base_s * (2**attempt))
        return d * (1 - r.jitter) + self.rng.uniform(0, d * r.jitter)

    def _open_breaker(self, name: str) -> None:
        self._open_until[name] = self.clock() + self.settings.breaker.cooldown_s
        # after the cooldown the next 429 re-opens it immediately (half-open probe)
        self._consec_429[name] = self.settings.breaker.consecutive_429 - 1

    def _bucket_wait(self, name: str, tokens: float) -> float:
        waits = [0.0]
        if name in self._rpm:
            waits.append(self._rpm[name].wait_needed(1))
        for buckets in (self._tpm, self._tpd):
            if name in buckets:
                waits.append(buckets[name].wait_needed(tokens))
        return max(waits)

    def _bucket_take(self, name: str, tokens: float) -> None:
        if name in self._rpm:
            self._rpm[name].take(1)
        for buckets in (self._tpm, self._tpd):
            if name in buckets:
                buckets[name].take(tokens)

    def _bucket_settle(self, name: str, surplus: float) -> None:
        """A call reserved `estimate` tokens up front. When the response reports real usage, give
        back the part that was not used, or take the extra if it was more. 429s and other
        failures keep their full reservation (the safe side: it is not known whether the
        provider counted them)."""
        for buckets in (self._tpm, self._tpd):
            if name in buckets:
                if surplus >= 0:
                    buckets[name].give_back(surplus)
                else:
                    buckets[name].take(-surplus)

    def breaker_state(self, name: str) -> str:
        return "open" if self._open_until.get(name, 0) > self.clock() else "closed"


def rec_to_response(resp: LLMResponse, rec: CallRecord) -> LLMResponse:
    """Return the response with the (possibly estimated) token counts the ledger used."""
    return dataclasses.replace(
        resp,
        prompt_tokens=rec.prompt_tokens,
        completion_tokens=rec.completion_tokens,
        tokens_estimated=rec.tokens_estimated,
    )
