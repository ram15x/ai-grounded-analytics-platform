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
    JSON_REJECTED,
    MALFORMED,
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


# 400 json_validate_failed answers in a row, per provider and question, after which the provider
# gets no more requests for that question
JSON_STREAK_LIMIT = 2
# An empty reply that ended on finish_reason `length` is retried once with the completion allowance
# doubled, never above this.
LENGTH_RETRY_CAP = 4096
EMPTY_KINDS = frozenset(["content_empty", "content_null"])


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
            elif run.json_streak.get(name, 0) >= JSON_STREAK_LIMIT:
                # this provider refused JSON mode twice in a row for this question: no more
                # requests to it for the question; the next provider (if any) is tried
                run.add(CallRecord(0, req.task, name, "-", "json_streak_skip"))
            else:
                eligible.append(name)
        if not eligible and any(r.outcome == "json_streak_skip" for r in run.records):
            raise AllProvidersFailed(
                [{"provider": r.provider, "outcome": r.outcome}
                 for r in run.records if r.outcome == "json_streak_skip"]
            )  # fmt: skip
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
        json_rejects = 0
        rl_retries = 0
        len_retried = False
        attempt = 0
        while attempt <= retry.max_retries:
            wait = self._bucket_wait(name, reserved)
            if wait > self.settings.bucket.max_wait_s or wait == float("inf"):
                fail("throttled", model, attempt=attempt)
                return None
            if wait > 0:
                run.waits.append({"provider": name, "task": req.task, "kind": "bucket",
                                  "seconds": round(wait, 3)})  # fmt: skip
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
                        max_tokens=req.max_tokens,
                        json_mode_off=not req.json_mode and json_rejects > 0,
                        max_tokens_doubled=len_retried,
                        prompt_tokens=exc.prompt_tokens,
                        completion_tokens=exc.completion_tokens,
                        reasoning_tokens=exc.reasoning_tokens,
                        finish_reason=exc.finish_reason,
                        raw_body=exc.raw_body,
                        malformed_kind=exc.malformed_kind,
                    )  # fmt: skip
                )
                attempts.append({"provider": name, "outcome": exc.kind, "attempt": attempt})
                wait_ms = 0.0
                if exc.kind == JSON_REJECTED:
                    # JSON mode refused the model's own output. Retry ONCE with JSON mode off for
                    # this call (our own parse and schema validation still apply; the identical
                    # request was seen to fail again, assumptions row 33). It is a real HTTP
                    # request, it does not use up the normal retry budget, and the probe's request
                    # cap counts it. UNVERIFIED: that JSON mode off helps. If the provider still
                    # answers JSON_REJECTED, hand the text back as unparseable (repair path).
                    self._consec_429[name] = 0
                    json_rejects += 1
                    run.json_streak[name] = run.json_streak.get(name, 0) + 1
                    if run.json_streak[name] >= JSON_STREAK_LIMIT:
                        return None  # second in a row for this question: stop, next provider
                    if json_rejects == 1 and req.json_mode:
                        req = dataclasses.replace(req, json_mode=False)
                        continue
                    return LLMResponse(
                        text=exc.failed_generation or "",
                        provider=name,
                        model=model,
                        latency_ms=elapsed,
                        finish_reason=JSON_REJECTED,
                    )
                if exc.kind == RATE_LIMIT:
                    n = self._consec_429.get(name, 0) + 1
                    self._consec_429[name] = n
                    if n >= self.settings.breaker.consecutive_429:
                        self._open_breaker(name)
                        return None
                    # one retry, after the wait the provider asked for; with no Retry-After, after
                    # the token guard says the request fits again (never a blind retry). A wait
                    # above the cap is not shortened: the provider is given up for this call.
                    rl = self.settings.rate_limit
                    if rl_retries >= rl.retries:
                        return None
                    if exc.retry_after is not None:
                        delay, basis = exc.retry_after, "retry_after"
                    else:
                        need = self._bucket_wait(name, reserved)
                        delay = max(
                            self.settings.retry.base_s, 0.0 if need == float("inf") else need
                        )
                        basis = "token_guard"
                    if delay > rl.max_wait_s:
                        return None
                    run.waits.append({"provider": name, "task": req.task,
                                      "kind": "rate_limit_retry", "basis": basis,
                                      "seconds": round(delay, 3)})  # fmt: skip
                    wait_ms += delay * 1000
                    self.sleep(delay)
                    rl_retries += 1
                    continue
                self._consec_429[name] = 0
                if (
                    exc.kind == MALFORMED
                    and exc.malformed_kind in EMPTY_KINDS
                    and exc.finish_reason == "length"
                    and not len_retried
                    and min(req.max_tokens * 2, LENGTH_RETRY_CAP) > req.max_tokens
                ):
                    # the model used its whole allowance (reasoning) and wrote nothing: ask once
                    # more with a larger allowance, never with the identical request
                    len_retried = True
                    req = dataclasses.replace(
                        req, max_tokens=min(req.max_tokens * 2, LENGTH_RETRY_CAP)
                    )
                    reserved = estimate_tokens(req)
                    continue
                if json_rejects > 0 and exc.kind == MALFORMED:
                    # the JSON-off retry came back as an unusable HTTP 200 (seen: empty content,
                    # finish length). Do not repeat it: the text goes back unparseable, as for a
                    # second refusal.
                    return LLMResponse(
                        text="",
                        provider=name,
                        model=model,
                        latency_ms=elapsed,
                        finish_reason=JSON_REJECTED,
                    )
                empty_length = (
                    exc.kind == MALFORMED
                    and exc.malformed_kind in EMPTY_KINDS
                    and exc.finish_reason == "length"
                )
                if json_rejects > 0 or empty_length:
                    # the JSON-off or doubled-allowance retry failed, or the allowance cannot be
                    # raised any further: an identical request would only repeat the failure
                    return None
                if exc.kind not in RETRYABLE or attempt == retry.max_retries:
                    return None
                delay = self._delay(exc, attempt)
                wait_ms += delay * 1000
                self.sleep(delay)
                attempt += 1
                continue
            self._consec_429[name] = 0
            run.json_streak[name] = 0
            rec = self._record(req, resp, "ok", attempt, wait_ms)
            rec = dataclasses.replace(rec, max_tokens_doubled=len_retried)
            rec = dataclasses.replace(rec, json_mode_off=json_rejects > 0 and not req.json_mode)
            run.add(rec)
            if not rec.tokens_estimated:  # settle the reservation against what the provider said
                self._bucket_settle(name, reserved - (rec.prompt_tokens + rec.completion_tokens))
            return rec_to_response(resp, rec)
        return None

    # ---- helpers ----
    def _for_provider(self, name: str, req: LLMRequest) -> LLMRequest:
        """The request as this provider will see it: its configured reasoning effort (only for
        Groq gpt-oss models) and, for compose and repair on a model that reports reasoning
        tokens, a completion allowance of at least `reasoning_min_tokens`. Both become part of
        the request and so of the cassette key."""
        cfg = self.settings.providers[name]
        try:
            model = cfg.model_for(req.task)
        except ConfigError:
            model = ""
        effort = cfg.reasoning_effort
        if effort and "gpt-oss" in model.lower() and req.reasoning_effort != effort:
            req = dataclasses.replace(req, reasoning_effort=effort)
        floor = cfg.reasoning_min_tokens
        if (
            floor
            and req.task in ("compose", "repair")
            and not req.max_tokens_fixed
            and req.max_tokens < floor
        ):
            req = dataclasses.replace(req, max_tokens=floor)
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
            max_tokens=req.max_tokens,
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
