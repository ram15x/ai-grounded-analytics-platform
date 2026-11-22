"""First real provider calls, as tooling the owner runs on their own machine.

`list_models` and `run_probe` are the only code in this repository that is meant to reach a real
provider, and only when invoked by `grounded live ...` with a key in the environment. Tests mock the
transport. Hard limits: at most 60 HTTP requests per probe run (retries included; 25 unless the
owner passes --max-requests), a token cap from config, and a stop on the first 401/403.
Output is raw per-model numbers with n, never an
accuracy claim and never a ranking. A key is redacted from everything printed or written.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ConfigError
from ..semantic import SemanticLayer
from . import usage
from .accounting import RunContext, merge_summaries, percentile
from .cassette import (
    KEY_ENV_VARS,
    Cassette,
    CassetteContext,
    check_cassettes,
    layer_hash,
    scan_for_secrets,
    schema_hash,
    sha256_text,
    warehouse_content_hash,
)
from .pipeline import QuestionRun, run_question
from .prompts import active_prompt_hashes, catalogue_text, plan_request, prompt_hashes
from .providers import PLACEHOLDER_PREFIX, build_one
from .router import Router
from .settings import RouterSettings
from .transport import (
    Getter,
    HttpResponse,
    Transport,
    TransportError,
    TransportTimeout,
    rate_headers,
    urllib_get,
    urllib_transport,
)
from .types import PUBLIC, LLMRequest, ProviderError, SensitivityRefusal

HARD_MAX_REQUESTS = 60  # per probe run; the default cap is 25 (config `probe.max_requests`)
PROBE_BUCKET_MAX_WAIT_S = 60.0  # the most one wait may last; a probe has no fallback, so it waits
ERROR_BODY_CHARS = 300
_sleep: Callable[[float], None] = time.sleep  # tests replace this


# ---------------------------------------------------------------- redaction


def secret_values(env: Mapping[str, str]) -> list[str]:
    return [v for k in KEY_ENV_VARS if (v := env.get(k)) and len(v) >= 8]


def redact(text: str, env: Mapping[str, str]) -> str:
    for v in secret_values(env):
        text = text.replace(v, "<redacted>")
    return text


_BEARER = re.compile(r"(?i)bearer\s+\S+")


def safe_error_body(body: bytes, env: Mapping[str, str], limit: int = ERROR_BODY_CHARS) -> str:
    """A provider's error body made safe to print and store: decoded leniently, whitespace
    collapsed, any key value from the environment and any `Bearer ...` text replaced, and cut to
    `limit` characters. The request is never included, only what the provider sent back."""
    text = " ".join(body.decode("utf-8", errors="replace").split())
    text = redact(_BEARER.sub("Bearer <redacted>", text), env)
    return text if len(text) <= limit else text[:limit] + f"... [truncated, {len(text)} chars]"


# ---------------------------------------------------------------- 429 quota details

_TOKEN_OK = re.compile(r"[^A-Za-z0-9_./:\- ()]")
_PERIOD = re.compile(r"per[ -](second|minute|min|hour|day)|per(second|minute|hour|day)", re.I)
_MSG_LIMIT = re.compile(r"\blimit[: ]+(\d[\d,.]*)", re.I)
_MSG_KIND = re.compile(
    r"\bon ([a-z ]{3,30}?) per (second|minute|hour|day)(?: \(([A-Za-z]{2,5})\))?"
)
_MSG_METRIC = re.compile(r"metric: ([\w./\-]{3,160})")
_MSG_RETRY = re.compile(r"try again in ([\w.]{1,12})", re.I)
_MSG_FREE = re.compile(r"\b(free-models-per-(?:min|day)|[a-z-]+-per-(?:min|day))\b")
QUOTA_FIELD_CHARS = 120


def _tok(value: Any, limit: int = QUOTA_FIELD_CHARS) -> str | None:
    """A short provider-supplied name or number, with anything outside a small safe alphabet
    removed (so no id, key or free text is carried), or None."""
    if not isinstance(value, str | int | float) or isinstance(value, bool):
        return None
    text = _TOKEN_OK.sub("", str(value)).strip()[:limit]
    return text or None


def quota_info(body: bytes) -> dict[str, Any] | None:
    """The quota metric and limit names a 429 body carries, and nothing else: no message text,
    no organisation or project id, no request content. Returns None when none were found.
    Shapes handled (all UNVERIFIED against a live body except where a report shows them):
    Google RPC `error.details[].violations[]` (quotaMetric, quotaId, quotaValue) and RetryInfo,
    Groq/OpenAI style `error.type`/`error.code` plus the limit name in the message, and
    OpenRouter's `free-models-per-...` limit name."""
    try:
        data = json.loads(body[:20000].decode("utf-8", errors="replace"))
    except ValueError:
        return None
    if isinstance(data, list) and data:
        data = data[0]
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict):
        return None
    out: dict[str, Any] = {}
    for key, name in (("status", "status"), ("type", "type"), ("code", "code")):
        v = _tok(err.get(key))
        if v and not (key == "code" and v == "429"):
            out[name] = v
    violations: list[dict[str, str]] = []
    for d in err.get("details") or []:
        if not isinstance(d, dict):
            continue
        for v in d.get("violations") or []:
            if isinstance(v, dict):
                item = {
                    k: t
                    for k, raw in (
                        ("quota_metric", v.get("quotaMetric")),
                        ("quota_id", v.get("quotaId")),
                        ("quota_value", v.get("quotaValue")),
                    )
                    if (t := _tok(raw))
                }
                if item:
                    violations.append(item)
        if d.get("retryDelay") is not None:
            rd = _tok(d.get("retryDelay"), 20)
            if rd:
                out["retry_delay"] = rd
    if violations:
        out["violations"] = violations[:4]
    msg = err.get("message") if isinstance(err.get("message"), str) else ""
    if m := _MSG_KIND.search(msg):
        out["limit_on"] = _tok(f"{m.group(1)} per {m.group(2)}")
        if m.group(3):
            out["limit_name"] = _tok(m.group(3))
    if m := _MSG_LIMIT.search(msg):
        out["limit_value"] = _tok(m.group(1), 20)
    if m := _MSG_METRIC.search(msg):
        out.setdefault("quota_metric", _tok(m.group(1)))
    if m := _MSG_FREE.search(msg):
        out["limit_name"] = _tok(m.group(1))
    if "retry_delay" not in out and (m := _MSG_RETRY.search(msg)):
        out["retry_delay"] = _tok(m.group(1).rstrip("."), 20)
    out = {k: v for k, v in out.items() if v}
    return out or None


def quota_reading(info: dict[str, Any]) -> str:
    """A keyword reading of a quota_info result: which window the limit names, and whether a
    limit of zero was reported. It only looks for words in the names the provider sent."""
    names = " ".join(
        [str(info.get(k, "")) for k in ("limit_on", "limit_name", "quota_metric", "quota_id")]
        + [
            f"{v.get('quota_id', '')} {v.get('quota_metric', '')}"
            for v in info.get("violations", [])
        ]
    )
    windows = []
    for m in _PERIOD.finditer(names):
        w = (m.group(1) or m.group(2)).lower()
        windows.append({"min": "minute"}.get(w, w))
    values = [v["quota_value"] for v in info.get("violations", []) if "quota_value" in v]
    if "limit_value" in info:
        values.append(info["limit_value"])
    parts = []
    parts.append(
        "window named: " + "/".join(sorted(set(windows))) if windows else "no window named"
    )
    if any(v.strip().strip(".,") == "0" for v in values):
        parts.append("a limit of 0 was reported")
    return "; ".join(parts)


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for k of n; None when n is 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(name: str, k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"name": name, "k": k, "n": n, "wilson95": [round(x, 4) for x in ci] if ci else None}


def render_rate(r: dict[str, Any]) -> str:
    if r["n"] == 0:
        return f"- {r['name']}: n/a (n=0)"
    lo, hi = r["wilson95"]
    return (
        f"- {r['name']}: {r['k']} of {r['n']} (n={r['n']}; 95% Wilson interval "
        f"{lo * 100:.1f}%-{hi * 100:.1f}%)"
    )


# ---------------------------------------------------------------- counting transport


class RequestCapReached(Exception):
    """The hard request cap for this probe run was reached; no request was sent."""


class AuthStop(Exception):
    """A 401 or 403 came back. The probe stops on the first one."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


MAX_CONSECUTIVE_FAILURES = 5  # consecutive failed requests (not 401/403) before the probe stops
MODEL_GONE_PHRASES = (
    "not found",
    "no longer available",
    "does not exist",
    "no endpoints found",
    "model_not_found",
    "is not supported",
)


def is_model_gone(status: int | None, body: bytes) -> bool:
    """An HTTP 404 whose body says the model is not found / no longer available. A 404 whose
    body says nothing of the kind is not matched (it counts as an ordinary failure)."""
    if status != 404:
        return False
    text = body[:4000].decode("utf-8", errors="replace").lower()
    return any(p in text for p in MODEL_GONE_PHRASES)


class ProviderUnavailable(Exception):
    """MAX_CONSECUTIVE_FAILURES requests in a row failed (timeout, network error, or non-2xx
    other than 401/403). The probe stops: the provider is not usable right now."""

    def __init__(self, statuses: list[str], any_success: bool):
        self.statuses = statuses
        self.any_success = any_success
        super().__init__(f"{len(statuses)} consecutive failed requests ({', '.join(statuses)})")


class ModelGone(Exception):
    """A 404 saying the model is not found or no longer available. Stops on the first one."""

    def __init__(self, body: str):
        super().__init__("HTTP 404: model not found / no longer available")
        self.body = body


@dataclass
class HttpEvent:
    seq: int
    status: int | None  # None for a timeout or connection error
    error: str | None
    latency_ms: float
    rate_headers: dict[str, str] = field(default_factory=dict)
    error_body: str | None = None  # redacted, truncated; only for a non-2xx response
    quota: dict[str, Any] | None = None  # names/limits parsed from a 429 body; replaces its body


class CountingTransport:
    """Wraps the real transport: counts every request, enforces the cap, records status codes and
    rate-limit headers for every response (successful or not), and stops on the first 401/403."""

    def __init__(
        self,
        inner: Transport,
        max_requests: int,
        clock: Callable[[], float] = time.perf_counter,
        env: Mapping[str, str] | None = None,
    ):
        self.inner = inner
        self.env: Mapping[str, str] = env or {}
        self.max_requests = min(max_requests, HARD_MAX_REQUESTS)
        self.clock = clock
        self.events: list[HttpEvent] = []
        # True while a request that is MEANT to fail with a model-not-found 404 is in flight
        # (the quota check's invalid-model step). It neither stops the probe nor counts toward
        # the consecutive-failure streak.
        self.expect_model_404 = False
        self._streak: list[str] = []
        self._any_success = False

    def _failed(self, label: str) -> None:
        self._streak.append(label)
        if len(self._streak) >= MAX_CONSECUTIVE_FAILURES:
            raise ProviderUnavailable(list(self._streak), self._any_success)

    @property
    def count(self) -> int:
        return len(self.events)

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float):
        if self.count >= self.max_requests:
            raise RequestCapReached(f"request cap of {self.max_requests} reached")
        seq = self.count + 1
        start = self.clock()
        try:
            resp = self.inner(url, headers, body, timeout)
        except (TransportTimeout, TransportError) as exc:
            kind = "timeout" if isinstance(exc, TransportTimeout) else "network"
            self.events.append(HttpEvent(seq, None, kind, (self.clock() - start) * 1000))
            try:
                self._failed(kind)
            except ProviderUnavailable as stop:
                raise stop from exc
            raise
        quota = quota_info(resp.body) if resp.status == 429 else None
        if 200 <= resp.status < 300 or quota:
            # a 429 whose quota names were found keeps those names, not the body
            error_body = None
        else:
            error_body = safe_error_body(resp.body, self.env)
        self.events.append(
            HttpEvent(
                seq,
                resp.status,
                None,
                (self.clock() - start) * 1000,
                rate_headers(resp.headers),
                error_body,
                quota,
            )
        )
        if resp.status in (401, 403):
            raise AuthStop(resp.status)
        if self.expect_model_404 and is_model_gone(resp.status, resp.body):
            return resp
        if is_model_gone(resp.status, resp.body):
            raise ModelGone(self.events[-1].error_body or "")
        if 200 <= resp.status < 300:
            self._streak.clear()
            self._any_success = True
        else:
            self._failed(str(resp.status))
        return resp


# ---------------------------------------------------------------- grounded live models

LIST_URLS = {
    "groq": "https://api.groq.com/openai/v1/models",  # UNVERIFIED
    "openrouter": "https://openrouter.ai/api/v1/models",  # UNVERIFIED
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",  # UNVERIFIED
}
MAX_LIST_PAGES = 3


def _list_headers(kind: str, key: str) -> dict[str, str]:
    if kind == "gemini":
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


def _parse_models(kind: str, data: Any) -> tuple[list[dict[str, Any]], str | None]:
    out: list[dict[str, Any]] = []
    if kind == "gemini":
        for m in data.get("models", []):
            mid = str(m["name"]).removeprefix("models/")
            methods = m.get("supportedGenerationMethods") or []
            out.append({"id": mid, "generate": "generateContent" in methods})
        return out, data.get("nextPageToken") or None
    for m in data["data"]:
        entry: dict[str, Any] = {"id": str(m["id"])}
        if kind == "openrouter":
            pr = m.get("pricing") or {}
            entry["free"] = str(pr.get("prompt")) == "0" and str(pr.get("completion")) == "0"
        else:
            entry["active"] = m.get("active")
        out.append(entry)
    return out, None


def configured_ids(settings: RouterSettings, name: str) -> dict[str, str]:
    """task-or-default -> model id for a provider, skipping REPLACE placeholders."""
    return {
        t: m
        for t, m in settings.providers[name].models.items()
        if m and not m.startswith(PLACEHOLDER_PREFIX)
    }


HEADROOM_MAX_TOKENS = (
    16  # a reasoning model still spent 93 tokens in total (measured, Groq gpt-oss-20b)
)


def has_remaining_tokens(rate: Mapping[str, str]) -> bool:
    return any("remaining" in k and "token" in k for k in rate)


def measure_headroom(
    settings: RouterSettings,
    name: str,
    model: str | None,
    env: Mapping[str, str],
    transport: Transport | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """One tiny real chat call so the provider's rate-limit headers can be read.

    Why: a provider's model-list call is free but (observed on Groq, 2026-11-22) carries no
    rate-limit headers; they come on chat calls. This sends a 16-token-capped request ("Reply
    with the word OK.", no dataset content, tagged public) and returns only the limit-related
    headers and how many tokens the call used. The key is never returned. Whether other
    providers send such headers on chat calls is not verified here."""
    cfg = settings.providers[name]
    out: dict[str, Any] = {
        "http_status": None, "error": None, "rate_headers": {}, "tokens_spent": None, "model": None,
    }  # fmt: skip
    key = env.get(cfg.env_key or "")
    if not key:
        out["error"] = "no key"
        return out
    model = model or cfg.models.get("default")
    out["model"] = model
    if not model:
        out["error"] = "name a model first"
        return out
    adapter = build_one(settings, name, env=env, transport=transport or urllib_transport)
    req = LLMRequest(
        task="probe", messages=(("user", "Reply with the word OK."),), json_mode=False,
        max_tokens=HEADROOM_MAX_TOKENS, max_tokens_fixed=True, sensitivity=PUBLIC,
    )  # fmt: skip
    try:
        body = json.dumps(adapter._body(req, model)).encode("utf-8")  # noqa: SLF001
        resp = adapter.transport(adapter._url(model), adapter._headers(key), body, timeout)  # noqa: SLF001
    except TransportTimeout:
        out["error"] = "timeout"
        return out
    except TransportError as exc:
        out["error"] = f"network: {exc}"
        return out
    out["http_status"] = resp.status
    out["rate_headers"] = rate_headers(resp.headers)
    if resp.status != 200:
        out["error"] = _error_text(resp)
        return out
    try:
        used = json.loads(resp.body.decode("utf-8")).get("usage", {})
        out["tokens_spent"] = used.get("total_tokens")
    except (ValueError, AttributeError):
        pass
    return out


def list_provider_models(
    settings: RouterSettings,
    name: str,
    env: Mapping[str, str],
    getter: Getter | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    getter = getter or urllib_get
    """Result for one provider: status, model ids, rate-limit headers, configured-id check."""
    cfg = settings.providers[name]
    res: dict[str, Any] = {
        "provider": name,
        "env_key": cfg.env_key,
        "key_present": bool(env.get(cfg.env_key or "")),
        "http_status": None,
        "error": None,
        "models": [],
        "rate_headers": {},
        "requests": 0,
        "configured": {},
    }
    if not res["key_present"]:
        return res
    key = env[cfg.env_key]  # type: ignore[index]
    url, token = LIST_URLS[cfg.kind], None
    for _ in range(MAX_LIST_PAGES):
        page_url = url if token is None else f"{url}&pageToken={token}"
        res["requests"] += 1
        try:
            resp: HttpResponse = getter(page_url, _list_headers(cfg.kind, key), timeout)
        except TransportTimeout:
            res["error"] = "timeout"
            break
        except TransportError as exc:
            res["error"] = f"network: {exc}"
            break
        res["http_status"] = resp.status
        res["rate_headers"].update(rate_headers(resp.headers))
        if resp.status != 200:
            res["error"] = _error_text(resp)
            break
        try:
            models, token = _parse_models(cfg.kind, json.loads(resp.body.decode("utf-8")))
        except (ValueError, KeyError, TypeError, AttributeError):
            res["error"] = "unexpected response shape"
            break
        res["models"].extend(models)
        if not token:
            break
    ids = {m["id"] for m in res["models"]}
    for task, mid in configured_ids(settings, name).items():
        res["configured"][task] = {"id": mid, "present": mid in ids}
    return res


def _error_text(resp: HttpResponse) -> str:
    """HTTP status plus the provider's error message if it is JSON. Never the request."""
    msg = ""
    try:
        err = json.loads(resp.body.decode("utf-8")).get("error")
        msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
    except (ValueError, AttributeError, UnicodeDecodeError):
        pass
    return f"HTTP {resp.status}" + (f": {msg[:200]}" if msg else "")


def render_models(results: list[dict[str, Any]], show_all: bool = False) -> str:
    lines: list[str] = []
    for r in results:
        lines.append(f"== {r['provider']} ({r['env_key']}) ==")
        if not r["key_present"]:
            lines += [f"skipped: environment variable {r['env_key']} is not set", ""]
            continue
        if r["error"]:
            lines.append(f"FAILED after {r['requests']} request(s): {r['error']}")
        else:
            lines.append(f"HTTP {r['http_status']}: {len(r['models'])} models listed")
        if r["rate_headers"]:
            lines.append("rate-limit headers seen:")
            lines += [f"  {k}: {v}" for k, v in sorted(r["rate_headers"].items())]
        else:
            lines.append("rate-limit headers seen: none")
        for task, c in r["configured"].items():
            lines.append(
                f"configured model ({task}): {c['id']} -> {'PRESENT' if c['present'] else 'ABSENT'}"
            )
        if not r["configured"]:
            lines.append("configured model: none set in config/router.yaml (placeholder)")
        models = r["models"]
        if r["provider"] and models and "generate" in models[0]:
            models = [m for m in models if m["generate"]]
            lines.append(f"models supporting generateContent: {len(models)}")
        elif models and "free" in models[0] and not show_all:
            free = [m for m in models if m["free"]]
            lines.append(
                f"zero-price models: {len(free)} of {len(models)} (use --all for every ID)"
            )
            models = free
        lines += [f"  {m['id']}" for m in sorted(models, key=lambda m: m["id"])]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- grounded live probe


@dataclass
class ProbeOptions:
    provider: str
    model: str
    out_dir: Path
    max_requests: int | None = None  # None: the config default (25); at most HARD_MAX_REQUESTS
    token_cap: int | None = None
    max_tokens: int | None = None
    quota_check: bool = False
    record: bool = True
    reasoning_effort: str | None = None  # overrides the config value for this run


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "model"


def default_out_dir(provider: str, model: str, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "live_runs" / f"{stamp}_{provider}_{slug(model)}"


def probe_settings(base: RouterSettings, opts: ProbeOptions) -> RouterSettings:
    """A copy of the router config pinned to ONE provider and ONE model, with no fallback."""
    if opts.provider not in base.providers:
        raise ConfigError(f"unknown provider {opts.provider!r}; known: {', '.join(base.providers)}")
    cfg = base.providers[opts.provider]
    if cfg.kind == "fake" or not cfg.hosted:
        raise ConfigError(f"{opts.provider} is not a hosted provider; probe is for hosted ones")
    if not opts.model or opts.model.startswith(PLACEHOLDER_PREFIX):
        raise ConfigError("give a real model ID (see `grounded live models`)")
    s = base.model_copy(deep=True)
    s.providers[opts.provider].models = {"default": opts.model}
    s.tasks = {t: [opts.provider] for t in ("plan", "compose", "repair")}
    if opts.max_requests is not None:
        if not 1 <= opts.max_requests <= HARD_MAX_REQUESTS:
            raise ConfigError(f"--max-requests must be between 1 and {HARD_MAX_REQUESTS}")
        s.probe.max_requests = opts.max_requests
    # one provider, no fallback: waiting for the token bucket is better than abstaining
    s.bucket.max_wait_s = min(s.bucket.max_wait_s, PROBE_BUCKET_MAX_WAIT_S)
    if opts.reasoning_effort:
        if s.providers[opts.provider].kind != "groq":
            raise ConfigError("--reasoning-effort is only wired for groq")
        s.providers[opts.provider].reasoning_effort = opts.reasoning_effort
    if opts.token_cap:
        s.probe.token_cap = min(opts.token_cap, s.probe.token_cap)
    return s


def quota_check(
    provider: Any, model: str, counting: CountingTransport, timeout: float
) -> dict[str, Any]:
    """Three requests: good, deliberately invalid (unknown model ID), good. Reports the raw
    rate-limit headers after each. Whether a failed request used quota is only observable if the
    provider reports a remaining-requests header; otherwise this says so."""
    good = LLMRequest(
        task="plan",
        messages=(("user", 'Reply with the JSON object {"ok": true}'),),
        max_tokens=64,
        sensitivity=PUBLIC,
    )
    steps = []
    for label, mid in (
        ("good_1", model),
        ("invalid_model", "va-invalid-model-id"),
        ("good_2", model),
    ):
        before = counting.count
        outcome = "ok"
        counting.expect_model_404 = label == "invalid_model"
        try:
            provider.complete(good, mid, timeout)
        except ProviderError as exc:
            outcome = exc.kind
        finally:
            counting.expect_model_404 = False
        ev = counting.events[before] if counting.count > before else None
        steps.append(
            {
                "step": label,
                "outcome": outcome,
                "http_status": ev.status if ev else None,
                "rate_headers": ev.rate_headers if ev else {},
                "error_body": ev.error_body if ev else None,
            }
        )
    rem = [
        {k: v for k, v in s["rate_headers"].items() if "remaining" in k and "request" in k}
        for s in steps
    ]
    observable = bool(rem[0]) and bool(rem[2]) and rem[0].keys() == rem[2].keys()
    return {
        "steps": steps,
        "observable": observable,
        "remaining_request_headers": rem,
        "how_to_read": (
            "If the invalid request did not use quota, the remaining-requests value after good_2 "
            "is about one lower than after good_1; if it did, about two lower. Limits also refill "
            "with time, so a small gap is inconclusive."
            if observable
            else "No remaining-requests header was reported on both good requests, so whether a "
            "failed request counts toward quota is NOT observable from this run."
        ),
    }


def _events_by(runs: list[QuestionRun], step: tuple[str, ...]) -> list[dict[str, Any]]:
    return [e for r in runs for e in r.events if e["step"] in step]


def build_rates(completed: list[QuestionRun]) -> list[dict[str, Any]]:
    plans = _events_by(completed, ("plan",))
    plan_valid = [e for e in plans if e["schema_ok"] and not e["abstained"]]
    composes = _events_by(completed, ("compose", "repair"))
    first = _events_by(completed, ("compose",))
    valid_compose = [e for e in composes if e["schema_ok"]]
    valid_first = [e for e in first if e["schema_ok"]]
    n = len(completed)
    outcomes = [r.outcome for r in completed]
    rates = [
        rate("plan responses that parse as JSON", sum(e["json_ok"] for e in plans), len(plans)),
        rate(
            "plan responses valid against the PlanOutput schema",
            sum(e["schema_ok"] for e in plans),
            len(plans),
        ),  # fmt: skip
        rate(
            "schema-valid plans that compiled and executed",
            sum(bool(e["exec_ok"]) for e in plan_valid),
            len(plan_valid),
        ),  # fmt: skip
        rate(
            "compose/repair responses that parse as JSON",
            sum(e["json_ok"] for e in composes),
            len(composes),
        ),  # fmt: skip
        rate(
            "compose/repair responses valid against the ComposeOutput schema",
            sum(e["schema_ok"] for e in composes),
            len(composes),
        ),  # fmt: skip
        rate(
            "first compose answers that typed a digit outside a placeholder",
            sum(bool(e["typed_digit"]) for e in valid_first),
            len(valid_first),
        ),  # fmt: skip
        rate(
            "all schema-valid compose/repair answers that typed a digit outside a placeholder",
            sum(bool(e["typed_digit"]) for e in valid_compose),
            len(valid_compose),
        ),  # fmt: skip
        rate(
            "questions that needed at least one repair (plan or compose)",
            sum(r.repairs_used > 0 for r in completed),
            n,
        ),  # fmt: skip
        rate("questions that abstained", outcomes.count("abstain"), n),
        rate("questions partially verified", outcomes.count("partially_verified"), n),
        rate(
            "questions with every number verified (NOT an accuracy figure)",
            outcomes.count("verified"),
            n,
        ),  # fmt: skip
    ]
    return rates


def _headers_summary(events: list[HttpEvent]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in events:
        for k, v in e.rate_headers.items():
            d = out.setdefault(k, {"first": v, "last": v, "seen": 0})
            d["last"] = v
            d["seen"] += 1
    return out


def build_report(
    opts: ProbeOptions,
    s: RouterSettings,
    layer: SemanticLayer,
    runs: list[RunContext],
    completed: list[QuestionRun],
    cut: list[str],
    counting: CountingTransport,
    stop_reason: str,
    quota: dict[str, Any] | None,
    started: datetime,
) -> dict[str, Any]:
    agg = merge_summaries(runs) if runs else None
    by_status: dict[str, int] = {}
    for e in counting.events:
        k = str(e.status) if e.status is not None else f"({e.error})"
        by_status[k] = by_status.get(k, 0) + 1
    lat = [e.latency_ms for e in counting.events if e.status == 200]
    retry_after = [
        {"seq": e.seq, "status": e.status, "value": v}
        for e in counting.events
        for k, v in e.rate_headers.items()
        if k == "retry-after"
    ]
    return {
        "header": {
            "provider": opts.provider,
            "model": opts.model,
            "started_utc": started.isoformat(timespec="seconds"),
            "mode": "live probe",
            "dataset": layer.source.relation,
            "dataset_sensitivity": s.sensitivity_of((layer.source.relation,)),
            "prompts": active_prompt_hashes(),
            "caps": {
                "max_requests": s.probe.max_requests,
                "token_cap": s.probe.token_cap,
                "max_tokens": opts.max_tokens or 1024,
            },
            "bucket": {
                "fraction": s.bucket.fraction,
                "tokens_per_minute": s.providers[opts.provider].tpm,
                "requests_per_minute": s.providers[opts.provider].rpm,
                "max_wait_s": s.bucket.max_wait_s,
            },
            "reasoning_effort_sent": (
                s.providers[opts.provider].reasoning_effort if "gpt-oss" in opts.model else None
            ),
            "scope": "one model, public data only; raw results; no accuracy claim, no ranking",
            "prompt_note": "prompts are first drafts, not tuned or proven "
            f"({', '.join(sorted(active_prompt_hashes()))}): "
            "results describe this model with these prompts together",
        },
        "stop": {"reason": stop_reason, "questions_completed": len(completed), "cut": cut},
        "http": {
            "requests_sent": counting.count,
            "by_status": by_status,
            "retry_after_headers": retry_after,
            "quota_429": [
                {"seq": e.seq, **e.quota, "reading": quota_reading(e.quota)}
                for e in counting.events
                if e.quota
            ],
            "error_bodies": [
                {"seq": e.seq, "status": e.status, "body": e.error_body}
                for e in counting.events
                if e.error_body is not None
            ],
            "latency_ms_p50_http200": percentile(lat, 50),
            "latency_ms_p95_http200": percentile(lat, 95),
            "latency_n": len(lat),
            "rate_headers": _headers_summary(counting.events),
        },
        "accounting": agg,
        "tokens_spent": usage.tokens_spent(runs),
        "rates": build_rates(completed),
        "questions": [
            {
                "id": r.question,
                "outcome": r.outcome,
                "reason": r.reason,
                "repairs_used": r.repairs_used,
                "plan_failures": r.plan_failures,
                "attempt_statuses": [a.status for a in r.attempts],
                "model_responses": len(r.events),
                "abstain_detail": r.abstain_detail,
            }
            for r in completed
        ],
        "abstains": abstain_section(layer, completed),
        "quota_check": quota,
    }


def abstain_section(layer: SemanticLayer, completed: list[QuestionRun]) -> dict[str, Any]:
    """Every abstain with the words behind it, and what the plan prompt's catalogue held. The
    catalogue text is exactly what `plan_request` puts in the prompt (names, units,
    descriptions, synonyms, allowed dimensions), so it is the same for every question."""
    text = catalogue_text(layer)
    cat = json.loads(text)
    items = [
        {"question": r.question, "reason": r.reason, **(r.abstain_detail or {})}
        for r in completed
        if r.outcome == "abstain"
    ]
    return {
        "plan_prompt_id": plan_request(layer, "x").prompt_id,
        "catalogue_sha256": sha256_text(text),
        "catalogue_metrics": sorted(cat["metrics"]),
        "catalogue_dimensions": sorted(cat["dimensions"]),
        "catalogue": cat,
        "items": items,
    }


def render_quota(q: dict[str, Any]) -> str:
    parts = [
        f"{k}={q[k]}"
        for k in ("status", "type", "code", "limit_on", "limit_name", "limit_value", "retry_delay")
        if k in q
    ]
    for v in q.get("violations", []):
        parts.append("violation(" + ", ".join(f"{k}={v[k]}" for k in sorted(v)) + ")")
    if "quota_metric" in q:
        parts.append(f"quota_metric={q['quota_metric']}")
    parts.append(f"reading: {q['reading']}")
    return "; ".join(parts)


def render_report(rep: dict[str, Any]) -> str:
    h, st, ht = rep["header"], rep["stop"], rep["http"]
    f = lambda v: "n/a" if v is None else f"{v:.0f}"  # noqa: E731
    lines = [
        f"# Live probe: {h['provider']} / {h['model']}",
        f"Started {h['started_utc']}. Dataset {h['dataset']} ({h['dataset_sensitivity']}).",
        f"Scope: {h['scope']}.",
        f"Note: {h['prompt_note']}.",
        f"Caps: {h['caps']['max_requests']} requests, {h['caps']['token_cap']} tokens, "
        f"max_tokens {h['caps']['max_tokens']} per call.",
        f"Client-side buckets at {h['bucket']['fraction']:.0%} of the configured limits "
        f"(tokens/minute {h['bucket']['tokens_per_minute'] or 'not set'}, "
        f"requests/minute {h['bucket']['requests_per_minute'] or 'not set'}); waits up to "
        f"{h['bucket']['max_wait_s']:g} s for a bucket. reasoning_effort sent: "
        f"{h['reasoning_effort_sent'] or 'no (not configured)'}.",
        *(
            ["", f"**{st['reason']}**"]
            if st.get("kind") in ("provider_unavailable", "model_gone", "auth")
            else []
        ),
        "",
        "## Run",
        f"- stopped: {st['reason']}; questions completed (n={st['questions_completed']})"
        + (f"; cut by a cap: {', '.join(st['cut'])}" if st["cut"] else ""),
        f"- HTTP requests sent: {ht['requests_sent']}; by status: {ht['by_status']}",
        f"- retry-after headers: {ht['retry_after_headers'] or 'none seen'}",
        *(
            [
                "- HTTP 429 quota details (names and limits only; the body is not kept; the "
                "reading is a keyword match on those names):"
            ]
            + [f"  - request {q['seq']}: {render_quota(q)}" for q in ht.get("quota_429", [])]
            if ht.get("quota_429")
            else []
        ),
        *(
            [
                "- non-2xx response bodies (provider text only; redacted; cut at "
                f"{ERROR_BODY_CHARS} chars):"
            ]
            + [
                f"  - request {b['seq']}, HTTP {b['status']}: {b['body'] or '(empty body)'}"
                for b in ht.get("error_bodies", [])
            ]
            if ht.get("error_bodies")
            else ["- non-2xx response bodies: none"]
        ),
        f"- HTTP-200 latency, request to response: p50 {f(ht['latency_ms_p50_http200'])} ms, "
        f"p95 {f(ht['latency_ms_p95_http200'])} ms (n={ht['latency_n']})",
        "",
        "## Rates (each with n and a 95% Wilson interval; verified is not accuracy)",
        *[render_rate(r) for r in rep["rates"]],
        "",
    ]
    a = rep["accounting"]
    if a:
        lines += [
            "## Tokens",
            f"- prompt {a['prompt_tokens']}, completion {a['completion_tokens']}, "
            f"reasoning reported {a['reasoning_tokens_reported']} "
            "(0 may mean not reported; it may or may not be inside completion), "
            f"total counted {a['tokens_used']}; estimated-usage calls: "
            f"{a['tokens_estimated_calls']}",
            f"- finish reasons over {a['calls_ok']} successful calls: {a['finish_reasons']}",
            f"- latency over successful calls (n={a['latency_n']}): "
            f"p50 {f(a['latency_ms_p50'])} ms, p95 {f(a['latency_ms_p95'])} ms",
            f"- errors by provider: { {k: v['errors'] for k, v in a['by_provider'].items()} }",
            "",
        ]
    lines.append("## Rate-limit headers seen (first -> last, count)")
    if ht["rate_headers"]:
        lines += [f"- {k}: {v['first']} -> {v['last']} ({v['seen']}x)"
                  for k, v in sorted(ht["rate_headers"].items())]  # fmt: skip
    else:
        lines.append("- none")
    lines += ["", "## Questions (raw)"]
    lines += [
        f"- {q['id'][:60]}: {q['outcome']}{' (' + q['reason'] + ')' if q['reason'] else ''}, "
        f"repairs {q['repairs_used']}, plan failures {q['plan_failures']}, "
        f"attempts {q['attempt_statuses']}"
        for q in rep["questions"]
    ]
    ab = rep["abstains"]
    if ab["items"]:
        lines += [
            "",
            "## Abstains (raw)",
            f"The plan prompt ({ab['plan_prompt_id']}) holds a catalogue of metric names, units, "
            "descriptions, synonyms and allowed dimensions (from semantic/metrics.yaml); it is the "
            "same for every "
            f"question (sha256 {ab['catalogue_sha256'][:12]}).",
            f"- catalogue metrics in the prompt ({len(ab['catalogue_metrics'])}): "
            + ", ".join(ab["catalogue_metrics"]),
            f"- catalogue dimensions in the prompt ({len(ab['catalogue_dimensions'])}): "
            + ", ".join(ab["catalogue_dimensions"]),
        ]
        for it in ab["items"]:
            who = (
                f"the planner replied {it['planner_reply']!r}"
                if it.get("source") == "planner"
                else f"not the planner; last model step: {it.get('last_model_step')}"
            )
            lines.append(f"- {it['question']}: reason code {it['reason']!r}; {who}")
    qc = rep["quota_check"]
    if qc:
        lines += ["", "## Does a failed request use quota? (raw)"]
        lines += [f"- {s['step']}: {s['outcome']} (HTTP {s['http_status']}) {s['rate_headers']}"
                  for s in qc["steps"]]  # fmt: skip
        lines.append(f"- observable: {qc['observable']}. {qc['how_to_read']}")
    return "\n".join(lines) + "\n"


def early_stop(exc: Exception) -> tuple[str, str]:
    """The stop line and kind for a provider-unavailable or model-gone stop."""
    if isinstance(exc, ModelGone):
        body = exc.body or "(empty body)"
        return (
            "STOPPED on the first HTTP 404 saying the model is not found or no longer "
            f"available: the configured model ID is probably wrong or retired. Provider said: "
            f"{body}",
            "model_gone",
        )
    assert isinstance(exc, ProviderUnavailable)
    if not exc.any_success:
        head = "STOPPED: provider unavailable, nothing measured"
    else:
        head = (
            "STOPPED: provider unavailable; the numbers below cover only what was measured "
            "before it stopped"
        )
    return f"{head} ({exc})", "provider_unavailable"


def run_probe(
    settings: RouterSettings,
    opts: ProbeOptions,
    *,
    layer: SemanticLayer,
    readonly_url: str,
    questions: list[str],
    known_values: dict[str, set[str]] | None,
    env: Mapping[str, str],
    inner_transport: Transport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the fixed battery against one model. Returns {"report", "markdown", "files", ...}."""
    s = probe_settings(settings, opts)
    if s.sensitivity_of((layer.source.relation,)) != PUBLIC:
        raise SensitivityRefusal(
            f"refusing to probe: {layer.source.relation} is not tagged public in config/router.yaml"
        )
    cfg = s.providers[opts.provider]
    if not env.get(cfg.env_key or ""):
        raise ConfigError(f"environment variable {cfg.env_key} is not set")
    counting = CountingTransport(inner_transport or urllib_transport, s.probe.max_requests, env=env)
    provider = build_one(s, opts.provider, transport=counting, env=env)
    cassette: Cassette | None = None
    ctx: CassetteContext | None = None
    if opts.record:
        ctx = CassetteContext(layer_hash(), schema_hash(), warehouse_content_hash(readonly_url))
        cassette = Cassette(opts.out_dir / "cassettes", "record", ctx, env=env)
    router = Router(s, {opts.provider: provider}, cassette=cassette, clock=clock, sleep=_sleep)
    started = datetime.now(UTC)
    runs: list[RunContext] = []
    completed: list[QuestionRun] = []
    cut: list[str] = []
    quota: dict[str, Any] | None = None
    stop = "all questions run"
    stop_kind: str | None = None
    used_tokens = 0
    try:
        if opts.quota_check:
            quota = quota_check(provider, opts.model, counting, cfg.timeout_s)
        for q in questions:
            left = s.probe.token_cap - used_tokens
            if counting.count + 2 > s.probe.max_requests:
                stop = "request cap: not enough requests left for another question"
                break
            if left <= 0:
                stop = "token cap reached"
                break
            run = router.new_run(q[:40], token_cap=left)
            try:
                res = run_question(
                    q, layer=layer, readonly_url=readonly_url, router=router, run=run,
                    known_values=known_values, max_tokens=opts.max_tokens,
                )  # fmt: skip
            except RequestCapReached:
                runs.append(run)
                cut.append(q)
                stop = "request cap reached mid-question"
                break
            except (ProviderUnavailable, ModelGone) as exc:
                runs.append(run)
                cut.append(q)
                stop, stop_kind = early_stop(exc)
                break
            runs.append(run)
            used_tokens += run.tokens_used
            if res.outcome == "abstain" and res.reason == "budget":
                cut.append(q)
                stop = "token cap reached mid-question"
                break
            completed.append(res)
    except AuthStop as exc:
        stop = f"STOPPED on first auth error ({exc})"
        stop_kind = "auth"
    except (ProviderUnavailable, ModelGone) as exc:
        stop, stop_kind = early_stop(exc)
    except RequestCapReached:
        stop = "request cap reached"
    rep = build_report(opts, s, layer, runs, completed, cut, counting, stop, quota, started)
    rep["stop"]["kind"] = stop_kind
    md = render_report(rep)
    out: dict[str, Any] = {
        "report": rep,
        "markdown": md,
        "stopped_on_auth": stop_kind == "auth",
        "stopped_early": stop_kind in ("provider_unavailable", "model_gone"),
    }
    out.update(write_outputs(opts.out_dir, rep, md, cassette, ctx, env))
    return out


def write_outputs(
    out_dir: Path,
    rep: dict[str, Any],
    md: str,
    cassette: Cassette | None,
    ctx: CassetteContext | None,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Scan everything for credentials, then write the report files. Returns the scan result."""
    text_json = json.dumps(rep, indent=2, sort_keys=True, default=str)
    leak = scan_for_secrets(text_json + md, env)
    if leak:
        raise ConfigError(f"refusing to write the report: it contains a credential ({leak})")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(text_json, encoding="utf-8")
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    result: dict[str, Any] = {"out_dir": str(out_dir), "cassette": None}
    if cassette is not None and ctx is not None:
        cas_dir = out_dir / "cassettes"
        raw = (
            (cas_dir / "calls.jsonl").read_text("utf-8")
            if (cas_dir / "calls.jsonl").exists()
            else ""
        )
        leak = scan_for_secrets(raw, env)
        check = check_cassettes(
            cas_dir,
            {"layer": ctx.layer_hash, "schema": ctx.schema_hash, "warehouse": ctx.warehouse_hash},
            prompt_hashes(),
        )
        result["cassette"] = {
            "dir": str(cas_dir),
            "entries": check["entries"],
            "skipped_internal": cassette.skipped_internal,
            "key_scan": "FOUND A CREDENTIAL: " + leak if leak else "clean",
            "check_ok": check["ok"],
        }
    return result


def default_questions_path() -> Path:
    return Path(__file__).resolve().parents[3] / "eval" / "questions" / "dev.yaml"


def load_probe_questions(path: Path | None = None, limit: int = 10) -> list[str]:
    """The first `limit` T1 dev questions that are expressible as a MetricQuery (author-written)."""
    import yaml  # noqa: PLC0415

    try:
        data = yaml.safe_load((path or default_questions_path()).read_text("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read the dev questions: {exc}") from exc
    return [q["question"] for q in data["questions"] if q.get("expressible_as_metric_query")][
        :limit
    ]
