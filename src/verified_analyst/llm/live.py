"""First real provider calls, as tooling the owner runs on their own machine.

`list_models` and `run_probe` are the only code in this repository that is meant to reach a real
provider, and only when invoked by `va live ...` with a key in the environment. Tests mock the
transport. Hard limits: at most 25 HTTP requests per probe run (retries included), a token cap
from config, and a stop on the first 401/403. Output is raw per-model numbers with n, never an
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
from .accounting import RunContext, merge_summaries, percentile
from .cassette import (
    KEY_ENV_VARS,
    Cassette,
    CassetteContext,
    check_cassettes,
    layer_hash,
    scan_for_secrets,
    schema_hash,
    warehouse_content_hash,
)
from .pipeline import QuestionRun, run_question
from .prompts import prompt_hashes
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

HARD_MAX_REQUESTS = 25
_sleep: Callable[[float], None] = time.sleep  # tests replace this


# ---------------------------------------------------------------- redaction


def secret_values(env: Mapping[str, str]) -> list[str]:
    return [v for k in KEY_ENV_VARS if (v := env.get(k)) and len(v) >= 8]


def redact(text: str, env: Mapping[str, str]) -> str:
    for v in secret_values(env):
        text = text.replace(v, "<redacted>")
    return text


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


@dataclass
class HttpEvent:
    seq: int
    status: int | None  # None for a timeout or connection error
    error: str | None
    latency_ms: float
    rate_headers: dict[str, str] = field(default_factory=dict)


class CountingTransport:
    """Wraps the real transport: counts every request, enforces the cap, records status codes and
    rate-limit headers for every response (successful or not), and stops on the first 401/403."""

    def __init__(
        self,
        inner: Transport,
        max_requests: int,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.inner = inner
        self.max_requests = min(max_requests, HARD_MAX_REQUESTS)
        self.clock = clock
        self.events: list[HttpEvent] = []

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
            raise
        self.events.append(
            HttpEvent(
                seq, resp.status, None, (self.clock() - start) * 1000, rate_headers(resp.headers)
            )
        )
        if resp.status in (401, 403):
            raise AuthStop(resp.status)
        return resp


# ---------------------------------------------------------------- va live models

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


# ---------------------------------------------------------------- va live probe


@dataclass
class ProbeOptions:
    provider: str
    model: str
    out_dir: Path
    max_requests: int = HARD_MAX_REQUESTS
    token_cap: int | None = None
    max_tokens: int | None = None
    quota_check: bool = False
    record: bool = True


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
        raise ConfigError("give a real model ID (see `va live models`)")
    s = base.model_copy(deep=True)
    s.providers[opts.provider].models = {"default": opts.model}
    s.tasks = {t: [opts.provider] for t in ("plan", "compose", "repair")}
    s.probe.max_requests = min(opts.max_requests, s.probe.max_requests, HARD_MAX_REQUESTS)
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
        try:
            provider.complete(good, mid, timeout)
        except ProviderError as exc:
            outcome = exc.kind
        ev = counting.events[before] if counting.count > before else None
        steps.append(
            {
                "step": label,
                "outcome": outcome,
                "http_status": ev.status if ev else None,
                "rate_headers": ev.rate_headers if ev else {},
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
            "dataset": layer.source.relation,
            "dataset_sensitivity": s.sensitivity_of((layer.source.relation,)),
            "prompts": prompt_hashes(),
            "caps": {
                "max_requests": s.probe.max_requests,
                "token_cap": s.probe.token_cap,
                "max_tokens": opts.max_tokens or 1024,
            },
            "scope": "one model, public data only; raw results; no accuracy claim, no ranking",
            "prompt_note": "prompts are the untuned minimum (plan.v0, compose.v0, repair.v0): "
            "results describe this model with these prompts together",
        },
        "stop": {"reason": stop_reason, "questions_completed": len(completed), "cut": cut},
        "http": {
            "requests_sent": counting.count,
            "by_status": by_status,
            "retry_after_headers": retry_after,
            "latency_ms_p50_http200": percentile(lat, 50),
            "latency_ms_p95_http200": percentile(lat, 95),
            "latency_n": len(lat),
            "rate_headers": _headers_summary(counting.events),
        },
        "accounting": agg,
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
            }
            for r in completed
        ],
        "quota_check": quota,
    }


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
        "",
        "## Run",
        f"- stopped: {st['reason']}; questions completed (n={st['questions_completed']})"
        + (f"; cut by a cap: {', '.join(st['cut'])}" if st["cut"] else ""),
        f"- HTTP requests sent: {ht['requests_sent']}; by status: {ht['by_status']}",
        f"- retry-after headers: {ht['retry_after_headers'] or 'none seen'}",
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
    qc = rep["quota_check"]
    if qc:
        lines += ["", "## Does a failed request use quota? (raw)"]
        lines += [f"- {s['step']}: {s['outcome']} (HTTP {s['http_status']}) {s['rate_headers']}"
                  for s in qc["steps"]]  # fmt: skip
        lines.append(f"- observable: {qc['observable']}. {qc['how_to_read']}")
    return "\n".join(lines) + "\n"


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
    counting = CountingTransport(inner_transport or urllib_transport, s.probe.max_requests)
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
            runs.append(run)
            used_tokens += run.tokens_used
            if res.outcome == "abstain" and res.reason == "budget":
                cut.append(q)
                stop = "token cap reached mid-question"
                break
            completed.append(res)
    except AuthStop as exc:
        stop = f"STOPPED on first auth error ({exc})"
    except RequestCapReached:
        stop = "request cap reached"
    rep = build_report(opts, s, layer, runs, completed, cut, counting, stop, quota, started)
    md = render_report(rep)
    out: dict[str, Any] = {"report": rep, "markdown": md, "stopped_on_auth": "auth error" in stop}
    out.update(write_outputs(opts, rep, md, cassette, ctx, env))
    return out


def write_outputs(
    opts: ProbeOptions,
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
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    (opts.out_dir / "report.json").write_text(text_json, encoding="utf-8")
    (opts.out_dir / "report.md").write_text(md, encoding="utf-8")
    result: dict[str, Any] = {"out_dir": str(opts.out_dir), "cassette": None}
    if cassette is not None and ctx is not None:
        cas_dir = opts.out_dir / "cassettes"
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
