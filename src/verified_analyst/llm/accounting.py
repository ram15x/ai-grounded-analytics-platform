"""Per-call and per-run token and latency accounting (PHASE_1C.md section e).

No accuracy figure is produced here. Percentiles are over the calls in one run and are printed
with their n."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CallRecord:
    seq: int
    task: str
    provider: str
    model: str
    outcome: (
        str  # ok | cassette | rate_limit | server | timeout | ... | refused_sensitivity | budget
    )
    attempt: int = 0  # 0 = first try for this provider
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    reasoning_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    wait_ms: float = 0.0  # time slept before this attempt (bucket or backoff)
    retry_after_s: float | None = None
    from_cassette: bool = False
    max_tokens: int | None = None  # the completion allowance this request carried
    json_mode_off: bool = False  # the retry after a JSON-mode refusal sends JSON mode off

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


@dataclass
class RunContext:
    """One run's budget and ledger. Create one per question (or per eval run)."""

    run_id: str
    token_cap: int
    records: list[CallRecord] = field(default_factory=list)
    tokens_used: int = 0

    def add(self, rec: CallRecord) -> None:
        rec.seq = len(self.records) + 1
        self.records.append(rec)
        if rec.outcome in ("ok", "cassette"):
            self.tokens_used += rec.tokens

    def summary(self) -> dict[str, Any]:
        ok = [r for r in self.records if r.outcome in ("ok", "cassette")]
        by_provider: dict[str, dict[str, Any]] = {}
        for r in self.records:
            p = by_provider.setdefault(
                r.provider,
                {"attempts": 0, "ok": 0, "tokens": 0, "errors": {}, "_lat": []},
            )
            p["attempts"] += 1
            if r.outcome in ("ok", "cassette"):
                p["ok"] += 1
                p["tokens"] += r.tokens
                p["_lat"].append(r.latency_ms)
            else:
                p["errors"][r.outcome] = p["errors"].get(r.outcome, 0) + 1
        for p in by_provider.values():
            lat = p.pop("_lat")
            p["latency_ms_p50"] = percentile(lat, 50)
            p["latency_ms_p95"] = percentile(lat, 95)
            p["latency_n"] = len(lat)
        lat_all = [r.latency_ms for r in ok]
        return {
            "run_id": self.run_id,
            "token_cap": self.token_cap,
            "tokens_used": self.tokens_used,
            "prompt_tokens": sum(r.prompt_tokens for r in ok),
            "completion_tokens": sum(r.completion_tokens for r in ok),
            "reasoning_tokens_reported": sum(r.reasoning_tokens for r in ok),
            "finish_reasons": _count(r.finish_reason or "(none)" for r in ok),
            "calls_ok": len(ok),
            "attempts": len(self.records),
            "tokens_estimated_calls": sum(r.tokens_estimated for r in ok),
            "slept_ms": sum(r.wait_ms for r in self.records),
            "latency_ms_p50": percentile(lat_all, 50),
            "latency_ms_p95": percentile(lat_all, 95),
            "latency_n": len(lat_all),
            "by_provider": by_provider,
            "cost_usd": 0,
            "cost_note": "0 on free tiers; no would-be paid estimate is computed in this phase",
        }

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary(), "calls": [asdict(r) for r in self.records]}


def merge_summaries(runs: list[RunContext]) -> dict[str, Any]:
    """Aggregate several runs (for an eval). Latencies are pooled over successful calls."""
    pooled = RunContext("aggregate", token_cap=sum(r.token_cap for r in runs))
    for r in runs:
        for rec in r.records:
            pooled.add(CallRecord(**{**asdict(rec), "seq": 0}))
    out = pooled.summary()
    out["runs"] = len(runs)
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f}"

    lines = [
        f"Run `{summary['run_id']}`: {summary['calls_ok']} successful calls in "
        f"{summary['attempts']} attempts, {summary['tokens_used']} tokens "
        f"(cap {summary['token_cap']}).",
        f"Latency over successful calls (n={summary['latency_n']}): "
        f"p50 {f(summary['latency_ms_p50'])} ms, p95 {f(summary['latency_ms_p95'])} ms.",
        f"Cost: {summary['cost_usd']} ({summary['cost_note']}).",
        "",
        "| Provider | Attempts | OK | Tokens | p50 ms | p95 ms | n | Errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, p in sorted(summary["by_provider"].items()):
        errs = ", ".join(f"{k}={v}" for k, v in sorted(p["errors"].items())) or "none"
        lines.append(
            f"| {name} | {p['attempts']} | {p['ok']} | {p['tokens']} | "
            f"{f(p['latency_ms_p50'])} | {f(p['latency_ms_p95'])} | {p['latency_n']} | {errs} |"
        )
    return "\n".join(lines)
