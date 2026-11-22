"""A daily token tally from our own reports, a token budget estimate, and the guard that refuses to
start a run that would go over the daily cap.

The provider does not report what is left of a daily limit (assumptions row 53: Groq's 429 body
names it only once it is hit). So this module adds up what OUR reports say was spent in the last
24 hours for the same provider and model. It undercounts whatever is not in a report: other tools
using the same key, a run whose report was deleted, requests refused before they were answered. It
is a planning aid, not the provider's number.

Sources, per run folder:
- `report.json` under data/live_runs: `tokens_spent.total` (every call that reported usage,
  replays from cassettes excluded) when the report has it; older reports only have
  `accounting.tokens_used` (successful calls only) and are marked approximate.
- the cassettes of data/binder_model_labels_work (one folder, resumed across runs, so its
  `report.json` only covers the last run): every recorded call, by its `recorded_at`.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import ConfigError
from .accounting import RunContext

ROOT = Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    """data/ under the repository, or the folder in GA_DATA_DIR (the test suite points this at an
    empty folder so that a developer's own reports never change a test result)."""
    env = os.environ.get("GA_DATA_DIR")
    return Path(env) if env else ROOT / "data"


def default_report_roots() -> tuple[Path, ...]:
    return (_data_dir() / "live_runs",)


def default_cassette_roots() -> tuple[Path, ...]:
    return (_data_dir() / "binder_model_labels_work",)


WINDOW_H = 24
ESTIMATE_REPORTS = 3  # the most recent reports of the same provider, model and mode

CAP_SOURCE_MEASURED = "measured for this model (assumptions row 53)"
CAP_SOURCE_ASSUMED = "UNVERIFIED for this model: the documented provider value in config"


def tokens_spent(runs: Sequence[RunContext]) -> dict[str, Any]:
    """Every call record that reported usage, except replays from a cassette (no request was
    sent for those). Includes unusable HTTP 200 replies, which use the daily allowance too."""
    prompt = sum(c.prompt_tokens for r in runs for c in r.records if c.outcome != "cassette")
    comp = sum(c.completion_tokens for r in runs for c in r.records if c.outcome != "cassette")
    return {"prompt": prompt, "completion": comp, "total": prompt + comp,
            "basis": "calls that reported usage, cassette replays excluded"}  # fmt: skip


def _when(text: str | None, fallback: float) -> datetime:
    if text:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback, UTC)


@dataclass
class Tally:
    provider: str
    model: str
    total: int = 0
    reports: int = 0
    approximate: int = 0  # reports that only had accounting.tokens_used
    items: list[dict[str, Any]] = field(default_factory=list)


def _report_tokens(rep: dict[str, Any]) -> tuple[int | None, bool]:
    spent = rep.get("tokens_spent")
    if isinstance(spent, dict) and isinstance(spent.get("total"), int):
        return spent["total"], False
    acc = rep.get("accounting")
    if isinstance(acc, dict) and isinstance(acc.get("tokens_used"), int):
        return acc["tokens_used"], True
    return None, False


def _identity(rep: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    h = rep.get("header") or rep.get("meta") or {}
    return h.get("provider"), h.get("model"), h.get("started_utc") or h.get("at")


def collect_tally(
    provider: str,
    model: str,
    *,
    report_roots: Sequence[Path] | None = None,
    cassette_roots: Sequence[Path] | None = None,
    now: datetime | None = None,
    hours: float = WINDOW_H,
) -> Tally:
    report_roots = default_report_roots() if report_roots is None else report_roots
    cassette_roots = default_cassette_roots() if cassette_roots is None else cassette_roots
    now = now or datetime.now(UTC)
    since = now - timedelta(hours=hours)
    t = Tally(provider, model)
    for root in report_roots:
        for f in sorted(root.rglob("report.json")) if root.is_dir() else []:
            try:
                rep = json.loads(f.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            p, m, started = _identity(rep)
            if (p, m) != (provider, model):
                continue
            at = _when(started, f.stat().st_mtime)
            total, approx = _report_tokens(rep)
            if total is None or at < since or at > now + timedelta(minutes=5):
                continue
            t.total += total
            t.reports += 1
            t.approximate += approx
            t.items.append({"folder": f.parent.name, "at": at.isoformat(timespec="seconds"),
                            "tokens": total, "approximate": approx})  # fmt: skip
    for root in cassette_roots:
        for f in sorted(root.rglob("calls.jsonl")) if root.is_dir() else []:
            spent = 0
            for line in f.read_text("utf-8").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if (e.get("provider"), e.get("model")) != (provider, model):
                    continue
                at = _when(e.get("recorded_at"), f.stat().st_mtime)
                if since <= at <= now + timedelta(minutes=5):
                    tok = e.get("tokens") or {}
                    spent += int(tok.get("prompt", 0)) + int(tok.get("completion", 0))
            if spent:
                t.total += spent
                t.reports += 1
                t.items.append({"folder": f"{f.parent.parent.name} (cassettes)", "at": "",
                                "tokens": spent, "approximate": False})  # fmt: skip
    return t


def cap_for(settings: Any, provider: str, model: str) -> tuple[int | None, str]:
    p = settings.providers.get(provider)
    if p is None:
        return None, "unknown provider"
    if model in p.daily_tokens:
        return p.daily_tokens[model], CAP_SOURCE_MEASURED
    if p.tpd:
        return p.tpd, CAP_SOURCE_ASSUMED
    return None, "no daily cap in config for this provider"


@dataclass
class Estimate:
    per_question: float | None
    reports_used: int
    questions_planned: int
    run_tokens: int | None  # None: no earlier report to estimate from
    note: str


def estimate_run(
    provider: str,
    model: str,
    mode: str,
    questions_planned: int,
    *,
    report_roots: Sequence[Path] | None = None,
    multiplier: int = 1,
) -> Estimate:
    """Tokens per question from the last reports of the same provider, model and mode (the median
    of up to 3), times the questions planned (times `multiplier` runs)."""
    report_roots = default_report_roots() if report_roots is None else report_roots
    found: list[tuple[str, float]] = []
    for root in report_roots:
        for f in sorted(root.rglob("report.json")) if root.is_dir() else []:
            try:
                rep = json.loads(f.read_text("utf-8"))
            except (OSError, ValueError):
                continue
            p, m, started = _identity(rep)
            if (p, m) != (provider, model) or (rep.get("header") or {}).get("mode", "") != mode:
                continue
            total, _ = _report_tokens(rep)
            stop = rep.get("stop") if isinstance(rep.get("stop"), dict) else {}
            n = sum(int(v) for k, v in stop.items() if k.endswith("questions_run"))
            if not n:
                n = len(rep.get("questions") or [])
            if total and n:
                found.append((_when(started, f.stat().st_mtime).isoformat(), total / n))
    used = [v for _, v in sorted(found)[-ESTIMATE_REPORTS:]]
    if not used:
        return Estimate(None, 0, questions_planned, None,
                        f"no earlier report for {provider}/{model} in mode '{mode}'")  # fmt: skip
    per_q = statistics.median(used)
    return Estimate(per_q, len(used), questions_planned,
                    round(per_q * questions_planned * multiplier),
                    f"median of {len(used)} earlier report(s) in mode '{mode}'")  # fmt: skip


def usage_lines(tally: Tally, cap: int | None, source: str, est: Estimate | None = None,
                ceiling: int | None = None) -> list[str]:  # fmt: skip
    cap_txt = f"{cap}" if cap is not None else "an unknown cap"
    lines = [
        f"about {tally.total} of {cap_txt} tokens used in the last {WINDOW_H:g} h "
        f"({tally.provider}/{tally.model}), from our own reports ({tally.reports} report(s)"
        + (f", {tally.approximate} approximate: successful calls only" if tally.approximate else "")
        + "); the provider does not report the daily remainder. Anything not in a report is not "
        "counted.",
        f"daily cap {cap_txt}: {source}.",
    ]
    if est is not None:
        if est.run_tokens is None:
            lines.append(
                f"token budget for this run: no estimate ({est.note})"
                + (f"; the run's own token cap is {ceiling}" if ceiling else "")
            )
        else:
            lines.append(
                f"token budget for this run: about {est.run_tokens} tokens = "
                f"{est.per_question:.0f} per question x {est.questions_planned} questions "
                f"({est.note}; n={est.reports_used}; a run is capped at {ceiling})"
                if ceiling
                else f"token budget for this run: about {est.run_tokens} tokens "
                f"({est.per_question:.0f} per question x {est.questions_planned}; {est.note})"
            )
    return lines


def start(
    settings: Any,
    provider: str,
    model: str,
    *,
    mode: str,
    questions: int,
    ceiling: int | None,
    force: bool,
    say: Callable[[str], None] = print,
    multiplier: int = 1,
    planned_override: int | None = None,
    report_roots: Sequence[Path] | None = None,
    cassette_roots: Sequence[Path] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Print the usage line, then refuse (ConfigError) when the tally plus the planned budget
    would exceed the daily cap and `force` is not set. The planned budget is the estimate from the
    last reports, or, with none, the run's own token cap."""
    report_roots = default_report_roots() if report_roots is None else report_roots
    cassette_roots = default_cassette_roots() if cassette_roots is None else cassette_roots
    tally = collect_tally(provider, model, report_roots=report_roots,
                          cassette_roots=cassette_roots, now=now)  # fmt: skip
    cap, source = cap_for(settings, provider, model)
    est = estimate_run(provider, model, mode, questions, report_roots=report_roots,
                       multiplier=multiplier)  # fmt: skip
    if planned_override is not None:
        est = Estimate(est.per_question, est.reports_used, questions, planned_override,
                       "from the batches still to ask")  # fmt: skip
    for line in usage_lines(tally, cap, source, est, ceiling):
        say(line)
    planned = est.run_tokens if est.run_tokens is not None else (ceiling or 0) * multiplier
    basis = "the estimate" if est.run_tokens is not None else "the run's token cap"
    over = cap is not None and tally.total + planned > cap
    if over and not force:
        raise ConfigError(
            f"refusing to start: about {tally.total} tokens used in the last {WINDOW_H:g} h plus "
            f"{planned} planned ({basis}) is {tally.total + planned}, above the daily cap of "
            f"{cap} ({source}). Wait for the window to pass, lower --token-cap, or pass --force."
        )
    if over:
        say(f"--force given: starting although {tally.total} + {planned} is above {cap}.")
    return {"tally": tally.total, "cap": cap, "cap_source": source, "planned": planned,
            "planned_basis": basis, "forced": bool(over and force)}  # fmt: skip
