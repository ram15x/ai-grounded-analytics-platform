"""The daily token tally, the budget estimate and the start guard. Invented report folders only."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_helpers import settings
from verified_analyst.config import ConfigError
from verified_analyst.llm import usage
from verified_analyst.llm.accounting import CallRecord, RunContext
from verified_analyst.llm.settings import load_router_settings

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
M = "openai/gpt-oss-20b"


def report(
    root: Path,
    name: str,
    *,
    hours_ago: float,
    tokens: int | None = None,
    used: int | None = None,
    mode="route plus full answer",
    model=M,
    questions=(12, 5),
):
    d = root / name
    d.mkdir(parents=True)
    rep = {
        "header": {"provider": "groq", "model": model, "mode": mode,
                   "started_utc": (NOW - timedelta(hours=hours_ago)).isoformat()},
        "stop": {"router_questions_run": questions[0], "injection_questions_run": questions[1]},
    }  # fmt: skip
    if tokens is not None:
        rep["tokens_spent"] = {"total": tokens}
    if used is not None:
        rep["accounting"] = {"tokens_used": used}
    (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")


def cassettes(root: Path, entries):
    d = root / "cassettes"
    d.mkdir(parents=True)
    (d / "calls.jsonl").write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )


def entry(hours_ago, p=300, c=200, model=M):
    return {"provider": "groq", "model": model, "tokens": {"prompt": p, "completion": c},
            "recorded_at": (NOW - timedelta(hours=hours_ago)).isoformat()}  # fmt: skip


def test_tokens_spent_counts_every_call_that_reported_usage_but_not_cassette_replays():
    run = RunContext("q", token_cap=10_000)
    run.add(CallRecord(0, "compose", "groq", "m", "ok", prompt_tokens=100, completion_tokens=50))
    run.add(CallRecord(0, "compose", "groq", "m", "malformed", prompt_tokens=80,
                       completion_tokens=1024))  # fmt: skip
    run.add(CallRecord(0, "compose", "groq", "m", "cassette", prompt_tokens=999,
                       completion_tokens=999))  # fmt: skip
    run.add(CallRecord(0, "compose", "groq", "m", "rate_limit"))
    s = usage.tokens_spent([run])
    assert s["total"] == 100 + 50 + 80 + 1024 and s["prompt"] == 180


def test_tally_sums_the_last_24_h_for_the_same_provider_and_model_only(tmp_path):
    live = tmp_path / "live_runs"
    report(live, "a", hours_ago=1, tokens=50_000)
    report(live, "b", hours_ago=23, tokens=20_000)
    report(live, "old", hours_ago=25, tokens=90_000)  # outside the window
    report(live, "other_model", hours_ago=1, tokens=70_000, model="other/model")
    report(live, "older_format", hours_ago=2, used=6_000)  # only accounting.tokens_used
    (live / "broken").mkdir()
    (live / "broken" / "report.json").write_text("{not json", encoding="utf-8")
    work = tmp_path / "work"
    cassettes(work, [entry(3), entry(4, 100, 100), entry(30), entry(1, model="x/y")])
    t = usage.collect_tally("groq", M, report_roots=[live], cassette_roots=[work], now=NOW)
    assert t.total == 50_000 + 20_000 + 6_000 + 500 + 200
    assert t.reports == 4 and t.approximate == 1  # three folders and the cassette folder
    assert {i["folder"] for i in t.items} >= {"a", "b", "older_format"}


def test_the_cap_is_measured_for_gpt_oss_20b_and_unverified_for_other_models():
    real = load_router_settings()
    assert usage.cap_for(real, "groq", M) == (200000, usage.CAP_SOURCE_MEASURED)
    cap, src = usage.cap_for(real, "groq", "openai/gpt-oss-120b")
    assert cap == 200000 and "UNVERIFIED" in src  # the documented value, not a measurement
    assert usage.cap_for(real, "gemini", "x")[0] is None
    assert "measured" in usage.CAP_SOURCE_MEASURED


def test_the_estimate_is_the_median_of_the_last_three_reports_of_the_same_mode(tmp_path):
    live = tmp_path
    for i, tok in enumerate([170_000, 17_000, 34_000, 51_000]):  # per question 10000, 1000...
        report(live, f"r{i}", hours_ago=10 - i, tokens=tok)
    report(live, "other_mode", hours_ago=0.5, tokens=999_999, mode="route only")
    e = usage.estimate_run("groq", M, "route plus full answer", 17, report_roots=[live])
    assert e.reports_used == 3 and e.per_question == 2000.0 and e.run_tokens == 34_000
    none = usage.estimate_run("groq", M, "nothing", 5, report_roots=[live])
    assert none.run_tokens is None and "no earlier report" in none.note


def test_start_prints_the_usage_line_and_refuses_over_the_cap_unless_forced(tmp_path):
    cfg = settings()
    cfg.providers["groq"].daily_tokens = {M: 200_000}
    live = tmp_path / "live"
    report(live, "x", hours_ago=2, tokens=180_000, questions=(12, 5))
    out = []
    with pytest.raises(ConfigError, match=r"refusing to start.*daily cap of 200000.*--force"):
        usage.start(cfg, "groq", M, mode="route plus full answer", questions=17, ceiling=60_000,
                    force=False, say=out.append, report_roots=[live], cassette_roots=[],
                    now=NOW)  # fmt: skip
    first = out[0]
    assert "about 180000 of 200000 tokens used in the last 24 h" in first
    assert "from our own reports" in first and "does not report the daily remainder" in first
    assert any("token budget for this run: about 180000" in x for x in out)
    out.clear()
    r = usage.start(cfg, "groq", M, mode="route plus full answer", questions=17, ceiling=60_000,
                    force=True, say=out.append, report_roots=[live], cassette_roots=[],
                    now=NOW)  # fmt: skip
    assert r["forced"] and any("--force given" in x for x in out)


def test_without_an_estimate_the_run_token_cap_is_the_planned_budget(tmp_path):
    cfg = settings()
    cfg.providers["groq"].daily_tokens = {M: 200_000}
    live = tmp_path / "live"
    report(live, "x", hours_ago=1, tokens=150_000, mode="something else")
    out = []
    with pytest.raises(ConfigError, match=r"60000 planned \(the run's token cap\)"):
        usage.start(cfg, "groq", M, mode="route only", questions=12, ceiling=60_000, force=False,
                    say=out.append, report_roots=[live], cassette_roots=[], now=NOW)  # fmt: skip
    assert any("no estimate" in x and "token cap is 60000" in x for x in out)
    ok = usage.start(cfg, "groq", M, mode="route only", questions=12, ceiling=30_000, force=False,
                     say=out.append, report_roots=[live], cassette_roots=[], now=NOW)  # fmt: skip
    assert not ok["forced"] and ok["planned"] == 30_000


def test_a_model_with_no_known_cap_prints_the_tally_and_is_never_refused(tmp_path):
    cfg = settings()  # the test provider has no tpd and no daily_tokens
    out = []
    r = usage.start(
        cfg, "groq", "any/model", mode="m", questions=1, ceiling=10**9, force=False,
        say=out.append, report_roots=[tmp_path], cassette_roots=[], now=NOW,
    )  # fmt: skip
    assert r["cap"] is None and "an unknown cap" in out[0]

