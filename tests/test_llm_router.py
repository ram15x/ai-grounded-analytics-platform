"""Router behaviour with mocked HTTP and a fake clock. No sleeping, no network."""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest

from llm_helpers import (
    CONN_ERROR,
    KEYS,
    TIMEOUT,
    MockTransport,
    ok_gemini,
    ok_openai,
    req,
    settings,
    status,
)
from verified_analyst.config import ConfigError
from verified_analyst.llm.accounting import merge_summaries, render_markdown
from verified_analyst.llm.fake import FakeProvider
from verified_analyst.llm.providers import GeminiProvider, GroqProvider, OpenRouterProvider
from verified_analyst.llm.router import Router, TokenBucket, estimate_tokens
from verified_analyst.llm.settings import load_router_settings
from verified_analyst.llm.transport import HttpResponse
from verified_analyst.llm.types import (
    AllProvidersFailed,
    BudgetExceeded,
    ProviderError,
    SensitivityRefusal,
)

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def rig(cfg=None, *, groq=(), gemini=(), openrouter=(), local=None, **router_kw):
    cfg = cfg or settings()
    clock = Clock()
    tg, tm, to = MockTransport(groq), MockTransport(gemini), MockTransport(openrouter)
    kw = {"env": KEYS, "clock": clock}
    providers = {
        "groq": GroqProvider("groq", "GROQ_API_KEY", transport=tg, **kw),
        "gemini": GeminiProvider("gemini", "GEMINI_API_KEY", transport=tm, **kw),
        "openrouter": OpenRouterProvider("openrouter", "OPENROUTER_API_KEY", transport=to, **kw),
    }
    if local is not None:
        providers["local"] = local
    router = Router(
        cfg, providers, clock=clock, sleep=clock.sleep, rng=random.Random(1), **router_kw
    )
    return router, router.new_run("t"), {"groq": tg, "gemini": tm, "openrouter": to}, clock


# ---- retry, backoff, retry-after ----


def test_success_first_try_records_tokens_and_latency():
    router, run, tr, _ = rig(groq=[ok_openai("{}", 100, 20)])
    r = router.call(req(), run)
    assert r.provider == "groq" and r.model == "m-groq"
    assert run.tokens_used == 120
    assert run.records[0].outcome == "ok" and run.records[0].prompt_tokens == 100
    assert len(tr["groq"].calls) == 1 and tr["gemini"].calls == []


def test_429_honours_retry_after_then_succeeds():
    router, run, tr, clock = rig(groq=[status(429, {"Retry-After": "4"}), ok_openai()])
    r = router.call(req(), run)
    assert r.provider == "groq"
    assert clock.slept == [4.0]
    assert [x.outcome for x in run.records] == ["rate_limit", "ok"]
    assert run.records[0].retry_after_s == 4.0
    assert run.records[1].wait_ms == 4000.0


def test_5xx_timeout_and_malformed_back_off_with_jitter_and_retry():
    router, run, tr, clock = rig(
        groq=[status(500), TIMEOUT, status(200, body=b"nope"), ok_openai()]
    )
    r = router.call(req(), run)
    assert r.provider == "groq"
    assert [x.outcome for x in run.records] == ["server", "timeout", "malformed", "ok"]
    base = [1.0, 2.0, 4.0]  # base_s * 2**attempt, jitter keeps each within [d/2, d]
    assert len(clock.slept) == 3
    for slept, d in zip(clock.slept, base, strict=True):
        assert d * 0.5 <= slept <= d
    assert len(set(clock.slept)) == 3  # jitter, not a fixed schedule


def test_backoff_is_capped_and_deterministic_with_a_seeded_rng():
    cfg = settings(retry={"max_retries": 5, "base_s": 10, "cap_s": 12, "jitter": 0})
    router, run, _, clock = rig(cfg, groq=[status(500)] * 5 + [ok_openai()])
    router.call(req(), run)
    assert clock.slept == [10, 12, 12, 12, 12]


def test_connection_error_retries():
    router, run, _, _ = rig(groq=[CONN_ERROR, ok_openai()])
    router.call(req(), run)
    assert [x.outcome for x in run.records] == ["network", "ok"]


def test_retries_exhausted_falls_through_to_next_provider():
    router, run, tr, _ = rig(groq=[status(500)] * 4, gemini=[ok_gemini()])
    r = router.call(req(), run)
    assert r.provider == "gemini"
    assert len(tr["groq"].calls) == 4  # first try + 3 retries
    assert tr["openrouter"].calls == []


def test_non_retryable_errors_fall_through_without_retrying():
    for bad in (status(401), status(400)):
        router, run, tr, clock = rig(groq=[bad], gemini=[ok_gemini()])
        assert router.call(req(), run).provider == "gemini"
        assert len(tr["groq"].calls) == 1 and clock.slept == []


def test_retry_after_longer_than_cap_falls_through_immediately():
    router, run, tr, clock = rig(groq=[status(429, {"Retry-After": "3600"})], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert clock.slept == [] and len(tr["groq"].calls) == 1


def test_missing_key_and_placeholder_model_fall_through():
    cfg = settings()
    cfg.providers["gemini"].models["default"] = "REPLACE-ME"
    router, run, tr, _ = rig(
        cfg, groq=[status(401)], gemini=[ok_gemini()], openrouter=[ok_openai()]
    )
    router.providers["groq"].env = {}  # no key
    r = router.call(req(), run)
    assert r.provider == "openrouter"
    outcomes = {x.provider: x.outcome for x in run.records}
    assert outcomes["groq"] == "no_key" and outcomes["gemini"] == "config"
    assert tr["groq"].calls == [] and tr["gemini"].calls == []


# ---- circuit breaker ----


def test_breaker_opens_after_three_consecutive_429s_and_skips_provider():
    router, run, tr, clock = rig(
        groq=[status(429, {"Retry-After": "1"})] * 3, gemini=[ok_gemini(), ok_gemini()]
    )
    assert router.call(req(), run).provider == "gemini"
    assert router.breaker_state("groq") == "open"
    assert len(tr["groq"].calls) == 3
    # a second call does not touch groq at all while the breaker is open
    assert router.call(req(), run).provider == "gemini"
    assert len(tr["groq"].calls) == 3
    assert any(x.outcome == "breaker_open" for x in run.records)


def test_breaker_half_open_probe_and_reopen():
    router, run, tr, clock = rig(
        groq=[status(429, {"Retry-After": "1"})] * 4 + [ok_openai()], gemini=[ok_gemini()] * 3
    )
    router.call(req(), run)
    assert router.breaker_state("groq") == "open"
    clock.t += 61  # cooldown over
    assert router.breaker_state("groq") == "closed"
    router.call(req(), run)  # one probe, 429 again: re-opens at once (1 call, not 3)
    assert len(tr["groq"].calls) == 4 and router.breaker_state("groq") == "open"
    clock.t += 61
    assert router.call(req(), run).provider == "groq"  # probe succeeds, breaker closes
    assert router._consec_429["groq"] == 0


def test_non_429_error_resets_the_consecutive_count():
    router, run, tr, _ = rig(
        settings(retry={"max_retries": 6}),
        groq=[status(429, {"Retry-After": "1"}), status(429, {"Retry-After": "1"}),
              status(500), status(429, {"Retry-After": "1"}), ok_openai()]
    )  # fmt: skip
    assert router.call(req(), run).provider == "groq"
    assert router.breaker_state("groq") == "closed"


def test_success_resets_the_consecutive_count():
    router, run, tr, _ = rig(
        groq=[status(429, {"Retry-After": "1"}), status(429, {"Retry-After": "1"}), ok_openai(),
              status(429, {"Retry-After": "1"}), status(429, {"Retry-After": "1"}), ok_openai()]
    )  # fmt: skip
    router.call(req(), run)
    router.call(req(), run)
    assert router.breaker_state("groq") == "closed"


def test_all_providers_failing_raises_with_the_attempt_list():
    router, run, _, _ = rig(
        groq=[status(401)], gemini=[status(401)], openrouter=[status(401)],
        local=FakeProvider([ProviderError("server", "boom")] * 4, name="local"),
    )  # fmt: skip
    with pytest.raises(AllProvidersFailed) as ei:
        router.call(req(), run)
    assert [a["provider"] for a in ei.value.attempts[:3]] == ["groq", "gemini", "openrouter"]
    assert "all providers failed" in str(ei.value)
    assert run.tokens_used == 0


# ---- token buckets ----


def test_bucket_math():
    c = Clock()
    b = TokenBucket(10, 1, c)
    assert b.wait_needed(10) == 0
    b.take(10)
    assert b.wait_needed(4) == pytest.approx(4)
    c.t += 4
    assert b.wait_needed(4) == pytest.approx(0)
    assert b.wait_needed(11) == float("inf")


def test_buckets_are_sized_at_half_the_documented_limit():
    cfg = settings()
    cfg.providers["groq"].rpm, cfg.providers["groq"].tpd = 30, 200000
    router, *_ = rig(cfg)
    assert router._rpm["groq"].capacity == 15
    assert router._tpd["groq"].capacity == 100000
    assert "gemini" not in router._rpm  # no documented limit: no bucket


def test_rpm_bucket_makes_the_router_wait_then_proceed():
    cfg = settings(bucket={"fraction": 0.5, "max_wait_s": 60})
    cfg.providers["groq"].rpm = 2  # capacity 1 request, 1 per 60 s
    router, run, tr, clock = rig(cfg, groq=[ok_openai(), ok_openai()])
    router.call(req(), run)
    router.call(req(), run)
    assert tr["groq"].calls and len(tr["groq"].calls) == 2
    assert clock.slept == [pytest.approx(60)]


def test_a_wait_longer_than_the_cap_falls_through_instead_of_sleeping():
    cfg = settings(bucket={"fraction": 0.5, "max_wait_s": 5})
    cfg.providers["groq"].rpm = 2  # one request, then a 60 s wait
    router, run, tr, clock = rig(cfg, groq=[ok_openai()], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "groq"
    assert router.call(req(), run).provider == "gemini"
    assert clock.slept == []
    assert any(x.outcome == "throttled" for x in run.records)


def test_a_request_larger_than_the_daily_bucket_is_throttled_not_sent():
    cfg = settings()
    cfg.providers["groq"].tpd = 100  # capacity 50 tokens, below any estimate
    router, run, tr, _ = rig(cfg, gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert tr["groq"].calls == []


# ---- budget ----


def test_budget_cap_raises_before_any_call_and_is_recorded():
    router, run, tr, _ = rig(groq=[ok_openai()])
    run.token_cap = estimate_tokens(req()) - 1
    with pytest.raises(BudgetExceeded):
        router.call(req(), run)
    assert tr["groq"].calls == [] and run.records[-1].outcome == "budget"


def test_budget_accumulates_across_calls():
    router, run, _, _ = rig(groq=[ok_openai("{}", 400, 100)] * 3)
    run.token_cap = 10_000
    router.call(req(), run)
    router.call(req(), run)
    assert run.tokens_used == 1000
    run.token_cap = 1000 + estimate_tokens(req()) - 1
    with pytest.raises(BudgetExceeded):
        router.call(req(), run)


def test_default_cap_comes_from_config():
    cfg = settings(budget={"per_run_tokens": 1234})
    router, *_ = rig(cfg)
    assert router.new_run("x").token_cap == 1234
    assert router.new_run("x", token_cap=99).token_cap == 99


def test_missing_usage_is_estimated_and_flagged_in_the_ledger():
    import json

    body = json.dumps({"choices": [{"message": {"content": "abcdefgh" * 10}}]}).encode()
    from verified_analyst.llm.transport import HttpResponse

    router, run, _, _ = rig(groq=[HttpResponse(200, body)])
    r = router.call(req(text="x" * 400), run)
    assert r.tokens_estimated and run.records[0].tokens_estimated
    assert run.tokens_used == run.records[0].tokens > 0
    assert run.summary()["tokens_estimated_calls"] == 1


# ---- accounting ----


def test_run_summary_aggregates_per_provider_with_n():
    router, run, _, _ = rig(
        groq=[status(500), ok_openai("{}", 10, 10)], gemini=[ok_gemini("{}", 5, 5)]
    )
    router.call(req(), run)
    router.providers["groq"].transport.script += [status(401)]
    router.call(req(), run)
    s = run.summary()
    assert s["tokens_used"] == 20 + 10
    assert s["by_provider"]["groq"]["errors"] == {"server": 1, "auth": 1}
    assert s["by_provider"]["groq"]["ok"] == 1
    assert s["by_provider"]["gemini"]["tokens"] == 10
    assert s["latency_n"] == 2 and s["cost_usd"] == 0
    md = render_markdown(s)
    assert "n=2" in md and "| groq |" in md and "accuracy" not in md.lower()


def test_merge_summaries_pools_runs():
    router, run1, _, _ = rig(groq=[ok_openai("{}", 10, 10), ok_openai("{}", 1, 1)])
    router.call(req(), run1)
    run2 = router.new_run("r2")
    router.call(req(), run2)
    m = merge_summaries([run1, run2])
    assert m["runs"] == 2 and m["tokens_used"] == 22 and m["latency_n"] == 2


def test_percentiles():
    from verified_analyst.llm.accounting import percentile

    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10


# ---- data sensitivity ----

INTERNAL = ("hr.salaries",)


@pytest.mark.parametrize(
    "datasets", [INTERNAL, ("not.tagged",), ("analytics.sales_lines", "hr.salaries")]
)
def test_internal_or_untagged_payload_never_reaches_a_hosted_adapter(datasets):
    local = FakeProvider([{"ok": 1}], name="local", hosted=False)
    router, run, tr, _ = rig(local=local, groq=[ok_openai()], gemini=[ok_gemini()],
                             openrouter=[ok_openai()])  # fmt: skip
    r = router.call(req(datasets=datasets), run)
    assert r.provider == "local"
    assert all(t.calls == [] for t in tr.values())
    assert [x.outcome for x in run.records].count("refused_sensitivity") == 3


def test_no_hosted_call_even_when_the_local_provider_fails_and_the_chain_falls_through():
    local = FakeProvider([ProviderError("server", "down")] * 5, name="local", hosted=False)
    router, run, tr, _ = rig(local=local, groq=[ok_openai()], gemini=[ok_gemini()],
                             openrouter=[ok_openai()])  # fmt: skip
    with pytest.raises(AllProvidersFailed):
        router.call(req(datasets=INTERNAL), run)
    assert all(t.calls == [] for t in tr.values())


def test_internal_payload_with_only_hosted_providers_is_refused():
    cfg = settings(tasks={"plan": ["groq", "gemini", "openrouter"]})
    router, run, tr, _ = rig(cfg, groq=[ok_openai()])
    with pytest.raises(SensitivityRefusal):
        router.call(req(datasets=INTERNAL), run)
    assert all(t.calls == [] for t in tr.values())


def test_public_payload_still_uses_hosted_and_falls_back_among_them():
    router, run, tr, _ = rig(groq=[status(500)] * 4, gemini=[ok_gemini()])
    assert router.call(req(datasets=("analytics.sales_lines",)), run).provider == "gemini"


def test_a_public_request_cannot_be_downgraded_by_the_caller_stamp():
    # the caller may claim "public" on the request object; the router recomputes it from datasets
    forged = dataclasses.replace(req(datasets=INTERNAL), sensitivity="public")
    local = FakeProvider([{"ok": 1}], name="local")
    router, run, tr, _ = rig(local=local, groq=[ok_openai()])
    assert router.call(forged, run).provider == "local"
    assert tr["groq"].calls == []
    assert local.requests[0].sensitivity == "internal"


def test_sensitivity_is_checked_again_for_every_fallback_step():
    # groq (public data) fails, falls to gemini, then local: the same eligible list is used
    local = FakeProvider([{"ok": 1}], name="local")
    router, run, tr, _ = rig(local=local, groq=[status(500)] * 4, gemini=[status(500)] * 4,
                             openrouter=[status(500)] * 4)  # fmt: skip
    r = router.call(req(datasets=("analytics.sales_lines",)), run)
    assert r.provider == "local"
    assert local.requests[0].sensitivity == "public"


def test_sensitivity_rule_is_pure_and_fail_closed():
    cfg = settings()
    assert cfg.sensitivity_of(()) == "public"
    assert cfg.sensitivity_of(("analytics.sales_lines",)) == "public"
    assert cfg.sensitivity_of(("hr.salaries",)) == "internal"
    assert cfg.sensitivity_of(("unknown",)) == "internal"
    assert cfg.sensitivity_of(("analytics.sales_lines", "unknown")) == "internal"


# ---- config ----


def test_shipped_config_loads_and_has_no_secrets_or_real_ids_in_code():
    cfg = load_router_settings(ROOT / "config" / "router.yaml")
    assert cfg.tasks["plan"][0] == "groq"
    assert cfg.datasets["analytics.sales_lines"] == "public"
    text = (ROOT / "config" / "router.yaml").read_text()
    assert "UNVERIFIED" in text
    for key in ("gsk_", "sk-or", "AIza"):
        assert key not in text


def test_model_ids_appear_only_in_config_never_in_source():
    cfg = load_router_settings(ROOT / "config" / "router.yaml")
    ids = {m for p in cfg.providers.values() for m in p.models.values()}
    assert "openai/gpt-oss-20b" in ids
    src = "".join(p.read_text("utf-8") for p in (ROOT / "src").rglob("*.py"))
    for model_id in ids:
        assert model_id not in src, model_id


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        settings(tasks={"plan": ["nope"]})
    with pytest.raises(ValueError):
        settings(tasks={"plan": []})
    with pytest.raises(ValueError):
        settings(datasets={"x": "secret"})


def test_unknown_task_is_a_config_error():
    router, run, _, _ = rig()
    with pytest.raises(ConfigError):
        router.call(req(task="nonsense"), run)


def test_shipped_config_refuses_to_call_with_placeholder_models():
    cfg = load_router_settings(ROOT / "config" / "router.yaml")
    t = {n: MockTransport([ok_gemini(), ok_openai()]) for n in ("gemini", "openrouter")}
    providers = {
        "gemini": GeminiProvider("gemini", "GEMINI_API_KEY", transport=t["gemini"], env=KEYS),
        "openrouter": OpenRouterProvider(
            "openrouter", "OPENROUTER_API_KEY", transport=t["openrouter"], env=KEYS
        ),
    }
    clock = Clock()
    cfg.tasks["plan"] = ["gemini", "openrouter"]
    router = Router(cfg, providers, clock=clock, sleep=clock.sleep)
    with pytest.raises(AllProvidersFailed):
        router.call(req(datasets=("analytics.sales_lines",)), router.new_run("x"))
    assert t["gemini"].calls == [] and t["openrouter"].calls == []


# ---- tokens per minute, settled against real usage ----


def tpm_cfg(tpm, max_wait=200):
    cfg = settings(bucket={"fraction": 0.5, "max_wait_s": max_wait})
    cfg.providers["groq"].tpm = tpm
    return cfg


def test_tpm_bucket_is_sized_at_half_the_limit_and_refills_over_a_minute():
    router, *_ = rig(tpm_cfg(8000))
    b = router._tpm["groq"]
    assert b.capacity == 4000 and b.refill_per_s == pytest.approx(4000 / 60)
    assert "gemini" not in router._tpm  # no configured limit: no bucket


def test_an_unused_part_of_the_reservation_is_given_back_after_the_response():
    r = req(max_tokens=100)
    est = estimate_tokens(r)
    router, run, _, _ = rig(tpm_cfg(240), groq=[ok_openai("{}", 10, 5)])  # capacity 120
    router.call(r, run)
    # reserved `est`, used 15, so about 120 - 15 remains (no time has passed on the fake clock)
    assert router._tpm["groq"].level == pytest.approx(120 - 15)
    assert est > 15  # the reservation really was larger than the use


def test_usage_above_the_reservation_is_charged_and_the_next_call_waits():
    r = req(max_tokens=100)
    router, run, tr, clock = rig(
        tpm_cfg(240), groq=[ok_openai("{}", 100, 100), ok_openai("{}", 10, 5)]
    )  # capacity 120, refill 2 tokens/s
    router.call(r, run)  # reserves about 103, really used 200
    level = router._tpm["groq"].level
    assert level == pytest.approx(120 - 200)
    router.call(r, run)
    need = estimate_tokens(r)
    assert clock.slept == [pytest.approx((need - level) / 2.0)]
    assert len(tr["groq"].calls) == 2


def test_tpm_wait_beyond_the_cap_falls_through_to_the_next_provider():
    r = req(max_tokens=100)
    router, run, tr, clock = rig(
        tpm_cfg(240, max_wait=5), groq=[ok_openai("{}", 100, 100)], gemini=[ok_gemini()]
    )
    assert router.call(r, run).provider == "groq"
    assert router.call(r, run).provider == "gemini"  # groq would need a long wait
    assert clock.slept == [] and any(x.outcome == "throttled" for x in run.records)


def test_a_request_larger_than_the_minute_bucket_is_throttled_not_sent():
    cfg = tpm_cfg(100)  # capacity 50 tokens, below any estimate
    router, run, tr, _ = rig(cfg, gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert tr["groq"].calls == []


def test_estimated_usage_keeps_the_full_reservation():
    r = req(max_tokens=100)
    est = estimate_tokens(r)
    no_usage = HttpResponse(200, b'{"choices": [{"message": {"content": "{}"}}]}')
    router, run, _, _ = rig(tpm_cfg(240), groq=[no_usage])
    router.call(r, run)
    assert run.records[0].tokens_estimated
    assert router._tpm["groq"].level == pytest.approx(120 - est)


def test_the_shipped_config_has_the_observed_groq_minute_limit_and_no_reasoning_effort():
    s = load_router_settings(ROOT / "config" / "router.yaml")
    assert s.providers["groq"].tpm == 8000
    assert all(p.reasoning_effort is None for p in s.providers.values())  # UNVERIFIED: not set


# ---- reasoning_effort from config ----


def test_configured_reasoning_effort_is_sent_to_groq_and_only_then():
    cfg = settings()
    cfg.providers["groq"].reasoning_effort = "low"
    router, run, tr, _ = rig(cfg, groq=[ok_openai()])
    router.call(req(), run)
    assert tr["groq"].calls[0]["body"]["reasoning_effort"] == "low"
    router, run, tr, _ = rig(settings(), groq=[ok_openai()])
    router.call(req(), run)
    assert "reasoning_effort" not in tr["groq"].calls[0]["body"]


def test_reasoning_effort_on_another_provider_is_a_config_error():
    from pydantic import ValidationError

    from verified_analyst.llm.settings import RouterSettings

    cfg = settings().model_dump()
    cfg["providers"]["gemini"]["reasoning_effort"] = "low"
    with pytest.raises(ValidationError, match="only wired for groq"):
        RouterSettings.model_validate(cfg)
    cfg["providers"]["gemini"]["reasoning_effort"] = None
    cfg["providers"]["groq"]["reasoning_effort"] = "Low; drop table"  # not a plain word
    with pytest.raises(ValidationError):
        RouterSettings.model_validate(cfg)


# ---- Groq json_validate_failed (HTTP 400): one retry, then the text goes to the repair path ----


def jv(generation="", code="json_validate_failed"):
    body = {"error": {"message": "Failed to validate JSON.", "type": "invalid_request_error",
                      "code": code, "failed_generation": generation}}  # fmt: skip
    return status(400, body=json.dumps(body).encode())


def test_json_validate_failed_is_retried_once_with_json_mode_off_and_can_succeed():
    router, run, tr, clock = rig(groq=[jv(), ok_openai('{"a": 1}')])
    r = router.call(req(), run)
    assert r.text == '{"a": 1}' and r.finish_reason != "json_rejected"
    first, second = (c["body"] for c in tr["groq"].calls)
    assert "response_format" in first and "response_format" not in second
    assert {k: v for k, v in first.items() if k != "response_format"} == second  # nothing else
    assert [x.outcome for x in run.records] == ["json_rejected", "ok"]
    assert [x.json_mode_off for x in run.records] == [False, True]
    assert all(x.max_tokens == first["max_tokens"] for x in run.records)
    assert clock.slept == []


def test_a_retry_that_returns_non_json_text_is_handed_back_as_text_with_no_third_request():
    router, run, tr, _ = rig(groq=[jv(), ok_openai("{ half a json")])
    r = router.call(req(), run)
    assert (r.text, r.provider) == ("{ half a json", "groq")
    assert len(tr["groq"].calls) == 2  # exactly one retry, never a third request


def test_an_empty_failed_generation_is_retried_the_same_way():
    router, run, tr, _ = rig(groq=[jv(""), ok_openai('{"a": 1}')])
    assert router.call(req(), run).text == '{"a": 1}' and len(tr["groq"].calls) == 2


def test_if_the_json_off_retry_is_itself_refused_the_next_provider_is_tried():
    router, run, tr, _ = rig(groq=[jv(), jv()], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert len(tr["groq"].calls) == 2


def test_the_json_retry_does_not_use_up_the_normal_retry_budget():
    cfg = settings(retry={"max_retries": 0})
    router, run, tr, _ = rig(cfg, groq=[jv(), ok_openai()])
    assert router.call(req(), run).finish_reason != "json_rejected"
    assert len(tr["groq"].calls) == 2


def empty_200(content="", finish="stop", c=0):
    body = {
        "choices": [{"message": {"content": content, "reasoning": "thinking"},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 40, "completion_tokens": c},
    }  # fmt: skip
    return HttpResponse(200, json.dumps(body).encode())


def test_a_server_error_on_the_json_off_retry_ends_the_path_with_two_requests():
    router, run, tr, _ = rig(groq=[jv(), status(500), ok_openai()], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert len(tr["groq"].calls) == 2  # not three: the normal retry budget is not used here


def test_an_empty_200_after_the_json_off_retry_is_not_repeated_and_the_reply_is_recorded():
    # seen on Groq gpt-oss-20b: 400 json_validate_failed, then HTTP 200 with empty content and 0
    # completion tokens, which used to be retried as `malformed` (max_retries = 3, four calls)
    router, run, tr, _ = rig(groq=[jv(), empty_200(), ok_openai(), ok_openai(), ok_openai()])
    r = router.call(req(), run)
    assert len(tr["groq"].calls) == 2
    assert (r.text, r.finish_reason, r.provider) == ("", "json_rejected", "groq")
    rej, mal = run.records
    assert (rej.outcome, mal.outcome) == ("json_rejected", "malformed")
    assert mal.json_mode_off and mal.malformed_kind == "content_empty"
    assert (mal.completion_tokens, mal.prompt_tokens, mal.finish_reason) == (0, 40, "stop")
    assert mal.raw_body.startswith('{"choices"') and len(mal.raw_body) <= 300


def test_a_malformed_reply_without_a_json_refusal_keeps_the_normal_retries():
    router, run, tr, _ = rig(groq=[empty_200(), empty_200(), ok_openai()])
    assert router.call(req(), run).text == '{"ok": true}'
    assert [x.outcome for x in run.records] == ["malformed", "malformed", "ok"]
    assert all(x.malformed_kind == "content_empty" for x in run.records[:2])


def test_content_null_and_non_json_bodies_are_recorded_with_their_kind_and_the_key_is_redacted():
    null = HttpResponse(200, b'{"choices":[{"message":{"content":null},"finish_reason":"length"}]}')
    leak = HttpResponse(200, b"oops Bearer abc123 and gsk-test-key here")
    router, run, tr, _ = rig(groq=[null, leak, ok_openai()])
    router.call(req(), run)
    a, b, _ = run.records
    assert (a.malformed_kind, a.finish_reason) == ("content_null", "length")
    assert b.malformed_kind == "body_not_json"
    assert "abc123" not in b.raw_body and "Bearer <redacted>" in b.raw_body


def test_the_second_json_refusal_in_a_row_stops_the_provider_for_the_question():
    router, run, tr, _ = rig(
        groq=[jv(), empty_200(), jv(), ok_openai()], gemini=[ok_gemini(), ok_gemini()]
    )
    first = router.call(req(), run)  # 400, then an empty JSON-off reply: handed back unparseable
    assert first.finish_reason == "json_rejected"
    second = router.call(req(), run)  # the second 400 in a row: groq is dropped, gemini answers
    assert second.provider == "gemini" and len(tr["groq"].calls) == 3
    third = router.call(req(), run)  # groq gets no more requests for this question
    assert third.provider == "gemini" and len(tr["groq"].calls) == 3
    assert [x.outcome for x in run.records if x.provider == "groq"][-1] == "json_streak_skip"


def test_a_success_between_refusals_resets_the_streak():
    router, run, tr, _ = rig(groq=[jv(), ok_openai(), jv(), ok_openai(), ok_openai()])
    for _i in range(3):
        assert router.call(req(), run).provider == "groq"
    assert len(tr["groq"].calls) == 5


def test_with_no_other_provider_the_second_refusal_in_a_row_ends_the_question():
    cfg = settings()
    cfg.tasks = {t: ["groq"] for t in ("route", "plan", "compose", "repair")}
    router, run, tr, _ = rig(cfg, groq=[jv(), empty_200(), jv(), ok_openai()])
    router.call(req(), run)
    with pytest.raises(AllProvidersFailed):
        router.call(req(), run)
    with pytest.raises(AllProvidersFailed):
        router.call(req(), run)
    assert len(tr["groq"].calls) == 3


@pytest.mark.parametrize(
    "resp",
    [
        jv(code="model_not_found"),  # another 400 from Groq
        status(400, body=b"not json"),
        status(400, body=b"{}"),
    ],
    ids=["other_code", "not_json", "empty_object"],
)
def test_other_400s_are_still_not_retried(resp):
    router, run, tr, _ = rig(groq=[resp], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"  # fell through, groq not retried
    assert len(tr["groq"].calls) == 1
    assert run.records[0].outcome == "bad_request"


def test_a_json_validate_failed_400_without_json_mode_is_not_retried():
    router, run, tr, _ = rig(groq=[jv()], gemini=[ok_gemini()])
    assert router.call(req(json_mode=False), run).provider == "gemini"
    assert len(tr["groq"].calls) == 1 and run.records[0].outcome == "bad_request"


def test_only_adapters_that_know_the_wording_classify_it():
    router, run, tr, _ = rig(openrouter_only(), openrouter=[jv()], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert len(tr["openrouter"].calls) == 1 and run.records[0].outcome == "bad_request"


def openrouter_only():
    cfg = settings()
    cfg.tasks = {t: ["openrouter", "gemini"] for t in ("route", "plan", "compose", "repair")}
    return cfg


def test_timeout_is_per_provider_in_the_shipped_config():
    cfg = load_router_settings(ROOT / "config" / "router.yaml")
    assert cfg.providers["groq"].timeout_s == 30
    assert cfg.providers["gemini"].timeout_s == 90
    text = (ROOT / "config" / "router.yaml").read_text()
    gemini_block = text.split("  gemini:")[1].split("  openrouter:")[0]
    assert "UNVERIFIED" in gemini_block.split("timeout_s")[0]


def test_each_provider_call_gets_that_providers_timeout():
    cfg = settings()
    cfg.providers["groq"].timeout_s = 30
    cfg.providers["gemini"].timeout_s = 90
    # a long retry-after sends the call from groq on to gemini
    router, run, tr, _ = rig(cfg, groq=[status(429, {"Retry-After": "600"})], gemini=[ok_gemini()])
    assert router.call(req(), run).provider == "gemini"
    assert tr["groq"].calls[0]["timeout"] == 30
    assert tr["gemini"].calls[0]["timeout"] == 90
