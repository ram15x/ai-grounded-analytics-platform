"""End to end, with the real warehouse fixture and NO real model: a FakeProvider returns scripted
JSON. plan -> compile -> guard -> execute -> compose -> verify -> render, one repair loop, and
the abstain paths. These are unit tests of the plumbing, not a measure of any model."""

from __future__ import annotations

import json

import pytest

from llm_helpers import KEYS, MockTransport, ok_openai, settings
from verified_analyst.cli import catalogue
from verified_analyst.llm.cassette import (
    Cassette,
    CassetteContext,
    layer_hash,
    schema_hash,
    warehouse_content_hash,
)
from verified_analyst.llm.fake import FakeProvider
from verified_analyst.llm.pipeline import run_question
from verified_analyst.llm.providers import GroqProvider
from verified_analyst.llm.router import Router
from verified_analyst.semantic import load_semantic_layer

pytestmark = pytest.mark.db

QUESTION = "Which country had the highest net revenue, and what was the runner-up?"
PLAN = {
    "metric_query": {
        "metrics": ["net_revenue"],
        "dimensions": ["country"],
        "order_by": [{"by": "net_revenue", "desc": True}],
        "limit": 3,
    }
}


def cell(country):
    return {"kind": "cell", "query_id": "q1", "column": "net_revenue",
            "row_key": {"country": country}}  # fmt: skip


def claim(cid, country):
    return {"id": cid, "label": "net revenue", "unit": "currency", "provenance": cell(country)}


GOOD = {
    "template": "The United Kingdom led with {uk} of net revenue, ahead of EIRE at {eire}.",
    "claims": [claim("uk", "United Kingdom"), claim("eire", "EIRE")],
}
TYPED_NUMBER = {
    "template": "The United Kingdom led with {uk} of net revenue, ahead of EIRE at {eire}, "
    "up 14% on last year.",
    "claims": GOOD["claims"],
}
NO_PLACEHOLDER_USED = {"template": "Revenue was high in the United Kingdom.", "claims": []}


@pytest.fixture
def env(warehouse):
    layer = load_semantic_layer()
    return layer, warehouse.readonly_url, catalogue(layer, warehouse.readonly_url)


def build(script, env_, cfg=None, **kw):
    layer, url, known = env_
    cfg = cfg or settings(datasets={"analytics.sales_lines": "public"})
    fake = FakeProvider(script, name="local")
    cfg.tasks = {t: ["local"] for t in ("plan", "compose", "repair")}
    router = Router(cfg, {"local": fake}, **kw)
    return router, fake, layer, url, known


def go(script, env_, token_cap=None, **kw):
    router, fake, layer, url, known = build(script, env_, **kw)
    run = router.new_run("e2e", token_cap)
    out = run_question(QUESTION, layer=layer, readonly_url=url, router=router, run=run,
                       known_values=known)  # fmt: skip
    return out, fake, run


def test_verified_on_the_first_attempt(env):
    out, fake, run = go({"plan": [PLAN], "compose": [GOOD]}, env)
    assert out.outcome == "verified" and out.reason == ""
    assert "£10,638.54" in out.rendered and "£3,374.20" in out.rendered
    assert out.repairs_used == 0 and [a.status for a in out.attempts] == ["verified"]
    assert [r.task for r in fake.requests] == ["plan", "compose"]
    assert fake.remaining == 0
    assert out.restatement.startswith("Compute net revenue")
    # the compose prompt carries the result tables and the restatement, and no SQL
    user = fake.requests[1].messages[1][1]
    assert "10638.54" in user and out.restatement in user and "SELECT" not in user.upper()


def test_one_repair_loop_fixes_a_number_the_model_typed(env):
    out, fake, run = go({"plan": [PLAN], "compose": [TYPED_NUMBER], "repair": [GOOD]}, env)
    assert out.outcome == "verified"
    assert [a.status for a in out.attempts] == ["partially_verified", "verified"]
    assert [a.kind for a in out.attempts] == ["compose", "repair"]
    assert out.repairs_used == 1
    assert [r.task for r in fake.requests] == ["plan", "compose", "repair"]
    repair_user = fake.requests[2].messages[1][1]
    assert "number_outside_placeholder" in repair_user and "up 14%" in repair_user
    assert "14%" not in out.rendered
    s = run.summary()
    assert s["calls_ok"] == 3 and s["tokens_used"] > 0 and s["by_provider"]["local"]["ok"] == 3


def test_abstain_when_the_planner_says_forecast(env):
    out, fake, run = go({"plan": [{"abstain": "forecast"}]}, env)
    assert out.outcome == "abstain" and out.reason == "forecast"
    assert [r.task for r in fake.requests] == ["plan"] and out.rendered == ""
    assert out.restatement == "" and out.attempts == []


def test_abstain_budget_before_any_call(env):
    out, fake, run = go({"plan": [PLAN]}, env, token_cap=10)
    assert out.outcome == "abstain" and out.reason == "budget"
    assert fake.requests == [] and run.records[-1].outcome == "budget"


def test_abstain_budget_after_the_plan_call(env):
    # enough for the plan call, not for the compose call
    out, fake, run = go({"plan": [PLAN], "compose": [GOOD]}, env, token_cap=1500)
    assert (out.outcome, out.reason) == ("abstain", "budget")
    assert [r.task for r in fake.requests] == ["plan"]
    assert run.tokens_used > 0


def test_plan_schema_failure_then_success_counts_against_repairs(env):
    out, fake, _ = go({"plan": ["this is not json", PLAN], "compose": [GOOD]}, env)
    assert out.outcome == "verified" and out.plan_failures == 1 and out.repairs_used == 1
    assert "failed" in fake.requests[1].messages[1][1]


def test_plan_naming_an_unknown_metric_is_fed_back(env):
    bad = {"metric_query": {"metrics": ["profit_margin"]}}
    out, fake, _ = go({"plan": [bad, PLAN], "compose": [GOOD]}, env)
    assert out.outcome == "verified" and out.plan_failures == 1
    assert "profit_margin" in fake.requests[1].messages[1][1]


def test_plan_with_both_fields_is_invalid(env):
    both = {**PLAN, "abstain": "forecast"}
    out, _, _ = go({"plan": [both, PLAN], "compose": [GOOD]}, env)
    assert out.outcome == "verified" and out.plan_failures == 1


def test_plan_failures_exhaust_the_shared_repair_budget(env):
    out, fake, _ = go({"plan": ["x", "y", "z"]}, env)
    assert (out.outcome, out.reason) == ("abstain", "plan_failed")
    assert len(fake.requests) == 3  # first try plus the two repairs


def test_compose_schema_failure_counts_as_an_attempt_then_repairs(env):
    out, fake, _ = go({"plan": [PLAN], "compose": ["not json"], "repair": [GOOD]}, env)
    assert out.outcome == "verified"
    assert [a.status for a in out.attempts] == ["schema_failure", "verified"]
    assert "schema_failure" in fake.requests[2].messages[1][1]


def test_compose_with_extra_fields_is_a_schema_failure(env):
    bad = {**GOOD, "value": 123}  # the composer has no place to put a number
    out, _, _ = go({"plan": [PLAN], "compose": [bad], "repair": [GOOD]}, env)
    assert out.attempts[0].status == "schema_failure" and out.outcome == "verified"


def test_still_unverified_after_two_repairs_is_partially_verified_with_flags(env):
    out, fake, _ = go(
        {"plan": [PLAN], "compose": [TYPED_NUMBER], "repair": [TYPED_NUMBER, TYPED_NUMBER]}, env
    )
    assert out.outcome == "partially_verified" and out.repairs_used == 2
    assert len(out.attempts) == 3 and len(fake.requests) == 4
    assert out.attempts[0].verified_numbers == 2 and out.attempts[0].total_numbers == 3
    assert out.verification["status"] == "partially_verified"
    assert "UNSUPPORTED" in json.dumps(out.verification)


def test_no_verified_number_in_any_attempt_abstains(env):
    out, _, _ = go(
        {"plan": [PLAN], "compose": [NO_PLACEHOLDER_USED],
         "repair": [NO_PLACEHOLDER_USED, NO_PLACEHOLDER_USED]}, env
    )  # fmt: skip
    assert (out.outcome, out.reason) == ("abstain", "unverified")
    assert out.rendered == ""
    # a text with no figure passes the verifier vacuously; the pipeline refuses to call it verified
    assert [a.status for a in out.attempts] == ["no_figures"] * 3


def test_provider_failure_abstains_with_reason(env):
    from verified_analyst.llm.types import ProviderError

    out, _, _ = go({"plan": [ProviderError("auth", "nope")] * 3}, env)
    assert (out.outcome, out.reason) == ("abstain", "provider_error")


def test_report_is_json_and_markdown_and_has_no_accuracy_figure(env):
    out, _, _ = go({"plan": [PLAN], "compose": [TYPED_NUMBER], "repair": [GOOD]}, env)
    d = json.loads(json.dumps(out.as_dict()))
    assert d["outcome"] == "verified" and d["accounting"]["cost_usd"] == 0
    assert d["accounting"]["latency_n"] == 3
    md = out.render_md()
    assert (
        "Query, restated" in md and "No accuracy figure" in md and "%" not in md.split("Answer")[0]
    )


# ---- cassettes end to end ----


def test_record_then_replay_the_whole_pipeline_with_no_provider(env, tmp_path, warehouse):
    layer, url, known = env
    ctx = CassetteContext(layer_hash(), schema_hash(), warehouse_content_hash(url))
    cas = Cassette(tmp_path / "e2e", "record", ctx, env={})
    live, fake, _ = go({"plan": [PLAN], "compose": [TYPED_NUMBER], "repair": [GOOD]}, env,
                       cassette=cas)  # fmt: skip
    assert live.outcome == "verified" and len(cas.entries) == 3
    cas2 = Cassette(tmp_path / "e2e", "replay", ctx, env={})
    router, _, layer, url, known = build({}, env, cassette=cas2)
    router.providers.clear()  # replay needs no provider at all
    run = router.new_run("replay")
    again = run_question(QUESTION, layer=layer, readonly_url=url, router=router, run=run,
                         known_values=known)  # fmt: skip
    assert again.outcome == live.outcome and again.rendered == live.rendered
    assert [a.status for a in again.attempts] == [a.status for a in live.attempts]
    assert all(r.outcome == "cassette" for r in run.records)
    assert run.tokens_used == live.run.tokens_used
    assert len(cas2.usage()["hit_keys"]) == 3 and cas2.usage()["missed"] == {}


def test_replay_with_a_changed_warehouse_misses_and_the_pipeline_abstains(env, tmp_path):
    layer, url, known = env
    ctx = CassetteContext(layer_hash(), schema_hash(), "warehouse-v1")
    go({"plan": [PLAN], "compose": [GOOD]}, env,
       cassette=Cassette(tmp_path / "c", "record", ctx, env={}))  # fmt: skip
    ctx2 = CassetteContext(layer_hash(), schema_hash(), "warehouse-v2")
    router, _, _, _, _ = build({}, env, cassette=Cassette(tmp_path / "c", "replay", ctx2, env={}))
    out = run_question(QUESTION, layer=layer, readonly_url=url, router=router,
                       run=router.new_run("r"), known_values=known)  # fmt: skip
    assert (out.outcome, out.reason) == ("abstain", "provider_error")


def test_warehouse_hash_is_stable_and_reflects_the_data(env, warehouse_factory):
    import pandas as pd

    _, url, _ = env
    assert warehouse_content_hash(url) == warehouse_content_hash(url)
    raw = pd.read_csv("tests/fixtures/mini_retail.csv").iloc[:200]
    other = warehouse_factory(raw)
    assert warehouse_content_hash(other.readonly_url) != warehouse_content_hash(url)


# ---- data sensitivity end to end ----


def test_internal_dataset_keeps_result_tables_off_the_hosted_provider(env):
    layer, url, known = env
    cfg = settings(datasets={"analytics.sales_lines": "internal"})  # same data, tagged internal
    t = MockTransport([ok_openai(json.dumps(PLAN), 30, 20)])
    groq = GroqProvider("groq", "GROQ_API_KEY", transport=t, env=KEYS)
    fake = FakeProvider({"compose": [GOOD]}, name="local")
    cfg.tasks = {
        "plan": ["groq", "local"],
        "compose": ["groq", "local"],
        "repair": ["groq", "local"],
    }
    router = Router(cfg, {"groq": groq, "local": fake})
    run = router.new_run("s")
    out = run_question(QUESTION, layer=layer, readonly_url=url, router=router, run=run,
                       known_values=known)  # fmt: skip
    assert out.outcome == "verified"
    # the plan prompt holds only the catalogue, so groq may serve it; the compose prompt holds
    # result rows, so it must go to the local model
    assert len(t.calls) == 1
    assert "10638.54" not in json.dumps(t.calls[0]["body"])
    assert [r.task for r in fake.requests] == ["compose"]
    assert "10638.54" in fake.requests[0].messages[1][1]
    assert any(r.outcome == "refused_sensitivity" for r in run.records)


def test_internal_dataset_with_only_hosted_providers_abstains_with_sensitivity(env):
    layer, url, known = env
    cfg = settings(datasets={"analytics.sales_lines": "internal"})
    t = MockTransport([ok_openai(json.dumps(PLAN), 30, 20)])
    groq = GroqProvider("groq", "GROQ_API_KEY", transport=t, env=KEYS)
    cfg.tasks = {"plan": ["groq"], "compose": ["groq"], "repair": ["groq"]}
    router = Router(cfg, {"groq": groq})
    run = router.new_run("s")
    out = run_question(QUESTION, layer=layer, readonly_url=url, router=router, run=run,
                       known_values=known)  # fmt: skip
    assert (out.outcome, out.reason) == ("abstain", "sensitivity")
    assert len(t.calls) == 1  # the plan only; the compose payload never left
    assert "10638.54" not in json.dumps(t.calls[0]["body"])


def test_repair_prompt_carries_field_paths_not_just_a_count(env):
    bad = {
        "template": "The United Kingdom led with {uk}.",
        "claims": [{"id": "uk", "label": "net revenue", "unit": "GBP", "provenance": cell("EIRE")}],
    }
    out, fake, _ = go({"plan": [PLAN], "compose": [bad], "repair": [GOOD]}, env)
    assert out.outcome == "verified"
    repair_user = fake.requests[2].messages[1][1]
    assert "claims.0.unit" in repair_user and "Input should be" in repair_user
    assert "GBP" not in repair_user.split("Problems:")[1]  # the bad value is not echoed in problems
    assert out.attempts[0].problems and "claims.0.unit" in out.attempts[0].problems[0]


def test_a_derived_percent_of_claim_shaped_like_the_prompt_example_verifies(env):
    """The compose prompt's worked example uses a percent_of derivation over two cells. This runs
    the same shape against the real warehouse so the example is not a shape that cannot verify."""
    derived = {
        "template": "The United Kingdom had net revenue of {uk}, and EIRE had {eire}. "
        "The United Kingdom's was {pct} of EIRE's.",
        "claims": [
            claim("uk", "United Kingdom"),
            claim("eire", "EIRE"),
            {
                "id": "pct",
                "label": "United Kingdom net revenue as a percent of EIRE",
                "unit": "percent",
                "provenance": {
                    "kind": "derived",
                    "op": "percent_of",
                    "args": [cell("United Kingdom"), cell("EIRE")],
                },
            },
        ],
    }
    out, _, _ = go({"plan": [PLAN], "compose": [derived]}, env)
    assert out.outcome == "verified", [(a.status, a.problems) for a in out.attempts]
    assert out.repairs_used == 0
    assert "%" in out.rendered
