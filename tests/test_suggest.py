"""Model-drafted metric suggestions: what the model sees, and which suggestions survive."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from grounded_analytics.datasets import profile as dp
from grounded_analytics.datasets import suggest as sg

EX = Path(__file__).resolve().parent.parent / "examples" / "datasets"
PROF = dp.profile_frame(dp.read_table(EX / "saas_subscriptions.csv"), "saas.csv")
MAN = yaml.safe_load((EX / "saas_subscriptions.manifest.yaml").read_text("utf-8"))
FORM = {k: MAN[k] for k in ("name", "description", "grain", "currency", "columns")}
FORM |= {"visibility": "public", "metrics": {}}


def test_the_model_never_sees_personal_or_ignored_columns_or_rows():
    text, usable = sg.context(PROF, FORM)
    assert "billing_email" not in text and "example.com" not in text
    assert "billing_email" not in usable and "mrr_usd" in usable
    assert text.count("\n- ") == len(usable)  # one line per usable column, no rows


def reply(*metrics):
    return json.dumps({"metrics": list(metrics)})


def test_valid_suggestions_are_kept_unapproved_and_rule_breakers_are_dropped_with_a_reason():
    out = sg.check(
        reply(
            {
                "name": "mrr",
                "description": "Monthly revenue",
                "aggregate": "sum",
                "column": "mrr_usd",
                "unit": "currency",
                "decimals": 0,
            },
            {
                "name": "subs",
                "description": "Subscriptions",
                "aggregate": "count",
                "unit": "count",
                "decimals": 0,
            },
            {
                "name": "churned",
                "description": "Churned",
                "aggregate": "count",
                "filter": {"column": "status", "value": "churned"},
                "unit": "count",
                "decimals": 0,
            },
            {
                "name": "churn_rate",
                "description": "Churn",
                "numerator": "churned",
                "denominator": "subs",
                "unit": "ratio",
                "decimals": 4,
            },
            {
                "name": "emails",
                "description": "x",
                "aggregate": "count_distinct",
                "column": "billing_email",
                "unit": "count",
            },
            {
                "name": "avg_plan",
                "description": "x",
                "aggregate": "avg",
                "column": "plan",
                "unit": "count",
            },
            {
                "name": "bad_ratio",
                "description": "x",
                "numerator": "nope",
                "denominator": "subs",
                "unit": "ratio",
                "decimals": 4,
            },
        ),  # fmt: skip
        PROF,
        FORM,
        sg.context(PROF, FORM)[1],
    )
    kept = {s["name"]: s for s in out["suggestions"]}
    assert set(kept) == {"mrr", "subs", "churned", "churn_rate"}
    assert all(s["approve"] is False and s["known"] == "" for s in kept.values())
    assert kept["churn_rate"]["aggregate"] == "ratio"
    assert kept["churned"]["filterCol"] == "status" and kept["churned"]["filterVal"] == "churned"
    reasons = " | ".join(out["dropped"])
    assert "emails" in reasons and "avg_plan" in reasons and "bad_ratio" in reasons


def test_money_without_a_currency_and_existing_names_are_refused():
    form = FORM | {"currency": "", "metrics": {"subs": {}}}
    out = sg.check(
        reply(
            {"name": "mrr", "aggregate": "sum", "column": "mrr_usd", "unit": "currency"},
            {"name": "subs", "aggregate": "count", "unit": "count"},
        ),
        PROF, form, sg.context(PROF, form)[1],
    )  # fmt: skip
    assert out["suggestions"] == []
    assert any("currency" in d for d in out["dropped"]) and any(
        "exists" in d for d in out["dropped"]
    )


def test_a_reply_that_is_not_json_is_a_plain_error():
    import pytest

    with pytest.raises(sg.SuggestError):
        sg.check("not json", PROF, FORM, set())


def test_internal_tables_are_refused_before_any_model_call():
    from grounded_analytics.cli import _ws_suggest

    out = _ws_suggest(None, "tok", FORM | {"visibility": "internal"}, "groq", None, "k")
    assert "internal" in out["error"]
