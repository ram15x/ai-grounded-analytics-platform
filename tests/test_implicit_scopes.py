"""Implicit scopes: a metric whose definition already applies a scope ("merchandise units")
may be described with that scope word; nothing else may.

Origin: a real Groq probe (2026-11-20, plan.v1) answered "The total merchandise units were {c1}."
and the binder returned UNSUPPORTED scope_not_in_query, although units_sold is DEFINED as
merchandise sale lines. The fix is declared per metric in semantic/metrics.yaml (implicit_scopes),
honoured for that metric only. These tests pin both directions: the true description verifies, and
every lookalike lie still does not. Synthetic data only; unit tests, not an accuracy claim.
"""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from adversarial_cases import COUNTRIES, KNOWN, LAYER, _claim, cases, cell, queries
from verified_analyst.claims import TemplateAnswer
from verified_analyst.semantic import SemanticLayer, default_path
from verified_analyst.template import verify_template
from verified_analyst.verifier import Verifier

QS = queries()
KNOWN_K = {**KNOWN, "product_kind": {"merchandise", "non_merchandise"}}  # as the pipeline loads
TOTAL_UNITS = cell("qt", "units_sold")  # 777, one row, no filter


def prose_run(prose, claims, known=KNOWN_K):
    return Verifier(LAYER, QS, None, None, known).verify(prose, claims)


def count_claim(text, value, prov, label="units", cid="c1"):
    return _claim(cid, text, value, "count", label, prov)


def money_claim(text, value, prov, label, cid="c1"):
    return _claim(cid, text, value, "currency", label, prov)


def tmpl_run(template, claims, known=KNOWN_K):
    ans = TemplateAnswer.model_validate({"template": template, "claims": claims})
    return verify_template(LAYER, ans, QS, None, known)


def tclaim(cid, prov, label, unit):
    return {"id": cid, "label": label, "unit": unit, "provenance": prov}


def reasons(report):
    return {v.reason for v in report.verdicts}


# ---- the layer declares it, for one metric ------------------------------------------------------


def test_only_units_sold_declares_an_implicit_scope():
    declared = {n: m.implicit_scopes for n, m in LAYER.metrics.items() if m.implicit_scopes}
    assert declared == {"units_sold": {"product_kind": ["merchandise"]}}


def test_the_plan_catalogue_does_not_carry_implicit_scopes():
    """The planner prompt (plan.v1) is unchanged by this feature."""
    from verified_analyst.llm.prompts import catalogue_text

    assert "implicit_scopes" not in catalogue_text(LAYER)


# ---- the true description now verifies ----------------------------------------------------------


def test_the_probe_sentence_verifies_in_placeholder_form():
    r = tmpl_run(
        "The total merchandise units were {c1}.",
        [tclaim("c1", TOTAL_UNITS, "total merchandise units", "count")],
    )
    assert r.status == "verified", (r.issues, [v.as_dict() for v in r.report.verdicts])


def test_the_probe_sentence_verifies_in_prose_form():
    r = prose_run("The total merchandise units were 777.", [count_claim("777", "777", TOTAL_UNITS)])
    assert r.status == "verified", [v.as_dict() for v in r.verdicts]


def test_a_row_of_units_sold_with_the_scope_word_verifies():
    prov = cell("qc", "units_sold", country="France")
    r = prose_run("France merchandise units were 101.", [count_claim("101", "101", prov)])
    assert r.status == "verified", [v.as_dict() for v in r.verdicts]


def test_a_sentence_without_the_scope_word_still_verifies():
    r = prose_run("Total units were 777.", [count_claim("777", "777", TOTAL_UNITS)])
    assert r.status == "verified"


# ---- lookalike lies stay rejected ---------------------------------------------------------------


@pytest.mark.parametrize("who", ["UK", "United Kingdom"])
def test_a_country_on_merchandise_units_with_no_country_filter_is_unsupported(who):
    prose = f"{who} merchandise units were 777."
    r = prose_run(prose, [count_claim("777", "777", TOTAL_UNITS)])
    assert r.status != "verified"
    assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "scope_not_in_query")]
    t = tmpl_run(
        f"{who} merchandise units were {{c1}}.", [tclaim("c1", TOTAL_UNITS, "units", "count")]
    )
    assert t.status != "verified" and "scope_not_in_query" in reasons(t.report)


def test_implicit_scope_does_not_cover_a_different_scope_word_on_the_same_cell():
    r = prose_run("France merchandise units were 777.", [count_claim("777", "777", TOTAL_UNITS)])
    assert r.verdicts[0].reason == "scope_not_in_query"  # France is not in the query


@pytest.mark.parametrize("phrase", ["non-merchandise", "non merchandise", "non_merchandise"])
def test_the_opposite_scope_is_unsupported_for_units_sold(phrase):
    r = prose_run(f"The total {phrase} units were 777.", [count_claim("777", "777", TOTAL_UNITS)])
    assert r.status != "verified" and r.verdicts[0].reason == "scope_not_in_query"


METRIC_LABELS = {
    "net_revenue": "net revenue",
    "gross_revenue": "gross revenue",
    "cancelled_revenue": "cancelled revenue",
    "discount_amount": "discount amount",
    "order_count": "order count",
    "active_customers": "active customers",
    "aov": "average order value",
    "revenue_per_customer": "revenue per customer",
}


@pytest.mark.parametrize("metric", sorted(METRIC_LABELS))
def test_merchandise_on_any_metric_without_the_implicit_scope_is_unsupported(metric):
    """qt holds one row per metric, no filter: 'merchandise <metric>' adds a scope nobody ran."""
    unit = LAYER.metrics[metric].unit_kind
    value = {"net_revenue": "9000.09", "gross_revenue": "9500.95", "cancelled_revenue": "400.04",
             "discount_amount": "100.01", "order_count": "100", "active_customers": "50",
             "aov": "90.09", "revenue_per_customer": "180.18"}[metric]  # fmt: skip
    prov = cell("qt", metric)
    claim = _claim(
        "c1",
        value,
        value,
        "currency" if unit == "currency" else "count",
        METRIC_LABELS[metric],
        prov,
    )
    r = prose_run(f"Merchandise {METRIC_LABELS[metric]} was {value}.", [claim])
    assert r.status != "verified", metric
    assert r.verdicts[0].reason == "scope_not_in_query", (metric, r.verdicts[0].as_dict())


def test_merchandise_gross_revenue_in_placeholder_form_is_unsupported():
    t = tmpl_run(
        "Merchandise gross revenue was {c1}.",
        [tclaim("c1", cell("qt", "gross_revenue"), "gross revenue", "currency")],
    )
    assert t.status != "verified" and "scope_not_in_query" in reasons(t.report)


def test_the_scope_is_not_borrowed_across_claims_in_one_sentence():
    """units_sold has the scope; gross_revenue in the same sentence does not."""
    claims = [
        count_claim("777", "777", cell("qt", "units_sold"), cid="a"),
        money_claim("£9,500.95", "9500.95", cell("qt", "gross_revenue"), "gross revenue", cid="b"),
    ]
    r = prose_run(
        "Merchandise units were 777, and merchandise gross revenue was £9,500.95.", claims
    )
    by_id = {v.claim_id: v for v in r.verdicts}
    assert by_id["a"].ok and not by_id["b"].ok
    assert by_id["b"].reason == "scope_not_in_query" and r.status != "verified"


def test_a_filter_on_the_scope_dimension_is_still_judged_by_the_filter_not_the_metric():
    """With a product_kind filter in the query the old rule applies; the metric's own scope is
    only a fallback when the query has no filter on that dimension."""
    from adversarial_cases import _q

    q = _q(
        "qk", ["gross_revenue"], [], [(__import__("decimal").Decimal("1.00"),)],
        filters=[{"dimension": "product_kind", "op": "eq", "value": "merchandise"}],
    )  # fmt: skip
    ok = Verifier(LAYER, {"qk": q}, None, None, KNOWN_K).verify(
        "Merchandise gross revenue was £1.00.",
        [money_claim("£1.00", "1.00", cell("qk", "gross_revenue"), "gross revenue")],
    )
    assert ok.status == "verified"
    bad = Verifier(LAYER, {"qk": q}, None, None, KNOWN_K).verify(
        "Non-merchandise gross revenue was £1.00.",
        [money_claim("£1.00", "1.00", cell("qk", "gross_revenue"), "gross revenue")],
    )
    assert bad.status != "verified" and bad.verdicts[0].reason == "scope_mismatch"


# ---- aliases are covered in test_aliases.py; the units-specific cases stay here ------------------


def _units_query(qid, **kw):
    from decimal import Decimal

    from adversarial_cases import _q

    return _q(qid, ["units_sold"], [], [(Decimal("55"),)], **kw)


def _run_with(extra, prose, claim):
    return Verifier(LAYER, {**QS, **extra}, None, None, KNOWN_K).verify(prose, [claim])


@pytest.mark.parametrize("who", ["UK", "United Kingdom"])
def test_merchandise_units_with_a_united_kingdom_filter_verifies(who):
    q = _units_query(
        "qu", filters=[{"dimension": "country", "op": "eq", "value": "United Kingdom"}]
    )
    claim = count_claim("55", "55", cell("qu", "units_sold"))
    r = _run_with({"qu": q}, f"{who} merchandise units were 55.", claim)
    assert r.status == "verified", [v.as_dict() for v in r.verdicts]


@pytest.mark.parametrize("who", ["UK", "United Kingdom"])
def test_merchandise_units_with_a_france_filter_is_unsupported(who):
    q = _units_query("qu", filters=[{"dimension": "country", "op": "eq", "value": "France"}])
    claim = count_claim("55", "55", cell("qu", "units_sold"))
    r = _run_with({"qu": q}, f"{who} merchandise units were 55.", claim)
    assert r.status != "verified" and r.verdicts[0].reason == "scope_mismatch"


# ---- the 30 adversarial answers, with the scope dimension's values known -------------------------

CASES = cases()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_the_30_stay_unverified_with_product_kind_values_loaded(case):
    r = prose_run(case.prose, case.claims)
    assert r.status != "verified", [v.as_dict() for v in r.verdicts]


def test_false_accepts_are_zero_of_30_in_prose_and_placeholder_form_with_product_kind_known():
    from test_template import to_template

    prose_accepts = [c.name for c in CASES if prose_run(c.prose, c.claims).status == "verified"]
    t_accepts = []
    for c in CASES:
        template, claims = to_template(c)
        if tmpl_run(template, claims).status == "verified":
            t_accepts.append(c.name)
    assert len(CASES) == 30 and prose_accepts == [] and t_accepts == []


# ---- the layer refuses a declaration its expr does not back up ----------------------------


def _raw():
    return copy.deepcopy(yaml.safe_load(default_path().read_text("utf-8")))


def _load(mutate):
    data = _raw()
    mutate(data)
    return SemanticLayer.model_validate(data)


def test_the_shipped_layer_loads():
    assert _load(lambda d: None).metrics["units_sold"].implicit_scopes


def _declare(metric, dim, values, **extra):
    """Mutation: give `metric` an implicit scope, taking `dim` out of its allowed dimensions
    first so the declaration is judged on its expr and not on the grouping rule."""

    def mutate(d):
        m = d["metrics"][metric]
        m["allowed_dimensions"] = [x for x in m["allowed_dimensions"] if x != dim]
        m["implicit_scopes"] = {dim: values}
        m.update(extra)

    return mutate


NEGATED = "COALESCE(SUM(l.quantity) FILTER (WHERE l.product_kind != 'merchandise'), 0)"
BAD_DECLARATIONS = {
    "expr lacks it": (_declare("gross_revenue", "product_kind", ["merchandise"]), "expr does not"),
    "wrong value": (_declare("units_sold", "product_kind", ["non_merchandise"]), "expr does not"),
    "unknown dim": (_declare("units_sold", "planet", ["x"]), "unknown dimension"),
    "not enumerable": (_declare("units_sold", "stock_code", ["x"]), "enumerable text"),
    "empty": (_declare("units_sold", "product_kind", []), "non-empty list"),
    "ratio metric": (_declare("aov", "product_kind", ["merchandise"]), "only a simple metric"),
    "negated expr": (_declare("units_sold", "product_kind", ["merchandise"], expr=NEGATED),
                     "expr does not"),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(BAD_DECLARATIONS))
def test_bad_implicit_scope_declarations_are_refused(name):
    mutate, match = BAD_DECLARATIONS[name]
    with pytest.raises(ValueError, match=match):
        _load(mutate)


def test_a_metric_cannot_be_both_fixed_to_and_grouped_by_the_scope():
    def mutate(d):
        d["metrics"]["units_sold"]["allowed_dimensions"].append("product_kind")

    with pytest.raises(ValueError, match="both be fixed to and grouped by"):
        _load(mutate)


def test_alias_declarations_are_checked():
    with pytest.raises(ValueError, match="empty or used twice"):
        _load(lambda d: d["dimensions"]["country"].update(
            value_aliases={"United Kingdom": ["UK"], "France": ["UK"]}))  # fmt: skip
    with pytest.raises(ValueError, match="text dimension"):
        _load(lambda d: d["dimensions"]["month"].update(value_aliases={"x": ["y"]}))


def test_the_countries_helper_still_lists_the_uk():
    assert "United Kingdom" in COUNTRIES and json.dumps(sorted(KNOWN_K["product_kind"]))
