"""Verifier behaviour on in-memory executed results (no database needed)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from grounded_analytics.claims import AnswerDoc, Claim, QuerySpec
from grounded_analytics.compiler import MetricQuery, compile_query
from grounded_analytics.semantic import load_semantic_layer
from grounded_analytics.verifier import ExecutedQuery, Verifier, verify_answer

LAYER = load_semantic_layer()
D = Decimal


def compiled(qid, metrics, dims, rows, **kw):
    mq = MetricQuery(metrics=metrics, dimensions=dims, **kw)
    cq = compile_query(LAYER, mq)
    return ExecutedQuery(qid, cq.sql, cq.columns, rows, metric_query=mq)


@pytest.fixture
def q1():
    return compiled(
        "q1",
        ["net_revenue", "order_count", "cancellation_rate"],
        ["country"],
        [
            ("France", D("1234.56"), D("12"), D("0.054")),
            ("Germany", D("2000.00"), D("20"), D("0.1")),
        ],
    )


def cell(qid="q1", col="net_revenue", row=None):
    return {"kind": "cell", "query_id": qid, "column": col, "row_key": row or {"country": "France"}}


def claim(cid, text, value, unit, label, prov):
    return Claim.model_validate(
        {"id": cid, "text": text, "value": value, "unit": unit, "label": label, "provenance": prov}
    )


def run(prose, claims, queries, passages=None, shadow=None):
    qs = {q.id: q for q in queries}
    return Verifier(LAYER, qs, passages, shadow).verify(prose, claims)


def verdicts(report):
    return [v.verdict for v in report.verdicts]


# ---- happy paths --------------------------------------------------------------------------


def test_cell_verified_with_rounding(q1):
    r = run(
        "France net revenue was £1,234.56.",
        [claim("c1", "£1,234.56", "1234.56", "currency", "net revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED"] and r.status == "verified"


def test_cell_rounded_to_millions_scale(q1):
    big = compiled("q2", ["net_revenue"], [], [(D("1234567.89"),)])
    r = run(
        "Net revenue was £1.2m.",
        [
            claim(
                "c",
                "£1.2m",
                "1200000",
                "currency",
                "revenue",
                {"kind": "cell", "query_id": "q2", "column": "net_revenue"},
            )
        ],
        [big],
    )
    assert verdicts(r) == ["VERIFIED"]


def test_ratio_metric_shown_as_percent(q1):
    r = run(
        "France cancellation rate was 5.4%.",
        [claim("c", "5.4%", "5.4", "percent", "cancellation rate", cell(col="cancellation_rate"))],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED"]


def test_derived_percent_change_with_direction(q1):
    prose = "Germany is up 62.0% on France."
    prov = {
        "kind": "derived",
        "op": "percent_change",
        "args": [cell(), cell(row={"country": "Germany"})],
    }
    r = run(prose, [claim("c", "62.0%", "62.0", "percent", "difference", prov)], [q1])
    assert verdicts(r) == ["VERIFIED_DERIVED"], r.verdicts


def test_derived_sum_and_literal_must_be_in_prose(q1):
    prov = {"kind": "derived", "op": "sum", "args": [cell(), cell(row={"country": "Germany"})]}
    r = run(
        "Together they made £3,234.56.",
        [claim("c", "£3,234.56", "3234.56", "currency", "revenue", prov)],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED_DERIVED"]
    prov2 = {"kind": "derived", "op": "sum", "args": [cell(), {"kind": "literal", "value": "999"}]}
    r2 = run(
        "Total £2,233.56.", [claim("c", "£2,233.56", "2233.56", "currency", "revenue", prov2)], [q1]
    )
    assert verdicts(r2) == ["UNSUPPORTED"] and r2.verdicts[0].reason == "literal_not_in_prose"


def test_passage_quote():
    ps = {"p1": "The board expects revenue of about £2.5 million next year."}
    r = run(
        "The board expects £2.5 million.",
        [
            claim(
                "c",
                "£2.5 million",
                "2500000",
                "currency",
                "expected revenue",
                {"kind": "passage", "passage_id": "p1", "span": [0, 55]},
            )
        ],
        [],
        ps,
    )
    assert verdicts(r) == ["VERIFIED_QUOTE"]


def test_passage_number_absent():
    ps = {"p1": "The board expects revenue of £2.5 million."}
    r = run(
        "The board expects £3 million.",
        [
            claim(
                "c",
                "£3 million",
                "3000000",
                "currency",
                "expected revenue",
                {"kind": "passage", "passage_id": "p1", "span": [0, 40]},
            )
        ],
        [],
        ps,
    )
    assert verdicts(r) == ["UNSUPPORTED"]


# ---- a passage must come from a store; the answer file's own passage text is never used

FORGED = "The board expects revenue of about £2.5 million next year."
PASSAGE_CLAIM = {
    "id": "c",
    "text": "£2.5 million",
    "value": "2500000",
    "unit": "currency",
    "label": "expected revenue",
    "provenance": {"kind": "passage", "passage_id": "p1", "span": [0, 55]},
}


def _answer_with_forged_passage():
    return AnswerDoc.model_validate(
        {
            "prose": "The board expects £2.5 million.",
            "passages": [{"id": "p1", "text": FORGED}],
            "claims": [PASSAGE_CLAIM],
        }
    )


def test_verify_answer_without_a_store_rejects_a_model_written_passage():
    r = verify_answer(LAYER, _answer_with_forged_passage(), {})
    assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "no_passage_store")]
    assert r.status != "verified"


def test_a_store_that_lacks_the_passage_is_a_different_reason():
    r = verify_answer(LAYER, _answer_with_forged_passage(), {}, passages={})
    assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "unknown_passage")]


def test_the_store_text_wins_over_the_answer_file_text():
    store = {"p1": "The board expects revenue of about £9.9 million next year."}
    r = verify_answer(LAYER, _answer_with_forged_passage(), {}, passages=store)
    assert [(v.verdict, v.reason) for v in r.verdicts] == [
        ("UNSUPPORTED", "number_not_in_cited_span")
    ]


def test_a_test_may_opt_in_to_the_answer_file_text_only_by_name():
    r = verify_answer(
        LAYER,
        _answer_with_forged_passage(),
        {},
        trust_passages_written_in_the_answer_file=True,
    )
    assert [v.verdict for v in r.verdicts] == ["VERIFIED_QUOTE"]
    with pytest.raises(TypeError):  # not available positionally
        verify_answer(LAYER, _answer_with_forged_passage(), {}, None, None, None, True)


def test_a_verifier_built_without_passages_has_no_store():
    r = Verifier(LAYER, {}).verify(
        "The board expects £2.5 million.", [Claim.model_validate(PASSAGE_CLAIM)]
    )
    assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "no_passage_store")]


def test_row_count_and_parameter(q1):
    r = run(
        "Top 2 countries, 2 rows.",
        [
            claim("a", "2", "2", "count", "countries", {"kind": "row_count", "query_id": "q1"}),
            claim("b", "2", "2", "count", "rows", {"kind": "row_count", "query_id": "q1"}),
        ],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED", "VERIFIED"]
    lim = compiled("q3", ["net_revenue"], ["country"], [("France", D("1"))], limit=5)
    r2 = run(
        "Top 5 countries.",
        [claim("c", "5", "5", "count", "limit", {"kind": "parameter", "query_id": "q3"})],
        [lim],
    )
    assert verdicts(r2) == ["VERIFIED"]


def test_no_numbers_is_verified(q1):
    assert run("Nothing numeric here.", [], [q1]).status == "verified"


# ---- reconciliation ---------------------------------------------------------------------------


def test_number_without_claim_is_unsupported(q1):
    r = run(
        "France made £1,234.56 from 99 orders.",
        [claim("c", "£1,234.56", "1234.56", "currency", "net revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED", "UNSUPPORTED"]
    assert r.verdicts[1].reason == "no_claim" and r.status == "partially_verified"


def test_claim_not_in_prose_is_dropped(q1):
    r = run("No numbers.", [claim("c", "5", "5", "count", "x", cell())], [q1])
    assert r.dropped_claims == ["c"] and r.status == "verified"


def test_ambiguous_extraction_never_verified(q1):
    r = run("About 5m units sold.", [], [q1])
    assert verdicts(r) == ["AMBIGUOUS"] and r.status == "partially_verified"


def test_dates_and_ids_ignored(q1):
    r = run("Invoice 536365 on 2011-03-04 in Q4.", [], [q1])
    assert r.verdicts == [] and len(r.excluded) >= 3


# ---- label binding and metric definition -----------------------------------------------------


def test_label_binds_via_synonym(q1):
    r = run(
        "France revenue was £1,234.56.",
        [claim("c", "£1,234.56", "1234.56", "currency", "revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED"]


def test_wrong_label_for_column_is_metric_mismatch(q1):
    # 12 is France's order_count; the claim says it is net revenue
    r = run(
        "France order count was 12.",
        [claim("c", "12", "12", "count", "net revenue", cell(col="order_count"))],
        [q1],
    )
    assert verdicts(r) == ["METRIC_MISMATCH"]


def test_unit_mismatch_percent_vs_currency(q1):
    r = run(
        "France revenue was 1,234.56%.",
        [claim("c", "1,234.56%", "1234.56", "percent", "net revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["UNSUPPORTED"]


def test_currency_symbol_mismatch(q1):
    r = run(
        "France revenue was $1,234.56.",
        [claim("c", "$1,234.56", "1234.56", "currency", "net revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["UNSUPPORTED"] and r.verdicts[0].reason == "currency_mismatch"


def test_tampered_compiled_sql_is_metric_mismatch(q1):
    q1.sql = q1.sql.replace("FROM", "FROM /* x */").replace("l.country", "l.country")
    q1.sql = q1.sql + " WHERE 1=1"
    r = run(
        "France revenue was £1,234.56.",
        [claim("c", "£1,234.56", "1234.56", "currency", "net revenue", cell())],
        [q1],
    )
    assert verdicts(r) == ["METRIC_MISMATCH"]


def freeform(rows):
    return ExecutedQuery(
        "f1",
        "SELECT country, SUM(line_amount) AS net_revenue FROM analytics.sales_lines GROUP BY 1",
        ["country", "net_revenue"],
        rows,
        declared_metric_query=MetricQuery(metrics=["net_revenue"], dimensions=["country"]),
    )


def test_free_form_needs_declared_canonical():
    q = ExecutedQuery("f1", "SELECT 1", ["net_revenue"], [(D("5"),)])
    r = run(
        "Revenue was £5.",
        [
            claim(
                "c",
                "£5",
                "5",
                "currency",
                "net revenue",
                {"kind": "cell", "query_id": "f1", "column": "net_revenue"},
            )
        ],
        [q],
    )
    assert verdicts(r) == ["AMBIGUOUS"] and r.verdicts[0].reason == "no_canonical_query"


def test_shadow_recomputation_agrees_and_disagrees():
    canon = compiled("s", ["net_revenue"], ["country"], [("France", D("100")), ("Germany", D("7"))])

    def shadow(_):
        return canon

    prov = {
        "kind": "cell",
        "query_id": "f1",
        "column": "net_revenue",
        "row_key": {"country": "France"},
    }
    good = run(
        "France revenue £100.",
        [claim("c", "£100", "100", "currency", "revenue", prov)],
        [freeform([("France", D("100"))])],
        shadow=shadow,
    )
    assert verdicts(good) == ["VERIFIED"]
    bad = run(
        "France revenue £120.",
        [claim("c", "£120", "120", "currency", "revenue", prov)],
        [freeform([("France", D("120"))])],
        shadow=shadow,
    )
    assert verdicts(bad) == ["METRIC_MISMATCH"]
    assert bad.verdicts[0].reason == "disagrees_with_metric_definition"


# ---- qualifiers, direction, row selection --------------------------------------------------------


def test_lower_and_upper_bounds(q1):
    base = lambda t, v: claim("c", t, v, "currency", "net revenue", cell())  # noqa: E731
    assert verdicts(run("France net revenue was over £1,000.", [base("£1,000", "1000")], [q1])) == [
        "VERIFIED"
    ]
    assert verdicts(run("France net revenue was over £2,000.", [base("£2,000", "2000")], [q1])) == [
        "UNSUPPORTED"
    ]
    assert verdicts(
        run("France net revenue was under £2,000.", [base("£2,000", "2000")], [q1])
    ) == ["VERIFIED"]
    assert verdicts(
        run("France net revenue was under £1,000.", [base("£1,000", "1000")], [q1])
    ) == ["UNSUPPORTED"]


def test_direction_word_must_match_sign(q1):
    prov = {
        "kind": "derived",
        "op": "difference",
        "args": [cell(), cell(row={"country": "Germany"})],
    }
    c = lambda t, v: claim("c", t, v, "currency", "net revenue", prov)  # noqa: E731
    assert verdicts(run("France fell by £765.44.", [c("£765.44", "765.44")], [q1])) == [
        "VERIFIED_DERIVED"
    ]
    assert verdicts(run("France rose by £765.44.", [c("£765.44", "765.44")], [q1])) == [
        "DERIVATION_MISMATCH"
    ]


def test_row_not_found_and_ambiguous_rows(q1):
    r = run(
        "Spain revenue was £1.",
        [claim("c", "£1", "1", "currency", "net revenue", cell(row={"country": "Spain"}))],
        [q1],
    )
    assert r.verdicts[0].reason == "row_not_found"
    dup = compiled("d", ["net_revenue"], ["country"], [("France", D("1")), ("France", D("2"))])
    r2 = run(
        "France revenue was £1.",
        [claim("c", "£1", "1", "currency", "net revenue", cell(qid="d"))],
        [dup],
    )
    assert verdicts(r2) == ["AMBIGUOUS"]


def test_division_by_zero_and_wrong_arity(q1):
    z = compiled("z", ["order_count"], [], [(D("0"),)])
    zc = {"kind": "cell", "query_id": "z", "column": "order_count"}
    r = run(
        "Ratio 5.",
        [
            claim(
                "c",
                "5",
                "5",
                "number",
                "ratio",
                {"kind": "derived", "op": "ratio", "args": [zc, zc]},
            )
        ],
        [z],
    )
    assert r.verdicts[0].reason == "division_by_zero"


def test_claim_value_must_equal_prose(q1):
    r = run(
        "France revenue was £1,234.56.",
        [claim("c", "£1,234.56", "9999", "currency", "net revenue", cell())],
        [q1],
    )
    assert r.verdicts[0].reason == "claim_value_differs_from_text"


def test_repeated_numbers_consume_claims_in_order(q1):
    r = run(
        "France 12 orders; Germany 12 orders.",
        [
            claim("a", "12", "12", "count", "order count", cell(col="order_count")),
            claim(
                "b",
                "12",
                "12",
                "count",
                "order count",
                cell(col="order_count", row={"country": "Germany"}),
            ),
        ],
        [q1],
    )
    assert verdicts(r) == ["VERIFIED", "UNSUPPORTED"]  # Germany really has 20


def test_schema_rejects_bad_claims():
    with pytest.raises(ValidationError):
        Claim.model_validate(
            {
                "id": "x",
                "text": "1",
                "value": "abc",
                "unit": "count",
                "label": "l",
                "provenance": {"kind": "row_count", "query_id": "q"},
            }
        )
    with pytest.raises(ValidationError):
        QuerySpec(id="a")  # neither metric_query nor sql
    with pytest.raises(ValidationError):
        QuerySpec(id="a", sql="SELECT 1", metric_query=MetricQuery(metrics=["net_revenue"]))
    with pytest.raises(ValidationError):
        AnswerDoc.model_validate({"prose": "x", "claims": [], "extra": 1})
