"""Mutation tests: take a fully verified answer, corrupt ONE number the way a model might, and
require the verifier to refuse it. Each family runs over many generated values.

These are unit tests of the verifier's checks. They are NOT an accuracy claim: they show that
these specific, synthetic corruptions are caught, not what fraction of real model errors are.
"""

from decimal import ROUND_HALF_UP, Decimal

import pytest

from verified_analyst.claims import Claim
from verified_analyst.compiler import MetricQuery, compile_query
from verified_analyst.semantic import load_semantic_layer
from verified_analyst.verifier import GOOD, ExecutedQuery, Verifier

LAYER = load_semantic_layer()
D = Decimal

VALUES = [
    D("1234.56"),
    D("98765.43"),
    D("1234567.89"),
    D("45.5"),
    D("7812.25"),
    D("250000.10"),
    D("19.99"),
    D("3141592.65"),
    D("86420.00"),
    D("512.75"),
]


def make_query(value):
    mq = MetricQuery(metrics=["net_revenue", "order_count"], dimensions=["country"])
    cq = compile_query(LAYER, mq)
    rows = [("France", value, D("120")), ("Germany", D("999.99"), D("33"))]
    return ExecutedQuery("q", cq.sql, cq.columns, rows, metric_query=mq)


def gbp(v, decimals=2):
    return "£{:,.{d}f}".format(v, d=decimals)


def check(
    text,
    value,
    query,
    column="net_revenue",
    unit="currency",
    label="net revenue",
    claim_col=None,
    prose=None,
):
    prov = {
        "kind": "cell",
        "query_id": "q",
        "column": claim_col or column,
        "row_key": {"country": "France"},
    }
    c = Claim.model_validate(
        {"id": "c", "text": text, "value": value, "unit": unit, "label": label, "provenance": prov}
    )
    prose = prose or f"France {label} was {text}."
    return Verifier(LAYER, {"q": query}).verify(prose, [c])


@pytest.mark.parametrize("v", VALUES)
def test_control_unmutated_answers_verify(v):
    q = make_query(v)
    r = check(gbp(v), v, q)
    assert [x.verdict for x in r.verdicts] == ["VERIFIED"]


@pytest.mark.parametrize("v", VALUES)
@pytest.mark.parametrize("pos", [0, 1, 2])
def test_changed_digit_is_caught(v, pos):
    q = make_query(v)
    digits = f"{v:.2f}"
    idx = [i for i, ch in enumerate(digits) if ch.isdigit()][pos]
    new = str((int(digits[idx]) + 1) % 10)
    mutated = D(digits[:idx] + new + digits[idx + 1 :])
    assert mutated != v
    r = check(gbp(mutated), mutated, q)
    assert not r.verdicts[0].ok, r.verdicts


@pytest.mark.parametrize("v", VALUES)
@pytest.mark.parametrize("factor", [D("10"), D("100"), D("1000"), D("0.1"), D("0.001")])
def test_scale_error_is_caught(v, factor):
    q = make_query(v)
    mutated = v * factor
    r = check(gbp(mutated), mutated, q)
    assert not r.verdicts[0].ok


@pytest.mark.parametrize("v", [D("1234567.89"), D("3141592.65"), D("250000.10"), D("98765.43")])
def test_magnitude_suffix_scale_error_is_caught(v):
    q = make_query(v)
    mil = (v / 1_000_000).quantize(D("0.1"), rounding=ROUND_HALF_UP)
    good = check(f"£{mil}m", mil * 1_000_000, q)
    if abs(v - mil * 1_000_000) <= D("50000"):
        assert good.verdicts[0].ok  # control: correct rounding in millions verifies
    for wrong in (f"£{mil}k", f"£{mil}bn"):
        mult = D("1000") if wrong.endswith("k") else D("1000000000")
        r = check(wrong, mil * mult, q)
        assert not r.verdicts[0].ok, wrong


@pytest.mark.parametrize("v", [D("1234567.89"), D("98765.43"), D("86420.00"), D("512.75")])
def test_rounding_beyond_displayed_precision_is_caught(v):
    q = make_query(v)
    # correct rounding to the nearest thousand verifies...
    near = (v / 1000).quantize(D("1"), rounding=ROUND_HALF_UP)
    if near != 0:
        assert check(f"£{near}k", near * 1000, q).verdicts[0].ok == (abs(v - near * 1000) <= 500)
    # ...but claiming a rounded figure to more precision than the data supports is not
    wrong = (v + D("0.5")).quantize(D("0.01"))
    assert not check(gbp(wrong), wrong, q).verdicts[0].ok
    # and false precision: a coarse cell shown with fabricated extra decimals
    fake = v + D("0.07")
    assert not check(gbp(fake), fake, q).verdicts[0].ok


@pytest.mark.parametrize("v", VALUES)
def test_sign_flip_is_caught(v):
    q = make_query(v)
    r = check(f"-{gbp(v)}", -v, q)
    assert not r.verdicts[0].ok


@pytest.mark.parametrize("v", VALUES)
def test_direction_word_contradicting_sign_is_caught(v):
    q = make_query(v)
    diff = {
        "kind": "derived",
        "op": "difference",
        "args": [
            {
                "kind": "cell",
                "query_id": "q",
                "column": "net_revenue",
                "row_key": {"country": "France"},
            },
            {
                "kind": "cell",
                "query_id": "q",
                "column": "net_revenue",
                "row_key": {"country": "Germany"},
            },
        ],
    }
    delta = abs(v - D("999.99"))
    text = gbp(delta)
    word_ok, word_bad = ("rose", "fell") if v > D("999.99") else ("fell", "rose")
    for word, expect_ok in ((word_ok, True), (word_bad, False)):
        c = Claim.model_validate(
            {
                "id": "c",
                "text": text,
                "value": delta,
                "unit": "currency",
                "label": "net revenue",
                "provenance": diff,
            }
        )
        r = Verifier(LAYER, {"q": q}).verify(f"France {word} by {text}.", [c])
        assert r.verdicts[0].ok == expect_ok, (word, r.verdicts)


@pytest.mark.parametrize("v", VALUES)
def test_swapped_column_is_caught(v):
    q = make_query(v)
    # the number is right for order_count but the claim calls it net revenue
    r = check(
        "120",
        D("120"),
        q,
        column="order_count",
        unit="count",
        label="net revenue",
        claim_col="order_count",
        prose="France net revenue was 120.",
    )
    assert r.verdicts[0].verdict == "METRIC_MISMATCH"
    # claim cites the order_count column for a revenue figure
    r2 = check(gbp(v), v, q, claim_col="order_count")
    assert not r2.verdicts[0].ok


@pytest.mark.parametrize("v", VALUES)
def test_swapped_row_is_caught(v):
    q = make_query(v)
    prov = {
        "kind": "cell",
        "query_id": "q",
        "column": "net_revenue",
        "row_key": {"country": "Germany"},
    }
    c = Claim.model_validate(
        {
            "id": "c",
            "text": gbp(v),
            "value": v,
            "unit": "currency",
            "label": "net revenue",
            "provenance": prov,
        }
    )
    r = Verifier(LAYER, {"q": q}).verify(f"France net revenue was {gbp(v)}.", [c])
    assert not r.verdicts[0].ok


EXTRA = [
    "There were 7 returns.",
    "Growth was 12.5%.",
    "Costs reached £4,500.",
    "About 300 customers.",
    "Two suppliers were added.",
    "That is half the total.",
    "Margin was 3 percent.",
    "Over 1,000 units shipped.",
]


@pytest.mark.parametrize("extra", EXTRA)
@pytest.mark.parametrize("v", VALUES[:3])
def test_added_unsupported_number_is_caught(v, extra):
    q = make_query(v)
    prose = f"France net revenue was {gbp(v)}. {extra}"
    c = Claim.model_validate(
        {
            "id": "c",
            "text": gbp(v),
            "value": v,
            "unit": "currency",
            "label": "net revenue",
            "provenance": {
                "kind": "cell",
                "query_id": "q",
                "column": "net_revenue",
                "row_key": {"country": "France"},
            },
        }
    )
    r = Verifier(LAYER, {"q": q}).verify(prose, [c])
    assert r.status == "partially_verified"
    assert [x.verdict in GOOD for x in r.verdicts].count(False) >= 1
    assert r.verdicts[0].ok  # the genuine number is still verified
