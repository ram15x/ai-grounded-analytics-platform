"""30 adversarial answers must never come back 'verified' (false-accept count must be 0).

They were written BEFORE the sentence-context check existed; against the verifier as it stood at
that point all 30 were falsely accepted (30/30). The extension (src/grounded_analytics/context.py)
brought that to 0. Controls guard the other direction: legitimate answers must still verify.
"""

import pytest

from adversarial_cases import KNOWN, LAYER, cases, cell, gbp, queries
from grounded_analytics.verifier import Verifier

CASES = cases()


def run(prose, claims, known=KNOWN):
    return Verifier(LAYER, queries(), None, None, known).verify(prose, claims)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_adversarial_answer_is_not_verified(case):
    report = run(case.prose, case.claims)
    assert report.status != "verified", [v.as_dict() for v in report.verdicts]


def test_false_accept_count_is_zero():
    accepted = [c.name for c in CASES if run(c.prose, c.claims).status == "verified"]
    assert len(CASES) == 30
    assert accepted == []


def test_categories_are_all_covered():
    cats = {c.category for c in CASES}
    assert cats == {"wrong_row", "wrong_period", "wrong_dimension_value", "wrong_metric_same_unit"}


CONTROLS = [
    (
        "France net revenue was £1,111.11.",
        "£1,111.11",
        "1111.11",
        "net revenue",
        cell("qc", "net_revenue", country="France"),
    ),
    (
        "Germany net revenue was £2,222.22.",
        "£2,222.22",
        "2222.22",
        "net revenue",
        cell("qc", "net_revenue", country="Germany"),
    ),
    (
        "EIRE made £3,333.33 in net revenue.",
        "£3,333.33",
        "3333.33",
        "net revenue",
        cell("qc", "net_revenue", country="EIRE"),
    ),
    (
        "In February net revenue was £5,202.02.",
        "£5,202.02",
        "5202.02",
        "net revenue",
        cell("qm", "net_revenue", month="2011-02-01"),
    ),
    (
        "Net revenue for March 2011 was £5,303.03.",
        "£5,303.03",
        "5303.03",
        "net revenue",
        cell("qm", "net_revenue", month="2011-03-01"),
    ),
    (
        "Net revenue for 2011-01 was £5,101.01.",
        "£5,101.01",
        "5101.01",
        "net revenue",
        cell("qm", "net_revenue", month="2011-01-01"),
    ),
    (
        "Gross revenue was £9,500.95.",
        "£9,500.95",
        "9500.95",
        "gross revenue",
        cell("qt", "gross_revenue"),
    ),
    ("Sales were £9,000.09.", "£9,000.09", "9000.09", "net revenue", cell("qt", "net_revenue")),
    (
        "France net revenue was £1,111.11.",
        "£1,111.11",
        "1111.11",
        "net revenue",
        cell("qf", "net_revenue"),
    ),
    (
        "France and EIRE together made £3,333.33 of net revenue.",
        "£3,333.33",
        "3333.33",
        "net revenue",
        cell("qi", "net_revenue"),
    ),
    (
        "January net revenue was £5,101.01.",
        "£5,101.01",
        "5101.01",
        "net revenue",
        cell("qj", "net_revenue"),
    ),
    (
        "Cancelled revenue was £400.04.",
        "£400.04",
        "400.04",
        "cancelled revenue",
        cell("qt", "cancelled_revenue"),
    ),
]


@pytest.mark.parametrize("prose,text,val,label,prov", CONTROLS, ids=[c[0][:40] for c in CONTROLS])
def test_legitimate_answers_still_verify(prose, text, val, label, prov):
    report = run(prose, [gbp(text, val, label, prov)])
    assert report.status == "verified", [v.as_dict() for v in report.verdicts]


def test_two_numbers_two_rows_one_sentence_verifies():
    prose = "France made £1,111.11, and Germany made £2,222.22, of net revenue."
    claims = [
        gbp(
            "£1,111.11", "1111.11", "net revenue", cell("qc", "net_revenue", country="France"), "a"
        ),
        gbp(
            "£2,222.22", "2222.22", "net revenue", cell("qc", "net_revenue", country="Germany"), "b"
        ),
    ]
    assert run(prose, claims).status == "verified"


# ---- documented limits (NOT counted in the 30); behaviour pinned so a change is noticed ----


def test_limit_two_numbers_in_one_clause_are_ambiguous_not_guessed():
    """Convention for composers: one number per clause (comma or semicolon between them)."""
    prose = "France made £1,111.11 and Germany made £2,222.22 of net revenue."
    claims = [
        gbp(
            "£1,111.11", "1111.11", "net revenue", cell("qc", "net_revenue", country="France"), "a"
        ),
        gbp(
            "£2,222.22", "2222.22", "net revenue", cell("qc", "net_revenue", country="Germany"), "b"
        ),
    ]
    r = run(prose, claims)
    assert r.status != "verified"
    assert r.verdicts[0].verdict == "AMBIGUOUS"  # equidistant from both names: not guessed


def test_limit_name_not_in_layer_is_ambiguous_not_verified():
    prov = cell("qc", "net_revenue", country="United Kingdom")
    r = run(
        "Britain net revenue was £4,444.44.", [gbp("£4,444.44", "4444.44", "net revenue", prov)]
    )
    assert r.status != "verified" and r.verdicts[0].verdict == "AMBIGUOUS"


def test_alias_listed_in_layer_acts_like_the_full_name():
    """ "UK" is listed as an alias of "United Kingdom" in the layer; see test_aliases.py."""
    prov = cell("qc", "net_revenue", country="United Kingdom")
    r = run("UK net revenue was £4,444.44.", [gbp("£4,444.44", "4444.44", "net revenue", prov)])
    full = run(
        "United Kingdom net revenue was £4,444.44.", [gbp("£4,444.44", "4444.44", "net", prov)]
    )
    assert r.status == full.status == "verified"


def test_limit_together_phrasing_can_still_fool_nearest_mention():
    """KNOWN GAP: 'France and Germany together £X' attaches X to the nearer name."""
    prov = cell("qc", "net_revenue", country="Germany")
    r = run(
        "France and Germany together made £2,222.22.",
        [gbp("£2,222.22", "2222.22", "net revenue", prov)],
    )
    assert (
        r.status == "verified"
    )  # a false accept this heuristic cannot see; see docs/VERIFIER_LIMITS.md
