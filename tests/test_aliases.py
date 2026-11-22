"""A value alias behaves exactly like the value it stands for ("UK" == "United Kingdom").

Background: before this, the binder did not see "UK", so "UK net revenue was £9,000.09." on a
one-row global total, or on a France-filtered query, came back VERIFIED. These tests pin that the
lies are rejected, the true sentences verify, and the two spellings never differ. Synthetic data;
unit tests by the author of the check, not an accuracy claim."""

from __future__ import annotations

import pytest

from adversarial_cases import LAYER
from alias_cases import (
    ALL,
    CONTROLS,
    KNOWN_K,
    LIES,
    WHO,
    alias_queries,
    prose_claims,
    template_claims,
)
from grounded_analytics.claims import TemplateAnswer
from grounded_analytics.template import verify_template
from grounded_analytics.verifier import Verifier

QS = alias_queries()


def prose_result(case, who):
    prose, claims = prose_claims(case, who)
    r = Verifier(LAYER, QS, None, None, KNOWN_K).verify(prose, claims)
    return r.status, [(v.verdict, v.reason) for v in r.verdicts]


def template_result(case, who):
    template, claims = template_claims(case, who)
    ans = TemplateAnswer.model_validate({"template": template, "claims": claims})
    r = verify_template(LAYER, ans, QS, None, KNOWN_K)
    return r.status, [(v.verdict, v.reason) for v in r.report.verdicts]


def test_the_shipped_layer_lists_uk_for_the_united_kingdom_only():
    assert LAYER.dimensions["country"].value_aliases == {"United Kingdom": ["UK"]}
    assert all(not d.value_aliases for n, d in LAYER.dimensions.items() if n != "country")


@pytest.mark.parametrize("who", WHO)
@pytest.mark.parametrize("case", LIES, ids=[c.name for c in LIES])
def test_alias_lies_are_not_verified_in_prose_and_placeholder_form(case, who):
    assert prose_result(case, who)[0] != "verified", (who, prose_result(case, who))
    assert template_result(case, who)[0] != "verified", (who, template_result(case, who))


@pytest.mark.parametrize("who", WHO)
@pytest.mark.parametrize("case", CONTROLS, ids=[c.name for c in CONTROLS])
def test_true_alias_sentences_verify_in_prose_and_placeholder_form(case, who):
    assert prose_result(case, who)[0] == "verified", (who, prose_result(case, who))
    assert template_result(case, who)[0] == "verified", (who, template_result(case, who))


@pytest.mark.parametrize("case", ALL, ids=[c.name for c in ALL])
def test_the_alias_and_the_full_name_give_the_same_verdicts_and_reasons(case):
    assert prose_result(case, "UK") == prose_result(case, "United Kingdom")
    assert template_result(case, "UK") == template_result(case, "United Kingdom")


def test_false_accepts_of_the_alias_lies_are_zero():
    accepted = [
        (c.name, who)
        for c in LIES
        for who in WHO
        if prose_result(c, who)[0] == "verified" or template_result(c, who)[0] == "verified"
    ]
    assert len(LIES) == 10 and accepted == []


def test_a_name_the_layer_does_not_list_is_still_not_read():
    """Documented limit, unchanged: only listed aliases are read. 'Britain' is not listed."""
    from adversarial_cases import _claim, cell

    prov = cell("qc", "net_revenue", country="United Kingdom")
    claim = _claim("c1", "£4,444.44", "4444.44", "currency", "net revenue", prov)
    r = Verifier(LAYER, QS, None, None, KNOWN_K).verify(
        "Britain net revenue was £4,444.44.", [claim]
    )
    assert r.status != "verified" and r.verdicts[0].verdict == "AMBIGUOUS"


def test_an_alias_is_read_only_where_its_value_is_in_play():
    """Like the full name: with no country values known and no country column, nothing is read."""
    from adversarial_cases import _claim, cell

    claim = _claim(
        "c1", "£9,000.09", "9000.09", "currency", "net revenue", cell("qt", "net_revenue")
    )
    for who in WHO:
        r = Verifier(LAYER, QS, None, None, {}).verify(f"{who} net revenue was £9,000.09.", [claim])
        assert r.status == "verified", who  # the same documented blindness for both spellings
