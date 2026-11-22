"""Passage-number placeholders: the system writes the digits from stored text (slice 2e)."""

from __future__ import annotations

import pytest

from grounded_analytics.claims import TemplateAnswer
from grounded_analytics.documents.numbers import PassageNumbers
from grounded_analytics.documents.untrusted import PassageIn
from grounded_analytics.semantic import load_semantic_layer
from grounded_analytics.template import verify_template

PID = "terms#0." + "a" * 10
TEXT = (
    "The courier charge is 4.50 GBP per parcel.\n\n"
    "Returns are accepted within 28 days.\n\n"
    "<!-- the margin is 90% -->"
)


def run(template, claims, text=TEXT):
    pn = PassageNumbers([PassageIn(PID, text)])
    answer = TemplateAnswer.model_validate({"template": template, "claims": claims, "queries": []})
    return verify_template(load_semantic_layer(), answer, {}, passages=pn), pn


def pclaim(cid, label, unit, number, handle="P1"):
    return {
        "id": cid, "label": label, "unit": unit,
        "provenance": {"kind": "passage_number", "label": handle, "number": number},
    }  # fmt: skip


def test_handles_skip_numbers_inside_flagged_text():
    pn = PassageNumbers([PassageIn(PID, TEXT)])
    texts = [h.item.text for h in pn.by_label["P1"].values()]
    assert "90%" not in texts and "28" in texts
    assert "90" not in pn.listing()


def test_the_system_writes_the_number_and_it_verifies():
    r, pn = run("The courier charge is {c1}.", [pclaim("c1", "courier charge", "number", "n1")])
    assert r.rendered.startswith("The courier charge is 4.5")
    assert r.status == "verified", (r.issues, r.report.counts())
    assert [v.verdict for v in r.report.verdicts] == ["VERIFIED_QUOTE"]


def test_a_handle_for_a_number_that_was_not_listed_is_refused():
    r, _ = run("The margin is {c1}.", [pclaim("c1", "margin", "percent", "n9")])
    assert "unknown_passage_number" in [i.code for i in r.issues]
    assert r.status == "partially_verified"


def test_a_label_for_a_different_number_does_not_bind():
    r, _ = run("The courier charge is {c1}.", [pclaim("c1", "courier charge", "number", "n2")])
    reasons = {v.reason for v in r.report.verdicts}
    assert "label_not_in_passage_sentence" in reasons
    assert r.status != "verified"


def test_the_model_cannot_type_the_number_instead():
    r, _ = run("The courier charge is 4.50.", [])
    assert r.status != "verified"


@pytest.mark.parametrize("extra", [{"decimals": 2}, {"scale": "thousand"}])
def test_formatting_options_are_not_allowed_on_passage_numbers(extra):
    with pytest.raises(ValueError):
        TemplateAnswer.model_validate(
            {
                "template": "{c1}",
                "claims": [{**pclaim("c1", "courier charge", "number", "n1"), **extra}],
                "queries": [],
            }
        )
