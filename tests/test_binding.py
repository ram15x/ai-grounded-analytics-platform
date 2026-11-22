"""Label binding for passage numbers (slice 2e): R1 window, R2 sentence, R3 entities."""

from __future__ import annotations

import pytest

from grounded_analytics.documents.binding import CURRENT, check_binding

TEXT = (
    "The courier charge is 4.50 GBP per parcel. The scratch charge is 12 GBP. Returns in 28 days."
)


def at(needle):
    i = TEXT.index(needle)
    return i, i + len(needle)


def ok(label, sentence, needle):
    return check_binding(label, sentence, TEXT, *at(needle))


def test_a_label_next_to_its_number_binds():
    assert ok("courier charge", "The courier charge is 4.50.", "4.50") is None


def test_a_label_from_another_sentence_of_the_passage_does_not_bind():
    r = ok("courier charge", "The courier charge is 12.", "12")
    assert r and r[0] == "label_not_in_passage_sentence"
    old = check_binding("courier charge", "The courier charge is 12.", TEXT, *at("12"), CURRENT)
    assert old and old[0] == "label_not_in_passage_window"


def test_the_prose_must_name_what_the_label_says():
    r = ok("courier charge", "The total is 4.50.", "4.50")
    assert r and r[0] == "label_not_in_sentence"


def test_an_entity_the_passage_never_mentions_is_refused():
    r = ok("courier charge", "The courier charge at Acme is 4.50.", "4.50")
    assert r and r[0] == "entity_not_in_passage"


@pytest.mark.parametrize("sentence", ["Returns close in 28.", "Return window is 28."])
def test_simple_inflections_match(sentence):
    assert ok("returns", sentence, "28") is None


def test_an_empty_label_fails_closed():
    assert ok("the", "The value is 4.50.", "4.50")[0] == "label_empty"


def test_calendar_words_are_not_entities():
    assert ok("returns", "Returns in March are 28.", "28") is None


def test_explain_gives_label_sentence_window_and_missing_words_cut_to_200():
    import json

    from grounded_analytics.documents.binding import explain

    i, j = at("12")
    d = json.loads(explain("label_not_in_passage_window", "courier charge " * 30,
                           "The courier charge is 12. " * 20, TEXT, i, j))  # fmt: skip
    assert set(d) == {"label", "sentence", "window_words", "missing_label_words"}
    assert all(len(v) <= 200 for v in d.values())
    assert "<number>" in d["window_words"] and "courier" in d["missing_label_words"]
    d = json.loads(explain("label_not_in_passage_sentence", "courier charge",
                           "The courier charge is 12.", TEXT, i, j))  # fmt: skip
    assert d["window_words"] == "The scratch charge is <number> GBP" and (
        d["missing_label_words"] == "courier"
    )
