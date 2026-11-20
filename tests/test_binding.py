"""Label binding for passage numbers (slice 2e): R1 window, R2 sentence, R3 entities."""

from __future__ import annotations

import pytest

from verified_analyst.documents.binding import check_binding

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
    assert r and r[0] == "label_not_in_passage_window"


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
