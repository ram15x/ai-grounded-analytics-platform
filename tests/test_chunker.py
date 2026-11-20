"""The chunker is pure: exact slices, no overlap, nothing lost, same output every time."""

import random

import pytest

from verified_analyst.documents.chunker import MIN_MAX_CHARS, chunk_text
from verified_analyst.documents.store import DocumentError, normalize_text

PARA = "Customers may return items within 28 days. Refunds take 5 working days. "


def _check(text, max_chars):
    chunks = chunk_text(text, max_chars)
    assert chunks == chunk_text(text, max_chars)
    covered = set()
    prev_end = 0
    for c in chunks:
        piece = text[c.start : c.end]
        assert piece and piece == piece.strip()
        assert c.end - c.start <= max_chars
        assert c.start >= prev_end, "chunks overlap or are out of order"
        prev_end = c.end
        covered.update(range(c.start, c.end))
    lost = [i for i, ch in enumerate(text) if not ch.isspace() and i not in covered]
    assert not lost, f"characters outside every chunk: {lost[:5]}"
    return chunks


def test_a_short_document_is_one_chunk_with_its_exact_span():
    text = "  Returns within 28 days.  \n"
    (c,) = chunk_text(text, 200)
    assert text[c.start : c.end] == "Returns within 28 days."


def test_empty_and_blank_text_make_no_chunks():
    assert chunk_text("", 200) == []
    assert chunk_text(" \n\n \t", 200) == []


def test_paragraphs_are_packed_up_to_the_limit():
    text = "\n\n".join(["alpha " * 10, "beta " * 10, "gamma " * 10])
    chunks = _check(text, 150)
    assert 1 < len(chunks) < 3 + 1


def test_a_long_paragraph_is_cut_at_sentence_ends():
    text = PARA * 20
    chunks = _check(text, 160)
    assert all(text[c.end - 1] in ".!?" for c in chunks)


def test_a_sentence_longer_than_the_limit_is_cut_at_a_space():
    text = ("word " * 100).strip()
    chunks = _check(text, 100)
    assert len(chunks) > 1
    assert all(not text[c.start - 1 : c.start].strip() for c in chunks[1:])


def test_a_single_unbreakable_run_is_hard_cut():
    chunks = _check("x" * 1000, 120)
    assert len(chunks) == 9


def test_unicode_is_cut_by_character_not_byte():
    _check("£4.95 flat charge. " * 60 + "日本語のテキスト。" * 40, 200)


def test_too_small_a_limit_is_refused():
    with pytest.raises(ValueError):
        chunk_text("abc", MIN_MAX_CHARS - 1)


@pytest.mark.parametrize("seed", range(25))
def test_random_documents_keep_every_property(seed):
    rnd = random.Random(seed)
    words = ["alpha", "£12.50", "return.", "Refund", "5", "x" * 40, "\n\n", "\n", "  ", "3.", "A"]
    text = " ".join(rnd.choice(words) for _ in range(rnd.randint(1, 400)))
    _check(text, rnd.choice([80, 120, 300, 800]))


def test_normalize_text_changes_only_line_endings_and_bom():
    assert normalize_text("﻿a\r\nb\rc") == "a\nb\nc"
    assert normalize_text("tab\there  and  spaces") == "tab\there  and  spaces"


@pytest.mark.parametrize("bad", ["", "  \n", "has\x00nul"])
def test_normalize_text_refuses_unusable_documents(bad):
    with pytest.raises(DocumentError):
        normalize_text(bad)
