"""The synthetic corpus and its questions: size, provenance, label uniqueness, difficulty guards.

These tests read both question files, as label checks; nothing here tunes anything. The held-out
questions are not edited by the corpus growth."""

from pathlib import Path

import pytest

from grounded_analytics.documents.chunker import chunk_text
from grounded_analytics.documents.metrics import (
    Retrieved,
    lexical_overlap,
    load_questions,
    norm,
    unreachable_anchors,
)
from grounded_analytics.documents.store import normalize_text

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "corpus" / "synthetic"
QDIR = ROOT / "eval" / "doc_questions"
ORIGINAL_14 = {
    "returns-policy",
    "returns-policy-2010",
    "delivery-policy",
    "wholesale-terms",
    "supplier-note-giftware",
    "stock-policy",
    "customer-service-sla",
    "revenue-definitions",
    "christmas-peak-memo",
    "pricing-memo",
    "data-quality-notes",
    "regional-sales-review",
    "supplier-email-forwarded",
    "office-handbook-extract",
}


def docs() -> dict[str, str]:
    return {
        p.stem: normalize_text(p.read_text(encoding="utf-8")) for p in sorted(CORPUS.glob("*.md"))
    }


def chunks(max_chars=800):
    return [
        Retrieved(k, t[s.start : s.end])
        for k, t in docs().items()
        for s in chunk_text(t, max_chars)
    ]


def test_the_generator_writes_exactly_the_committed_files(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_synthetic", ROOT / "corpus" / "build_synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built = mod.build()
    assert not {n.removesuffix(".md") for n in built} & ORIGINAL_14, (
        "the generator must not touch them"
    )
    for name, text in built.items():
        assert (CORPUS / name).read_text(encoding="utf-8") == text, name
    committed = {p.name for p in CORPUS.glob("*.md")}
    assert committed == set(built) | {f"{k}.md" for k in ORIGINAL_14}


def test_size_and_every_document_says_it_is_invented():
    d = docs()
    assert 60 <= len(d) <= 100
    for key, text in d.items():
        assert text.startswith("# "), key
        assert "SYNTHETIC DOCUMENT" in text, key


def test_hard_negatives_exist_and_are_marked_as_such():
    d = docs()
    marked = [
        k
        for k, t in d.items()
        if any(
            w in t for w in ("superseded", "not adopted", "not approved", "PROPOSAL ONLY", "DRAFT")
        )
    ]
    assert len(marked) >= 8
    hostile = [k for k, t in d.items() if "deliberately hostile" in t]
    assert len(hostile) >= 5
    families = ("delivery-", "supplier-note-", "category-notes-", "review-")
    for fam in families:
        assert sum(k.startswith(fam) for k in d) >= 4, fam


@pytest.mark.parametrize("name", ["dev", "heldout"])
def test_each_anchor_is_in_its_document_and_in_no_other(name):
    d = {k: norm(t) for k, t in docs().items()}
    for q in load_questions(QDIR / f"{name}.yaml"):
        for e in q.evidence:
            holders = [k for k, t in d.items() if norm(e.contains) in t]
            assert holders == [e.doc], f"{q.id}: anchor {e.contains!r} is in {holders}"


@pytest.mark.parametrize("name", ["dev", "heldout"])
def test_every_anchor_is_inside_one_chunk_at_the_default_size(name):
    qs = load_questions(QDIR / f"{name}.yaml")
    assert unreachable_anchors(qs, chunks()) == []


def test_dev_set_has_enough_unanswerable_and_hard_questions():
    qs = load_questions(QDIR / "dev.yaml")
    none = [q for q in qs if not q.evidence]
    assert len(qs) >= 60
    assert len(none) * 40 >= 8 * len(qs), "at least 8 no-answer questions per 40"
    kinds = {q.kind for q in qs}
    assert {
        "paraphrase_low_overlap",
        "near_duplicate",
        "table",
        "figure",
        "superseded",
        "hostile_doc",
        "no_answer",
    } <= kinds
    assert len({q.question for q in qs}) == len(qs)


def test_paraphrase_questions_mostly_avoid_the_documents_words():
    d = docs()
    qs = [q for q in load_questions(QDIR / "dev.yaml") if q.kind == "paraphrase_low_overlap"]
    assert len(qs) >= 10
    too_close = [
        (q.id, round(lexical_overlap(q.question, d[q.evidence[0].doc]), 2))
        for q in qs
        if lexical_overlap(q.question, d[q.evidence[0].doc]) > 0.5
    ]
    assert not too_close, too_close


def test_the_overlap_measure_itself():
    assert lexical_overlap("refund within 28 days", "A refund is paid within 28 days.") == 1.0
    assert lexical_overlap("send back the goods", "Returns within 28 days.") == 0.0
    assert lexical_overlap("", "x") == 0.0


def test_heldout_is_not_edited_by_this_work():
    held = load_questions(QDIR / "heldout.yaml")
    assert len(held) == 12 and sum(not q.evidence for q in held) == 1
