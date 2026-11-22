"""Retrieval metrics, checked by hand on tiny cases."""

import math
from pathlib import Path

import pytest

from grounded_analytics.documents.metrics import (
    Evidence,
    Question,
    Retrieved,
    covers,
    evaluate,
    load_questions,
    score_question,
    unreachable_anchors,
)
from grounded_analytics.documents.store import make_passage_id, parse_passage_id, sha256_hex

Q = Question("q", "lookup", "?", (Evidence("a", "28 days"),))


def R(doc, text):
    return Retrieved(doc, text)


def test_covers_ignores_case_and_whitespace_but_not_the_document():
    assert covers("a", "Within  28\nDAYS of delivery", Evidence("a", "within 28 days"))
    assert not covers("b", "within 28 days", Evidence("a", "within 28 days"))
    assert not covers("a", "within 14 days", Evidence("a", "within 28 days"))


def test_a_hit_at_rank_one_scores_one_everywhere():
    s = score_question([R("a", "28 days"), R("a", "x")], Q, 1)
    assert s["mrr"] == 1 and s["recall@1"] == 1 and s["ndcg@3"] == 1


def test_a_hit_at_rank_three():
    ranked = [R("b", "x"), R("a", "y"), R("a", "28 days")]
    s = score_question(ranked, Q, 1)
    assert s["mrr"] == pytest.approx(1 / 3)
    assert s["recall@1"] == 0 and s["recall@3"] == 1
    assert s["ndcg@3"] == pytest.approx((1 / math.log2(4)) / 1)


def test_no_hit_scores_zero():
    s = score_question([R("b", "x")], Q, 1)
    assert s["mrr"] == 0 and s["recall@10"] == 0 and s["ndcg@10"] == 0


def test_recall_counts_each_evidence_anchor():
    q = Question("q", "lookup", "?", (Evidence("a", "28 days"), Evidence("b", "5 days")))
    s = score_question([R("a", "28 days")], q, 2)
    assert s["recall@1"] == 0.5


def test_ndcg_is_one_when_all_relevant_chunks_lead():
    ranked = [R("a", "28 days"), R("a", "28 days again"), R("b", "x")]
    assert score_question(ranked, Q, 2)["ndcg@3"] == pytest.approx(1)


def test_evaluate_averages_and_reports_unanswerable_separately():
    qs = [Q, Question("n", "no_answer", "?", ())]
    corpus = [R("a", "28 days")]
    rep = evaluate(lambda q, k: [R("a", "28 days")], qs, corpus)
    assert rep["questions_scored"] == 1
    assert rep["overall"]["recall@1"] == 1
    assert rep["unanswerable"] == {"n": 1, "returned_something": 1}
    assert rep["by_kind"]["lookup"]["n"] == 1


def test_unreachable_anchors_are_listed_not_scored_silently():
    bad = unreachable_anchors([Q], [R("a", "within 14 days")])
    assert bad and "q" in bad[0]
    assert unreachable_anchors([Q], [R("a", "28 days")]) == []


def test_passage_ids_round_trip_and_reject_lookalikes():
    sha = sha256_hex("text")
    pid = make_passage_id("returns-policy", 3, sha)
    assert parse_passage_id(pid) == ("returns-policy", 3, sha[:10])
    for bad in [
        "",
        "x",
        "a#1",
        "a#1.zz",
        "A#1." + sha[:10],
        "a#-1." + sha[:10],
        "a#1." + sha[:9],
        "../a#1." + sha[:10],
        "a#1." + sha[:10] + "x",
        None,
        5,
    ]:
        assert parse_passage_id(bad) is None


def test_question_files_are_well_formed_and_do_not_overlap():
    root = Path(__file__).parent.parent / "eval" / "doc_questions"
    dev, held = load_questions(root / "dev.yaml"), load_questions(root / "heldout.yaml")
    assert len(dev) >= 20 and len(held) >= 10
    assert not {q.id for q in dev} & {q.id for q in held}
    assert not {q.question for q in dev} & {q.question for q in held}
    assert {q.kind for q in dev + held} <= {
        "lookup",
        "paraphrase",
        "exact_term",
        "table",
        "conflict",
        "no_answer",
        "paraphrase_low_overlap",
        "near_duplicate",
        "figure",
        "superseded",
        "hostile_doc",
    }
