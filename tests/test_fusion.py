"""Rank fusion, hybrid and reranked search on hand-made lists (no database)."""

import pytest

from grounded_analytics.documents.embed import (
    EmbeddingError,
    FakeEmbedder,
    batched,
    check_vectors,
    l2_normalize,
    vector_literal,
)
from grounded_analytics.documents.fusion import fuse_hits, hybrid, reranked, rrf
from grounded_analytics.documents.store import Hit


def hit(pid, score=1.0):
    return Hit(pid, pid.split("#")[0], "t", 0, 0, 5, f"text {pid}", score)


def test_rrf_scores_by_reciprocal_rank():
    out = dict(rrf([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert out["a"] == pytest.approx(1 / 61)
    assert [k for k, _ in rrf([["a", "b", "c"], ["b", "c", "d"]])] == ["b", "c", "a", "d"]


def test_an_item_in_both_lists_beats_one_that_tops_a_single_list():
    assert rrf([["x", "a"], ["y", "a"]])[0][0] == "a"


def test_rrf_ties_break_by_first_list_then_position():
    assert [k for k, _ in rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_a_repeated_key_in_one_list_counts_once():
    assert dict(rrf([["a", "a", "a"]]))["a"] == pytest.approx(1 / 61)


def test_fuse_hits_keeps_one_hit_per_passage_with_the_fused_score():
    fused = fuse_hits([[hit("a#0.1", 9), hit("b#0.2", 8)], [hit("b#0.2", 0.5)]], k=5)
    assert [h.passage_id for h in fused] == ["b#0.2", "a#0.1"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_uses_both_searches_and_the_depth():
    seen = []

    def kw(q, k):
        seen.append(("kw", k))
        return [hit("a#0.1"), hit("b#0.2")]

    def vec(q, k):
        seen.append(("vec", k))
        return [hit("c#0.3"), hit("b#0.2")]

    out = hybrid(kw, vec, depth=7)("q", 2)
    assert seen == [("kw", 7), ("vec", 7)]
    assert [h.passage_id for h in out][0] == "b#0.2" and len(out) == 2


def test_reranked_orders_by_score_and_keeps_base_order_on_ties():
    base = lambda q, k: [hit("a#0.1"), hit("b#0.2"), hit("c#0.3"), hit("d#0.4")][:k]  # noqa: E731
    out = reranked(base, lambda q, hits: [0.1, 0.9, 0.9, 0.2], depth=4)("q", 3)
    assert [h.passage_id for h in out] == ["b#0.2", "c#0.3", "d#0.4"]
    assert out[0].score == 0.9


def test_reranked_refuses_a_score_list_of_the_wrong_length():
    base = lambda q, k: [hit("a#0.1"), hit("b#0.2")]  # noqa: E731
    with pytest.raises(ValueError, match="1 scores for 2"):
        reranked(base, lambda q, hits: [1.0])("q", 2)
    assert reranked(lambda q, k: [], lambda q, h: [])("q", 3) == []


# ---- embedding helpers


def test_fake_embedder_is_deterministic_normalised_and_similar_for_similar_text():
    e = FakeEmbedder(dims=64)
    a, b, c = e.embed(
        ["refund within 28 days", "refund within 28 days", "night shift rota"], "query"
    )
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9
    cos = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    near = e.embed(["a refund within 28 days of delivery"], "document")[0]
    assert cos(a, near) > cos(a, c)
    assert e.model_id == "fake:64" and e.hosted is False


def test_vector_checks_refuse_bad_responses():
    ok = [[0.1, 0.2], [0.3, 0.4]]
    check_vectors(ok, 2, 2)
    for bad, n, d in (
        (ok, 3, None),
        ([[0.1], [0.1, 0.2]], 2, None),
        ([[]], 1, None),
        ([[float("nan"), 1.0]], 1, None),
        ([["x", 1.0]], 1, None),
        (ok, 2, 3),
    ):
        with pytest.raises(EmbeddingError):
            check_vectors(bad, n, d)


def test_l2_normalize_and_literals_and_batches():
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    with pytest.raises(EmbeddingError):
        l2_normalize([0.0, 0.0])
    assert vector_literal([1, 0.5]) == "[1.0,0.5]"
    assert [list(b) for b in batched(["a", "b", "c"], 2)] == [["a", "b"], ["c"]]
