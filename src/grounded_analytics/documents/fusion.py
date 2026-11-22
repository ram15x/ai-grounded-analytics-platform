"""Rank fusion and the retrieval methods built from keyword search, vector search and a reranker.

`rrf` is pure. Reciprocal rank fusion scores an item by the sum of 1 / (k + rank) over the lists
that contain it, so it needs no comparable scores between keyword and vector search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from grounded_analytics.documents.store import Hit

RRF_K = 60  # the constant from the original RRF paper; a measured choice, not tuned here


def rrf(
    rankings: Sequence[Sequence[str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fused (key, score), best first. Ties break by the earliest list, then position.

    `weights` (one per list, default all 1) scale each list's contribution; equal weights give
    the plain RRF of the original paper."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError("one non-negative weight per ranking is required")
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for li, ranking in enumerate(rankings):
        for pos, key in enumerate(ranking):
            if key in scores and first_seen[key][0] == li:
                continue  # a key repeated inside one list counts once
            scores[key] = scores.get(key, 0.0) + weights[li] / (k + pos + 1)
            first_seen.setdefault(key, (li, pos))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


Search = Callable[[str, int], list[Hit]]


def fuse_hits(
    lists: Sequence[Sequence[Hit]],
    k: int,
    rrf_k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    by_id: dict[str, Hit] = {}
    for hits in lists:
        for h in hits:
            by_id.setdefault(h.passage_id, h)
    fused = rrf([[h.passage_id for h in hits] for hits in lists], rrf_k, weights)
    return [replace(by_id[pid], score=score) for pid, score in fused[:k]]


def hybrid(
    keyword: Search,
    vector: Search,
    depth: int = 50,
    rrf_k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> Search:
    """Keyword and vector search to `depth` each, fused by (optionally weighted) RRF.
    Weights are (keyword, vector)."""

    def search(query: str, k: int) -> list[Hit]:
        return fuse_hits([keyword(query, depth), vector(query, depth)], k, rrf_k, weights)

    return search


def reranked(
    base: Search, rerank: Callable[[str, list[Hit]], list[float]], depth: int = 20
) -> Search:
    """Take `depth` candidates from `base`, score them with the reranker, keep the best k.

    A score list of the wrong length is an error, not a silent truncation. Ties keep base order."""

    def search(query: str, k: int) -> list[Hit]:
        cands = base(query, max(depth, k))
        if not cands:
            return []
        scores = rerank(query, cands)
        if len(scores) != len(cands):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(cands)} candidates")
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
        return [replace(cands[i], score=float(scores[i])) for i in order[:k]]

    return search
