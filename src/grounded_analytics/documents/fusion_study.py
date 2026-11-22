"""A dev-only study of how keyword and vector rankings are fused.

It runs a fixed grid (RRF constant, candidate depth, keyword weight), scores every setting on the
same dev questions, and reports each one next to the default with a paired bootstrap interval on
the per-question change. It picks nothing: the table is in grid order, not sorted by score, and a
setting is marked only as "interval excludes zero" or not. One embedder on one 65-question set
cannot justify a setting; the same table from a second embedder is what would.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from grounded_analytics.documents.fusion import RRF_K, Search, fuse_hits
from grounded_analytics.documents.metrics import KS, Question, Retrieved, evaluate
from grounded_analytics.documents.store import Hit

DEFAULT = ("default", RRF_K, 50, 1.0)  # what `hybrid` uses today
GRID_K = (10, 30, 60)
GRID_DEPTH = (10, 20, 50)
GRID_KEYWORD_WEIGHT = (1.0, 0.5, 0.25)
MAX_DEPTH = 50  # keyword search allows at most 50
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261120
PAIRED_METRICS = ("recall@1", "mrr")


@dataclass(frozen=True)
class Setting:
    rrf_k: int
    depth: int
    keyword_weight: float

    @property
    def label(self) -> str:
        return f"k={self.rrf_k} depth={self.depth} kw_weight={self.keyword_weight:g}"


BASELINE = Setting(DEFAULT[1], DEFAULT[2], DEFAULT[3])
# Pre-registered in docs/adr/0009 before any held-out scoring; changing it is a decision.
CANDIDATE_B = Setting(rrf_k=10, depth=20, keyword_weight=0.25)


def grid() -> list[Setting]:
    return [Setting(k, d, w) for w in GRID_KEYWORD_WEIGHT for d in GRID_DEPTH for k in GRID_K]


def _cached(fn: Search, depth: int = MAX_DEPTH) -> Search:
    store: dict[str, list[Hit]] = {}

    def search(query: str, k: int) -> list[Hit]:
        if query not in store:
            store[query] = fn(query, depth)
        return store[query][:k]

    return search


def fused(keyword: Search, vector: Search, s: Setting) -> Search:
    def search(query: str, k: int) -> list[Hit]:
        return fuse_hits(
            [keyword(query, s.depth), vector(query, s.depth)],
            k,
            s.rrf_k,
            (s.keyword_weight, 1.0),
        )

    return search


def paired_bootstrap(
    before: Sequence[float], after: Sequence[float], seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Mean change (after - before) over the same questions, with a 95% percentile interval from
    resampling questions. Deterministic for a fixed seed."""
    if len(before) != len(after) or not before:
        raise ValueError("paired lists must be the same non-zero length")
    diffs = [a - b for a, b in zip(after, before, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES)
    )
    lo = means[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    mean = sum(diffs) / n
    return {
        "mean_change": mean,
        "ci95": [lo, hi],
        "excludes_zero": lo > 0 or hi < 0,
        "questions_better": sum(d > 1e-12 for d in diffs),
        "questions_worse": sum(d < -1e-12 for d in diffs),
    }


def run_study(
    keyword: Search,
    vector: Search,
    questions: Sequence[Question],
    corpus: Sequence[Retrieved],
    settings: Sequence[Setting] | None = None,
) -> dict[str, Any]:
    kw, vec = _cached(keyword), _cached(vector)

    def score(fn: Search) -> dict[str, Any]:
        return evaluate(
            lambda q, k: [Retrieved(h.doc_key, h.text) for h in fn(q, k)], questions, corpus, KS
        )

    base = Setting(DEFAULT[1], DEFAULT[2], DEFAULT[3])
    default = score(fused(kw, vec, base))
    ids = sorted(default["per_question"])

    def pq(res: dict[str, Any], metric: str) -> list[float]:
        return [res["per_question"][i][metric] for i in ids]

    def row(label: str, res: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "label": label,
            "overall": res["overall"],
            "by_kind_mrr": {k: v["mrr"] for k, v in res["by_kind"].items()},
        }
        out["vs_default"] = {
            m: paired_bootstrap(pq(default, m), pq(res, m)) for m in PAIRED_METRICS
        }
        return out

    rows = [row(s.label, score(fused(kw, vec, s))) for s in (settings or grid())]
    return {
        "scope": "dev questions only",
        "questions_scored": default["questions_scored"],
        "default": {
            "label": base.label,
            "overall": default["overall"],
            "by_kind_mrr": {k: v["mrr"] for k, v in default["by_kind"].items()},
        },
        "keyword_only": row("keyword only", score(kw)),
        "vector_only": row("vector only", score(vec)),
        "grid": rows,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "note": (
            "Nothing is selected here. Rows are in grid order. One embedder and one question set "
            "cannot justify a setting; repeat with another embedder and look for changes that "
            "agree. A bootstrap interval that excludes zero describes these 65 questions only."
        ),
    }


def render_markdown(st: dict[str, Any]) -> str:
    cols = ["recall@1", "recall@5", "mrr", "ndcg@5"]
    d = st["default"]

    def cells(o: dict[str, float]) -> str:
        return " | ".join(f"{o[c]:.3f}" for c in cols)

    def delta(r: dict[str, Any], m: str) -> str:
        v = r["vs_default"][m]
        flag = "*" if v["excludes_zero"] else ""
        return f"{v['mean_change']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{flag}"

    lines = [
        "## Fusion study (dev questions only)",
        "",
        f"> {st['note']}",
        "",
        f"{st['questions_scored']} scored questions. Change columns are after minus the default "
        f"({d['label']}), with a 95% bootstrap interval; `*` means the interval excludes zero.",
        "",
        "| setting | "
        + " | ".join(cols)
        + " | recall@1 change | mrr change | better / worse (mrr) |",
        "|---|" + "---|" * (len(cols) + 3),
        f"| **default** {d['label']} | {cells(d['overall'])} | - | - | - |",
    ]
    for r in [st["vector_only"], st["keyword_only"], *st["grid"]]:
        v = r["vs_default"]["mrr"]
        lines.append(
            f"| {r['label']} | {cells(r['overall'])} | {delta(r, 'recall@1')} | {delta(r, 'mrr')} "
            f"| {v['questions_better']} / {v['questions_worse']} |"
        )
    return "\n".join(lines) + "\n"
