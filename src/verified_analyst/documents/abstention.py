"""Can the top search score tell a question the documents answer from one they do not?

Nearest-neighbour and keyword search always return something for any question, so search alone
never says "nothing relevant". This measures how well the best score separates the two groups on
the dev questions. It reports facts and sets no threshold: cosine and keyword scores are not
comparable between models, and a cut-off chosen on these questions would be tuned to them.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Any

from verified_analyst.documents.fusion import Search
from verified_analyst.documents.metrics import Question

NOTE = (
    "Search always returns the nearest passages, so it cannot say that nothing answers a "
    "question. Separation (AUC) near 0.5 means the top score does not tell the groups apart; "
    "near 1.0 means it does on these questions. No threshold is set: scores differ between "
    "models, and one chosen here would be fitted to the dev questions. Deciding that an answer "
    "is unsupported stays with the passage check, not with search."
)


def auc(answerable: Sequence[float], unanswerable: Sequence[float]) -> float | None:
    """Chance that a random answerable question has a higher top score than a random unanswerable
    one (ties count half). None when either group is empty."""
    if not answerable or not unanswerable:
        return None
    wins = sum(1.0 if a > u else 0.5 if a == u else 0.0 for a in answerable for u in unanswerable)
    return wins / (len(answerable) * len(unanswerable))


def _summary(xs: Sequence[float]) -> dict[str, float] | None:
    return {"min": min(xs), "median": median(xs), "max": max(xs)} if xs else None


def profile(search: Search, questions: Sequence[Question]) -> dict[str, Any]:
    ans: list[float] = []
    non: list[float] = []
    empty = {"answerable": 0, "unanswerable": 0}
    for q in questions:
        hits = search(q.question, 1)
        group, name = (ans, "answerable") if q.evidence else (non, "unanswerable")
        if hits:
            group.append(hits[0].score)
        else:
            group.append(float("-inf"))  # no result at all ranks below every score
            empty[name] += 1
    finite = lambda xs: [x for x in xs if x != float("-inf")]  # noqa: E731
    return {
        "answerable": {"n": len(ans), "top_score": _summary(finite(ans))},
        "unanswerable": {"n": len(non), "top_score": _summary(finite(non))},
        "returned_nothing": empty,
        "auc": auc(ans, non),
    }
