"""Retrieval metrics over labelled questions. Pure functions; no database, no model.

A label is an evidence anchor: a document key and a string that must occur in it. A retrieved chunk
is relevant to a question when it comes from an evidence document and contains that document's
string (case and whitespace ignored). Anchoring to text, not chunk ids, keeps labels valid when
the chunking changes; an anchor that no chunk contains is reported, never silently scored as a miss.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Evidence:
    doc: str
    contains: str


@dataclass(frozen=True)
class Question:
    id: str
    kind: str
    question: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Retrieved:
    doc_key: str
    text: str


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "are",
        "was",
        "were",
        "what",
        "which",
        "when",
        "where",
        "who",
        "whom",
        "how",
        "many",
        "much",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "may",
        "must",
        "will",
        "with",
        "from",
        "that",
        "this",
        "than",
        "then",
        "into",
        "out",
        "over",
        "under",
        "after",
        "before",
        "about",
        "each",
        "every",
        "any",
        "all",
        "per",
        "not",
        "his",
        "her",
        "its",
        "our",
        "their",
        "you",
        "your",
    ]
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9£$€%.]+", text.casefold())
    out = set()
    for w in words:
        w = w.strip(".")
        if len(w) >= 3 and w not in _STOP:
            out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def lexical_overlap(question: str, text: str) -> float:
    """Share of the question's content words (crude stemming, no stopwords) found in `text`.

    A guard for the claim that a question is a paraphrase: a low value means the question mostly
    does not repeat the document's words. It is a rough check, not a measure of difficulty."""
    q = _tokens(question)
    return len(q & _tokens(text)) / len(q) if q else 0.0


def load_questions(path: Path) -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for q in raw["questions"]:
        ev = tuple(Evidence(e["doc"], e["contains"]) for e in q.get("evidence") or [])
        out.append(Question(q["id"], q["kind"], q["question"], ev))
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return out


def covers(doc_key: str, text: str, ev: Evidence) -> bool:
    return doc_key == ev.doc and norm(ev.contains) in norm(text)


def is_relevant(chunk: Retrieved, evidence: Sequence[Evidence]) -> bool:
    return any(covers(chunk.doc_key, chunk.text, e) for e in evidence)


def unreachable_anchors(questions: Sequence[Question], corpus: Sequence[Retrieved]) -> list[str]:
    """Evidence no chunk contains (a typo in the label, or the chunker cut the anchor in two)."""
    bad = []
    for q in questions:
        for e in q.evidence:
            if not any(covers(c.doc_key, c.text, e) for c in corpus):
                bad.append(f"{q.id}: {e.doc!r} / {e.contains!r}")
    return bad


def _dcg(flags: Sequence[bool]) -> float:
    return sum(1 / math.log2(i + 2) for i, f in enumerate(flags) if f)


def score_question(
    ranked: Sequence[Retrieved],
    q: Question,
    relevant_in_corpus: int,
    ks: Sequence[int] = KS,
) -> dict[str, float]:
    """recall@k (share of evidence anchors found in the top k), mrr, ndcg@k (binary)."""
    out: dict[str, float] = {}
    flags = [is_relevant(c, q.evidence) for c in ranked]
    first = next((i for i, f in enumerate(flags) if f), None)
    out["mrr"] = 0.0 if first is None else 1 / (first + 1)
    for k in ks:
        top = ranked[:k]
        found = sum(any(covers(c.doc_key, c.text, e) for c in top) for e in q.evidence)
        out[f"recall@{k}"] = found / len(q.evidence)
        ideal = _dcg([True] * min(k, relevant_in_corpus))
        out[f"ndcg@{k}"] = _dcg(flags[:k]) / ideal if ideal else 0.0
    return out


def evaluate(
    search: Callable[[str, int], Sequence[Retrieved]],
    questions: Sequence[Question],
    corpus: Sequence[Retrieved],
    ks: Sequence[int] = KS,
) -> dict:
    """Mean metrics over the questions that have evidence, overall and per kind.

    Questions without evidence (nothing in the corpus answers them) are not scored for recall; the
    report gives how many of them returned no passage at all at the largest k."""
    depth = max(ks)
    rows: list[tuple[Question, dict[str, float]]] = []
    unanswerable_returning = 0
    unanswerable = 0
    for q in questions:
        ranked = list(search(q.question, depth))
        if not q.evidence:
            unanswerable += 1
            unanswerable_returning += bool(ranked)
            continue
        n_rel = sum(is_relevant(c, q.evidence) for c in corpus)
        rows.append((q, score_question(ranked, q, n_rel, ks)))

    def mean(sel: Sequence[dict[str, float]]) -> dict[str, float]:
        keys = sel[0].keys() if sel else []
        return {k: sum(r[k] for r in sel) / len(sel) for k in keys}

    by_kind: dict[str, list[dict[str, float]]] = defaultdict(list)
    for q, r in rows:
        by_kind[q.kind].append(r)
    return {
        "questions_scored": len(rows),
        "overall": mean([r for _, r in rows]),
        "by_kind": {k: {"n": len(v), **mean(v)} for k, v in sorted(by_kind.items())},
        "per_question": {q.id: r for q, r in rows},
        "unanswerable": {"n": unanswerable, "returned_something": unanswerable_returning},
        "unreachable_anchors": unreachable_anchors(questions, corpus),
    }
