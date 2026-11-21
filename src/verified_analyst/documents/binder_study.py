"""A measured test set for the passage label binder (slice 2e follow-up).

Cases are generated, with no model, from the synthetic corpus. For every number in every chunk:

  correct cases   the number with a label taken from its own sentence, in four styles a composer
                  writes: `nearest` (words just before the number), `subject` (leading content
                  words of the sentence), `paraphrase` (a stock head such as "time limit for" or
                  "amount of" plus the subject words) and `distant` (only words far from the
                  number: the style that failed on a real answer);
  swapped cases   the same label attached to another number: one in a different sentence of the
                  same chunk, and one in a different document. A pair is dropped if at least half
                  of the label's content words occur in the other number's own sentence (the label
                  would then be a fair description of it).

Documents are split in two by a hash of the document key. Variants are chosen and tuned on the dev
half only. The locked half is built only by `binder_locked.py`, which runs once for the variants
registered in docs/adr/0011-binder-variants.md. The corpus is synthetic and written by the project
author, the label styles are generated rather than written by a model, and a few swapped pairs may
still be fair; this is a screen for gross failures, not a benchmark of real answers.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..context import sentence_span
from ..extract import extract_numbers
from .binding import _STOP, DEFAULT, BinderConfig, check_binding, content, stem
from .chunker import chunk_text
from .untrusted import blocked_overlap

SEED = 20261121
CORPUS_DIR = Path(__file__).resolve().parents[3] / "corpus" / "synthetic"
STYLES = ("nearest", "subject", "paraphrase", "distant")
SWAP_KINDS = ("same_passage", "other_document")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TIME = frozenset(
    ["day", "days", "week", "weeks", "month", "months", "hour", "hours", "year", "years", "working"]
)
FAR = 6  # a label word is "distant" if it is more than this many words before the number


def half_of(doc_key: str) -> str:
    """dev or locked, from a hash of the document key (stable, independent of corpus order)."""
    return "dev" if hashlib.sha256(doc_key.encode()).digest()[0] % 2 == 0 else "locked"


@dataclass(frozen=True)
class Number:
    doc_key: str
    chunk: int
    text: str  # the chunk
    start: int
    end: int
    shown: str  # the number as written
    unit: str
    s_start: int
    s_end: int

    @property
    def id(self) -> str:
        return f"{self.doc_key}#{self.chunk}@{self.start}"

    @property
    def sentence(self) -> str:
        return self.text[self.s_start : self.s_end]


@dataclass(frozen=True)
class Case:
    kind: str  # correct | same_passage | other_document
    style: str
    label: str
    prose: str
    number: Number  # the number the label is attached to
    source: Number  # the number the label was taken from (== number when correct)


def load_numbers(half: str, corpus_dir: Path = CORPUS_DIR) -> list[Number]:
    out: list[Number] = []
    for path in sorted(corpus_dir.glob("*.md")):
        key = path.stem
        if half_of(key) != half:
            continue
        text = path.read_text("utf-8")
        for ordinal, sp in enumerate(chunk_text(text)):
            chunk = text[sp.start : sp.end]
            for it in extract_numbers(chunk).ok:
                if it.value is None or blocked_overlap(chunk, it.start, it.end):
                    continue
                a, b = sentence_span(chunk, it.start, it.end)
                out.append(Number(key, ordinal, chunk, it.start, it.end, it.text, it.unit, a, b))
    return out


def _words_around(n: Number) -> tuple[list[str], list[str]]:
    before = [w for w in _WORD.findall(n.text[n.s_start : n.start])]
    after = [w for w in _WORD.findall(n.text[n.end : n.s_end])]
    return before, after


def _content_list(words: list[str]) -> list[str]:
    return [w.lower() for w in words if w.lower() not in _STOP and len(w) > 1]


def labels_for(n: Number) -> dict[str, str]:
    before, after = _words_around(n)
    near = _content_list(before)[-3:] or _content_list(after)[:3]
    subject = _content_list(before + after)[:6]
    nxt = after[0].lower() if after else ""
    head = ("time limit for" if nxt in _TIME else "amount of" if n.unit == "currency"
            else "rate of" if n.unit == "percent" else "number of")  # fmt: skip
    far = _content_list(before[:-FAR])[:3]
    out = {
        "nearest": " ".join(near),
        "subject": " ".join(subject),
        "paraphrase": f"{head} {' '.join(subject[:5])}",
        "distant": " ".join(far),
    }
    return {k: v for k, v in out.items() if content(v)}


def _fair(label: str, other: Number) -> bool:
    want = content(label)
    return bool(want) and len(want & content(other.sentence)) / len(want) >= 0.5


def build_cases(numbers: list[Number]) -> list[Case]:
    cases: list[Case] = []
    by_chunk: dict[tuple[str, int], list[Number]] = {}
    for n in numbers:
        by_chunk.setdefault((n.doc_key, n.chunk), []).append(n)
    for n in numbers:
        for style, label in labels_for(n).items():
            prose = f"The {label} is {n.shown}."
            cases.append(Case("correct", style, label, prose, n, n))
            rng = random.Random(f"{SEED}:{n.id}:{style}")
            span = (n.s_start, n.s_end)
            same = [
                m
                for m in by_chunk[(n.doc_key, n.chunk)]
                if (m.s_start, m.s_end) != span and not _fair(label, m)
            ]
            other = [m for m in numbers if m.doc_key != n.doc_key and not _fair(label, m)]
            for kind, pool in (("same_passage", same), ("other_document", other)):
                if pool:
                    m = rng.choice(sorted(pool, key=lambda x: x.id))
                    cases.append(Case(kind, style, label, f"The {label} is {m.shown}.", m, n))
    return cases


def judge(case: Case, cfg: BinderConfig) -> bool:
    """True if the binder accepts the case."""
    n = case.number
    return check_binding(case.label, case.prose, n.text, n.start, n.end, cfg) is None


def evaluate(cfg: BinderConfig, cases: list[Case]) -> dict[str, Any]:
    correct = [c for c in cases if c.kind == "correct"]
    swapped = [c for c in cases if c.kind != "correct"]
    rej = [c for c in correct if not judge(c, cfg)]
    acc = [c for c in swapped if judge(c, cfg)]
    return {
        "variant": cfg.name,
        "false_reject": {"k": len(rej), "n": len(correct)},
        "false_accept": {"k": len(acc), "n": len(swapped)},
        "false_reject_by_style": {s: _kn(rej, correct, s) for s in STYLES},
        "false_accept_by_kind": {
            k: {"k": sum(c.kind == k for c in acc), "n": sum(c.kind == k for c in swapped)}
            for k in SWAP_KINDS
        },
    }


def _kn(bad: list[Case], pool: list[Case], style: str) -> dict[str, int]:
    return {"k": sum(c.style == style for c in bad), "n": sum(c.style == style for c in pool)}


def variants() -> list[BinderConfig]:
    out = []
    for skip in (False, True):
        out += [BinderConfig(f, w, ig, False, skip) for ig in (False, True)
                for f in (1.0, 0.75, 0.6, 0.5) for w in (6, 8, 10)]  # fmt: skip
        out += [BinderConfig(1.0, w, False, True, skip) for w in (6, 8, 10)]
    return out


def dev_cases(corpus_dir: Path = CORPUS_DIR) -> list[Case]:
    return build_cases(load_numbers("dev", corpus_dir))


def dev_table(cases: list[Case]) -> list[dict[str, Any]]:
    return [evaluate(v, cases) for v in variants()]


def render(rows: list[dict[str, Any]], half: str, cases: list[Case]) -> str:
    docs = len({c.number.doc_key for c in cases})
    kinds = Counter(c.kind for c in cases)
    lines = [
        f"Binder study, {half} half: {docs} documents, "
        + ", ".join(f"{k} cases {v}" for k, v in sorted(kinds.items())),
        "Synthetic corpus, generated labels; counts, not a benchmark. DEFAULT is the shipped rule"
        f" ({DEFAULT.name}).",
        "",
        f"{'variant':38} {'false reject (correct)':>24} {'false accept (swapped)':>24}  by style",
    ]
    for r in rows:
        fr, fa = r["false_reject"], r["false_accept"]
        styles = " ".join(
            f"{s[:4]} {v['k']}/{v['n']}" for s, v in r["false_reject_by_style"].items()
        )
        mark = " *" if r["variant"] == DEFAULT.name else ""
        lines.append(
            f"{r['variant'] + mark:32} {fr['k']:>4} of {fr['n']:<5} (n={fr['n']})"
            f" {fa['k']:>4} of {fa['n']:<5} (n={fa['n']})  {styles}"
        )
    return "\n".join(lines) + "\n"


__all__ = ["stem"]
