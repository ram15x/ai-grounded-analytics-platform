"""Label binding for numbers quoted from a passage (slice 2e).

A cell number is bound to its label by the semantic layer. A passage has no layer, so the binder
checks three things from the stored text alone, and fails closed:

  R1  the claim label's content words occur within WINDOW words of the number in the passage
      (every label word the passage uses must be there; at least one must be);
  R2  the label's content words occur in the prose sentence that carries the number;
  R3  every capitalised, non-initial word in that sentence occurs in the cited chunk
      (an entity the passage never mentions cannot be attached to its number).

These are heuristics: they catch a number moved to a different subject, not every misreading.
See docs/INJECTION.md and docs/VERIFIER_LIMITS.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

WINDOW = 6
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_STOP = frozenset(
    str.split(
        "a an and are as at be by can for from has have in is it its of on or per than that the "
        "their this to was were will with within your our we you not no yes if when what which "
        "who how much many about over under into up down after before each all any"
    )
)
_CALENDAR = frozenset(
    str.split(
        "january february march april may june july august september october november december "
        "monday tuesday wednesday thursday friday saturday sunday"
    )
)


def stem(word: str) -> str:
    w = word.lower().strip("'-")
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content(text: str) -> set[str]:
    return {stem(w) for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 1}


_PREPS = frozenset(["for", "of", "to", "in", "on", "at", "by", "with", "within", "from"])


@dataclass(frozen=True)
class BinderConfig:
    """R1 variants measured in documents/binder_study.py. DEFAULT is the shipped rule.

    frac: share of the label words that the passage uses that must sit in the window (1.0: all).
    window: words either side of the number.
    ignore_repeated: label words that occur twice or more in the passage do not count as required.
    head_noun: if the label's head noun is used by the passage it must be in the window; the
        other label words then only need one of them in the window."""

    frac: float = 1.0
    window: int = WINDOW
    ignore_repeated: bool = False
    head_noun: bool = False
    skip_units: bool = (
        False  # unit and quantity words ("days", "units", "percent") are not evidence
    )

    @property
    def name(self) -> str:
        tail = "/skip_units" if self.skip_units else ""
        if self.head_noun:
            return f"head_noun/w{self.window}{tail}"
        return (
            f"frac{self.frac:g}/w{self.window}"
            + ("/ignore_repeated" if self.ignore_repeated else "")
            + tail
        )


DEFAULT = BinderConfig()
UNIT_WORDS = frozenset(
    [
        "day",
        "week",
        "month",
        "year",
        "hour",
        "minute",
        "second",
        "unit",
        "percent",
        "pound",
        "gbp",
        "working",
        "per",
        "cent",
    ]
)


def head_of(label: str) -> str | None:
    """Last content word of the label before its first preposition ("time limit for claiming a
    late parcel" gives "limit"), as a stem; None if there is none."""
    words = _WORD.findall(label)
    for i, w in enumerate(words):
        if w.lower() in _PREPS and i > 0:
            words = words[:i]
            break
    heads = [stem(w) for w in words if w.lower() not in _STOP and len(w) > 1]
    return heads[-1] if heads else None


def _r1_ok(label: str, passage: str, start: int, end: int, cfg: BinderConfig) -> bool:
    want = content(label)
    if cfg.skip_units:
        want -= {stem(w) for w in UNIT_WORDS}
    near = _window_words(passage, start, end, cfg.window)
    if not want & near:
        return False
    counts = Counter(stem(w) for w in _WORD.findall(passage) if w.lower() not in _STOP)
    used = want & set(counts)
    if cfg.head_noun:
        head = head_of(label)
        return head not in used or head in near
    required = {w for w in used if counts[w] < 2} if cfg.ignore_repeated else used
    if not required:
        return True
    return len(required & near) / len(required) >= cfg.frac - 1e-9


def _window_words(text: str, start: int, end: int, window: int = WINDOW) -> set[str]:
    before = _WORD.findall(text[:start])[-window:]
    after = _WORD.findall(text[end:])[:window]
    return content(" ".join([*before, *after]))


def entities(sentence: str) -> set[str]:
    """Capitalised words that are not the first word of the sentence and not calendar names."""
    words = list(_WORD.finditer(sentence))
    out: set[str] = set()
    for i, m in enumerate(words):
        w = m.group(0)
        if w.lower() in _CALENDAR or len(w) < 2:
            continue
        if w.isupper() or (i > 0 and w[0].isupper()):
            out.add(stem(w))
    return out


def check_binding(
    label: str, sentence: str, passage: str, start: int, end: int, cfg: BinderConfig = DEFAULT
) -> tuple[str, str] | None:
    """None when bound; otherwise (reason, detail)."""
    want = content(label)
    if not want:
        return "label_empty", ""
    if not _r1_ok(label, passage, start, end, cfg):
        near = _window_words(passage, start, end, cfg.window)
        used = want & content(passage)
        return "label_not_in_passage_window", ",".join(sorted((used - near) or want))
    if not want & content(sentence):
        return "label_not_in_sentence", ",".join(sorted(want))
    missing = entities(sentence) - content(passage) - want
    if missing:
        return "entity_not_in_passage", ",".join(sorted(missing))
    return None


def explain(
    reason: str, label: str, sentence: str, passage: str, start: int, end: int, cut: int = 200
) -> str:
    """JSON text for a refusal: the label, the sentence, the words near the number in the passage,
    and the label words that are missing (each part cut). For the probe report and for people;
    nothing decides on it."""
    want = content(label)
    near = _window_words(passage, start, end)
    if reason == "label_not_in_passage_window":
        missing = (want & content(passage)) - near or want
    elif reason == "label_not_in_sentence":
        missing = want - content(sentence)
    elif reason == "entity_not_in_passage":
        missing = entities(sentence) - content(passage) - want
    else:
        missing = set()
    window = " ".join(_WORD.findall(passage[:start])[-WINDOW:] + ["<number>"]
                      + _WORD.findall(passage[end:])[:WINDOW])  # fmt: skip
    return json.dumps(
        {
            "label": label[:cut],
            "sentence": sentence.strip()[:cut],
            "window_words": window[:cut],
            "missing_label_words": ", ".join(sorted(missing))[:cut],
        }
    )
