"""The numbers a composer may quote from passages, listed and resolved by the system (slice 2e).

The composer is shown `P2 n3` style handles; it never types a digit or a span. The list is built
from the same text the model is shown (flagged paragraphs withheld) and skips any number that
overlaps flagged text, so a handle can only point at text the verifier would accept."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..extract import ExtractedNumber, extract_numbers
from .untrusted import PassageIn, blocked_overlap, scan, withhold


@dataclass(frozen=True)
class Handle:
    label: str
    number: str  # n1, n2, ... within the passage
    passage_id: str
    item: ExtractedNumber


def handles_for(label: str, passage_id: str, text: str) -> list[Handle]:
    shown = withhold(text, scan(text))
    out: list[Handle] = []
    for item in extract_numbers(shown).ok:
        if item.value is None or blocked_overlap(text, item.start, item.end):
            continue
        out.append(Handle(label, f"n{len(out) + 1}", passage_id, item))
    return out


class PassageNumbers:
    """Handles for the passages shown to the composer, keyed by label then number id."""

    def __init__(self, passages: Sequence[PassageIn]):
        self.texts: dict[str, str] = {p.passage_id: p.text for p in passages}
        self.shown: dict[str, str] = {
            p.passage_id: withhold(p.text, scan(p.text)) for p in passages
        }
        self.by_label: dict[str, dict[str, Handle]] = {}
        for i, p in enumerate(passages, 1):
            label = f"P{i}"
            self.by_label[label] = {h.number: h for h in handles_for(label, p.passage_id, p.text)}

    def get(self, label: str, number: str) -> Handle | None:
        return self.by_label.get(label, {}).get(number)

    def listing(self) -> str:
        """For the prompt: each handle with the digits and the few words around it."""
        lines = []
        for label, hs in self.by_label.items():
            for h in hs.values():
                t = self.shown[h.passage_id]  # withheld text only
                near = " ".join(t[max(0, h.item.start - 40) : h.item.end + 40].split())
                lines.append(f"{label} {h.number} = {h.item.text}   (…{near}…)")
        return "\n".join(lines)

    @property
    def stored(self) -> Mapping[str, str]:
        return self.texts
