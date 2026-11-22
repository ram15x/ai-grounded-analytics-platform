"""Split a document into chunks that are exact slices of its text.

`chunk_text` is pure and deterministic. Every chunk is `text[start:end]` with no leading or
trailing whitespace, so a citation span inside a chunk maps to the characters the reader sees.
Chunks never overlap by default: with overlap, one sentence could sit in two chunks with two
different passage ids, which makes a citation ambiguous. Paragraphs (blank-line separated) are
packed up to `max_chars`; a paragraph longer than that is cut at sentence ends, and a sentence
longer than that is cut at the last space before the limit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 800
MIN_MAX_CHARS = 80

_PARA = re.compile(r"\n[ \t]*\n+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9£$€])")


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end) if end > start else None


def _pieces(text: str, start: int, end: int, max_chars: int) -> list[Span]:
    """Cut one paragraph [start, end) into pieces no longer than max_chars."""
    if end - start <= max_chars:
        s = _trim(text, start, end)
        return [s] if s else []
    out: list[Span] = []
    cursor = start
    for m in _SENT_END.finditer(text, start, end):
        out.extend(_fit(text, cursor, m.start(), max_chars))
        cursor = m.end()
    out.extend(_fit(text, cursor, end, max_chars))
    return out


def _fit(text: str, start: int, end: int, max_chars: int) -> list[Span]:
    out: list[Span] = []
    while end - start > max_chars:
        cut = text.rfind(" ", start + 1, start + max_chars + 1)
        if cut <= start:
            cut = start + max_chars
        s = _trim(text, start, cut)
        if s:
            out.append(s)
        start = cut
    s = _trim(text, start, end)
    if s:
        out.append(s)
    return out


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Span]:
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}")
    paragraphs: list[Span] = []
    cursor = 0
    for m in _PARA.finditer(text):
        paragraphs.append(Span(cursor, m.start()))
        cursor = m.end()
    paragraphs.append(Span(cursor, len(text)))

    pieces: list[Span] = []
    for p in paragraphs:
        pieces.extend(_pieces(text, p.start, p.end, max_chars))

    chunks: list[Span] = []
    for piece in pieces:
        if chunks and piece.end - chunks[-1].start <= max_chars:
            chunks[-1] = Span(chunks[-1].start, piece.end)
        else:
            chunks.append(piece)
    return chunks
