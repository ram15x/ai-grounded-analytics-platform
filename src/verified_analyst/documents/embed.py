"""Embedders: turn text into vectors. The interface, a deterministic fake for tests, and helpers.

A real embedder is a client of some HTTP service or a local model (embed_http.py). Nothing here
calls a network. `FakeEmbedder` hashes words and word pairs into a fixed number of buckets: it is
deterministic and similar texts get similar vectors, which is enough to test storage, fusion and
the probe plumbing. It is not a semantic model and its scores say nothing about real embeddings.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from typing import Literal, Protocol

Kind = Literal["document", "query"]


class EmbeddingError(Exception):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


class Embedder(Protocol):
    name: str
    model_id: str  # identifies model, dimensions and prefixes: a change makes new stored rows
    hosted: bool

    def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]: ...


def l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("bad_vector", "zero or non-finite vector")
    return [x / norm for x in vec]


def check_vectors(vectors: Sequence[Sequence[float]], expected: int, dims: int | None) -> None:
    """Refuse a response whose count, size or numbers are wrong, instead of storing it."""
    if len(vectors) != expected:
        raise EmbeddingError("malformed", f"got {len(vectors)} vectors for {expected} texts")
    size = len(vectors[0]) if vectors else 0
    for v in vectors:
        if len(v) != size or size == 0:
            raise EmbeddingError("malformed", "vectors differ in length or are empty")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
            raise EmbeddingError("malformed", "a vector holds a non-number")
    if dims is not None and size != dims:
        raise EmbeddingError("malformed", f"expected {dims} dimensions, got {size}")


def batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def vector_literal(vec: Sequence[float]) -> str:
    """pgvector's text form, for `%s::docs.vector`. Numbers only, so it cannot carry SQL."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


_WORD = re.compile(r"[a-z0-9£$€%.]+")


class FakeEmbedder:
    hosted = False

    def __init__(self, dims: int = 256, name: str = "fake"):
        if dims < 8:
            raise ValueError("dims must be at least 8")
        self.dims = dims
        self.name = name
        self.model_id = f"fake:{dims}"

    def _one(self, text: str) -> list[float]:
        words = [w.strip(".") for w in _WORD.findall(text.casefold())]
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w]
        grams = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vec = [0.0] * self.dims
        for g in grams:
            h = hashlib.sha256(g.encode("utf-8")).digest()
            vec[int.from_bytes(h[:4], "big") % self.dims] += 1.0 if h[4] & 1 else -1.0
        if not any(vec):
            vec[0] = 1.0
        return l2_normalize(vec)

    def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        return [self._one(t) for t in texts]
