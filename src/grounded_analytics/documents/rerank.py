"""Rerankers: score candidate passages for a question.

`ApiReranker` follows the common rerank request (model, query, documents, top_n) and reads
results[].index and relevance_score; Jina's page shows the request but not the response, so the
response shape is an assumption to be confirmed at the first call. `CrossEncoderReranker` uses the
optional fastembed package on this computer. `FakeReranker` is a word-overlap score
for tests. Nothing here calls the network except through the injected transport."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from grounded_analytics.documents.embed import EmbeddingError
from grounded_analytics.documents.embed_http import post_json
from grounded_analytics.documents.metrics import lexical_overlap
from grounded_analytics.documents.settings import RerankerSettings
from grounded_analytics.llm.transport import Transport, urllib_transport


class Reranker(Protocol):
    name: str
    hosted: bool
    depth: int

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FakeReranker:
    hosted = False

    def __init__(self, name: str = "fake_overlap", depth: int = 20):
        self.name, self.depth = name, depth

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [lexical_overlap(query, t) for t in texts]


class ApiReranker:
    def __init__(
        self,
        name: str,
        s: RerankerSettings,
        env: Mapping[str, str],
        transport: Transport | None = None,
    ):
        self.name, self.s, self.env = name, s, env
        self.hosted, self.depth = s.hosted, s.depth
        self.transport = transport or urllib_transport

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if self.s.model.startswith("REPLACE"):
            raise EmbeddingError("config", f"{self.name}: no real model name configured")
        key = self.env.get(self.s.env_key or "")
        if not key:
            raise EmbeddingError("no_key", f"environment variable {self.s.env_key} is not set")
        data = post_json(
            self.transport,
            self.s.url or "",
            {"Authorization": f"Bearer {key}"},
            {
                "model": self.s.model,
                "query": query,
                "documents": list(texts),
                "top_n": len(texts),
                "return_documents": False,
            },
            self.s.timeout_s,
            self.env,
            key,
        )
        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(texts):
            raise EmbeddingError("malformed", "results is missing or has the wrong length")
        scores: list[float | None] = [None] * len(texts)
        try:
            for r in results:
                i, sc = r["index"], r["relevance_score"]
                if not isinstance(i, int) or not 0 <= i < len(texts) or scores[i] is not None:
                    raise EmbeddingError("malformed", "a result index is out of range or repeated")
                if not isinstance(sc, (int, float)) or not math.isfinite(sc):
                    raise EmbeddingError("malformed", "a relevance score is not a number")
                scores[i] = float(sc)
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("malformed", "a result lacks index or relevance_score") from exc
        return [s for s in scores if s is not None]


class CrossEncoderReranker:
    """A cross-encoder on this computer's CPU through fastembed (`pip install -e ".[local]"`).

    `fastembed.rerank.cross_encoder.TextCrossEncoder(model_name=..., cache_dir=...)` with
    `.rerank(query, documents)` returning one score per document, in order, is written from
    memory and UNVERIFIED (docs/PHASE_2B_ASSUMPTIONS.md), as are the model names. A one-line
    notice is printed before the model loads, because the first load may download files."""

    hosted = False

    def __init__(self, name: str, s: RerankerSettings, library: Any = None, notice: Any = None):
        self.name, self.s, self.depth = name, s, s.depth
        self._library = library
        self._notice = notice or (lambda line: print(line, file=sys.stderr, flush=True))
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._library
        if factory is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "config",
                    "fastembed (with cross-encoder support) is not installed "
                    '(pip install -e ".[local]")',
                ) from exc

            def factory(model, cache_dir):
                return TextCrossEncoder(model_name=model, cache_dir=cache_dir)

        self._notice(
            f"loading local reranker {self.s.model}: the first use may download its files "
            "from the internet and can take a few minutes; later runs use the saved copy"
        )
        try:
            self._model = factory(self.s.model, self.s.cache_dir)
        except Exception as exc:
            raise EmbeddingError("load_failed", f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        model = self._load()
        try:
            scores = [float(x) for x in model.rerank(query, list(texts))]
        except Exception as exc:
            raise EmbeddingError(
                "malformed", f"local reranker failed: {type(exc).__name__}"
            ) from exc
        if len(scores) != len(texts) or not all(math.isfinite(x) for x in scores):
            raise EmbeddingError("malformed", "cross-encoder returned bad scores")
        return scores


def build_reranker(
    name: str,
    s: RerankerSettings,
    env: Mapping[str, str],
    transport: Transport | None = None,
    library: Any = None,
    notice: Any = None,
) -> Reranker:
    if s.kind == "fake":
        return FakeReranker(name, s.depth)
    if s.kind == "cross_encoder":
        return CrossEncoderReranker(name, s, library, notice)
    return ApiReranker(name, s, env, transport)
