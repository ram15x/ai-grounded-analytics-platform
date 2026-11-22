"""HTTP embedding clients: OpenAI-compatible (OpenRouter), Gemini, and Ollama on this computer.

Every request shape below was read from the provider's documentation and has not been tried
against a live service (docs/PHASE_2B_ASSUMPTIONS.md). The transport is injected, so tests use
mocked HTTP only. A client never retries: a probe wants to see the failure. Error text is cut and
redacted (llm.live.safe_error_body); the request, with the key, is never logged.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from grounded_analytics.documents.embed import (
    EmbeddingError,
    Kind,
    batched,
    check_vectors,
    l2_normalize,
)
from grounded_analytics.documents.settings import EmbedderSettings
from grounded_analytics.documents.vectorstore import embedding_model_id
from grounded_analytics.llm.live import safe_error_body
from grounded_analytics.llm.transport import (
    HttpResponse,
    Transport,
    TransportError,
    TransportTimeout,
    loopback_transport,
    urllib_transport,
)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def post_json(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    env: Mapping[str, str],
    secret: str = "",
) -> dict[str, Any]:
    """POST JSON and return the decoded body, or raise EmbeddingError with a short, safe reason."""
    body = json.dumps(payload).encode("utf-8")

    def safe(raw: bytes) -> str:
        text = safe_error_body(raw, env)
        return text.replace(secret, "<redacted>") if len(secret) >= 8 else text

    try:
        resp: HttpResponse = transport(
            url, {**headers, "Content-Type": "application/json"}, body, timeout
        )
    except TransportTimeout as exc:
        raise EmbeddingError("timeout", str(exc)) from exc
    except TransportError as exc:
        raise EmbeddingError("network", str(exc)) from exc
    if resp.status in (401, 403):
        raise EmbeddingError("auth", f"HTTP {resp.status}")
    if resp.status == 429:
        raise EmbeddingError("rate_limit", safe(resp.body))
    if resp.status >= 500:
        raise EmbeddingError("server", f"HTTP {resp.status} {safe(resp.body)}")
    if not 200 <= resp.status < 300:
        raise EmbeddingError("bad_request", f"HTTP {resp.status} {safe(resp.body)}")
    try:
        data = json.loads(resp.body)
    except ValueError as exc:
        raise EmbeddingError("malformed", "response is not JSON") from exc
    if not isinstance(data, dict):
        raise EmbeddingError("malformed", "response is not a JSON object")
    return data


class _Base:
    def __init__(
        self,
        name: str,
        s: EmbedderSettings,
        env: Mapping[str, str],
        transport: Transport | None,
        default_transport: Transport,
    ):
        self.name = name
        self.s = s
        self.env = env
        self.hosted = s.hosted
        self.transport = transport or default_transport
        self.model_id = embedding_model_id(
            name, s.model, s.dims, (s.query_prefix, s.document_prefix)
        )

    def _key(self) -> str:
        if not self.s.env_key:
            return ""
        key = self.env.get(self.s.env_key)
        if not key:
            raise EmbeddingError("no_key", f"environment variable {self.s.env_key} is not set")
        return key

    def _check_model(self) -> None:
        if self.s.model.startswith("REPLACE"):
            raise EmbeddingError("config", f"{self.name}: no real model name configured")

    def _text(self, text: str, kind: Kind) -> str:
        return (self.s.query_prefix if kind == "query" else self.s.document_prefix) + text

    def _finish(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        check_vectors(vectors, expected, self.s.dims)
        return [l2_normalize(v) for v in vectors] if self.s.normalize else vectors

    def _call(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        self._check_model()
        out: list[list[float]] = []
        for batch in batched(list(texts), self.s.batch_size):
            out += self._call(batch, kind)
        return out


class OpenAICompatEmbedder(_Base):
    """POST {base_url}/embeddings with {model, input[], encoding_format}. Response data[].embedding,
    each item carrying `index`; the client orders by index and refuses a missing or repeated one."""

    def __init__(self, name, s, env, transport=None):
        super().__init__(name, s, env, transport, urllib_transport)

    def _call(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "input": [self._text(t, kind) for t in texts],
            "encoding_format": "float",
        }
        if self.s.send_dimensions and self.s.dims:
            payload["dimensions"] = self.s.dims
        key = self._key()
        data = post_json(
            self.transport,
            f"{(self.s.base_url or '').rstrip('/')}/embeddings",
            {"Authorization": f"Bearer {key}"},
            payload,
            self.s.timeout_s,
            self.env,
            key,
        )
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError("malformed", "data is missing or has the wrong length")
        try:
            order = sorted(range(len(items)), key=lambda i: items[i]["index"])
            if [items[i]["index"] for i in order] != list(range(len(texts))):
                raise EmbeddingError("malformed", "data indexes are not 0..n-1")
            vectors = [items[i]["embedding"] for i in order]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("malformed", "data items lack index or embedding") from exc
        return self._finish(vectors, len(texts))


class GeminiEmbedder(_Base):
    """POST models/{model}:batchEmbedContents with one request per text.

    task_style "task_type": taskType RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY (gemini-embedding-001).
    task_style "prompt": the task is written into the text (gemini-embedding-2, no taskType).
    Response embeddings[].values (taken from SDK examples; the REST body was not shown)."""

    def __init__(self, name, s, env, transport=None):
        super().__init__(name, s, env, transport, urllib_transport)

    def _content(self, text: str, kind: Kind) -> str:
        text = self._text(text, kind)
        if self.s.task_style == "prompt":
            return (
                f"task: search result | query: {text}"
                if kind == "query"
                else f"title: none | text: {text}"
            )
        return text

    def _call(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        requests = []
        for t in texts:
            r: dict[str, Any] = {
                "model": f"models/{self.s.model}",
                "content": {"parts": [{"text": self._content(t, kind)}]},
            }
            if self.s.task_style == "task_type":
                r["taskType"] = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
            if self.s.dims:
                r["output_dimensionality"] = self.s.dims
            requests.append(r)
        key = self._key()
        data = post_json(
            self.transport,
            f"{(self.s.base_url or GEMINI_BASE).rstrip('/')}/models/{self.s.model}"
            ":batchEmbedContents",
            {"x-goog-api-key": key},
            {"requests": requests},
            self.s.timeout_s,
            self.env,
        )
        items = data.get("embeddings")
        if not isinstance(items, list) or len(items) != len(texts):
            raise EmbeddingError("malformed", "embeddings is missing or has the wrong length")
        try:
            vectors = [e["values"] for e in items]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError("malformed", "an embedding lacks values") from exc
        return self._finish(vectors, len(texts))


class OllamaEmbedder(_Base):
    """POST {base_url}/api/embed with {model, input[], truncate: false}. Response embeddings[][].

    truncate is false on purpose: a chunk longer than the model's window is an error, not a
    silently shortened text."""

    def __init__(self, name, s, env, transport=None):
        super().__init__(name, s, env, transport, loopback_transport)

    def _call(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.s.model,
            "input": [self._text(t, kind) for t in texts],
            "truncate": False,
        }
        if self.s.dims:
            payload["dimensions"] = self.s.dims
        data = post_json(
            self.transport,
            f"{(self.s.base_url or '').rstrip('/')}/api/embed",
            {},
            payload,
            self.s.timeout_s,
            self.env,
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list):
            raise EmbeddingError("malformed", "embeddings is missing")
        return self._finish(vectors, len(texts))


class LocalEmbedder(_Base):
    """Runs a small embedding model on this computer's CPU through the `fastembed` package (ONNX,
    no daemon, no network after the weights are on disk). Install with `pip install -e ".[local]"`.

    The package, its API (`TextEmbedding(model_name=..., cache_dir=...)` and `.embed(texts)`), the
    model names and their sizes were written from memory and are UNVERIFIED
    (docs/PHASE_2B_ASSUMPTIONS.md). The first load may download the model files, so a one-line
    notice is printed before the load. Nothing loads until a probe asks for an embedding."""

    def __init__(self, name, s, env, transport=None, library=None, notice=None):
        super().__init__(name, s, env, transport, urllib_transport)
        self._library = library  # a callable (model_name, cache_dir) -> object with .embed(texts)
        self._notice = notice or (lambda line: print(line, file=sys.stderr, flush=True))
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._library
        if factory is None:
            try:
                from fastembed import TextEmbedding  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "config", 'fastembed is not installed (pip install -e ".[local]")'
                ) from exc

            def factory(model, cache_dir):
                return TextEmbedding(model_name=model, cache_dir=cache_dir)

        self._notice(
            f"loading local model {self.s.model}: the first use may download its files "
            "from the internet and can take a few minutes; later runs use the saved copy"
        )
        try:
            self._model = factory(self.s.model, self.s.cache_dir)
        except Exception as exc:  # the library's own errors (download, unknown name) vary
            raise EmbeddingError("load_failed", f"{type(exc).__name__}: {str(exc)[:200]}") from exc
        return self._model

    def _call(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        model = self._load()
        try:
            raw = list(model.embed([self._text(t, kind) for t in texts]))
            vectors = [[float(x) for x in v] for v in raw]
        except Exception as exc:
            raise EmbeddingError("malformed", f"local model failed: {type(exc).__name__}") from exc
        return self._finish(vectors, len(texts))


def build_embedder(
    name: str,
    s: EmbedderSettings,
    env: Mapping[str, str],
    transport: Callable | None = None,
    library: Callable | None = None,
    notice: Callable[[str], None] | None = None,
):
    from grounded_analytics.documents.embed import FakeEmbedder  # noqa: PLC0415

    if s.kind == "fake":
        return FakeEmbedder(s.dims or 256, name)
    cls = {
        "openai_compat": OpenAICompatEmbedder,
        "gemini": GeminiEmbedder,
        "ollama": OllamaEmbedder,
        "local": LocalEmbedder,
    }[s.kind]
    if s.kind == "local":
        return LocalEmbedder(name, s, env, transport, library, notice)
    return cls(name, s, env, transport)
