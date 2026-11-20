"""Embedding storage and vector search over the `docs` schema (needs the pgvector extension).

Stored vectors are keyed by (document, chunk, model_id). `model_id` names the model, the
dimensions and the prefixes, so changing any of them makes new rows instead of mixing vectors that
cannot be compared. Search is exact (a scan with cosine distance); at this corpus size an index
would add nothing, and an approximate index would blur the comparison between methods.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources

import psycopg

from verified_analyst.config import connect_kwargs
from verified_analyst.documents.embed import Embedder, vector_literal
from verified_analyst.documents.fusion import Search
from verified_analyst.documents.store import DocumentError, Hit, _grant_reader, make_passage_id


def ensure_vector_schema(conn: psycopg.Connection, reader_role: str | None = None) -> None:
    ddl = (
        resources.files("verified_analyst.documents")
        .joinpath("vector_schema.sql")
        .read_text("utf-8")
    )
    try:
        conn.execute(ddl)
    except psycopg.errors.FeatureNotSupported as exc:
        raise DocumentError(
            "the pgvector extension is not available in this Postgres; use the pgvector image "
            "(docs/DOCKER_PGVECTOR_SWITCH.md)"
        ) from exc
    if reader_role:
        _grant_reader(conn, reader_role)


@dataclass(frozen=True)
class PendingChunk:
    doc_key: str
    ordinal: int
    text: str
    text_sha256: str
    sensitivity: str


def pending_chunks(conn: psycopg.Connection, model_id: str) -> list[PendingChunk]:
    """Chunks with no stored vector for this model, in a fixed order."""
    rows = conn.execute(
        "SELECT c.doc_key, c.ordinal, c.text, c.text_sha256, d.sensitivity "
        "FROM docs.chunks c JOIN docs.documents d USING (doc_key) "
        "WHERE NOT EXISTS (SELECT 1 FROM docs.chunk_embeddings e WHERE e.doc_key = c.doc_key "
        "AND e.ordinal = c.ordinal AND e.model_id = %s) ORDER BY c.doc_key, c.ordinal",
        (model_id,),
    ).fetchall()
    return [PendingChunk(*r) for r in rows]


def store_embeddings(
    conn: psycopg.Connection,
    model_id: str,
    chunks: Sequence[PendingChunk],
    vectors: Sequence[Sequence[float]],
) -> int:
    if len(chunks) != len(vectors):
        raise DocumentError("one vector per chunk is required")
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO docs.chunk_embeddings (doc_key, ordinal, model_id, text_sha256, dims, "
            "embedding) VALUES (%s, %s, %s, %s, %s, %s::docs.vector) "
            "ON CONFLICT (doc_key, ordinal, model_id) DO UPDATE SET "
            "text_sha256 = EXCLUDED.text_sha256, dims = EXCLUDED.dims, "
            "embedding = EXCLUDED.embedding",
            [
                (c.doc_key, c.ordinal, model_id, c.text_sha256, len(v), vector_literal(v))
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
    return len(chunks)


def vector_search(
    readonly_url: str, query_vec: Sequence[float], model_id: str, k: int = 5
) -> list[Hit]:
    """Nearest chunks by cosine similarity (1 is identical). Deterministic: ties by key."""
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 200:
        raise DocumentError("k must be an integer from 1 to 200")
    lit = vector_literal(query_vec)
    stmt = (
        "SELECT c.doc_key, d.title, c.ordinal, c.char_start, c.char_end, c.text, c.text_sha256, "
        "1 - (e.embedding OPERATOR(docs.<=>) %s::docs.vector) AS score "
        "FROM docs.chunk_embeddings e "
        "JOIN docs.chunks c ON c.doc_key = e.doc_key AND c.ordinal = e.ordinal "
        "JOIN docs.documents d ON d.doc_key = c.doc_key "
        "WHERE e.model_id = %s AND e.text_sha256 = c.text_sha256 "
        "ORDER BY e.embedding OPERATOR(docs.<=>) %s::docs.vector, c.doc_key, c.ordinal LIMIT %s"
    )
    try:
        with psycopg.connect(readonly_url, **connect_kwargs(readonly_url)) as conn:
            conn.read_only = True
            rows = conn.execute(stmt, (lit, model_id, lit, k)).fetchall()
    except psycopg.errors.DataException as exc:
        raise DocumentError("query vector and stored vectors differ in dimensions") from exc
    return [
        Hit(make_passage_id(r[0], r[2], r[6]), r[0], r[1], r[2], r[3], r[4], r[5], float(r[7]))
        for r in rows
    ]


class QueryVectors:
    """Query embeddings, computed once per text. The probe fills it in batches (few requests)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._cache: dict[str, list[float]] = {}

    def prefill(
        self, texts: Sequence[str], embed_batch: Callable[[Sequence[str]], list[list[float]]]
    ):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for t, v in zip(todo, embed_batch(todo), strict=True):
            self._cache[t] = v

    def get(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed([text], "query")[0]
        return self._cache[text]


def make_vector_search(readonly_url: str, model_id: str, queries: QueryVectors) -> Search:
    def search(query: str, k: int) -> list[Hit]:
        return vector_search(readonly_url, queries.get(query), model_id, k)

    return search


def embedding_model_id(name: str, model: str, dims: int | None, prefixes: tuple[str, str]) -> str:
    """A short stable id. Same model with different dimensions or prefixes is a different id."""
    tag = hashlib.sha256(repr((model, dims, prefixes)).encode()).hexdigest()[:8]
    return f"{name}:{tag}"
