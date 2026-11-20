"""Ingest documents, search them, and resolve stored passages for the verifier.

Passage ids look like `returns-policy#3.9f2c41ab07`: document key, chunk ordinal, and the first ten
hex digits of the chunk text's SHA-256. A citation therefore names one exact version of one exact
text. If the document is edited and re-ingested, an old id no longer resolves, and a claim that
cites it is UNSUPPORTED instead of silently pointing at different words.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import psycopg

from verified_analyst.config import connect_kwargs
from verified_analyst.documents.chunker import DEFAULT_MAX_CHARS, chunk_text

DOC_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PASSAGE_ID = re.compile(r"^([a-z0-9][a-z0-9_-]{0,63})#(\d{1,6})\.([0-9a-f]{10})$")
MAX_DOC_CHARS = 2_000_000
# Column per text-search configuration. The column name never comes from the caller.
_TSV = {"english": "tsv_english", "simple": "tsv_simple"}
MAX_K = 50


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Hit:
    passage_id: str
    doc_key: str
    title: str
    ordinal: int
    char_start: int
    char_end: int
    text: str
    score: float


@dataclass(frozen=True)
class IngestResult:
    doc_key: str
    chunks: int
    changed: bool


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(raw: str) -> str:
    """Line endings to `\\n`, BOM removed. Nothing else is touched, so spans match the source."""
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise DocumentError("document contains a NUL character")
    if not text.strip():
        raise DocumentError("document is empty")
    if len(text) > MAX_DOC_CHARS:
        raise DocumentError(f"document is longer than {MAX_DOC_CHARS} characters")
    return text


def make_passage_id(doc_key: str, ordinal: int, text_sha256: str) -> str:
    return f"{doc_key}#{ordinal}.{text_sha256[:10]}"


def parse_passage_id(passage_id: str) -> tuple[str, int, str] | None:
    m = _PASSAGE_ID.match(passage_id) if isinstance(passage_id, str) else None
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _ensure_schema(conn: psycopg.Connection) -> None:
    ddl = resources.files("verified_analyst.documents").joinpath("schema.sql").read_text("utf-8")
    conn.execute(ddl)


def ingest(
    conn: psycopg.Connection,
    doc_key: str,
    title: str,
    source: str,
    raw_text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    reader_role: str | None = None,
) -> IngestResult:
    """Store one document and its chunks in one transaction. Same text and size: no change."""
    if not DOC_KEY.match(doc_key):
        raise DocumentError("doc_key must be lowercase letters, digits, '-' or '_' (1-64 chars)")
    text = normalize_text(raw_text)
    spans = chunk_text(text, max_chars)
    digest = sha256_hex(text)
    with conn.transaction():
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT content_sha256 FROM docs.documents WHERE doc_key = %s", (doc_key,)
        ).fetchone()
        same_chunks = row is not None and row[0] == digest and _same_layout(conn, doc_key, spans)
        if not same_chunks:
            conn.execute(
                "INSERT INTO docs.documents (doc_key, title, source, content_sha256, text) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (doc_key) DO UPDATE SET "
                "title = EXCLUDED.title, source = EXCLUDED.source, "
                "content_sha256 = EXCLUDED.content_sha256, text = EXCLUDED.text, "
                "ingested_at = now()",
                (doc_key, title, source, digest, text),
            )
            conn.execute("DELETE FROM docs.chunks WHERE doc_key = %s", (doc_key,))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO docs.chunks (doc_key, ordinal, char_start, char_end, text, "
                    "text_sha256) VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (
                            doc_key,
                            i,
                            s.start,
                            s.end,
                            text[s.start : s.end],
                            sha256_hex(text[s.start : s.end]),
                        )
                        for i, s in enumerate(spans)
                    ],
                )
        if reader_role:
            _grant_reader(conn, reader_role)
    return IngestResult(doc_key, len(spans), not same_chunks)


def key_from_filename(name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", name.rsplit(".", 1)[0].lower()).strip("-")
    return stem[:64]


def read_document(path: Path) -> tuple[str, str, str]:
    """(doc_key, title, text) for a .md or .txt file. The source recorded is the file name only."""
    if path.suffix.lower() not in (".md", ".txt"):
        raise DocumentError(f"{path.name}: only .md and .txt files are supported")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"{path.name}: not valid UTF-8") from exc
    key = key_from_filename(path.name)
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    title = first.lstrip("# ").strip() or key
    return key, title[:200], text


def _same_layout(conn: psycopg.Connection, doc_key: str, spans: list) -> bool:
    stored = conn.execute(
        "SELECT char_start, char_end FROM docs.chunks WHERE doc_key = %s ORDER BY ordinal",
        (doc_key,),
    ).fetchall()
    return stored == [(s.start, s.end) for s in spans]


def _grant_reader(conn: psycopg.Connection, role: str) -> None:
    from psycopg import sql  # noqa: PLC0415

    r = sql.Identifier(role)
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA docs TO {}").format(r))
    conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA docs TO {}").format(r))


def search(
    readonly_url: str, query: str, k: int = 5, config: str = "english", mode: str = "any"
) -> list[Hit]:
    """Keyword search over chunks. Ranking is deterministic: score, then document, then ordinal.

    `mode="any"` matches a chunk that has any content word of the question and ranks by cover
    density, which suits a question written as a sentence. `mode="all"` needs every word (web
    search syntax: quotes, `-word`, `or`) and returns nothing for most sentences."""
    if config not in _TSV:
        raise DocumentError(f"config must be one of {sorted(_TSV)}")
    if mode not in ("any", "all"):
        raise DocumentError("mode must be 'any' or 'all'")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        raise DocumentError(f"k must be an integer from 1 to {MAX_K}")
    if not isinstance(query, str) or not query.strip():
        return []
    col = _TSV[config]
    parse = (
        f"websearch_to_tsquery('{config}', %s)"
        if mode == "all"
        else f"replace(plainto_tsquery('{config}', %s)::text, '&', '|')::tsquery"
    )
    stmt = (
        f"SELECT c.doc_key, d.title, c.ordinal, c.char_start, c.char_end, c.text, "
        f"c.text_sha256, ts_rank_cd(c.{col}, q.query, 32) AS score "
        f"FROM docs.chunks c JOIN docs.documents d USING (doc_key), "
        f"(SELECT {parse} AS query) AS q "
        f"WHERE c.{col} @@ q.query "
        f"ORDER BY score DESC, c.doc_key, c.ordinal LIMIT %s"
    )
    with psycopg.connect(readonly_url, **connect_kwargs(readonly_url)) as conn:
        conn.read_only = True
        rows = conn.execute(stmt, (query[:500], k)).fetchall()
    return [
        Hit(make_passage_id(r[0], r[2], r[6]), r[0], r[1], r[2], r[3], r[4], r[5], float(r[7]))
        for r in rows
    ]


def resolve_passages(readonly_url: str, passage_ids: list[str]) -> dict[str, str]:
    """Stored text for each id that names a chunk whose recorded hash still matches.

    The chunk must hash to its recorded SHA-256, that hash must start with the digits in the id,
    and the chunk must equal the slice of its document at the recorded offsets. Anything else is
    left out, and the verifier then reports the claim as UNSUPPORTED (unknown_passage)."""
    wanted = {pid: parse_passage_id(pid) for pid in set(passage_ids)}
    wanted = {pid: p for pid, p in wanted.items() if p is not None}
    if not wanted:
        return {}
    out: dict[str, str] = {}
    with psycopg.connect(readonly_url, **connect_kwargs(readonly_url)) as conn:
        conn.read_only = True
        for pid, (key, ordinal, sha10) in wanted.items():
            row = conn.execute(
                "SELECT c.text, c.text_sha256, "
                "substr(d.text, c.char_start + 1, c.char_end - c.char_start) = c.text "
                "FROM docs.chunks c JOIN docs.documents d USING (doc_key) "
                "WHERE c.doc_key = %s AND c.ordinal = %s",
                (key, ordinal),
            ).fetchone()
            if row is None:
                continue
            text, recorded, matches_document = row
            if matches_document and sha256_hex(text) == recorded and recorded.startswith(sha10):
                out[pid] = text
    return out


def all_chunks(readonly_url: str) -> list[tuple[str, int, str]]:
    """(doc_key, ordinal, text) of every chunk, in a fixed order. For label checks and metrics."""
    with psycopg.connect(readonly_url, **connect_kwargs(readonly_url)) as conn:
        conn.read_only = True
        rows = conn.execute("SELECT doc_key, ordinal, text FROM docs.chunks ORDER BY 1, 2")
        return [(r[0], r[1], r[2]) for r in rows.fetchall()]


def document_span(chunk_char_start: int, span: tuple[int, int]) -> tuple[int, int]:
    """Map a span relative to a chunk to offsets in the whole document."""
    return chunk_char_start + span[0], chunk_char_start + span[1]
