"""Document store, keyword search and stored-passage verification (needs Postgres)."""

from pathlib import Path

import psycopg
import pytest

from verified_analyst.documents.metrics import Retrieved, load_questions, unreachable_anchors
from verified_analyst.documents.store import (
    DocumentError,
    all_chunks,
    document_span,
    ingest,
    make_passage_id,
    resolve_passages,
    search,
    sha256_hex,
)
from verified_analyst.guard import GuardError, validate_sql

pytestmark = pytest.mark.db
ROOT = Path(__file__).parent.parent
TEXT = (
    "# Scratch policy\n\nThe scratch charge is £4.95 per parcel.\n\n"
    "Parcels above 10 kg pay £1.20 per extra kilogram."
)


@pytest.fixture
def scratch(docs_db):
    """A throwaway document; removed again so other tests see the corpus only."""
    key = "scratch-doc"
    with psycopg.connect(docs_db.admin_url) as c:
        ingest(c, key, "Scratch policy", "scratch.md", TEXT, 80, reader_role=docs_db.role)
    yield key
    with psycopg.connect(docs_db.admin_url) as c:
        c.execute("DELETE FROM docs.documents WHERE doc_key = %s", (key,))


def _chunk(docs_db, key, ordinal=0):
    with psycopg.connect(docs_db.admin_url) as c:
        return c.execute(
            "SELECT text, text_sha256, char_start FROM docs.chunks "
            "WHERE doc_key = %s AND ordinal = %s",
            (key, ordinal),
        ).fetchone()


# ---- ingestion


def test_the_corpus_is_stored_with_exact_spans(docs_db):
    with psycopg.connect(docs_db.admin_url) as c:
        rows = c.execute(
            "SELECT c.text, substr(d.text, c.char_start + 1, c.char_end - c.char_start) "
            "FROM docs.chunks c JOIN docs.documents d USING (doc_key)"
        ).fetchall()
    assert len(rows) >= 14 and all(a == b for a, b in rows)


def test_ingesting_the_same_text_again_changes_nothing(docs_db, scratch):
    with psycopg.connect(docs_db.admin_url) as c:
        r = ingest(c, scratch, "Scratch policy", "scratch.md", TEXT, 80)
    assert r.changed is False


def test_a_different_chunk_size_restores_the_chunks(docs_db, scratch):
    with psycopg.connect(docs_db.admin_url) as c:
        r = ingest(c, scratch, "Scratch policy", "scratch.md", TEXT, 400)
        n = c.execute("SELECT count(*) FROM docs.chunks WHERE doc_key = %s", (scratch,)).fetchone()
    assert r.changed and n[0] == r.chunks == 1


@pytest.mark.parametrize("key", ["", "Has Caps", "a b", "x" * 65, "-lead", "a#b", "../x"])
def test_bad_document_keys_are_refused(docs_db, key):
    with psycopg.connect(docs_db.admin_url) as c, pytest.raises(DocumentError):
        ingest(c, key, "t", "s", "some text")


def test_a_failed_ingest_leaves_nothing_behind(docs_db):
    with psycopg.connect(docs_db.admin_url) as c, pytest.raises(DocumentError):
        ingest(c, "half-done", "t", "s", "bad\x00text")
    with psycopg.connect(docs_db.admin_url) as c:
        assert (
            c.execute("SELECT 1 FROM docs.documents WHERE doc_key = 'half-done'").fetchone() is None
        )


# ---- the read-only role and the SQL guard


def test_the_reader_can_read_documents_but_not_change_them(docs_db):
    with psycopg.connect(docs_db.readonly_url) as c:
        assert c.execute("SELECT count(*) FROM docs.chunks").fetchone()[0] > 0
        for stmt in (
            "DELETE FROM docs.chunks",
            "UPDATE docs.documents SET title = 'x'",
            "INSERT INTO docs.documents (doc_key, title, source, content_sha256, text) "
            "VALUES ('a', 'a', 'a', 'a', 'a')",
            "DROP TABLE docs.chunks",
        ):
            with pytest.raises(psycopg.Error):
                c.execute(stmt)
            c.rollback()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT text FROM docs.chunks",
        "SELECT text FROM docs.documents",
        "SELECT * FROM chunks",
        "SELECT count(*) FROM analytics.sales_lines s, docs.chunks c",
    ],
)
def test_model_sql_can_never_read_documents(sql):
    with pytest.raises(GuardError):
        validate_sql(sql)


# ---- search


def test_a_sentence_question_finds_its_passage(docs_db):
    hits = search(docs_db.readonly_url, "How many days do customers have to return items?", 3)
    assert hits and hits[0].doc_key == "returns-policy"
    assert "28 days" in hits[0].text


def test_an_exact_code_is_found(docs_db):
    hits = search(docs_db.readonly_url, "85123A", 3)
    assert hits and hits[0].doc_key == "stock-policy"


def test_a_question_with_no_matching_word_returns_nothing(docs_db):
    assert search(docs_db.readonly_url, "zzyzx qqqq", 5) == []
    assert search(docs_db.readonly_url, "   ", 5) == []


def test_all_mode_needs_every_word(docs_db):
    url = docs_db.readonly_url
    assert search(url, "restocking fee opened", 5, mode="all")
    assert search(url, "restocking fee zzyzx", 5, mode="all") == []
    assert search(url, "restocking fee zzyzx", 5, mode="any")


def test_results_are_ranked_deterministically_and_k_is_respected(docs_db):
    a = search(docs_db.readonly_url, "working days delivery refund", 4)
    b = search(docs_db.readonly_url, "working days delivery refund", 4)
    assert [h.passage_id for h in a] == [h.passage_id for h in b] and len(a) <= 4
    assert [h.score for h in a] == sorted((h.score for h in a), reverse=True)


@pytest.mark.parametrize("k", [0, -1, 51, True, "5", 2.5])
def test_bad_k_is_refused(docs_db, k):
    with pytest.raises(DocumentError):
        search(docs_db.readonly_url, "returns", k)


@pytest.mark.parametrize("kw", [{"config": "english'; DROP TABLE docs.chunks; --"}, {"mode": "x"}])
def test_search_options_are_whitelisted(docs_db, kw):
    with pytest.raises(DocumentError):
        search(docs_db.readonly_url, "returns", 3, **kw)


@pytest.mark.parametrize(
    "query", ["'; DROP TABLE docs.chunks; --", "a & b | !c", '"unbalanced', "(((", "x:*", "\\"]
)
def test_odd_queries_never_break_search(docs_db, query):
    search(docs_db.readonly_url, query, 3)
    search(docs_db.readonly_url, query, 3, mode="all")
    assert all_chunks(docs_db.readonly_url)


def test_hits_carry_exact_spans_and_resolvable_ids(docs_db):
    (h, *_) = search(docs_db.readonly_url, "restocking fee", 1)
    assert resolve_passages(docs_db.readonly_url, [h.passage_id]) == {h.passage_id: h.text}
    with psycopg.connect(docs_db.admin_url) as c:
        doc = c.execute("SELECT text FROM docs.documents WHERE doc_key = %s", (h.doc_key,))
        assert doc.fetchone()[0][h.char_start : h.char_end] == h.text
    assert document_span(h.char_start, (2, 6)) == (h.char_start + 2, h.char_start + 6)


# ---- stored passages resolve only while their text is exactly what was stored


def test_an_edited_document_invalidates_the_old_passage_id(docs_db, scratch):
    text, sha, _ = _chunk(docs_db, scratch)
    pid = make_passage_id(scratch, 0, sha)
    assert resolve_passages(docs_db.readonly_url, [pid]) == {pid: text}
    with psycopg.connect(docs_db.admin_url) as c:
        ingest(c, scratch, "Scratch policy", "scratch.md", TEXT.replace("£4.95", "£9.95"), 80)
    assert resolve_passages(docs_db.readonly_url, [pid]) == {}


def test_a_chunk_changed_behind_the_store_is_not_trusted(docs_db, scratch):
    text, sha, _ = _chunk(docs_db, scratch)
    pid = make_passage_id(scratch, 0, sha)
    with psycopg.connect(docs_db.admin_url) as c:
        c.execute(
            "UPDATE docs.chunks SET text = replace(text, '£4.95', '£1.00') "
            "WHERE doc_key = %s AND ordinal = 0",
            (scratch,),
        )
    assert resolve_passages(docs_db.readonly_url, [pid]) == {}


def test_a_document_changed_behind_its_chunks_is_not_trusted(docs_db, scratch):
    text, sha, _ = _chunk(docs_db, scratch)
    pid = make_passage_id(scratch, 0, sha)
    with psycopg.connect(docs_db.admin_url) as c:
        c.execute(
            "UPDATE docs.documents SET text = replace(text, '£4.95', '£1.00') WHERE doc_key = %s",
            (scratch,),
        )
    assert resolve_passages(docs_db.readonly_url, [pid]) == {}


def test_unknown_malformed_and_wrong_hash_ids_do_not_resolve(docs_db, scratch):
    _, sha, _ = _chunk(docs_db, scratch)
    ids = [
        "nope#0." + sha[:10],
        make_passage_id(scratch, 99, sha),
        make_passage_id(scratch, 0, "0" * 64),
        "garbage",
        "",
    ]
    assert resolve_passages(docs_db.readonly_url, ids) == {}
    assert resolve_passages(docs_db.readonly_url, []) == {}


# ---- the corpus and its labels


def test_every_dev_and_heldout_anchor_is_in_the_default_chunks(docs_db):
    corpus = [Retrieved(k, t) for k, _, t in all_chunks(docs_db.readonly_url)]
    for name in ("dev", "heldout"):
        qs = load_questions(ROOT / "eval" / "doc_questions" / f"{name}.yaml")
        assert unreachable_anchors(qs, corpus) == [], name


def test_the_hostile_document_is_stored_as_plain_text(docs_db):
    with psycopg.connect(docs_db.admin_url) as c:
        text = c.execute(
            "SELECT text FROM docs.documents WHERE doc_key = 'supplier-email-forwarded'"
        ).fetchone()[0]
    assert "ignore all previous instructions" in text
    assert sha256_hex(text)
