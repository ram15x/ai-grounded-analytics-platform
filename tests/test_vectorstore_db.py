"""Embedding storage, vector search and the hybrid methods against real pgvector."""

from pathlib import Path

import psycopg
import pytest

from grounded_analytics.documents.embed import FakeEmbedder
from grounded_analytics.documents.fusion import hybrid
from grounded_analytics.documents.metrics import Retrieved, evaluate, load_questions
from grounded_analytics.documents.store import DocumentError, all_chunks, ingest, search
from grounded_analytics.documents.vectorstore import (
    QueryVectors,
    embedding_model_id,
    ensure_vector_schema,
    make_vector_search,
    pending_chunks,
    store_embeddings,
    vector_search,
)

pytestmark = pytest.mark.db
ROOT_DIR = Path(__file__).parent.parent
EMB = FakeEmbedder()


def test_every_chunk_has_a_vector_and_none_are_pending(vector_db):
    with psycopg.connect(vector_db.admin_url) as c:
        assert pending_chunks(c, EMB.model_id) == []
        n, dims = c.execute(
            "SELECT count(*), min(dims) FROM docs.chunk_embeddings WHERE model_id = %s",
            (EMB.model_id,),
        ).fetchone()
    assert n == len(all_chunks(vector_db.readonly_url)) and dims == 256


def test_the_schema_can_be_applied_twice(vector_db):
    with psycopg.connect(vector_db.admin_url) as c:
        ensure_vector_schema(c, vector_db.role)
        ensure_vector_schema(c, vector_db.role)


def test_a_chunk_is_its_own_nearest_neighbour(vector_db):
    chunks = all_chunks(vector_db.readonly_url)
    for key, ordinal, text in chunks[:12]:
        (top, *_) = vector_search(
            vector_db.readonly_url, EMB.embed([text], "query")[0], EMB.model_id, 3
        )
        assert (top.doc_key, top.ordinal) == (key, ordinal)
        assert top.score == pytest.approx(1.0, abs=1e-6)


def test_results_are_ordered_and_k_is_respected(vector_db):
    qv = EMB.embed(["how long do refunds take"], "query")[0]
    hits = vector_search(vector_db.readonly_url, qv, EMB.model_id, 5)
    assert len(hits) == 5 and [h.score for h in hits] == sorted(
        (h.score for h in hits), reverse=True
    )
    for bad in (0, 201, True, "5"):
        with pytest.raises(DocumentError):
            vector_search(vector_db.readonly_url, qv, EMB.model_id, bad)


def test_another_model_id_sees_nothing_and_a_wrong_size_is_refused(vector_db):
    qv = EMB.embed(["refund"], "query")[0]
    assert vector_search(vector_db.readonly_url, qv, "other:00000000", 3) == []
    with pytest.raises(DocumentError, match="dimensions"):
        vector_search(vector_db.readonly_url, qv[:10], EMB.model_id, 3)


def test_a_stale_vector_is_ignored_and_an_edited_chunk_loses_its_vectors(vector_db):
    key = "vec-scratch"
    text = "# Vector scratch\n\nThe scratch vector document mentions teapots and nothing else."
    with psycopg.connect(vector_db.admin_url) as c:
        ingest(c, key, "Vector scratch", "v.md", text, 400, reader_role=vector_db.role)
        todo = [p for p in pending_chunks(c, EMB.model_id) if p.doc_key == key]
        store_embeddings(c, EMB.model_id, todo, EMB.embed([p.text for p in todo], "document"))
    qv = EMB.embed(["teapots"], "query")[0]
    assert vector_search(vector_db.readonly_url, qv, EMB.model_id, 1)[0].doc_key == key
    with psycopg.connect(
        vector_db.admin_url
    ) as c:  # hash recorded with the vector no longer matches
        c.execute("UPDATE docs.chunk_embeddings SET text_sha256 = 'x' WHERE doc_key = %s", (key,))
    assert key not in {
        h.doc_key for h in vector_search(vector_db.readonly_url, qv, EMB.model_id, 200)
    }
    with psycopg.connect(vector_db.admin_url) as c:
        ingest(c, key, "Vector scratch", "v.md", text + " Edited.", 400)
        gone = c.execute("SELECT count(*) FROM docs.chunk_embeddings WHERE doc_key = %s", (key,))
        assert gone.fetchone()[0] == 0
        c.execute("DELETE FROM docs.documents WHERE doc_key = %s", (key,))


def test_the_reader_cannot_write_vectors(vector_db):
    with psycopg.connect(vector_db.readonly_url) as c:
        for stmt in (
            "DELETE FROM docs.chunk_embeddings",
            "UPDATE docs.chunk_embeddings SET dims = 1",
        ):
            with pytest.raises(psycopg.Error):
                c.execute(stmt)
            c.rollback()


def test_model_ids_separate_models_dimensions_and_prefixes():
    a = embedding_model_id("m", "model-x", 768, ("", ""))
    assert a == embedding_model_id("m", "model-x", 768, ("", ""))
    assert (
        len(
            {
                a,
                embedding_model_id("m", "model-x", 512, ("", "")),
                embedding_model_id("m", "model-x", 768, ("q: ", "")),
                embedding_model_id("m", "model-y", 768, ("", "")),
            }
        )
        == 4
    )


def test_hybrid_and_vector_methods_run_through_the_evaluator(vector_db):
    url = vector_db.readonly_url
    qv = QueryVectors(EMB)
    kw = lambda q, k: search(url, q, k)  # noqa: E731
    vec = make_vector_search(url, EMB.model_id, qv)
    corpus = [Retrieved(k, t) for k, _, t in all_chunks(url)]
    qs = load_questions(ROOT_DIR / "eval" / "doc_questions" / "dev.yaml")

    def as_retrieved(method):
        return lambda q, k: [Retrieved(h.doc_key, h.text) for h in method(q, k)]

    for method in (kw, vec, hybrid(kw, vec)):
        rep = evaluate(as_retrieved(method), qs, corpus)
        assert 0 <= rep["overall"]["recall@5"] <= 1 and rep["questions_scored"] > 20
