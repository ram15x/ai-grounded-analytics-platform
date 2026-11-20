"""`va docs probe` with fake and mocked models against real Postgres (pgvector for most tests)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import psycopg
import pytest

from verified_analyst.cli import main
from verified_analyst.documents.embed import FakeEmbedder
from verified_analyst.documents.metrics import load_questions
from verified_analyst.documents.probe import Paced, render_markdown, run_probe
from verified_analyst.documents.settings import (
    EmbedderSettings,
    RerankerSettings,
    RetrievalSettings,
)
from verified_analyst.documents.store import DocumentError, ingest
from verified_analyst.llm.transport import HttpResponse

ROOT = Path(__file__).parent.parent
QUESTIONS = load_questions(ROOT / "eval" / "doc_questions" / "dev.yaml")[:12]
KEY = "sk-probe-SECRET-98765"
DIMS = 64


def settings(batch=16):
    return RetrievalSettings(
        embedders={
            "fake": EmbedderSettings(kind="fake", hosted=False, dims=256),
            "hosted": EmbedderSettings(
                kind="openai_compat",
                hosted=True,
                model="mock-model",
                env_key="MOCK_KEY",
                base_url="https://mock.test/v1",
                dims=DIMS,
                batch_size=batch,
            ),
        },
        rerankers={
            "fake": RerankerSettings(kind="fake", hosted=False),
            "hosted": RerankerSettings(
                kind="rerank_api",
                hosted=True,
                model="rr",
                env_key="MOCK_KEY",
                url="https://mock.test/rerank",
                depth=5,
            ),
        },
    )


class MockProvider:
    """Embeds with the fake embedder and reranks by word overlap, over mocked HTTP."""

    def __init__(self, status=200):
        self.status, self.calls = status, []
        self.fake = FakeEmbedder(DIMS)

    def __call__(self, url, headers, body, timeout):
        payload = json.loads(body)
        self.calls.append((url, payload))
        if self.status != 200:
            return HttpResponse(self.status, f"nope {KEY}".encode(), {})
        if url.endswith("/embeddings"):
            vecs = self.fake.embed(payload["input"], "document")
            data = [{"index": i, "embedding": v} for i, v in enumerate(vecs)]
            return HttpResponse(200, json.dumps({"data": data}).encode(), {})
        docs = payload["documents"]
        q = set(payload["query"].lower().split())
        res = [
            {"index": i, "relevance_score": len(q & set(d.lower().split())) / (1 + len(q))}
            for i, d in enumerate(docs)
        ]
        return HttpResponse(200, json.dumps({"results": res}).encode(), {})


def probe(db, embedder, reranker=None, **kw):
    return run_probe(
        settings=kw.pop("settings", settings()),
        embedder_name=embedder,
        reranker_name=reranker,
        admin_url=db.admin_url,
        readonly_url=db.readonly_url,
        questions=QUESTIONS,
        env={"MOCK_KEY": KEY},
        reader_role=db.role,
        **kw,
    )


def drop_model(db, prefix):
    with psycopg.connect(db.admin_url) as c:
        c.execute("DELETE FROM docs.chunk_embeddings WHERE model_id LIKE %s", (prefix + ":%",))


@pytest.mark.db
def test_fake_models_run_all_methods_without_http(vector_db):
    res = probe(vector_db, "fake", "fake")
    r = res.report
    assert res.exit_code == 0 and r["stop"] is None
    assert set(r["methods"]) == {"keyword", "vector", "hybrid", "hybrid+rerank"}
    assert r["http"]["embedder"]["requests"] == 0 and r["scope"] == "dev questions only"
    assert "accuracy" in r["note"] and "Nothing here is an accuracy claim" in r["note"]
    for m in r["methods"].values():
        assert 0 <= m["overall"]["recall@5"] <= 1 and m["questions_scored"] > 0
    assert "| keyword |" in render_markdown(r)


@pytest.mark.db
def test_hosted_mock_counts_requests_embeds_missing_chunks_and_never_leaks_the_key(vector_db):
    mock = MockProvider()
    res = probe(
        vector_db,
        "hosted",
        "hosted",
        embed_transport=mock,
        rerank_transport=mock,
        sleep=lambda s: None,
    )
    try:
        r = res.report
        assert res.exit_code == 0, r["stop"]
        n = r["chunks"]
        embeds = [c for c in mock.calls if c[0].endswith("/embeddings")]
        docs_calls = math.ceil(n / 16)
        uniq = len({q.question for q in QUESTIONS})
        assert len(embeds) == docs_calls + math.ceil(uniq / 16)
        assert r["http"]["embedder"]["requests"] == len(embeds)
        assert r["http"]["reranker"]["requests"] == len(QUESTIONS)
        # a second run embeds no chunks again
        again = probe(vector_db, "hosted", None, embed_transport=MockProvider())
        assert again.report["chunks_embedded_this_run"] == 0
        assert KEY not in json.dumps(r) and KEY not in render_markdown(r)
    finally:
        drop_model(vector_db, "hosted")


@pytest.mark.db
def test_auth_failure_stops_on_the_first_request_with_exit_3(vector_db):
    mock = MockProvider(status=401)
    res = probe(vector_db, "hosted", embed_transport=mock)
    assert res.exit_code == 3 and res.report["stop"]["kind"] == "auth"
    assert len(mock.calls) == 1 and KEY not in json.dumps(res.report)


@pytest.mark.db
def test_rate_limit_is_reported_with_exit_4(vector_db):
    res = probe(vector_db, "hosted", embed_transport=MockProvider(status=429))
    assert res.exit_code == 4 and res.report["stop"]["kind"] == "rate_limit"
    assert KEY not in json.dumps(res.report)


@pytest.mark.db
def test_request_cap_stops_the_run(vector_db):
    mock = MockProvider()
    try:
        res = probe(vector_db, "hosted", embed_transport=mock, max_requests=2)
        assert res.exit_code == 4 and res.report["stop"]["kind"] == "request_cap"
        assert len(mock.calls) == 2
    finally:
        drop_model(vector_db, "hosted")


@pytest.mark.db
def test_a_missing_key_makes_no_request(vector_db):
    mock = MockProvider()
    res = run_probe(
        settings=settings(),
        embedder_name="hosted",
        reranker_name=None,
        admin_url=vector_db.admin_url,
        readonly_url=vector_db.readonly_url,
        questions=QUESTIONS,
        env={},
        embed_transport=mock,
    )
    assert res.exit_code == 2 and res.report["stop"]["kind"] == "no_key" and not mock.calls


@pytest.mark.db
def test_hosted_models_are_refused_when_a_document_is_not_public(vector_db):
    with psycopg.connect(vector_db.admin_url) as c:
        ingest(
            c,
            "zz-private-probe",
            "Private",
            "p.md",
            "Internal note about prices.",
            reader_role=vector_db.role,
            sensitivity="internal",
        )
    mock = MockProvider()
    try:
        with pytest.raises(DocumentError, match="public"):
            probe(vector_db, "hosted", embed_transport=mock)
        with pytest.raises(DocumentError, match="public"):
            probe(vector_db, "fake", "hosted", rerank_transport=mock)
        assert not mock.calls
        # a model on this computer is still allowed
        assert probe(vector_db, "fake", "fake").exit_code == 0
    finally:
        with psycopg.connect(vector_db.admin_url) as c:
            c.execute("DELETE FROM docs.documents WHERE doc_key = 'zz-private-probe'")


def test_unknown_names_are_refused_before_any_connection():
    for kw in ({"embedder_name": "nope"}, {"embedder_name": "fake", "reranker_name": "nope"}):
        with pytest.raises(DocumentError, match="unknown"):
            run_probe(
                settings=settings(),
                admin_url="postgresql://x/y",
                readonly_url="postgresql://x/y",
                questions=QUESTIONS,
                env={},
                **{"reranker_name": None, **kw},
            )


@pytest.mark.parametrize("n", [0, 151])
def test_request_cap_must_be_in_range(n):
    with pytest.raises(DocumentError, match="max-requests"):
        run_probe(
            settings=settings(),
            embedder_name="fake",
            reranker_name=None,
            admin_url="postgresql://x/y",
            readonly_url="postgresql://x/y",
            questions=QUESTIONS,
            env={},
            max_requests=n,
        )


def test_pacing_waits_between_requests_only():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    inner_calls = []
    p = Paced(lambda *a: inner_calls.append(a) or "ok", 0.7, sleep, lambda: now[0])
    p("u", {}, b"", 1)
    p("u", {}, b"", 1)
    now[0] += 5
    p("u", {}, b"", 1)
    assert len(inner_calls) == 3 and slept == [pytest.approx(0.7)]


def test_the_probe_path_never_names_the_locked_question_file():
    for rel in ("src/verified_analyst/documents/probe.py", "src/verified_analyst/cli.py"):
        assert "heldout.yaml" not in (ROOT / rel).read_text("utf-8"), rel


def test_cli_unknown_embedder_exits_2(monkeypatch, capsys):
    monkeypatch.setenv("VA_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "postgresql://x/y")
    assert main(["docs", "probe", "--embedder", "nope"]) == 2
    assert "unknown embedder" in capsys.readouterr().err


@pytest.mark.db
def test_probe_runs_a_local_embedder_through_an_injected_library(vector_db):
    st = RetrievalSettings(
        embedders={
            "loc": EmbedderSettings(
                kind="local", hosted=False, model="m/x", dims=64, batch_size=32, normalize=True
            )
        }
    )
    fake = FakeEmbedder(64)
    loads, notes = [], []

    def library(model, cache_dir):
        loads.append(model)
        return type("M", (), {"embed": lambda self, texts: fake.embed(list(texts), "document")})()

    try:
        res = probe(vector_db, "loc", settings=st, local_library=library, notice=notes.append)
        assert res.exit_code == 0 and res.report["stop"] is None
        assert loads == ["m/x"] and len(notes) == 1  # one load, one notice, before any embedding
        assert res.report["http"]["embedder"] == {"requests": 0}
        assert res.report["embedder"]["hosted"] is False and "vector" in res.report["methods"]
    finally:
        drop_model(vector_db, "loc")


@pytest.mark.db
def test_a_local_model_that_fails_to_load_exits_2_and_is_reported(vector_db):
    st = RetrievalSettings(
        embedders={"loc": EmbedderSettings(kind="local", hosted=False, model="m/x", dims=64)}
    )

    def library(model, cache_dir):
        raise OSError("cannot download")

    res = probe(vector_db, "loc", settings=st, local_library=library, notice=lambda s: None)
    assert res.exit_code == 2 and res.report["stop"]["kind"] == "load_failed"
