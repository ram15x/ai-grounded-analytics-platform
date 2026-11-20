"""Embedding and rerank clients against mocked HTTP only."""

from __future__ import annotations

import json
import math

import pytest

from verified_analyst.documents.embed import EmbeddingError
from verified_analyst.documents.embed_http import (
    GeminiEmbedder,
    OllamaEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
)
from verified_analyst.documents.rerank import ApiReranker, CrossEncoderReranker, FakeReranker
from verified_analyst.documents.settings import EmbedderSettings, RerankerSettings
from verified_analyst.llm.transport import HttpResponse, TransportError, TransportTimeout

KEY = "sk-test-SECRET-12345"


class Rec:
    def __init__(self, status=200, body=None, exc=None):
        self.status, self.body, self.exc, self.calls = status, body, exc, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body)))
        if self.exc:
            raise self.exc
        raw = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()
        return HttpResponse(self.status, raw)


def oa(**kw):
    d = dict(
        kind="openai_compat",
        hosted=True,
        model="m",
        env_key="K",
        base_url="https://x.test/api/v1",
        dims=8,
        batch_size=2,
    )
    return EmbedderSettings(**{**d, **kw})


def vec(n, seed=1.0):
    return [seed + i for i in range(n)]


def test_openai_orders_by_index_and_batches():
    def body(texts):
        return {
            "data": [{"index": i, "embedding": vec(8, i + 1)} for i in reversed(range(len(texts)))]
        }

    t = Rec()
    e = OpenAICompatEmbedder("a", oa(), {"K": KEY}, t)

    def post(url, headers, raw, timeout):
        payload = json.loads(raw)
        t.calls.append((url, headers, payload))
        return HttpResponse(200, json.dumps(body(payload["input"])).encode())

    e.transport = post
    out = e.embed(["a", "b", "c"], "document")
    assert len(out) == 3 and [len(c[2]["input"]) for c in t.calls] == [2, 1]
    assert out[0][0] == 1.0 and out[1][0] == 2.0
    assert t.calls[0][0] == "https://x.test/api/v1/embeddings"
    assert t.calls[0][1]["Authorization"] == f"Bearer {KEY}"
    assert "dimensions" not in t.calls[0][2]


def test_openai_sends_dimensions_only_when_asked_and_applies_prefix():
    t = Rec(body={"data": [{"index": 0, "embedding": vec(8)}]})
    e = OpenAICompatEmbedder("a", oa(send_dimensions=True, query_prefix="q: "), {"K": KEY}, t)
    e.embed(["hi"], "query")
    assert t.calls[0][2]["dimensions"] == 8 and t.calls[0][2]["input"] == ["q: hi"]


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},
        {"data": [{"index": 5, "embedding": vec(8)}]},
        {"data": [{"embedding": vec(8)}]},
        {"data": [{"index": 0, "embedding": vec(7)}]},
        {"data": [{"index": 0, "embedding": [float("nan")] * 8}]},
        {"nodata": 1},
    ],
)
def test_openai_malformed_is_refused(body):
    e = OpenAICompatEmbedder("a", oa(), {"K": KEY}, Rec(body=json.dumps(body).encode()))
    with pytest.raises(EmbeddingError):
        e.embed(["x"], "document")


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (403, "auth"), (429, "rate_limit"), (500, "server"), (400, "bad_request")],
)
def test_error_mapping_and_key_never_in_message(status, kind):
    e = OpenAICompatEmbedder("a", oa(), {"K": KEY}, Rec(status, f"oops {KEY}".encode()))
    with pytest.raises(EmbeddingError) as ei:
        e.embed(["x"], "document")
    assert ei.value.kind == kind and KEY not in str(ei.value)


def test_transport_failures_map():
    for exc, kind in ((TransportTimeout("slow"), "timeout"), (TransportError("down"), "network")):
        e = OpenAICompatEmbedder("a", oa(), {"K": KEY}, Rec(exc=exc))
        with pytest.raises(EmbeddingError) as ei:
            e.embed(["x"], "document")
        assert ei.value.kind == kind


def test_not_json_is_malformed_and_missing_key_makes_no_call():
    e = OpenAICompatEmbedder("a", oa(), {"K": KEY}, Rec(body=b"<html>"))
    with pytest.raises(EmbeddingError) as ei:
        e.embed(["x"], "document")
    assert ei.value.kind == "malformed"
    t = Rec(body={})
    with pytest.raises(EmbeddingError) as ei:
        OpenAICompatEmbedder("a", oa(), {}, t).embed(["x"], "document")
    assert ei.value.kind == "no_key" and not t.calls


def test_placeholder_model_is_refused_without_a_call():
    t = Rec(body={})
    with pytest.raises(EmbeddingError) as ei:
        OpenAICompatEmbedder("a", oa(model="REPLACE_ME"), {"K": KEY}, t).embed(["x"], "document")
    assert ei.value.kind == "config" and not t.calls


def gem(**kw):
    d = dict(
        kind="gemini",
        hosted=True,
        model="gemini-embedding-001",
        env_key="K",
        dims=8,
        task_style="task_type",
        normalize=True,
    )
    return EmbedderSettings(**{**d, **kw})


def test_gemini_task_type_request_and_normalisation():
    t = Rec(body={"embeddings": [{"values": [3.0, 4.0] + [0.0] * 6}]})
    out = GeminiEmbedder("g", gem(), {"K": KEY}, t).embed(["hello"], "query")
    url, headers, payload = t.calls[0]
    assert url.endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert headers["x-goog-api-key"] == KEY and "key=" not in url
    r = payload["requests"][0]
    assert r["taskType"] == "RETRIEVAL_QUERY" and r["output_dimensionality"] == 8
    assert r["model"] == "models/gemini-embedding-001"
    assert math.isclose(sum(x * x for x in out[0]), 1.0) and out[0][0] == pytest.approx(0.6)


def test_gemini_prompt_style_has_no_task_type():
    t = Rec(body={"embeddings": [{"values": vec(8)}] * 2})
    GeminiEmbedder(
        "g", gem(model="gemini-embedding-2", task_style="prompt", normalize=False), {"K": KEY}, t
    ).embed(["a", "b"], "document")
    reqs = t.calls[0][2]["requests"]
    assert all("taskType" not in r for r in reqs)
    assert reqs[0]["content"]["parts"][0]["text"] == "title: none | text: a"
    t2 = Rec(body={"embeddings": [{"values": vec(8)}]})
    GeminiEmbedder("g", gem(task_style="prompt", normalize=False), {"K": KEY}, t2).embed(
        ["q"], "query"
    )
    assert (
        t2.calls[0][2]["requests"][0]["content"]["parts"][0]["text"]
        == "task: search result | query: q"
    )


def test_gemini_malformed():
    for body in ({"embeddings": [{"vals": 1}]}, {"embeddings": []}, {}):
        with pytest.raises(EmbeddingError):
            GeminiEmbedder("g", gem(), {"K": KEY}, Rec(body=body)).embed(["x"], "document")


def test_ollama_sets_truncate_false_and_needs_no_key():
    s = EmbedderSettings(
        kind="ollama", hosted=False, model="m", base_url="http://127.0.0.1:11434", dims=8
    )
    t = Rec(body={"embeddings": [vec(8), vec(8, 2)]})
    out = OllamaEmbedder("o", s, {}, t).embed(["a", "b"], "document")
    assert len(out) == 2 and t.calls[0][0] == "http://127.0.0.1:11434/api/embed"
    assert t.calls[0][2]["truncate"] is False and "Authorization" not in t.calls[0][1]


def test_ollama_default_transport_refuses_non_loopback():
    s = EmbedderSettings(
        kind="ollama", hosted=False, model="m", base_url="http://example.com", dims=8
    )
    with pytest.raises(EmbeddingError) as ei:
        OllamaEmbedder("o", s, {}).embed(["a"], "document")
    assert ei.value.kind == "network"


def test_build_embedder_kinds():
    assert type(build_embedder("a", oa(), {})).__name__ == "OpenAICompatEmbedder"
    assert (
        type(build_embedder("f", EmbedderSettings(kind="fake", hosted=False, dims=16), {})).__name__
        == "FakeEmbedder"
    )


def rr(**kw):
    d = dict(kind="rerank_api", hosted=True, model="m", env_key="K", url="https://r.test/v1/rerank")
    return RerankerSettings(**{**d, **kw})


def test_api_reranker_orders_by_index():
    t = Rec(
        body={
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        }
    )
    out = ApiReranker("r", rr(), {"K": KEY}, t).score("q", ["a", "b"])
    assert out == [0.1, 0.9]
    p = t.calls[0][2]
    assert p["query"] == "q" and p["documents"] == ["a", "b"] and p["top_n"] == 2


@pytest.mark.parametrize(
    "body",
    [
        {"results": [{"index": 0, "relevance_score": 1}]},
        {"results": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 1}]},
        {"results": [{"index": 0, "relevance_score": "x"}, {"index": 1, "relevance_score": 1}]},
        {"results": [{"index": 9, "relevance_score": 1}, {"index": 1, "relevance_score": 1}]},
        {"results": [{"idx": 0}, {"idx": 1}]},
        {},
    ],
)
def test_api_reranker_malformed(body):
    with pytest.raises(EmbeddingError):
        ApiReranker("r", rr(), {"K": KEY}, Rec(body=body)).score("q", ["a", "b"])


def test_api_reranker_no_key_and_placeholder():
    with pytest.raises(EmbeddingError) as ei:
        ApiReranker("r", rr(), {}, Rec(body={})).score("q", ["a"])
    assert ei.value.kind == "no_key"
    with pytest.raises(EmbeddingError) as ei:
        ApiReranker("r", rr(model="REPLACE"), {"K": KEY}, Rec(body={})).score("q", ["a"])
    assert ei.value.kind == "config"


def test_cross_encoder_uses_injected_library_notice_once_and_checks_scores():
    notes, loads = [], []

    class M:
        def rerank(self, query, docs):
            return iter(float(len(d)) for d in docs)

    def lib(model, cache_dir):
        loads.append((model, cache_dir))
        return M()

    s = RerankerSettings(kind="cross_encoder", hosted=False, model="m", cache_dir="/tmp/c")
    r = CrossEncoderReranker("c", s, lib, notes.append)
    assert not loads and not notes
    assert r.score("q", ["a", "bbb"]) == [1.0, 3.0] and r.score("q", ["cc"]) == [2.0]
    assert loads == [("m", "/tmp/c")] and len(notes) == 1 and "download" in notes[0]

    class Bad:
        def rerank(self, query, docs):
            return [float("nan")] * len(docs)

    with pytest.raises(EmbeddingError):
        CrossEncoderReranker("c", s, lambda m, c: Bad(), lambda x: None).score("q", ["a"])

    def boom(m, c):
        raise OSError("offline")

    with pytest.raises(EmbeddingError) as ei:
        CrossEncoderReranker("c", s, boom, lambda x: None).score("q", ["a"])
    assert ei.value.kind == "load_failed"


def test_cross_encoder_without_fastembed_is_a_config_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fastembed", None)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", None)
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
    s = RerankerSettings(kind="cross_encoder", hosted=False, model="m")
    with pytest.raises(EmbeddingError) as ei:
        CrossEncoderReranker("c", s, notice=lambda x: None).score("q", ["a"])
    assert ei.value.kind == "config" and '".[local]"' in str(ei.value)


def test_hosted_and_local_rerankers_share_one_interface():
    from verified_analyst.documents.rerank import build_reranker

    hosted = build_reranker("h", rr(), {"K": KEY}, Rec(body={}))
    local = build_reranker("l", RerankerSettings(kind="cross_encoder", hosted=False, model="m"), {})
    fake = build_reranker("f", RerankerSettings(kind="fake", hosted=False), {})
    for r in (hosted, local, fake):
        assert callable(r.score) and isinstance(r.depth, int) and isinstance(r.hosted, bool)
    assert (hosted.hosted, local.hosted, fake.hosted) == (True, False, False)


def test_fake_reranker_prefers_overlap():
    a, b = FakeReranker().score(
        "refund policy days", ["shipping rates", "refund policy is 30 days"]
    )
    assert b > a


# ------------------------------------------------------------------ local (fastembed) embedder


def local_settings(**kw):
    d = dict(
        kind="local",
        hosted=False,
        model="some/model",
        dims=8,
        batch_size=2,
        normalize=True,
        query_prefix="Q: ",
    )
    return EmbedderSettings(**{**d, **kw})


class FakeLibrary:
    """Stands in for fastembed.TextEmbedding: records what it was asked; plain-list vectors."""

    def __init__(self):
        self.loads, self.seen = [], []

    def __call__(self, model, cache_dir):
        self.loads.append((model, cache_dir))
        outer = self

        class Model:
            def embed(self, texts):
                outer.seen.append(list(texts))
                return (iter([float(len(t)) + i for i in range(8)]) for t in texts)

        return Model()


def test_local_embedder_loads_once_prints_one_notice_and_batches():
    from verified_analyst.documents.embed_http import LocalEmbedder

    lib, notes = FakeLibrary(), []
    e = LocalEmbedder("l", local_settings(cache_dir="/tmp/m"), {}, library=lib, notice=notes.append)
    assert not lib.loads and not notes  # nothing happens at construction
    out = e.embed(["a", "bb", "ccc"], "query")
    e.embed(["d"], "document")
    assert lib.loads == [("some/model", "/tmp/m")] and len(notes) == 1
    assert "download" in notes[0] and "some/model" in notes[0]
    assert lib.seen[0] == ["Q: a", "Q: bb"] and lib.seen[2] == [
        "d"
    ]  # batches of 2, prefix on queries
    assert len(out) == 3 and math.isclose(sum(x * x for x in out[0]), 1.0)


def test_local_embedder_missing_package_is_a_config_error(monkeypatch):
    import sys

    from verified_analyst.documents.embed_http import LocalEmbedder

    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` fail
    with pytest.raises(EmbeddingError) as ei:
        LocalEmbedder("l", local_settings(), {}, notice=lambda s: None).embed(["x"], "document")
    assert ei.value.kind == "config" and '".[local]"' in str(ei.value)


def test_local_embedder_load_and_run_failures_are_mapped():
    from verified_analyst.documents.embed_http import LocalEmbedder

    def boom(model, cache_dir):
        raise OSError("no route to host " + "x" * 500)

    with pytest.raises(EmbeddingError) as ei:
        LocalEmbedder("l", local_settings(), {}, library=boom, notice=lambda s: None).embed(
            ["x"], "document"
        )
    assert ei.value.kind == "load_failed" and len(str(ei.value)) < 300

    class Bad:
        def __call__(self, m, c):
            return type("M", (), {"embed": lambda self, t: [[1.0] * 3 for _ in t]})()

    with pytest.raises(EmbeddingError):  # wrong dimensions
        LocalEmbedder("l", local_settings(), {}, library=Bad(), notice=lambda s: None).embed(
            ["x"], "document"
        )


def test_local_settings_rules_and_build():
    with pytest.raises(ValueError):
        local_settings(hosted=True)
    with pytest.raises(ValueError):
        local_settings(dims=None)
    assert type(build_embedder("l", local_settings(), {})).__name__ == "LocalEmbedder"
    assert build_embedder("l", local_settings(), {}).hosted is False
