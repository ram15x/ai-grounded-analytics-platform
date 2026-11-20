from __future__ import annotations

import pytest

from verified_analyst.documents.settings import (
    EmbedderSettings,
    RerankerSettings,
    load_retrieval_settings,
)
from verified_analyst.llm.transport import TransportError, loopback_transport


def test_shipped_config_loads_and_hosted_flags_are_consistent():
    s = load_retrieval_settings()
    assert {"fake_256", "gemini_embedding_2_768"} <= set(s.embedders)
    assert s.probe.max_requests <= 150
    for e in s.embedders.values():
        if e.kind == "ollama":
            assert not e.hosted
        if e.kind in ("gemini", "openai_compat"):
            assert e.hosted and e.env_key
    for r in s.rerankers.values():
        assert r.kind != "rerank_api" or (r.hosted and r.url and r.env_key)


def test_validation_rules():
    with pytest.raises(ValueError):
        EmbedderSettings(kind="gemini", hosted=True, model="m")  # no env_key
    with pytest.raises(ValueError):
        EmbedderSettings(kind="ollama", hosted=True, model="m", base_url="http://127.0.0.1")
    with pytest.raises(ValueError):
        EmbedderSettings(kind="openai_compat", hosted=True, model="m", env_key="K")  # no base_url
    with pytest.raises(ValueError):
        RerankerSettings(kind="cross_encoder", hosted=True, model="m")
    with pytest.raises(ValueError):
        RerankerSettings(kind="rerank_api", hosted=True, model="m")  # no url/key
    with pytest.raises(ValueError):
        EmbedderSettings(kind="fake", hosted=False, surprise=1)  # unknown field


def test_missing_file_is_a_config_error(tmp_path):
    from verified_analyst.config import ConfigError

    with pytest.raises(ConfigError):
        load_retrieval_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "url",
    ["http://example.com/api/embed", "https://127.0.0.1/api/embed", "http://user:pw@127.0.0.1/x"],
)
def test_loopback_transport_refuses_other_targets(url):
    with pytest.raises(TransportError):
        loopback_transport(url, {}, b"{}", 1.0)


def test_local_candidates_are_registered_as_not_hosted_with_dims():
    s = load_retrieval_settings()
    small, base = s.embedders["local_bge_small_en_v15"], s.embedders["local_bge_base_en_v15"]
    assert (small.kind, small.hosted, small.dims, small.model) == (
        "local",
        False,
        384,
        "BAAI/bge-small-en-v1.5",
    )
    assert (base.kind, base.hosted, base.dims) == ("local", False, 768)
    assert not any(e.kind == "ollama" for e in s.embedders.values())


def test_the_local_dependency_is_an_optional_extra_not_a_core_one():
    import tomllib
    from pathlib import Path

    meta = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text("utf-8"))
    core = " ".join(meta["project"]["dependencies"]).lower()
    assert "fastembed" not in core and "sentence-transformers" not in core
    assert any("fastembed" in d for d in meta["project"]["optional-dependencies"]["local"])


def test_local_rerankers_are_registered_as_not_hosted_cross_encoders():
    s = load_retrieval_settings()
    local = {n: r for n, r in s.rerankers.items() if r.kind == "cross_encoder"}
    assert set(local) == {"local_minilm_l6_cross_encoder", "local_bge_reranker_base"}
    assert all(not r.hosted for r in local.values())
