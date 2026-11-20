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
