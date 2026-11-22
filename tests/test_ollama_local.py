"""A local model may see internal data and needs no key; every remote adapter still may not."""

from __future__ import annotations

import dataclasses

import pytest

from grounded_analytics.llm.providers import GroqProvider, OllamaProvider
from grounded_analytics.llm.settings import ProviderSettings
from grounded_analytics.llm.types import SensitivityRefusal
from llm_helpers import MockTransport, ok_openai, req


def internal(r):
    return dataclasses.replace(r, sensitivity="internal")


def test_local_model_answers_internal_data_without_a_key():
    t = MockTransport([ok_openai("{}")])
    p = OllamaProvider("ollama", "", transport=t, env={})
    out = p.complete(internal(req()), "llama3.2:3b", 30)
    assert out.text == "{}" and t.calls[0]["url"].startswith("http://localhost:11434")
    assert t.calls[0]["body"]["model"] == "llama3.2:3b"


def test_a_remote_adapter_still_refuses_internal_data():
    p = GroqProvider("groq", "GROQ_API_KEY", transport=MockTransport(), env={"GROQ_API_KEY": "k"})
    with pytest.raises(SensitivityRefusal):
        p.complete(internal(req()), "m", 30)


def test_config_cannot_declare_a_remote_provider_as_local():
    with pytest.raises(ValueError, match="hosted: true"):
        ProviderSettings(kind="groq", hosted=False, env_key="GROQ_API_KEY")
    with pytest.raises(ValueError, match="hosted: false"):
        ProviderSettings(kind="ollama", hosted=True)
    ProviderSettings(kind="ollama", hosted=False)
