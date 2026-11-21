import pytest

from verified_analyst.llm.replay import ReplayProvider, request_key
from verified_analyst.llm.types import LLMRequest, ProviderError


def _req(task="plan", msgs=(("user", "hi"),)):
    return LLMRequest(task=task, messages=list(msgs))


def test_hit_returns_the_recorded_text_and_miss_is_an_error_not_an_invention():
    r = _req()
    p = ReplayProvider({request_key(r.task, r.messages): "recorded"})
    assert p.complete(r, "m", 1.0).text == "recorded"
    with pytest.raises(ProviderError):
        p.complete(_req(task="compose"), "m", 1.0)
    assert p.misses == 1


def test_shipped_recordings_load_and_hold_no_key_like_text():
    from pathlib import Path

    f = Path(__file__).resolve().parents[1] / "demo" / "recordings.jsonl"
    p = ReplayProvider.from_file(f)
    assert len(p.recordings) == 43
    blob = f.read_text(encoding="utf-8")
    assert "gsk_" not in blob and "AIza" not in blob
