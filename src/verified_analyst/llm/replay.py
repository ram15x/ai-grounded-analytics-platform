"""Replay of recorded model replies, so the pipeline can run with no key and no network.

A reply is found by its task and messages. A request with no recording raises ProviderError
("no_recording"); it is never answered with an invented reply."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .providers import Provider
from .types import LLMResponse, ProviderError


def request_key(task: str, messages) -> str:
    return hashlib.sha256(json.dumps([task, [list(m) for m in messages]]).encode()).hexdigest()


class ReplayProvider(Provider):
    name = "groq"
    hosted = True

    def __init__(self, recordings: dict[str, str]):
        self.recordings = recordings
        self.misses = 0

    @classmethod
    def from_file(cls, path: Path) -> ReplayProvider:
        rec = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            rec[r["key"]] = r["text"]
        return cls(rec)

    def complete(self, req, model, timeout):
        key = request_key(req.task, req.messages)
        if key not in self.recordings:
            self.misses += 1
            raise ProviderError("no_recording")
        return LLMResponse(
            text=self.recordings[key], provider=self.name, model=model, prompt_tokens=0,
            completion_tokens=0, latency_ms=0.0, finish_reason="stop",
        )  # fmt: skip
