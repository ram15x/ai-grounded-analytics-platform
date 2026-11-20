"""A scripted provider for tests and offline demos. It never touches the network."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from typing import Any

from .providers import Provider
from .types import LLMRequest, LLMResponse, ProviderError

Scripted = str | dict | list | ProviderError | Callable[[LLMRequest], Any]


class FakeProvider(Provider):
    """Returns scripted responses in order, per task (or from one shared queue).

    A script item is a string (returned as is), a dict or list (JSON-encoded), a ProviderError
    (raised), or a callable taking the request and returning one of those.
    """

    def __init__(
        self,
        script: dict[str, list[Scripted]] | list[Scripted],
        *,
        name: str = "fake",
        hosted: bool = False,
        latency_ms: float = 5.0,
    ):
        self.name = name
        self.hosted = hosted
        self.latency_ms = latency_ms
        if isinstance(script, dict):
            self._queues: dict[str, deque] = {k: deque(v) for k, v in script.items()}
            self._shared: deque | None = None
        else:
            self._queues = {}
            self._shared = deque(script)
        self.requests: list[LLMRequest] = []
        self.models_used: list[str] = []

    def complete(self, req: LLMRequest, model: str, timeout: float) -> LLMResponse:
        self.requests.append(req)
        self.models_used.append(model)
        queue = self._shared if self._shared is not None else self._queues.get(req.task)
        if not queue:
            raise AssertionError(f"FakeProvider: no scripted response left for task {req.task!r}")
        item = queue.popleft()
        if callable(item):
            item = item(req)
        if isinstance(item, ProviderError):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        # a stand-in token count (characters / 4); real adapters report the provider's own
        p = max(1, req.prompt_chars() // 4)
        c = max(1, len(text) // 4)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=p,
            completion_tokens=c,
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )

    @property
    def remaining(self) -> int:
        if self._shared is not None:
            return len(self._shared)
        return sum(len(q) for q in self._queues.values())
