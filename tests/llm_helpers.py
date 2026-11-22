"""Shared helpers for the model-router tests. Nothing here touches the network."""

from __future__ import annotations

import json

from grounded_analytics.llm.settings import RouterSettings
from grounded_analytics.llm.transport import HttpResponse, TransportError, TransportTimeout
from grounded_analytics.llm.types import LLMRequest

KEYS = {
    "GROQ_API_KEY": "gsk_test_key_groq_0123456789abcdef",
    "GEMINI_API_KEY": "AIzaTestKeyGemini0123456789abcdefgh",
    "OPENROUTER_API_KEY": "sk-or-test-key-openrouter-0123456789",
}


class MockTransport:
    """Scripted HTTP. Items: HttpResponse, an exception instance (raised), or a callable."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(
            {"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout}
        )
        if not self.script:
            raise AssertionError("MockTransport: no scripted response left")
        item = self.script.pop(0)
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item


def ok_openai(text='{"ok": true}', p=10, c=5):
    body = {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p, "completion_tokens": c},
    }
    return HttpResponse(200, json.dumps(body).encode())


def ok_gemini(text='{"ok": true}', p=10, c=5):
    body = {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": p, "candidatesTokenCount": c},
    }
    return HttpResponse(200, json.dumps(body).encode())


def status(code, headers=None, body=b""):
    return HttpResponse(code, body, {k.lower(): v for k, v in (headers or {}).items()})


TIMEOUT = TransportTimeout("timed out")
CONN_ERROR = TransportError("ConnectionResetError")


def req(task="plan", text="hello", datasets=(), **kw) -> LLMRequest:
    return LLMRequest(
        task=task,
        messages=(("system", "be brief"), ("user", text)),
        datasets=datasets,
        prompt_id=kw.pop("prompt_id", "plan.v0"),
        prompt_hash=kw.pop("prompt_hash", "h"),
        **kw,
    )


def settings(**over) -> RouterSettings:
    base = {
        "providers": {
            "groq": {"kind": "groq", "hosted": True, "env_key": "GROQ_API_KEY",
                     "models": {"default": "m-groq"}},
            "gemini": {"kind": "gemini", "hosted": True, "env_key": "GEMINI_API_KEY",
                       "models": {"default": "m-gem"}},
            "openrouter": {"kind": "openrouter", "hosted": True, "env_key": "OPENROUTER_API_KEY",
                           "models": {"default": "m-or"}},
            "local": {"kind": "fake", "hosted": False, "models": {"default": "m-local"}},
        },
        "tasks": {t: ["groq", "gemini", "openrouter", "local"]
                  for t in ("route", "plan", "compose", "repair")},
        "datasets": {"analytics.sales_lines": "public", "hr.salaries": "internal"},
    }  # fmt: skip
    base.update(over)
    return RouterSettings.model_validate(base)
