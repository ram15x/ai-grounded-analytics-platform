"""Provider adapters. Each reads its key from an environment variable at call time, puts it only
in a request header, and never includes it in an error message.

Everything about request and response shapes below is UNVERIFIED against live services: no
adapter has made a real call. See docs/PHASE_1C1_ASSUMPTIONS.md."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from ..config import ConfigError
from .transport import HttpResponse, Transport, TransportError, TransportTimeout, urllib_transport
from .types import (
    AUTH,
    BAD_REQUEST,
    BLOCKED,
    INTERNAL,
    MALFORMED,
    NETWORK,
    NO_KEY,
    RATE_LIMIT,
    SERVER,
    TIMEOUT,
    LLMRequest,
    LLMResponse,
    ProviderError,
    SensitivityRefusal,
)

PLACEHOLDER_PREFIX = "REPLACE"


class Provider(ABC):
    name: str
    hosted: bool

    @abstractmethod
    def complete(self, req: LLMRequest, model: str, timeout: float) -> LLMResponse: ...


def parse_retry_after(value: str | None) -> float | None:
    """Seconds only. HTTP-date form is not supported (treated as absent)."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if 0 <= seconds < 1e7 else None


class HostedProvider(Provider):
    hosted = True
    default_endpoint = ""
    accepts_reasoning_effort = False  # only adapters whose parameter name is known set this

    def __init__(
        self,
        name: str,
        env_key: str,
        *,
        endpoint: str | None = None,
        transport: Transport = urllib_transport,
        env: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.name = name
        self.env_key = env_key
        self.endpoint = endpoint or self.default_endpoint
        self.transport = transport
        self.env = os.environ if env is None else env
        self.clock = clock

    # -- subclass hooks --
    def _url(self, model: str) -> str:
        return self.endpoint

    def _headers(self, key: str) -> dict[str, str]:
        raise NotImplementedError

    def _body(self, req: LLMRequest, model: str) -> dict[str, Any]:
        raise NotImplementedError

    def _extract(self, data: Any) -> tuple[str, int, int, str, bool, int]:
        """Return (text, prompt, completion, finish_reason, usage_missing, reasoning_tokens)."""
        raise NotImplementedError

    # -- shared --
    def _key(self) -> str:
        key = self.env.get(self.env_key)
        if not key:
            raise ProviderError(NO_KEY, f"environment variable {self.env_key} is not set")
        return key

    def complete(self, req: LLMRequest, model: str, timeout: float) -> LLMResponse:
        # second line of defence behind the router's check: a hosted adapter refuses anything
        # the router has not stamped public
        if req.sensitivity == INTERNAL:
            raise SensitivityRefusal(f"{self.name}: refusing a request that is not tagged public")
        if not model or model.startswith(PLACEHOLDER_PREFIX):
            raise ConfigError(f"{self.name}: model id for task {req.task!r} is not configured")
        if req.reasoning_effort and not self.accepts_reasoning_effort:
            raise ConfigError(f"{self.name}: reasoning_effort is not wired for this adapter")
        key = self._key()
        body = json.dumps(self._body(req, model)).encode("utf-8")
        start = self.clock()
        try:
            resp = self.transport(self._url(model), self._headers(key), body, timeout)
        except TransportTimeout as exc:
            raise ProviderError(TIMEOUT, "request timed out") from exc
        except TransportError as exc:
            raise ProviderError(NETWORK, str(exc)) from exc
        latency_ms = (self.clock() - start) * 1000.0
        return self._to_response(resp, model, latency_ms, key)

    def _to_response(
        self, resp: HttpResponse, model: str, latency_ms: float, key: str
    ) -> LLMResponse:
        status = resp.status
        if status == 429:
            raise ProviderError(
                RATE_LIMIT,
                "HTTP 429",
                status=429,
                retry_after=parse_retry_after(resp.header("retry-after")),
            )
        if status in (401, 403):
            raise ProviderError(AUTH, f"HTTP {status}", status=status)
        if status == 408 or status >= 500:
            raise ProviderError(SERVER, f"HTTP {status}", status=status)
        if status != 200:
            raise ProviderError(BAD_REQUEST, f"HTTP {status}", status=status)
        try:
            data = json.loads(resp.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProviderError(MALFORMED, "response body is not JSON") from exc
        try:
            text, p_tok, c_tok, finish, usage_missing, r_tok = self._extract(data)
        except ProviderError:
            raise
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(MALFORMED, "response JSON has an unexpected shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderError(MALFORMED, "empty completion")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=latency_ms,
            finish_reason=finish,
            tokens_estimated=usage_missing,
            reasoning_tokens=r_tok,
        )


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class _OpenAICompatible(HostedProvider):
    def _headers(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _body(self, req: LLMRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": r, "content": c} for r, c in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.json_mode:
            body["response_format"] = {"type": "json_object"}
        if req.reasoning_effort:  # UNVERIFIED: sent only when configured
            body["reasoning_effort"] = req.reasoning_effort
        return body

    def _extract(self, data: Any) -> tuple[str, int, int, str, bool, int]:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage") or {}
        p, c = _int(usage.get("prompt_tokens")), _int(usage.get("completion_tokens"))
        details = usage.get("completion_tokens_details") or {}  # UNVERIFIED field name
        r = _int(details.get("reasoning_tokens") if isinstance(details, dict) else None) or 0
        fin = str(choice.get("finish_reason") or "")
        return text, p or 0, c or 0, fin, p is None or c is None, r


class GroqProvider(_OpenAICompatible):
    accepts_reasoning_effort = True  # UNVERIFIED: the parameter name and values are a guess
    default_endpoint = "https://api.groq.com/openai/v1/chat/completions"  # UNVERIFIED


class OpenRouterProvider(_OpenAICompatible):
    default_endpoint = "https://openrouter.ai/api/v1/chat/completions"  # UNVERIFIED


class GeminiProvider(HostedProvider):
    default_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"  # UNVERIFIED

    def _url(self, model: str) -> str:
        return f"{self.endpoint}/{model}:generateContent"

    def _headers(self, key: str) -> dict[str, str]:
        return {"x-goog-api-key": key, "Content-Type": "application/json"}

    def _body(self, req: LLMRequest, model: str) -> dict[str, Any]:
        system = [c for r, c in req.messages if r == "system"]
        turns = [
            {"role": "model" if r == "assistant" else "user", "parts": [{"text": c}]}
            for r, c in req.messages
            if r != "system"
        ]
        cfg: dict[str, Any] = {"temperature": req.temperature, "maxOutputTokens": req.max_tokens}
        if req.json_mode:
            cfg["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": turns, "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        return body

    def _extract(self, data: Any) -> tuple[str, int, int, str, bool, int]:
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise ProviderError(BLOCKED, f"prompt blocked ({block})")
        cand = data["candidates"][0]
        parts = cand["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        p, c = _int(usage.get("promptTokenCount")), _int(usage.get("candidatesTokenCount"))
        r = _int(usage.get("thoughtsTokenCount")) or 0  # UNVERIFIED field name
        fin = str(cand.get("finishReason") or "")
        return text, p or 0, c or 0, fin, p is None or c is None, r


ADAPTERS: dict[str, type[HostedProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def build_one(settings, name: str, **adapter_kwargs: Any) -> HostedProvider:
    """Instantiate one hosted adapter from the config."""
    p = settings.providers[name]
    return ADAPTERS[p.kind](name, p.env_key or "", endpoint=p.endpoint, **adapter_kwargs)
