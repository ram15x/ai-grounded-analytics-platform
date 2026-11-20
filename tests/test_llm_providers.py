"""Adapter tests with mocked HTTP only: request shape, key handling, and every error path."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from llm_helpers import (
    CONN_ERROR,
    KEYS,
    TIMEOUT,
    MockTransport,
    ok_gemini,
    ok_openai,
    req,
    status,
)
from verified_analyst.config import ConfigError
from verified_analyst.llm.providers import (
    ADAPTERS,
    GeminiProvider,
    GroqProvider,
    OpenRouterProvider,
    parse_retry_after,
)
from verified_analyst.llm.transport import HttpResponse, TransportError, urllib_transport
from verified_analyst.llm.types import (
    AUTH,
    BAD_REQUEST,
    BLOCKED,
    MALFORMED,
    NETWORK,
    NO_KEY,
    RATE_LIMIT,
    SERVER,
    ProviderError,
    SensitivityRefusal,
)
from verified_analyst.llm.types import (
    TIMEOUT as K_TIMEOUT,
)

CASES = [
    ("groq", GroqProvider, "GROQ_API_KEY", ok_openai),
    ("openrouter", OpenRouterProvider, "OPENROUTER_API_KEY", ok_openai),
    ("gemini", GeminiProvider, "GEMINI_API_KEY", ok_gemini),
]
IDS = [c[0] for c in CASES]


def make(cls, env_key, script):
    t = MockTransport(script)
    clock = iter(x / 1000 for x in range(0, 100000, 25))  # 25 ms per tick
    return cls(cls.__name__, env_key, transport=t, env=KEYS, clock=lambda: next(clock)), t


def public(**kw):
    return dataclasses.replace(req(**kw), sensitivity="public")


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_success_parses_text_tokens_latency(name, cls, env_key, ok):
    p, t = make(cls, env_key, [ok('{"a": 1}', 12, 7)])
    r = p.complete(public(), "model-x", 9.0)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.model) == ('{"a": 1}', 12, 7, "model-x")
    assert r.latency_ms == pytest.approx(25.0)
    assert not r.tokens_estimated
    assert t.calls[0]["timeout"] == 9.0


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_key_only_in_a_header_never_in_body_or_url(name, cls, env_key, ok):
    p, t = make(cls, env_key, [ok()])
    p.complete(public(), "model-x", 5)
    call = t.calls[0]
    key = KEYS[env_key]
    assert key in json.dumps(call["headers"])
    assert key not in json.dumps(call["body"])
    assert key not in call["url"]


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_request_is_deterministic_json_mode(name, cls, env_key, ok):
    p, t = make(cls, env_key, [ok()])
    p.complete(public(json_mode=True, max_tokens=321), "model-x", 5)
    body = t.calls[0]["body"]
    if cls is GeminiProvider:
        assert body["generationConfig"] == {
            "temperature": 0.0, "maxOutputTokens": 321, "responseMimeType": "application/json",
        }  # fmt: skip
        assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
        assert t.calls[0]["url"].endswith("/model-x:generateContent")
    else:
        assert body["temperature"] == 0.0 and body["max_tokens"] == 321
        assert body["response_format"] == {"type": "json_object"}
        assert body["model"] == "model-x"
    p2, t2 = make(cls, env_key, [ok()])
    p2.complete(public(json_mode=False), "model-x", 5)
    assert "response_format" not in t2.calls[0]["body"]
    assert "responseMimeType" not in t2.calls[0]["body"].get("generationConfig", {})


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
@pytest.mark.parametrize(
    ("resp", "kind", "retry_after"),
    [
        (status(429, {"Retry-After": "7"}), RATE_LIMIT, 7.0),
        (status(429, {"retry-after": "0.5"}), RATE_LIMIT, 0.5),
        (status(429), RATE_LIMIT, None),
        (status(429, {"Retry-After": "soon"}), RATE_LIMIT, None),
        (status(429, {"Retry-After": "-3"}), RATE_LIMIT, None),
        (status(500), SERVER, None),
        (status(503, {"Retry-After": "2"}), SERVER, None),
        (status(408), SERVER, None),
        (status(401), AUTH, None),
        (status(403), AUTH, None),
        (status(400), BAD_REQUEST, None),
        (status(404), BAD_REQUEST, None),
    ],
)
def test_http_status_mapping(name, cls, env_key, ok, resp, kind, retry_after):
    p, _ = make(cls, env_key, [resp])
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == kind
    assert ei.value.retry_after == retry_after


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
@pytest.mark.parametrize(
    "body",
    [
        b"not json at all",
        b"",
        b"\xff\xfe\x00",
        b"[]",
        b"{}",
        b'{"choices": []}',
        b'{"candidates": []}',
        b'{"choices": [{"message": {}}]}',
        b'{"choices": [{"message": {"content": null}}]}',
        b'{"choices": [{"message": {"content": "   "}}]}',
        b'{"candidates": [{"content": {"parts": []}}]}',
        b'{"candidates": [{"content": {}}]}',
        b'{"choices": 5}',
    ],
)
def test_malformed_200_bodies(name, cls, env_key, ok, body):
    p, _ = make(cls, env_key, [HttpResponse(200, body)])
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == MALFORMED


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_timeout_and_connection_error(name, cls, env_key, ok):
    p, _ = make(cls, env_key, [TIMEOUT])
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == K_TIMEOUT
    p, _ = make(cls, env_key, [CONN_ERROR])
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == NETWORK


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_missing_key_and_placeholder_model(name, cls, env_key, ok):
    t = MockTransport([ok()])
    p = cls(name, env_key, transport=t, env={})
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == NO_KEY and t.calls == []
    p = cls(name, env_key, transport=t, env=KEYS)
    with pytest.raises(ConfigError):
        p.complete(public(), "REPLACE-WITH-CURRENT-ID", 5)
    with pytest.raises(ConfigError):
        p.complete(public(), "", 5)
    assert t.calls == []


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_errors_never_contain_the_key(name, cls, env_key, ok):
    key = KEYS[env_key]
    bodies = [
        status(500, body=key.encode()),
        status(400, body=f"bad key {key}".encode()),
        HttpResponse(200, f"{key} not json".encode()),
    ]
    for resp in bodies:
        p, _ = make(cls, env_key, [resp])
        with pytest.raises(ProviderError) as ei:
            p.complete(public(), "m", 5)
        assert key not in str(ei.value) and key not in repr(ei.value.__dict__)


@pytest.mark.parametrize(("name", "cls", "env_key", "ok"), CASES, ids=IDS)
def test_hosted_adapter_refuses_untagged_or_internal_requests(name, cls, env_key, ok):
    t = MockTransport([ok(), ok()])
    p = cls(name, env_key, transport=t, env=KEYS)
    with pytest.raises(SensitivityRefusal):
        p.complete(req(), "m", 5)  # default sensitivity is internal (fail closed)
    with pytest.raises(SensitivityRefusal):
        p.complete(dataclasses.replace(req(), sensitivity="internal"), "m", 5)
    assert t.calls == []


def test_gemini_prompt_block_is_reported():
    body = json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode()
    p, _ = make(GeminiProvider, "GEMINI_API_KEY", [HttpResponse(200, body)])
    with pytest.raises(ProviderError) as ei:
        p.complete(public(), "m", 5)
    assert ei.value.kind == BLOCKED


def test_gemini_multiple_parts_are_joined_and_assistant_maps_to_model():
    body = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": '{"a":'}, {"text": " 1}"}]}}]}
    ).encode()
    p, t = make(GeminiProvider, "GEMINI_API_KEY", [HttpResponse(200, body)])
    r = dataclasses.replace(public(), messages=(("user", "q"), ("assistant", "a"), ("user", "q2")))
    out = p.complete(r, "m", 5)
    assert out.text == '{"a": 1}'
    assert [c["role"] for c in t.calls[0]["body"]["contents"]] == ["user", "model", "user"]
    assert out.tokens_estimated  # no usageMetadata in this body


@pytest.mark.parametrize(
    "usage", [{}, {"prompt_tokens": 5}, {"prompt_tokens": "x", "completion_tokens": 2}, None]
)
def test_missing_or_bad_usage_is_flagged_not_trusted(usage):
    body = {"choices": [{"message": {"content": "{}"}}]}
    if usage is not None:
        body["usage"] = usage
    p, _ = make(GroqProvider, "GROQ_API_KEY", [HttpResponse(200, json.dumps(body).encode())])
    r = p.complete(public(), "m", 5)
    assert r.tokens_estimated


def test_retry_after_parsing():
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after(" 1.5 ") == 1.5
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after(None) is None
    assert parse_retry_after("nan") is None
    assert parse_retry_after("inf") is None


def test_default_transport_refuses_plain_http_and_never_connects():
    with pytest.raises(TransportError):
        urllib_transport("http://example.com/x", {}, b"{}", 1)


def test_adapter_registry_matches_config_kinds():
    assert set(ADAPTERS) == {"groq", "gemini", "openrouter"}


def test_the_test_suite_blocks_outbound_network_connections():
    import socket

    # a literal address, so no DNS lookup happens before the guard is reached
    with socket.socket() as sock, pytest.raises(AssertionError, match="must not use the network"):
        sock.connect(("203.0.113.7", 443))


# ---- User-Agent (urllib layer, with urlopen mocked: no network) ----


class _FakeResp:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"{}"


def _capture(monkeypatch):
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def test_post_and_get_send_an_explicit_user_agent_not_the_python_default(monkeypatch):
    from verified_analyst import __version__
    from verified_analyst.llm.transport import urllib_get

    seen = _capture(monkeypatch)
    urllib_transport("https://example.test/x", {"Authorization": "Bearer k"}, b"{}", 5)
    urllib_get("https://example.test/y", {"x-goog-api-key": "k"}, 5)
    assert [r.get_method() for r in seen] == ["POST", "GET"]
    for r in seen:
        ua = r.get_header("User-agent")
        assert ua == f"verified-analyst/{__version__}"
        assert "python" not in ua.lower()


def test_an_explicit_user_agent_from_the_caller_is_kept(monkeypatch):
    seen = _capture(monkeypatch)
    urllib_transport("https://example.test/x", {"User-Agent": "custom/1"}, b"{}", 5)
    assert seen[0].get_header("User-agent") == "custom/1"


def test_user_agent_contains_no_key_or_host_data(monkeypatch):
    from verified_analyst.llm.transport import user_agent

    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    ua = user_agent()
    assert all(v not in ua for v in KEYS.values())
    assert re.fullmatch(r"verified-analyst/[0-9][0-9A-Za-z.+-]*", ua)


def test_adapter_requests_reach_the_transport_with_keys_only_in_auth_headers(monkeypatch):
    seen = _capture(monkeypatch)
    p = GroqProvider("groq", "GROQ_API_KEY", env=KEYS)  # default urllib transport, urlopen mocked
    body = ok_openai().body
    monkeypatch.setattr(_FakeResp, "read", lambda self: body)
    p.complete(public(), "m", 5)
    req_ = seen[0]
    assert req_.get_header("User-agent").startswith("verified-analyst/")
    assert KEYS["GROQ_API_KEY"] not in req_.get_header("User-agent")
    assert KEYS["GROQ_API_KEY"] in req_.get_header("Authorization")
