"""The public tryout: menu allow-list, per-visitor allowance, refund on failure, and the visitor's
key never coming back out. The model is faked; these test the rules around it."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from grounded_analytics import tryout as T
from grounded_analytics.web import app

KEY = "gsk_SECRETSECRETSECRET1234"


@pytest.fixture
def srv():
    cfg = T.load_config()
    calls: list[tuple] = []

    def byok(q, provider, model, key, tier):
        calls.append((q, provider, model, key, tier))
        if q.startswith("FAIL"):
            return {"outcome": "abstain", "reason": "provider_error", "rendered": "",
                    "error": f"bad key {key}"}  # fmt: skip
        return {"outcome": "verified", "reason": "", "rendered": "ok"}

    saved = {k: getattr(app._Handler, k) for k in ("tryout", "allowance", "byok_fn", "models_fn")}
    app._Handler.tryout = T.public_view(cfg)
    app._Handler.allowance = T.Allowance(cfg)
    app._Handler.byok_fn = staticmethod(byok)
    app._Handler.models_fn = None
    s = ThreadingHTTPServer(("127.0.0.1", 0), app._Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    s.cfg, s.calls = cfg, calls
    yield s
    s.shutdown()
    for k, v in saved.items():
        setattr(app._Handler, k, v)


def post(s, body, key=KEY, cookie=None):
    h = {"X-Provider-Key": key} if key is not None else {}
    if cookie:
        h["Cookie"] = cookie
    req = urllib.request.Request(
        f"http://127.0.0.1:{s.server_port}/api/ask", json.dumps(body).encode(), h, method="POST"
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r), r.headers.get("Set-Cookie")
    except urllib.error.HTTPError as e:
        return e.code, json.load(e), e.headers.get("Set-Cookie")


def test_config_is_twelve_questions_three_per_tier():
    cfg = T.load_config()
    assert [len(cfg["questions"][t]) for t in T.MENU_TIERS] == [3, 3, 3]
    assert cfg["own_slots"] == 3
    assert all(cfg["per_visitor"][t] == 1 for t in T.TIERS)


def test_visitor_gets_one_per_tier_then_is_stopped(srv):
    easy = srv.cfg["questions"]["easy"]
    code, out, cookie = post(srv, {"question": easy[0], "tier": "easy", "provider": "groq"})
    assert code == 200 and out["remaining"]["easy"] == 0 and out["remaining"]["hard"] == 1
    me = cookie.split(";")[0]
    code, out, _ = post(srv, {"question": easy[1], "tier": "easy", "provider": "groq"}, cookie=me)
    assert code == 429 and len(srv.calls) == 1
    hard = srv.cfg["questions"]["hard"][0]
    assert post(srv, {"question": hard, "tier": "hard", "provider": "groq"}, cookie=me)[0] == 200
    own = {"question": "anything I like", "tier": "own", "provider": "groq"}
    assert post(srv, own, cookie=me)[0] == 200
    assert post(srv, own, cookie=me)[0] == 429


def test_another_visitor_has_their_own_allowance(srv):
    q = {"question": srv.cfg["questions"]["easy"][0], "tier": "easy", "provider": "groq"}
    assert post(srv, q)[0] == 200
    assert post(srv, q)[0] == 200  # no cookie sent: a new visitor


def test_menu_tiers_refuse_text_that_is_not_on_the_menu(srv):
    code, out, _ = post(srv, {"question": "drop table", "tier": "easy", "provider": "groq"})
    assert code == 400 and not srv.calls
    wrong_tier = srv.cfg["questions"]["hard"][0]
    assert post(srv, {"question": wrong_tier, "tier": "easy", "provider": "groq"})[0] == 400


def test_a_key_is_required_and_the_provider_must_be_known(srv):
    q = {"question": srv.cfg["questions"]["easy"][0], "tier": "easy", "provider": "groq"}
    assert post(srv, q, key=None)[0] == 200  # menu questions may be served from stored replies
    own = {"question": "mine", "tier": "own", "provider": "groq"}
    assert post(srv, own, key=None)[0] == 400  # an own question always needs the visitor's key
    assert post(srv, {**q, "provider": "nope"})[0] == 400
    assert len(srv.calls) == 1  # only the key-less menu question above got through


def test_own_question_length_is_capped(srv):
    long = "x" * (srv.cfg["own_max_chars"] + 1)
    assert post(srv, {"question": long, "tier": "own", "provider": "groq"})[0] == 400


def test_a_failed_call_gives_the_slot_back(srv):
    code, out, cookie = post(srv, {"question": "FAIL me", "tier": "own", "provider": "groq"})
    assert code == 200 and out["remaining"]["own"] == 1


def test_default_groq_model_comes_from_config_and_key_is_passed_through(srv):
    q = {"question": srv.cfg["questions"]["easy"][0], "tier": "easy", "provider": "groq"}
    post(srv, q)
    assert (
        srv.calls[0][4] == "easy"
        and srv.calls[0][2] == srv.cfg["providers"]["groq"]["model"]
        and srv.calls[0][3] == KEY
    )


def test_tryout_config_endpoint_never_contains_a_key(srv):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/api/tryout") as r:
        body = r.read().decode()
    assert (
        "remaining" in body
        and KEY not in body
        and "gsk_" not in body.replace("Starts with gsk_", "")
    )


def test_bad_config_is_refused(tmp_path):
    bad = tmp_path / "t.yaml"
    bad.write_text("per_visitor: {easy: 1}\n", "utf-8")
    with pytest.raises(T.TryoutError):
        T.load_config(bad)


def test_byok_ask_scrubs_the_key_from_errors_and_stores_nothing(monkeypatch, tmp_path):
    from grounded_analytics import cli
    from grounded_analytics.llm.types import LLMError

    class Layer:
        class source:  # noqa: N801
            relation = "analytics.sales_lines"

    monkeypatch.setattr(cli, "load_semantic_layer", lambda: Layer)
    monkeypatch.setattr(cli, "readonly_database_url", lambda: "x")
    monkeypatch.setattr(cli, "catalogue", lambda *a: {})
    monkeypatch.setattr("grounded_analytics.llm.answer.make_retriever", lambda *a, **k: None)

    def boom(*a, **k):
        raise LLMError(f"provider said no for {KEY}")

    monkeypatch.setattr(cli, "_ask_one", boom)
    monkeypatch.chdir(tmp_path)
    out = cli._byok_ask("q", "groq", "openai/gpt-oss-20b", KEY)
    assert out["reason"] == "provider_error" and KEY not in json.dumps(out)
    assert list(tmp_path.iterdir()) == []  # no trace or key file written


def test_the_page_never_writes_the_key_to_browser_storage_or_a_url():
    html = app.PAGE.read_text("utf-8")
    for banned in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert banned not in html
    assert "X-Provider-Key" in html and "?key=" not in html


def test_cache_forgets_replies_from_a_run_that_did_not_verify(tmp_path):
    from grounded_analytics.llm.cassette import Cassette, CassetteContext

    c = Cassette(tmp_path, "resume", CassetteContext("l", "s", "w"), env={})
    c.entries["old"] = {"key": "old"}
    c.flush()
    before = set(c.entries)
    c.entries["new"] = {"key": "new"}
    assert c.discard_since(before) == 1
    again = Cassette(tmp_path, "resume", CassetteContext("l", "s", "w"), env={})
    assert set(again.entries) == {"old"}


def test_own_questions_are_never_cached(monkeypatch, tmp_path):
    from grounded_analytics import cli

    class Layer:
        class source:  # noqa: N801
            relation = "analytics.sales_lines"

    monkeypatch.setattr(cli, "load_semantic_layer", lambda: Layer)
    monkeypatch.setattr(cli, "readonly_database_url", lambda: "x")
    monkeypatch.setattr(cli, "catalogue", lambda *a: {})
    monkeypatch.setattr("grounded_analytics.llm.answer.make_retriever", lambda *a, **k: None)

    class Res:
        outcome = "verified"

        def as_dict(self):
            return {"outcome": "verified", "accounting": {"tokens_used": 5}}

    def fake_ask(question, qid, setup):
        return Res()

    monkeypatch.setattr(cli, "_ask_one", fake_ask)
    out = cli._byok_ask(
        "my private question", "groq", "openai/gpt-oss-20b", "gsk_x", "own", tmp_path
    )
    assert out["stored_replies_only"] is False
    assert list(tmp_path.iterdir()) == []  # nothing from an own question reaches the cache dir


def test_models_endpoint_returns_headroom_headers_and_scrubs_the_key(monkeypatch):
    from grounded_analytics import cli

    def fake_list(settings, name, env, *a, **k):
        echo = env[settings.providers[name].env_key]
        return {"provider": name, "models": [{"id": "m"}], "note": f"echo {echo}",
                "rate_headers": {"x-ratelimit-remaining-tokens": "7000"}}  # fmt: skip

    monkeypatch.setattr("grounded_analytics.llm.live.list_provider_models", fake_list)
    out = cli._byok_models("groq", KEY)
    assert out["rate_headers"]["x-ratelimit-remaining-tokens"] == "7000"
    assert KEY not in json.dumps(out)
