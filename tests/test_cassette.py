"""Cassette record/replay, key hashing, secret scan, and `va cassettes check`. No network."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from llm_helpers import KEYS, MockTransport, ok_openai, req, settings
from verified_analyst.cli import main
from verified_analyst.llm.cassette import (
    Cassette,
    CassetteContext,
    check_cassettes,
    layer_hash,
    make_key,
    read_entries,
    scan_for_secrets,
    schema_hash,
)
from verified_analyst.llm.fake import FakeProvider
from verified_analyst.llm.prompts import prompt_hashes
from verified_analyst.llm.providers import GroqProvider
from verified_analyst.llm.router import Router
from verified_analyst.llm.types import (
    CassetteLeak,
    CassetteMiss,
    LLMResponse,
    ProviderError,
)

CTX = CassetteContext("layer1", "schema1", "wh1")
PUB = ("analytics.sales_lines",)


def mk_router(tmp_path, mode, providers=None, ctx=CTX, env=None, now=time.time):
    cfg = settings()
    cas = Cassette(tmp_path / "suite", mode, ctx, env=env or {}, now=now)
    local = FakeProvider([{"a": 1}, {"b": 2}, {"c": 3}], name="local")
    router = Router(cfg, providers if providers is not None else {"local": local}, cassette=cas)
    cfg.tasks["plan"] = ["local"]
    return router, router.new_run("t"), cas, local


# ---- key ----


def test_key_is_stable_and_changes_with_every_component():
    r = req(datasets=PUB)
    base = make_key("groq", "m", r, CTX)
    assert base == make_key("groq", "m", r, CTX)
    variants = {
        "provider": make_key("gemini", "m", r, CTX),
        "model": make_key("groq", "m2", r, CTX),
        "message": make_key("groq", "m", req(text="other", datasets=PUB), CTX),
        "task": make_key("groq", "m", dataclasses.replace(r, task="compose"), CTX),
        "temperature": make_key("groq", "m", dataclasses.replace(r, temperature=0.7), CTX),
        "max_tokens": make_key("groq", "m", dataclasses.replace(r, max_tokens=5), CTX),
        "json_mode": make_key("groq", "m", dataclasses.replace(r, json_mode=False), CTX),
        "effort": make_key("groq", "m", dataclasses.replace(r, reasoning_effort="low"), CTX),
        "prompt_hash": make_key("groq", "m", dataclasses.replace(r, prompt_hash="x"), CTX),
        "layer": make_key("groq", "m", r, dataclasses.replace(CTX, layer_hash="x")),
        "schema": make_key("groq", "m", r, dataclasses.replace(CTX, schema_hash="x")),
        "warehouse": make_key("groq", "m", r, dataclasses.replace(CTX, warehouse_hash="x")),
    }
    assert len({base, *variants.values()}) == len(variants) + 1, variants


def test_key_ignores_fields_that_do_not_change_the_request():
    a = req(datasets=PUB)
    b = dataclasses.replace(a, sensitivity="public", datasets=("something", "else"))
    assert make_key("g", "m", a, CTX) == make_key("g", "m", b, CTX)


def test_real_hash_helpers_are_hex_and_stable():
    assert layer_hash() == layer_hash() and len(layer_hash()) == 64
    assert schema_hash() == schema_hash() and len(schema_hash()) == 64
    assert set(prompt_hashes()) == {"plan.v0", "compose.v1", "repair.v1"}


def test_layer_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_bytes(b"x: 1\ny: 2\n")
    b.write_bytes(b"x: 1\r\ny: 2\r\n")
    assert layer_hash(a) == layer_hash(b)
    b.write_bytes(b"x: 1\r\ny: 3\r\n")
    assert layer_hash(a) != layer_hash(b)


# ---- record and replay ----


def test_record_then_replay_returns_the_same_text_with_no_provider_call(tmp_path):
    router, run, cas, local = mk_router(tmp_path, "record")
    live = router.call(req(datasets=PUB), run)
    assert live.provider == "local" and not live.from_cassette
    # replay with NO providers at all: a hit cannot need one
    router2, run2, cas2, _ = mk_router(tmp_path, "replay", providers={})
    hit = router2.call(req(datasets=PUB), run2)
    assert hit.text == live.text and hit.from_cassette and hit.model == live.model
    assert run2.records[0].outcome == "cassette"
    assert run2.tokens_used == run.tokens_used > 0  # budget behaves the same on replay
    assert cas2.usage()["hit_keys"] == [next(iter(read_entries(tmp_path / "suite")))]


def test_replay_miss_raises_and_never_calls_a_provider(tmp_path):
    t = MockTransport([ok_openai()])
    groq = GroqProvider("groq", "GROQ_API_KEY", transport=t, env=KEYS)
    router, run, cas, _ = mk_router(tmp_path, "replay", providers={"groq": groq})
    router.settings.tasks["plan"] = ["groq"]
    with pytest.raises(CassetteMiss) as ei:
        router.call(req(datasets=PUB), run)
    assert "no network" in str(ei.value) or "never calls the network" in str(ei.value)
    assert t.calls == []
    assert len(cas.usage()["missed"]) == 1


def test_any_changed_input_turns_a_recorded_entry_into_a_miss(tmp_path):
    router, run, _, _ = mk_router(tmp_path, "record")
    router.call(req(datasets=PUB), run)
    for ctx, r in [
        (dataclasses.replace(CTX, layer_hash="new"), req(datasets=PUB)),
        (dataclasses.replace(CTX, schema_hash="new"), req(datasets=PUB)),
        (dataclasses.replace(CTX, warehouse_hash="new"), req(datasets=PUB)),
        (CTX, req(datasets=PUB, prompt_hash="edited prompt")),
        (CTX, req(datasets=PUB, text="a different question")),
    ]:
        router2, run2, _, _ = mk_router(tmp_path, "replay", providers={}, ctx=ctx)
        with pytest.raises(CassetteMiss):
            router2.call(r, run2)


def test_replay_miss_when_the_model_id_changes(tmp_path):
    router, run, _, _ = mk_router(tmp_path, "record")
    router.call(req(datasets=PUB), run)
    router2, run2, _, _ = mk_router(tmp_path, "replay", providers={})
    router2.settings.providers["local"].models["default"] = "a-newer-model"
    with pytest.raises(CassetteMiss):
        router2.call(req(datasets=PUB), run2)


def test_record_through_a_real_adapter_stores_no_header_or_key(tmp_path):
    t = MockTransport([ok_openai('{"x": 1}', 11, 3)])
    groq = GroqProvider("groq", "GROQ_API_KEY", transport=t, env=KEYS)
    router, run, cas, _ = mk_router(tmp_path, "record", providers={"groq": groq}, env=KEYS)
    router.settings.tasks["plan"] = ["groq"]
    router.call(req(datasets=PUB), run)
    raw = (tmp_path / "suite" / "calls.jsonl").read_text()
    for secret in KEYS.values():
        assert secret not in raw
    low = raw.lower()
    for word in ("authorization", "bearer", "x-goog-api-key", "headers", "api_key"):
        assert word not in low
    entry = json.loads(raw.splitlines()[0])
    assert entry["sensitivity"] == "public" and entry["provider"] == "groq"
    assert entry["tokens"] == {"prompt": 11, "completion": 3, "estimated": False, "reasoning": 0}
    assert set(entry["hashes"]) == {"prompt", "layer", "schema", "warehouse"}
    assert entry["request"]["temperature"] == 0.0
    assert entry["recorded_at"].endswith("+00:00")


def test_internal_payloads_are_never_recorded(tmp_path):
    router, run, cas, local = mk_router(tmp_path, "record")
    r = router.call(req(datasets=("hr.salaries",)), run)
    assert r.provider == "local"  # the local model may answer ...
    assert cas.skipped_internal == 1  # ... but nothing is written
    assert not (tmp_path / "suite" / "calls.jsonl").exists()
    r2 = router.call(req(datasets=("not.tagged",)), run)
    assert r2 and cas.skipped_internal == 2
    assert cas.entries == {}


def test_cassette_refuses_a_line_containing_a_key(tmp_path):
    cas = Cassette(tmp_path / "s", "record", CTX, env=KEYS)
    r = dataclasses.replace(req(text=f"echo {KEYS['GROQ_API_KEY']}"), sensitivity="public")
    resp = LLMResponse("{}", "groq", "m", 1, 1)
    with pytest.raises(CassetteLeak):
        cas.record("groq", "m", r, resp)
    leaky = LLMResponse("Bearer abcdefghijklmnop", "groq", "m", 1, 1)
    with pytest.raises(CassetteLeak):
        cas.record("groq", "m", dataclasses.replace(req(), sensitivity="public"), leaky)
    assert not (tmp_path / "s" / "calls.jsonl").exists()


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abcdefgh12345678",
        "x-goog-api-key: foo",
        "gsk_0123456789abcdefghij",
        "sk-or-v1-0123456789abcdefghij",
        "AIzaSyA0123456789abcdefghijk",
        '"authorization": "x"',
    ],
)
def test_secret_patterns_are_caught(text):
    assert scan_for_secrets(text, {}) is not None


def test_ordinary_text_is_not_flagged():
    assert scan_for_secrets("net revenue by country, top 3, ordered descending", KEYS) is None


def test_recording_is_deterministic_and_sorted(tmp_path):
    router, run, cas, local = mk_router(tmp_path, "record")
    for text in ("q1", "q2", "q3"):
        router.call(req(text=text, datasets=PUB), run)
    lines = (tmp_path / "suite" / "calls.jsonl").read_text().splitlines()
    keys = [json.loads(x)["key"] for x in lines]
    assert keys == sorted(keys) and len(keys) == 3


def test_failed_calls_are_not_recorded(tmp_path):
    local = FakeProvider([ProviderError("auth", "x")], name="local")
    router, run, cas, _ = mk_router(tmp_path, "record", providers={"local": local})
    with pytest.raises(Exception):  # noqa: B017
        router.call(req(datasets=PUB), run)
    assert cas.entries == {}


def test_bad_cassette_file_is_reported(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "calls.jsonl").write_text("not json\n")
    with pytest.raises(ValueError, match="not a cassette entry"):
        read_entries(d)


# ---- check ----

CURRENT = {"layer": "layer1", "schema": "schema1", "warehouse": "wh1"}


def record_one(tmp_path, now=None, prompt_id="plan.v0", **kw):
    now = now or time.time
    cas = Cassette(tmp_path / "suite", "record", CTX, env={}, now=now)
    r = dataclasses.replace(req(prompt_id=prompt_id, **kw), sensitivity="public")
    cas.record("local", "m", r, LLMResponse("{}", "local", "m", 1, 1, 5.0))
    return cas


def test_check_flags_stale_entries_by_component(tmp_path):
    record_one(tmp_path, prompt_hash=prompt_hashes()["plan.v0"])
    d = tmp_path / "suite"
    ph = prompt_hashes()
    assert check_cassettes(d, CURRENT, ph)["ok"]
    for part in ("layer", "schema", "warehouse"):
        res = check_cassettes(d, {**CURRENT, part: "changed"}, ph)
        assert not res["ok"] and res["stale"][0]["changed"] == [part]
    res = check_cassettes(d, CURRENT, {"plan.v0": "edited"})
    assert res["stale"][0]["changed"] == ["prompt"]
    res = check_cassettes(d, {**CURRENT, "warehouse": None}, ph)
    assert res["ok"] and not res["warehouse_checked"]


def test_check_reports_age(tmp_path):
    record_one(tmp_path, now=lambda: time.time() - 40 * 86400, prompt_hash="p")
    d = tmp_path / "suite"
    res = check_cassettes(d, {**CURRENT}, {"plan.v0": "p"}, max_age_days=30)
    assert res["too_old"] and not res["ok"]
    assert res["oldest_age_days"] == pytest.approx(40, abs=0.1)
    assert check_cassettes(d, {**CURRENT}, {"plan.v0": "p"}, max_age_days=90)["ok"]


def test_check_unused_and_missing_need_a_usage_file(tmp_path):
    cas = record_one(tmp_path, prompt_hash="p")
    d = tmp_path / "suite"
    key = next(iter(cas.entries))
    res = check_cassettes(d, CURRENT, {"plan.v0": "p"})
    assert res["unused"] is None and res["missing"] is None and res["ok"]
    res = check_cassettes(d, CURRENT, {"plan.v0": "p"}, usage={"hit_keys": [key], "missed": {}})
    assert res["ok"] and res["unused"] == [] and res["missing"] == []
    usage = {"hit_keys": [], "missed": {"deadbeef": {"task": "plan", "provider": "groq"}}}
    res = check_cassettes(d, CURRENT, {"plan.v0": "p"}, usage=usage)
    assert res["unused"] == [key] and res["missing"][0]["key"] == "deadbeef" and not res["ok"]


def test_cli_cassettes_check_exit_codes(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("VA_READONLY_DATABASE_URL", raising=False)
    d = tmp_path / "suite"
    from verified_analyst.llm.cassette import layer_hash as lh
    from verified_analyst.llm.cassette import schema_hash as sh

    real = CassetteContext(lh(), sh(), "wh")
    cas = Cassette(d, "record", real, env={})
    r = dataclasses.replace(
        req(prompt_id="plan.v0", prompt_hash=prompt_hashes()["plan.v0"]), sensitivity="public"
    )
    cas.record("local", "m", r, LLMResponse("{}", "local", "m", 1, 1, 5.0))
    assert main(["cassettes", "check", str(d), "--warehouse-hash", "wh"]) == 0
    out = capsys.readouterr().out
    assert "entries: 1" in out and out.strip().endswith("OK")
    assert main(["cassettes", "check", str(d), "--warehouse-hash", "other"]) == 1
    assert "STALE" in capsys.readouterr().out
    assert main(["cassettes", "check", str(d), "--warehouse-hash", "wh", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == 1
    usage = tmp_path / "usage.json"
    cas2 = Cassette(d, "replay", real, env={})
    cas2.lookup("local", "other-model", r)  # a miss
    cas2.write_usage(usage)
    assert (
        main(["cassettes", "check", str(d), "--warehouse-hash", "wh", "--usage", str(usage)]) == 1
    )
    out = capsys.readouterr().out
    assert "UNUSED" in out and "MISSING" in out


def test_cli_cassettes_check_on_an_empty_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("VA_READONLY_DATABASE_URL", raising=False)
    assert main(["cassettes", "check", str(Path(tmp_path) / "nothing")]) == 0
    assert "entries: 0" in capsys.readouterr().out


def test_a_request_without_reasoning_effort_keeps_its_old_key_material():
    from verified_analyst.llm.cassette import normalise_request

    r = req(datasets=PUB)
    assert "reasoning_effort" not in normalise_request(r)
    assert (
        normalise_request(dataclasses.replace(r, reasoning_effort="low"))["reasoning_effort"]
        == "low"
    )
