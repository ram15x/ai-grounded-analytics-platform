"""`va live models` and `va live probe` with mocked HTTP only. No real provider is contacted."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from llm_helpers import CONN_ERROR, KEYS, TIMEOUT, ok_openai, settings, status
from verified_analyst.cli import catalogue, main
from verified_analyst.config import ConfigError
from verified_analyst.llm import live
from verified_analyst.llm.cassette import read_entries
from verified_analyst.llm.live import (
    AuthStop,
    CountingTransport,
    ModelGone,
    ProbeOptions,
    ProviderUnavailable,
    RequestCapReached,
    list_provider_models,
    load_probe_questions,
    probe_settings,
    quota_info,
    quota_reading,
    redact,
    render_models,
    run_probe,
    safe_error_body,
    wilson,
)
from verified_analyst.llm.transport import HttpResponse
from verified_analyst.llm.types import CassetteLeak, SensitivityRefusal
from verified_analyst.semantic import load_semantic_layer

ROOT = Path(__file__).resolve().parents[1]
DEV = yaml.safe_load((ROOT / "eval" / "questions" / "dev.yaml").read_text())["questions"]
PLANS = {q["question"]: q["metric_query"] for q in DEV if q.get("expressible_as_metric_query")}


class Getter:
    def __init__(self, *items):
        self.items = list(items)
        self.calls: list[dict] = []

    def __call__(self, url, headers, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "timeout": timeout})
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def jbody(obj, code=200, headers=None):
    return HttpResponse(
        code, json.dumps(obj).encode(), {k.lower(): v for k, v in (headers or {}).items()}
    )


# ------------------------------------------------------------ redaction and statistics


def test_redact_removes_every_key_value():
    text = f"x {KEYS['GROQ_API_KEY']} y {KEYS['GEMINI_API_KEY']} z"
    out = redact(text, KEYS)
    assert "gsk_" not in out and "AIza" not in out and out.count("<redacted>") == 2
    assert redact("nothing here", KEYS) == "nothing here"
    assert redact("short", {"GROQ_API_KEY": "abc"}) == "short"  # too short to be a real key


def test_wilson_interval_known_values():
    lo, hi = wilson(5, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.237, 0.763)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and round(lo, 3) == 0.722
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and round(hi, 3) == 0.278
    assert wilson(0, 0) is None
    assert live.rate("x", 3, 0)["wilson95"] is None
    assert "n=0" in live.render_rate(live.rate("x", 0, 0))
    assert "(n=10;" in live.render_rate(live.rate("x", 5, 10))


# ------------------------------------------------------------ va live models


def cfg():
    s = settings()
    s.providers["gemini"].models["default"] = "REPLACE-ME"
    return s


def test_models_skips_providers_without_a_key_and_makes_no_request():
    g = Getter()
    for name in ("groq", "gemini", "openrouter"):
        r = list_provider_models(cfg(), name, {}, g)
        assert not r["key_present"] and r["requests"] == 0
    assert g.calls == []
    assert "is not set" in render_models([list_provider_models(cfg(), "groq", {}, g)])


def test_models_groq_lists_ids_headers_and_checks_the_configured_id():
    s = cfg()
    s.providers["groq"].models["default"] = "openai/gpt-oss-20b"
    body = {"data": [{"id": "openai/gpt-oss-20b", "active": True}, {"id": "llama-x"}]}
    g = Getter(jbody(body, headers={"X-RateLimit-Remaining-Requests": "29", "Content-Type": "x"}))
    r = list_provider_models(s, "groq", KEYS, g)
    assert [m["id"] for m in r["models"]] == ["openai/gpt-oss-20b", "llama-x"]
    assert r["rate_headers"] == {"x-ratelimit-remaining-requests": "29"}
    assert r["configured"]["default"] == {"id": "openai/gpt-oss-20b", "present": True}
    assert g.calls[0]["headers"]["Authorization"] == f"Bearer {KEYS['GROQ_API_KEY']}"
    text = render_models([r])
    assert "PRESENT" in text and "x-ratelimit-remaining-requests: 29" in text
    assert KEYS["GROQ_API_KEY"] not in text


def test_models_reports_an_absent_configured_id():
    s = cfg()
    s.providers["groq"].models["default"] = "not-a-real-model"
    r = list_provider_models(s, "groq", KEYS, Getter(jbody({"data": [{"id": "a"}]})))
    assert "ABSENT" in render_models([r])


def test_models_placeholder_ids_are_not_checked():
    r = list_provider_models(cfg(), "gemini", KEYS, Getter(jbody({"models": []})))
    assert r["configured"] == {}
    assert "placeholder" in render_models([r])


def test_models_gemini_paginates_strips_prefix_and_uses_a_header_not_the_url():
    p1 = {"models": [{"name": "models/g-a", "supportedGenerationMethods": ["generateContent"]}],
          "nextPageToken": "T2"}  # fmt: skip
    p2 = {"models": [{"name": "models/g-b", "supportedGenerationMethods": ["embedContent"]}]}
    g = Getter(jbody(p1), jbody(p2))
    r = list_provider_models(cfg(), "gemini", KEYS, g)
    assert [m["id"] for m in r["models"]] == ["g-a", "g-b"] and r["requests"] == 2
    assert "pageToken=T2" in g.calls[1]["url"]
    for c in g.calls:
        assert KEYS["GEMINI_API_KEY"] not in c["url"]
        assert c["headers"]["x-goog-api-key"] == KEYS["GEMINI_API_KEY"]
    text = render_models([r])
    assert "g-a" in text and "  g-b" not in text  # only generateContent models are listed


def test_models_pagination_is_bounded():
    page = {
        "models": [{"name": "models/x", "supportedGenerationMethods": []}],
        "nextPageToken": "T",
    }
    g = Getter(*[jbody(page)] * 10)
    assert list_provider_models(cfg(), "gemini", KEYS, g)["requests"] == 3


def test_models_openrouter_free_filter_and_all():
    body = {"data": [
        {"id": "a:free", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "paid", "pricing": {"prompt": "0.1", "completion": "0.2"}},
    ]}  # fmt: skip
    r = list_provider_models(cfg(), "openrouter", KEYS, Getter(jbody(body)))
    short = render_models([r])
    assert "a:free" in short and "paid" not in short and "1 of 2" in short
    assert "paid" in render_models([r], show_all=True)


@pytest.mark.parametrize(
    ("item", "needle"),
    [
        (
            status(401, body=b'{"error": {"message": "Invalid API Key"}}'),
            "HTTP 401: Invalid API Key",
        ),
        (status(429), "HTTP 429"),
        (status(500, body=b"<html>"), "HTTP 500"),
        (HttpResponse(200, b"not json"), "unexpected response shape"),
        (jbody({"nope": 1}), "unexpected response shape"),
        (TIMEOUT, "timeout"),
        (CONN_ERROR, "network"),
    ],
)
def test_models_failures_are_reported_not_raised(item, needle):
    r = list_provider_models(cfg(), "groq", KEYS, Getter(item))
    assert needle in r["error"]
    assert "FAILED" in render_models([r])


def test_models_cli_prints_no_key_even_if_the_provider_echoes_it(monkeypatch, capsys):
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    body = f'{{"error": {{"message": "bad key {KEYS["GROQ_API_KEY"]}"}}}}'.encode()
    g = Getter(status(401, body=body), jbody({"models": []}), jbody({"data": []}))
    monkeypatch.setattr(live, "urllib_get", g)
    code = main(["live", "models"])
    out = capsys.readouterr().out
    assert code == 1 and "<redacted>" in out
    for v in KEYS.values():
        assert v not in out


def test_models_cli_without_any_key_exits_nonzero(monkeypatch, capsys):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(live, "urllib_get", Getter())
    assert main(["live", "models"]) == 1
    assert "is not set" in capsys.readouterr().out
    assert main(["live", "models", "--provider", "nope"]) == 2


# ------------------------------------------------------------ counting transport


def test_counting_transport_counts_caps_and_captures_headers():
    resp = HttpResponse(200, b"{}", {"x-ratelimit-remaining-requests": "5", "server": "x"})
    t = CountingTransport(lambda *a: resp, max_requests=2)
    t("u", {}, b"", 1)
    t("u", {}, b"", 1)
    assert t.count == 2 and t.events[0].rate_headers == {"x-ratelimit-remaining-requests": "5"}
    with pytest.raises(RequestCapReached):
        t("u", {}, b"", 1)
    assert t.count == 2  # the refused request is not counted as sent


def test_counting_transport_never_exceeds_the_hard_ceiling():
    assert CountingTransport(lambda *a: None, max_requests=1000).max_requests == 60


@pytest.mark.parametrize("code", [401, 403])
def test_counting_transport_stops_on_the_first_auth_error(code):
    t = CountingTransport(lambda *a: HttpResponse(code, b"{}"), max_requests=5)
    with pytest.raises(AuthStop):
        t("u", {}, b"", 1)
    assert t.events[0].status == code


def test_counting_transport_records_timeouts():
    def boom(*a):
        raise TIMEOUT

    t = CountingTransport(boom, 5)
    with pytest.raises(type(TIMEOUT)):
        t("u", {}, b"", 1)
    assert t.events[0].status is None and t.events[0].error == "timeout"


def test_safe_error_body_redacts_collapses_and_truncates():
    key = KEYS["GROQ_API_KEY"]
    body = f'{{"error": {{"message": "bad\\n  json for {key}", "auth": "Bearer {key}"}}}}'.encode()
    out = safe_error_body(body, KEYS)
    assert key not in out and "Bearer <redacted>" in out and "  " not in out
    long = safe_error_body(b"x" * 1000, KEYS)
    assert long.startswith("x" * 300) and "truncated, 1000 chars" in long and len(long) < 340
    assert safe_error_body(b"\xff\xfe broken", KEYS)  # undecodable bytes do not raise


def test_counting_transport_keeps_error_bodies_only_for_non_2xx():
    key = KEYS["GROQ_API_KEY"]
    answers = iter(
        [
            HttpResponse(200, b'{"fine": true}'),
            HttpResponse(400, f'{{"error": {{"message": "json_validate_failed {key}"}}}}'.encode()),
            HttpResponse(429, b""),
        ]
    )
    t = CountingTransport(lambda *a: next(answers), max_requests=5, env=KEYS)
    for _ in range(3):
        t("u", {}, b"", 1)
    assert t.events[0].error_body is None
    assert "json_validate_failed" in t.events[1].error_body and key not in t.events[1].error_body
    assert t.events[2].error_body == ""


# ---- early stops: consecutive failures, model not found ----


def _scripted(*items):
    it = iter(items)

    def inner(*a):
        x = next(it)
        if isinstance(x, BaseException):
            raise x
        return x

    return inner


def test_five_consecutive_non_auth_failures_stop_the_probe():
    fails = [
        HttpResponse(503, b"{}"),
        TIMEOUT,
        HttpResponse(429, b""),
        TIMEOUT,
        HttpResponse(500, b""),
    ]
    t = CountingTransport(_scripted(*fails), max_requests=20)
    seen = 0
    with pytest.raises(ProviderUnavailable) as exc:
        for _ in fails:
            seen += 1
            try:
                t("u", {}, b"", 1)
            except type(TIMEOUT):
                continue
    assert seen == 5 and t.count == 5
    assert exc.value.statuses == ["503", "timeout", "429", "timeout", "500"]
    assert exc.value.any_success is False


def test_four_failures_then_a_success_do_not_stop_and_reset_the_streak():
    items = [HttpResponse(503, b"")] * 4 + [HttpResponse(200, b"{}")] + [HttpResponse(503, b"")] * 4
    t = CountingTransport(_scripted(*items), max_requests=20)
    for _ in items:
        t("u", {}, b"", 1)  # never raises: no run of 5
    assert t.count == 9


def test_a_streak_after_a_success_says_it_had_measured_something():
    items = [HttpResponse(200, b"{}")] + [HttpResponse(503, b"")] * 5
    t = CountingTransport(_scripted(*items), max_requests=20)
    t("u", {}, b"", 1)
    with pytest.raises(ProviderUnavailable) as exc:
        for _ in range(5):
            t("u", {}, b"", 1)
    assert exc.value.any_success is True


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_are_not_counted_as_the_failure_streak(code):
    t = CountingTransport(lambda *a: HttpResponse(code, b"{}"), max_requests=5)
    with pytest.raises(AuthStop):
        t("u", {}, b"", 1)


@pytest.mark.parametrize(
    "body",
    [
        b'{"error": {"message": "models/gemini-x is not found for API version v1beta"}}',
        b'{"error": {"message": "This model is no longer available to new users."}}',
        b'{"error": {"code": "model_not_found", "message": "The model `x` does not exist"}}',
    ],
)
def test_the_first_model_not_found_404_stops(body):
    t = CountingTransport(lambda *a: HttpResponse(404, body), max_requests=5, env=KEYS)
    with pytest.raises(ModelGone) as exc:
        t("u", {}, b"", 1)
    assert t.count == 1 and exc.value.body


def test_a_404_that_does_not_say_model_is_an_ordinary_failure():
    t = CountingTransport(lambda *a: HttpResponse(404, b"{}"), max_requests=5)
    t("u", {}, b"", 1)  # does not raise
    assert t.count == 1


def test_a_deliberate_invalid_model_404_neither_stops_nor_counts():
    body = b'{"error": {"code": "model_not_found", "message": "does not exist"}}'
    t = CountingTransport(lambda *a: HttpResponse(404, body), max_requests=5)
    t.expect_model_404 = True
    t("u", {}, b"", 1)
    assert t.count == 1 and t._streak == []


# ---- 429 quota names ----

GEMINI_429 = json.dumps(
    {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota. SECRET-PROJECT-TEXT",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/free_tier_requests",
                            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                            "quotaDimensions": {"model": "m", "location": "global"},
                            "quotaValue": "0",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"},
            ],
        }
    }
).encode()
GROQ_429 = json.dumps(
    {
        "error": {
            "message": "Rate limit reached for model `m` in organization `org_01SECRET` service "
            "tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7500, Requested 900. "
            "Please try again in 5.2s.",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
).encode()


def test_quota_info_keeps_names_and_limits_from_a_google_style_body():
    info = quota_info(GEMINI_429)
    assert info["violations"] == [
        {
            "quota_metric": "generativelanguage.googleapis.com/free_tier_requests",
            "quota_id": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            "quota_value": "0",
        }
    ]
    assert info["retry_delay"] == "23s" and info["status"] == "RESOURCE_EXHAUSTED"
    assert "SECRET" not in json.dumps(info)
    assert quota_reading(info) == "window named: minute; a limit of 0 was reported"


def test_quota_info_keeps_limit_names_from_a_groq_style_body_and_no_org_id():
    info = quota_info(GROQ_429)
    assert info["limit_on"] == "tokens per minute" and info["limit_name"] == "TPM"
    assert info["limit_value"] == "8000" and info["retry_delay"] == "5.2s"
    assert info["code"] == "rate_limit_exceeded" and info["type"] == "tokens"
    assert "SECRET" not in json.dumps(info) and "org_" not in json.dumps(info)
    assert quota_reading(info) == "window named: minute"


def test_a_daily_limit_is_read_as_a_day_window():
    body = json.dumps(
        {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000, Used 1."}}
    ).encode()
    assert quota_reading(quota_info(body)) == "window named: day"


@pytest.mark.parametrize("body", [b"", b"not json", b"[]", b'{"error": "x"}', b'{"error": {}}'])
def test_quota_info_returns_none_when_there_is_nothing_to_keep(body):
    assert quota_info(body) is None


def test_names_are_cleaned_and_cut():
    body = json.dumps(
        {"error": {"details": [{"violations": [{"quotaId": "ok-id\n<script>" + "x" * 500}]}]}}
    ).encode()
    qid = quota_info(body)["violations"][0]["quota_id"]
    assert "<" not in qid and "\n" not in qid and len(qid) <= 120


def test_a_429_with_names_keeps_the_names_not_the_body_and_without_names_keeps_the_body():
    answers = iter([HttpResponse(429, GEMINI_429), HttpResponse(429, b'{"x": "plain 429 text"}')])
    t = CountingTransport(lambda *a: next(answers), max_requests=5, env=KEYS)
    t("u", {}, b"", 1)
    t("u", {}, b"", 1)
    assert t.events[0].quota and t.events[0].error_body is None
    assert t.events[1].quota is None and "plain 429 text" in t.events[1].error_body


# ------------------------------------------------------------ probe settings


def opts(tmp_path, **kw):
    base = {"provider": "groq", "model": "m-groq", "out_dir": tmp_path / "run"}
    base.update(kw)
    return ProbeOptions(**base)


def test_probe_is_pinned_to_one_provider_and_one_model(tmp_path):
    s = probe_settings(settings(), opts(tmp_path, model="the-model"))
    assert s.providers["groq"].models == {"default": "the-model"}
    assert s.tasks == {t: ["groq"] for t in ("plan", "compose", "repair")}


def test_request_cap_defaults_to_25_can_be_raised_to_60_and_never_past_it(tmp_path):
    assert probe_settings(settings(), opts(tmp_path)).probe.max_requests == 25
    assert probe_settings(settings(), opts(tmp_path, max_requests=60)).probe.max_requests == 60
    assert probe_settings(settings(), opts(tmp_path, max_requests=7)).probe.max_requests == 7
    for bad in (61, 1000, 0, -3):
        with pytest.raises(ConfigError, match="between 1 and 60"):
            probe_settings(settings(), opts(tmp_path, max_requests=bad))


def test_the_token_cap_can_only_be_lowered(tmp_path):
    s = probe_settings(settings(), opts(tmp_path, token_cap=10**9))
    assert s.probe.token_cap == 60_000
    assert probe_settings(settings(), opts(tmp_path, token_cap=500)).probe.token_cap == 500


def test_a_probe_waits_for_the_bucket_longer_than_the_router_default(tmp_path):
    s = probe_settings(settings(bucket={"fraction": 0.5, "max_wait_s": 10}), opts(tmp_path))
    assert s.bucket.max_wait_s == 120
    s = probe_settings(settings(bucket={"fraction": 0.5, "max_wait_s": 500}), opts(tmp_path))
    assert s.bucket.max_wait_s == 500  # never lowered


@pytest.mark.parametrize(
    "kw", [{"provider": "nope"}, {"provider": "local"}, {"model": ""}, {"model": "REPLACE-ME"}]
)
def test_probe_rejects_bad_targets(tmp_path, kw):
    with pytest.raises(ConfigError):
        probe_settings(settings(), opts(tmp_path, **kw))


def test_the_shipped_config_defaults_the_probe_to_25_requests():
    from verified_analyst.llm.settings import load_router_settings

    s = load_router_settings()
    assert s.probe.max_requests == 25 and s.probe.token_cap > 0


def test_the_config_file_cannot_raise_the_probe_cap_above_60():
    from pydantic import ValidationError

    from verified_analyst.llm.settings import ProbeSettings

    assert ProbeSettings(max_requests=60).max_requests == 60
    with pytest.raises(ValidationError):
        ProbeSettings(max_requests=61)


def test_probe_questions_are_the_first_ten_expressible_t1_questions():
    qs = load_probe_questions()
    assert len(qs) == 10 and qs[0].startswith("What was total gross revenue")
    assert all(q in PLANS for q in qs)
    assert "How many invoice lines were sales?" not in qs  # not expressible as a MetricQuery


# ------------------------------------------------------------ probe, end to end with mocked HTTP

pytestmark_db = pytest.mark.db


class Model:
    """A scripted 'model' behind the mocked HTTP layer. It answers by reading the prompt."""

    def __init__(
        self, *, typed_digit_first=False, bad_json_plans=0, statuses=None, headers=None,
        abstain_plans=(),
    ):  # fmt: skip
        self.abstain_plans = dict(abstain_plans)  # question text -> abstain reason
        self.requests: list[dict] = []
        self.typed_digit_first = typed_digit_first
        self.bad_json_plans = bad_json_plans
        self.statuses = list(statuses or [])
        self.headers = headers or {}
        self.composed: set[str] = set()

    def __call__(self, url, headers, body, timeout):
        data = json.loads(body)
        self.requests.append({"url": url, "headers": headers, "body": data})
        if self.statuses:
            item = self.statuses.pop(0)
            if item is not None:
                return item
        system = data["messages"][0]["content"]
        user = data["messages"][-1]["content"]
        hdr = {k.lower(): v for k, v in self.headers.items()}
        if system.startswith("You turn one business question"):
            if self.bad_json_plans > 0:
                self.bad_json_plans -= 1
                text = "sorry, here is your plan"
            else:
                question = re.search(r"Question: (.*)", user).group(1).strip()
                if question in self.abstain_plans:
                    text = json.dumps({"abstain": self.abstain_plans[question]})
                else:
                    text = json.dumps({"metric_query": PLANS[question]})
        else:
            text = json.dumps(self.compose(user, repair="Problems:" in user))
        resp = ok_openai(text, 120, 40)
        resp.headers = hdr
        return resp

    def compose(self, user, repair):
        tables = json.loads(re.search(r"Result tables: (.*)", user).group(1))["q1"]
        claims, parts = [], []
        for i, col in enumerate(tables["columns"]):
            unit = "percent" if col == "cancellation_rate" else (
                "count" if col in ("units_sold", "order_count", "active_customers") else "currency"
            )  # fmt: skip
            prov = {"kind": "cell", "query_id": "q1", "column": col}
            claims.append({"id": f"c{i}", "label": col, "unit": unit, "provenance": prov})
            parts.append(f"The {col.replace('_', ' ')} was {{c{i}}}.")
        template = " ".join(parts)
        if self.typed_digit_first and not repair:
            template += " That is up 14% on last year."
        return {"template": template, "claims": claims}


@pytest.fixture
def env(warehouse, monkeypatch):
    monkeypatch.setattr(live, "_sleep", lambda s: None)
    layer = load_semantic_layer()
    return layer, warehouse.readonly_url, catalogue(layer, warehouse.readonly_url)


def probe(env, model, tmp_path, cfg=None, qs=None, **kw):
    layer, url, known = env
    return run_probe(
        cfg or settings(datasets={"analytics.sales_lines": "public"}),
        opts(tmp_path, **kw),
        layer=layer, readonly_url=url, questions=load_probe_questions() if qs is None else qs,
        known_values=known, env=KEYS, inner_transport=model,
    )  # fmt: skip


def rate_of(out, startswith):
    return next(r for r in out["report"]["rates"] if r["name"].startswith(startswith))


@pytest.mark.db
def test_probe_runs_ten_questions_in_twenty_requests_and_records_cassettes(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path)
    rep = out["report"]
    assert rep["stop"]["questions_completed"] == 10 and len(m.requests) == 20
    assert rep["http"]["requests_sent"] == 20 and rep["http"]["by_status"] == {"200": 20}
    verified = rate_of(out, "questions with every number verified")
    assert (verified["k"], verified["n"]) == (10, 10)
    assert rate_of(out, "plan responses that parse")["n"] == 10
    assert rate_of(out, "compose/repair responses valid")["n"] == 10
    assert rate_of(out, "first compose answers that typed a digit")["k"] == 0
    # every request was against the one pinned model, with the key in a header
    assert {r["body"]["model"] for r in m.requests} == {"m-groq"}
    assert all(KEYS["GROQ_API_KEY"] in json.dumps(r["headers"]) for r in m.requests)
    # cassettes: 20 entries, public only, scan clean, check passes
    c = out["cassette"]
    assert c["entries"] == 20 and c["key_scan"] == "clean" and c["check_ok"] is True
    assert c["skipped_internal"] == 0
    assert all(
        e["sensitivity"] == "public" for e in read_entries(tmp_path / "run" / "cassettes").values()
    )
    # report: n next to every rate; no key anywhere; no ranking or accuracy wording
    md = out["markdown"]
    for r in rep["rates"]:
        assert f"n={r['n']}" in md
    low = md.lower()
    assert "not an accuracy figure" in low and "no accuracy claim" in low
    assert "accurate" not in low and "better" not in low and "best" not in low
    assert "no ranking" in low and "rank" not in low.replace("no ranking", "")
    for f in ("report.md", "report.json"):
        assert not any(v in (tmp_path / "run" / f).read_text() for v in KEYS.values())
    assert all(
        v not in (tmp_path / "run" / "cassettes" / "calls.jsonl").read_text() for v in KEYS.values()
    )


@pytest.mark.db
def test_probe_report_shows_redacted_truncated_error_bodies(env, tmp_path):
    key = KEYS["GROQ_API_KEY"]
    msg = f"json_validate_failed for {key} " + "z" * 600
    bad = status(400, body=json.dumps({"error": {"message": msg}}).encode())
    out = probe(env, Model(statuses=[None, bad]), tmp_path, qs=load_probe_questions()[:1])
    bodies = out["report"]["http"]["error_bodies"]
    assert [b["status"] for b in bodies] == [400] and bodies[0]["seq"] == 2
    assert "json_validate_failed" in bodies[0]["body"] and "truncated" in bodies[0]["body"]
    assert "non-2xx response bodies" in out["markdown"] and "HTTP 400" in out["markdown"]
    for f in ("report.md", "report.json"):
        assert key not in (tmp_path / "run" / f).read_text()


@pytest.mark.db
def test_a_clean_probe_reports_no_error_bodies(env, tmp_path):
    out = probe(env, Model(), tmp_path, qs=load_probe_questions()[:1])
    assert out["report"]["http"]["error_bodies"] == []
    assert "non-2xx response bodies: none" in out["markdown"]


@pytest.mark.db
def test_va_cassettes_check_passes_on_a_probe_recording(env, tmp_path, capsys, monkeypatch):
    probe(env, Model(), tmp_path)
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", env[1])
    assert main(["cassettes", "check", str(tmp_path / "run" / "cassettes")]) == 0
    assert capsys.readouterr().out.strip().endswith("OK")


@pytest.mark.db
def test_placeholder_adherence_and_repair_are_counted(env, tmp_path):
    out = probe(env, Model(typed_digit_first=True), tmp_path, qs=load_probe_questions()[:3])
    first = rate_of(out, "first compose answers that typed a digit")
    assert (first["k"], first["n"]) == (3, 3)
    rep = rate_of(out, "questions that needed at least one repair")
    assert (rep["k"], rep["n"]) == (3, 3)
    assert rate_of(out, "questions with every number verified")["k"] == 3
    assert out["report"]["http"]["requests_sent"] == 9  # plan + compose + repair, three times


@pytest.mark.db
def test_unparseable_plan_counts_against_json_rate_and_triggers_a_repair(env, tmp_path):
    out = probe(env, Model(bad_json_plans=1), tmp_path, qs=load_probe_questions()[:2])
    pj = rate_of(out, "plan responses that parse")
    assert (pj["k"], pj["n"]) == (2, 3)
    assert rate_of(out, "questions that needed at least one repair")["k"] == 1


@pytest.mark.db
def test_request_cap_stops_before_the_next_question(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, max_requests=7)
    assert len(m.requests) == 6 and out["report"]["stop"]["questions_completed"] == 3
    assert "request cap" in out["report"]["stop"]["reason"]
    assert all(len(r["body"]["messages"]) >= 2 for r in m.requests)


@pytest.mark.db
def test_request_cap_mid_question_is_recorded_as_cut_not_counted(env, tmp_path):
    m = Model(typed_digit_first=True)  # each question needs plan + compose + repair = 3
    out = probe(
        env, m, tmp_path, max_requests=5
    )  # question 1 uses 3; question 2 gets cut at its 3rd
    assert len(m.requests) == 5
    assert out["report"]["stop"]["questions_completed"] == 1
    assert "mid-question" in out["report"]["stop"]["reason"]
    assert len(out["report"]["stop"]["cut"]) == 1


@pytest.mark.db
def test_token_cap_stops_the_run(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, token_cap=900)
    assert "token cap" in out["report"]["stop"]["reason"]
    assert out["report"]["accounting"]["tokens_used"] <= 900 + 200  # one call may overshoot
    assert len(m.requests) < 20


@pytest.mark.db
def test_first_auth_error_stops_the_whole_run(env, tmp_path):
    m = Model(statuses=[status(401, body=b'{"error": {"message": "bad key"}}')])
    out = probe(env, m, tmp_path)
    assert len(m.requests) == 1 and out["stopped_on_auth"]
    assert "auth error" in out["report"]["stop"]["reason"]
    assert out["report"]["stop"]["questions_completed"] == 0


@pytest.mark.db
def test_429_with_retry_after_and_5xx_are_reported(env, tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(live, "_sleep", slept.append)
    m = Model(statuses=[status(429, {"Retry-After": "3", "x-ratelimit-remaining-requests": "0"}),
                        status(503)])  # fmt: skip
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:1])
    http = out["report"]["http"]
    assert http["by_status"]["429"] == 1 and http["by_status"]["503"] == 1
    assert http["retry_after_headers"] == [{"seq": 1, "status": 429, "value": "3"}]
    assert http["rate_headers"]["x-ratelimit-remaining-requests"]["first"] == "0"
    assert 3.0 in slept
    assert out["report"]["stop"]["questions_completed"] == 1


@pytest.mark.db
def test_reasoning_tokens_and_finish_reasons_are_reported(env, tmp_path):
    def with_reasoning(url, headers, body, timeout):
        r = Model()(url, headers, body, timeout)
        d = json.loads(r.body)
        d["usage"]["completion_tokens_details"] = {"reasoning_tokens": 25}
        d["choices"][0]["finish_reason"] = "stop"
        return HttpResponse(200, json.dumps(d).encode(), {})

    out = probe(env, with_reasoning, tmp_path, qs=load_probe_questions()[:1])
    acc = out["report"]["accounting"]
    assert acc["reasoning_tokens_reported"] == 50 and acc["finish_reasons"] == {"stop": 2}
    assert acc["prompt_tokens"] == 240 and acc["completion_tokens"] == 80
    assert "reasoning reported 50" in out["markdown"]


@pytest.mark.db
def test_internal_dataset_is_refused_before_any_request(env, tmp_path):
    m = Model()
    with pytest.raises(SensitivityRefusal):
        probe(env, m, tmp_path, cfg=settings(datasets={"analytics.sales_lines": "internal"}))
    with pytest.raises(SensitivityRefusal):
        probe(env, m, tmp_path, cfg=settings(datasets={}))  # untagged is internal
    assert m.requests == [] and not (tmp_path / "run").exists()


@pytest.mark.db
def test_missing_key_is_refused_before_any_request(env, tmp_path):
    layer, url, known = env
    m = Model()
    with pytest.raises(ConfigError):
        run_probe(settings(), opts(tmp_path), layer=layer, readonly_url=url,
                  questions=["q"], known_values=known, env={}, inner_transport=m)  # fmt: skip
    assert m.requests == []


@pytest.mark.db
def test_no_record_writes_no_cassette(env, tmp_path):
    out = probe(env, Model(), tmp_path, qs=load_probe_questions()[:1], record=False)
    assert out["cassette"] is None and not (tmp_path / "run" / "cassettes").exists()
    assert (tmp_path / "run" / "report.md").exists()


@pytest.mark.db
def test_a_response_that_echoes_a_key_is_never_written_to_a_cassette(env, tmp_path):
    class Echo(Model):
        def compose(self, user, repair):
            d = super().compose(user, repair)
            d["template"] += f" {KEYS['GROQ_API_KEY']}"
            return d

    with pytest.raises(CassetteLeak):
        probe(env, Echo(), tmp_path, qs=load_probe_questions()[:1])
    f = tmp_path / "run" / "cassettes" / "calls.jsonl"
    assert not f.exists() or KEYS["GROQ_API_KEY"] not in f.read_text()


@pytest.mark.db
def test_quota_check_is_observable_only_with_remaining_headers(env, tmp_path):
    seq = iter([("5", 200), (None, 404), ("4", 200)])

    def transport(url, headers, body, timeout):
        rem, code = next(seq)
        h = {"x-ratelimit-remaining-requests": rem} if rem else {}
        if code == 200:
            return HttpResponse(200, ok_openai().body, h)
        return HttpResponse(code, b"{}", h)

    out = probe(env, transport, tmp_path, qs=[], quota_check=True, record=False)
    qc = out["report"]["quota_check"]
    assert [s["step"] for s in qc["steps"]] == ["good_1", "invalid_model", "good_2"]
    assert qc["steps"][1]["outcome"] == "bad_request" and qc["observable"] is True
    assert qc["remaining_request_headers"][0] == {"x-ratelimit-remaining-requests": "5"}
    assert out["report"]["http"]["requests_sent"] == 3
    # no headers at all: the report says it is not observable
    plain = lambda *a: HttpResponse(200, ok_openai().body, {})  # noqa: E731
    out = probe(env, plain, tmp_path / "b", qs=[], quota_check=True, record=False)
    assert out["report"]["quota_check"]["observable"] is False
    assert "NOT observable" in out["markdown"]


@pytest.mark.db
def test_probe_cli_prints_the_report_and_never_a_key(env, tmp_path, monkeypatch, capsys):
    layer, url, _ = env
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", url)
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    m = Model()
    monkeypatch.setattr(live, "urllib_transport", m)
    code = main(["live", "probe", "--provider", "groq", "--model", "openai/gpt-oss-20b",
                 "--out-dir", str(tmp_path / "cli"), "--max-requests", "4"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "# Live probe: groq / openai/gpt-oss-20b" in out
    assert "Nothing was committed" in out and "key scan: clean" in out
    assert all(v not in out for v in KEYS.values())
    assert len(m.requests) == 4


@pytest.mark.db
def test_probe_cli_exits_3_on_auth_error(env, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", env[1])
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(live, "urllib_transport", Model(statuses=[status(403)]))
    code = main(["live", "probe", "--provider", "groq", "--model", "x",
                 "--out-dir", str(tmp_path / "cli")])  # fmt: skip
    assert code == 3 and "STOPPED on first auth error" in capsys.readouterr().out


def test_probe_cli_refuses_without_a_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "postgresql://x")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("verified_analyst.cli.catalogue", lambda *a: {})
    code = main(["live", "probe", "--provider", "groq", "--model", "x", "--out-dir", str(tmp_path)])
    assert code == 2 and "GROQ_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["61", "0", "-1", "many"])
def test_cli_rejects_a_request_cap_outside_1_to_60(value, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["live", "probe", "--provider", "groq", "--model", "x", "--max-requests", value])
    assert exc.value.code == 2
    assert "--max-requests" in capsys.readouterr().err


@pytest.mark.db
def test_probe_with_a_cap_of_60_is_allowed_and_reported(env, tmp_path):
    out = probe(env, Model(), tmp_path, qs=load_probe_questions()[:2], max_requests=60)
    assert out["report"]["header"]["caps"]["max_requests"] == 60
    assert "Caps: 60 requests" in out["markdown"]


@pytest.mark.db
def test_report_states_the_bucket_and_whether_reasoning_effort_was_sent(env, tmp_path):
    cfg = settings(datasets={"analytics.sales_lines": "public"})
    cfg.providers["groq"].tpm = 8000
    out = probe(env, Model(), tmp_path, cfg=cfg, qs=load_probe_questions()[:1])
    h = out["report"]["header"]
    assert h["bucket"]["tokens_per_minute"] == 8000 and h["reasoning_effort_sent"] is None
    assert (
        "tokens/minute 8000" in out["markdown"] and "reasoning_effort sent: no" in out["markdown"]
    )
    cfg.providers["groq"].reasoning_effort = "low"
    m = Model()
    out = probe(env, m, tmp_path / "b", cfg=cfg, qs=load_probe_questions()[:1])
    assert out["report"]["header"]["reasoning_effort_sent"] == "low"
    assert {r["body"].get("reasoning_effort") for r in m.requests} == {"low"}
    assert "reasoning_effort sent: low" in out["markdown"]


def _jv():
    body = {"error": {"code": "json_validate_failed", "failed_generation": "", "message": "m"}}
    return status(400, body=json.dumps(body).encode())


@pytest.mark.db
def test_a_json_rejection_retry_is_a_counted_request_and_the_question_still_verifies(env, tmp_path):
    m = Model(statuses=[None, _jv()])  # plan ok, first compose rejected, the retry answers
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:1])
    rep = out["report"]
    assert rep["http"]["requests_sent"] == 3 and len(m.requests) == 3
    assert rep["http"]["by_status"] == {"200": 2, "400": 1}
    assert rate_of(out, "questions with every number verified")["k"] == 1
    assert rate_of(out, "questions that needed at least one repair")["k"] == 0
    assert rep["http"]["error_bodies"][0]["status"] == 400


@pytest.mark.db
def test_the_request_cap_stops_a_question_between_a_rejection_retry_and_its_repair(env, tmp_path):
    m = Model(statuses=[None, _jv(), _jv()])  # plan ok, compose rejected twice -> a repair needed
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:1], max_requests=3)
    assert out["report"]["http"]["requests_sent"] == 3  # plan, compose, its one retry; no fourth
    assert "request cap" in out["report"]["stop"]["reason"]
    assert out["report"]["accounting"]["by_provider"]["groq"]["errors"] == {"json_rejected": 2}


@pytest.mark.db
def test_every_abstain_prints_the_planners_words_and_the_catalogue_in_the_prompt(env, tmp_path):
    qs = load_probe_questions()[:3]
    m = Model(abstain_plans={qs[1]: "ambiguous"})
    out = probe(env, m, tmp_path, qs=qs)
    ab = out["report"]["abstains"]
    assert len(ab["items"]) == 1
    it = ab["items"][0]
    assert (it["question"], it["reason"], it["source"]) == (qs[1], "ambiguous", "planner")
    assert json.loads(it["planner_reply"]) == {"abstain": "ambiguous"}
    # the catalogue listed is exactly the one the model was sent
    plan_prompts = [r["body"]["messages"][-1]["content"] for r in m.requests]
    sent = next(p for p in plan_prompts if qs[1] in p)
    layer = env[0]
    assert set(ab["catalogue_metrics"]) == set(layer.metrics) and len(ab["catalogue_metrics"]) > 5
    assert ab["catalogue_dimensions"] == sorted(layer.dimensions)
    assert json.dumps(ab["catalogue"], sort_keys=True) in sent
    md = out["markdown"]
    assert "## Abstains (raw)" in md and "reason code 'ambiguous'" in md
    assert "the planner replied" in md and all(n in md for n in ab["catalogue_metrics"])
    assert ab["plan_prompt_id"] == "plan.v0" and "plan_failures" in out["report"]["questions"][0]


@pytest.mark.db
def test_a_non_planner_abstain_says_so_and_names_the_last_model_step(env, tmp_path):
    qs = load_probe_questions()[:1]
    m = Model(statuses=[None, status(400, body=b"{}")])  # plan ok, then a plain 400 on compose
    out = probe(env, m, tmp_path, qs=qs)
    it = out["report"]["abstains"]["items"][0]
    assert (it["reason"], it["source"], it["planner_reply"]) == ("provider_error", "pipeline", None)
    assert it["last_model_step"] == "plan"
    assert "not the planner; last model step: plan" in out["markdown"]


@pytest.mark.db
def test_a_run_with_no_abstain_has_no_abstain_section(env, tmp_path):
    out = probe(env, Model(), tmp_path, qs=load_probe_questions()[:1])
    assert out["report"]["abstains"]["items"] == []
    assert "## Abstains" not in out["markdown"]


def test_the_catalogue_text_holds_names_units_and_dimensions_only():
    """The report says the prompt has no definitions or synonyms. Guard that statement."""
    from verified_analyst.llm.prompts import catalogue_text

    cat = json.loads(catalogue_text(load_semantic_layer()))
    assert set(cat) == {"metrics", "dimensions"}
    assert all(set(m) == {"unit", "dimensions"} for m in cat["metrics"].values())


@pytest.mark.db
def test_five_failures_in_a_row_stop_with_nothing_measured(env, tmp_path):
    m = Model(statuses=[status(503)] * 20)
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:3], max_requests=30)
    rep = out["report"]
    assert "provider unavailable, nothing measured" in rep["stop"]["reason"]
    assert rep["stop"]["kind"] == "provider_unavailable" and out["stopped_early"]
    # question 1 used 4 attempts (abstained: provider_error); the 5th failure stopped question 2
    assert rep["stop"]["questions_completed"] == 1 and rep["questions"][0]["outcome"] == "abstain"
    assert len(m.requests) == 5 and len(rep["stop"]["cut"]) == 1
    assert rep["http"]["by_status"] == {"503": 5}
    assert "**STOPPED: provider unavailable, nothing measured" in out["markdown"]


@pytest.mark.db
def test_a_model_not_found_404_stops_on_the_first(env, tmp_path):
    body = b'{"error": {"message": "models/x is not found for API version v1beta"}}'
    m = Model(statuses=[status(404, body=body)])
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:3])
    rep = out["report"]
    assert len(m.requests) == 1 and rep["stop"]["kind"] == "model_gone"
    assert "not found or no longer available" in rep["stop"]["reason"]
    assert "models/x is not found" in rep["stop"]["reason"]


@pytest.mark.db
def test_the_quota_checks_invalid_model_step_does_not_stop_the_probe(env, tmp_path):
    seq = iter([200, 404, 200])
    gone = b'{"error": {"code": "model_not_found", "message": "The model does not exist"}}'

    def transport(url, headers, body, timeout):
        code = next(seq, 200)
        return HttpResponse(200, ok_openai().body, {}) if code == 200 else HttpResponse(404, gone)

    out = probe(env, transport, tmp_path, qs=[], quota_check=True, record=False)
    assert out["report"]["stop"]["kind"] is None
    assert [s["step"] for s in out["report"]["quota_check"]["steps"]][1] == "invalid_model"


@pytest.mark.db
def test_probe_cli_exits_4_when_the_provider_is_unavailable(env, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", env[1])
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(live, "urllib_transport", Model(statuses=[status(503)] * 20))
    code = main(["live", "probe", "--provider", "groq", "--model", "x",
                 "--out-dir", str(tmp_path / "cli")])  # fmt: skip
    assert code == 4 and "provider unavailable, nothing measured" in capsys.readouterr().out


@pytest.mark.db
def test_report_lists_429_quota_names_without_the_body(env, tmp_path, monkeypatch):
    monkeypatch.setattr(live, "_sleep", lambda s: None)
    m = Model(statuses=[status(429, {"Retry-After": "1"}, body=GEMINI_429)])
    out = probe(env, m, tmp_path, qs=load_probe_questions()[:1])
    http = out["report"]["http"]
    assert http["quota_429"][0]["seq"] == 1
    assert http["quota_429"][0]["violations"][0]["quota_value"] == "0"
    assert http["error_bodies"] == []  # the 429's body is not kept
    md = out["markdown"]
    assert "HTTP 429 quota details" in md and "a limit of 0 was reported" in md
    assert "SECRET-PROJECT-TEXT" not in md
    assert "SECRET-PROJECT-TEXT" not in (tmp_path / "run" / "report.json").read_text()
