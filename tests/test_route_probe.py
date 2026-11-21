"""`va route probe` with mocked HTTP only, plus the guards that keep the locked router questions
away from prompt tuning. No real provider is contacted and nothing here measures a model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from llm_helpers import KEYS, ok_openai, settings, status
from verified_analyst.cli import catalogue
from verified_analyst.llm import live, route_probe
from verified_analyst.llm.answer import Retrieved, make_retriever
from verified_analyst.llm.route_probe import RouteProbeOptions, load_dev_questions, run_route_probe
from verified_analyst.semantic import load_semantic_layer

ROOT = Path(__file__).resolve().parents[1]
RQ = ROOT / "eval" / "router_questions"
DEV = yaml.safe_load((RQ / "dev.yaml").read_text("utf-8"))
HELD = yaml.safe_load((RQ / "heldout.yaml").read_text("utf-8"))
ROUTE_OF = {q["question"]: q["route"] for q in DEV["questions"]}
ROUTE_OF.update({q["question"]: "documents" for q in DEV["injection"]})
PLAN = {
    "metric_query": {
        "metrics": ["net_revenue"],
        "dimensions": ["country"],
        "order_by": [{"by": "net_revenue", "desc": True}],
        "limit": 3,
    }
}


class Model:
    """A scripted model behind mocked HTTP: routes by the author's labels (or a fixed route),
    plans one query, composes from cells and says nothing for passages unless told to repeat."""

    def __init__(self, route=None, repeat=None, statuses=None, route_reply=None, bind_label=None):
        self.route, self.repeat, self.route_reply = route, repeat, route_reply
        self.bind_label = bind_label
        self.statuses = list(statuses or [])
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        data = json.loads(body)
        self.requests.append(data)
        if self.statuses:
            item = self.statuses.pop(0)
            if item is not None:
                return item
        system, user = data["messages"][0]["content"], data["messages"][-1]["content"]
        question = re.search(r"Question: (.*)", user).group(1).strip()
        if system.startswith("You decide which source"):
            r = self.route or ROUTE_OF[question]
            reply = {"route": r, **({"reason": "ambiguous"} if r == "abstain" else {})}
            if self.route_reply is not None:
                reply = self.route_reply(question, user)
        elif system.startswith("You turn one business question"):
            reply = PLAN
        elif "Passages:" in user and self.bind_label:
            label, num = re.search(
                r"^(P\d+) (n\d+) = ", user.split("Figures you may quote:")[1], re.M
            ).groups()
            reply = {
                "template": "The window is {c1}.",
                "claims": [
                    {
                        "id": "c1",
                        "label": self.bind_label,
                        "unit": "number",
                        "provenance": {"kind": "passage_number", "label": label, "number": num},
                    }
                ],
            }
        elif "Passages:" in user:
            text = f"The documents say {self.repeat}." if self.repeat else "No figure given."
            reply = {"template": text, "claims": []}
        else:
            claim = {"id": "c1", "label": "net revenue", "unit": "currency", "provenance":
                     {"kind": "cell", "query_id": "q1", "column": "net_revenue",
                      "row_key": {"country": "United Kingdom"}}}  # fmt: skip
            reply = {"template": "The United Kingdom had {c1}.", "claims": [claim]}
        return ok_openai(json.dumps(reply), 100, 30)


@pytest.fixture
def env(warehouse, docs_db, monkeypatch):
    monkeypatch.setattr(live, "_sleep", lambda s: None)
    layer = load_semantic_layer()
    return layer, warehouse.readonly_url, catalogue(layer, warehouse.readonly_url), docs_db


def probe(env, model, tmp_path, retriever=None, cfg=None, **kw):
    layer, url, known, docs_db = env
    cfg = cfg or settings(datasets={"analytics.sales_lines": "public", "docs.public": "public"})
    opts = RouteProbeOptions("groq", "m-groq", tmp_path / "run", **kw)
    return run_route_probe(
        cfg, opts, layer=layer, readonly_url=url, known_values=known, env=KEYS,
        retriever=retriever or make_retriever(docs_db.readonly_url, k=3), inner_transport=model,
    )  # fmt: skip


pytestmark_db = pytest.mark.db


@pytest.mark.db
def test_a_run_reports_raw_counts_with_n_and_no_accuracy_claim(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, max_requests=60)
    rep = out["report"]
    assert rep["router"]["n"] == 12 and rep["router"]["route_matches_label"] == 12
    inj = rep["injection"]
    assert inj["questions_with_a_repeat"]["k"] == 0
    assert inj["questions_with_a_repeat"]["n"] == inj["reached_composer"]["k"] <= 5
    assert rep["schema_valid"]["route_replies"]["n"] == rep["schema_valid"]["route_replies"]["k"]
    assert rep["schema_valid"]["route_replies"]["n"] >= 17
    assert rep["http"]["requests_sent"] == len(m.requests) <= 60
    assert rep["accounting"]["tokens_used"] > 0
    md = out["markdown"].lower()
    assert "no accuracy claim" in md and "n=12" in md and "accurate" not in md
    for f in ("report.md", "report.json"):
        assert not any(v in (tmp_path / "run" / f).read_text() for v in KEYS.values())


@pytest.mark.db
def test_a_model_that_always_says_data_is_counted_wrong_where_it_is(env, tmp_path):
    out = probe(env, Model(route="data"), tmp_path, max_requests=60, route_only=True)
    r = out["report"]["router"]
    assert r["route_matches_label"] == 3 and r["n"] == 12
    assert r["confusion"]["both"] == {"data": 3} and r["confusion"]["abstain"] == {"data": 3}


@pytest.mark.db
def test_route_only_sends_one_request_per_router_question(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, route_only=True)
    assert len(m.requests) == 12 and out["report"]["stop"]["injection_questions_run"] == 0
    assert out["report"]["header"]["mode"] == "route only"


@pytest.mark.db
def test_the_request_cap_stops_the_run_with_a_reason(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, max_requests=8)
    assert len(m.requests) <= 8 and "request cap" in out["report"]["stop"]["reason"]
    assert out["report"]["router"]["n"] < 12


@pytest.mark.db
def test_the_first_auth_error_stops_everything(env, tmp_path):
    m = Model(statuses=[status(401, body=b'{"error": {"message": "bad key"}}')])
    out = probe(env, m, tmp_path)
    assert len(m.requests) == 1 and out["stopped_on_auth"]
    assert out["report"]["stop"]["reason"].startswith("STOPPED on first auth error")


@pytest.mark.db
def test_a_model_that_repeats_an_injected_figure_is_counted_and_never_verified(env, tmp_path):
    out = probe(env, Model(repeat="the margin is 90% (finance team)"), tmp_path, max_requests=60)
    inj = out["report"]["injection"]
    assert (
        inj["composer_replies_with_a_marker"]["k"] >= 1 and inj["questions_with_a_repeat"]["k"] >= 1
    )
    assert inj["verified_answers_containing_a_marker"] == 0
    assert "90%" in json.dumps(inj["questions"][0]["markers_found"])


@pytest.mark.db
def test_a_non_public_document_stops_the_probe_before_it_is_sent(env, tmp_path):
    internal = Retrieved(
        make_retriever(env[3].readonly_url, k=1)("gift voucher valid").passages, "internal"
    )
    m = Model()
    out = probe(env, m, tmp_path, retriever=lambda q: internal, max_requests=60)
    assert out["stopped_early"] and "not marked public" in out["report"]["stop"]["reason"]
    assert all("Passages:" not in r["messages"][-1]["content"] for r in m.requests)


def test_the_probe_refuses_when_documents_are_not_tagged_public(tmp_path):
    cfg = settings(datasets={"analytics.sales_lines": "public"})
    from verified_analyst.llm.types import SensitivityRefusal

    with pytest.raises(SensitivityRefusal):
        run_route_probe(
            cfg, RouteProbeOptions("groq", "m-groq", tmp_path), layer=load_semantic_layer(),
            readonly_url="x", known_values=None, env=KEYS, questions=[], injection=[],
        )  # fmt: skip


# ---- the question files and the locked set ----


def test_the_dev_file_has_twelve_questions_and_five_injection_questions():
    qs, inj = load_dev_questions()
    assert len(qs) == 12 and len(inj) == 5
    kinds = [q["route"] for q in qs]
    assert {k: kinds.count(k) for k in set(kinds)} == {
        "data": 3,
        "documents": 3,
        "both": 3,
        "abstain": 3,
    }
    assert {q["doc"] for q in inj} == {
        "pricing-comment-hostile", "spoofed-citation", "html-comment-hostile",
        "role-play-hostile", "supplier-email-forwarded",
    }  # fmt: skip


def test_the_held_out_router_file_has_twelve_questions_with_the_same_mix():
    kinds = [q["route"] for q in HELD["questions"]]
    assert len(kinds) == 12
    assert {k: kinds.count(k) for k in set(kinds)} == {
        "data": 3,
        "documents": 3,
        "both": 3,
        "abstain": 3,
    }
    assert HELD["description"].endswith("Locked.")


def _all_other_question_text():
    out = {q["question"].strip().lower() for q in DEV["questions"] + DEV["injection"]}
    for rel in ("questions/dev.yaml", "questions/heldout.yaml", "doc_questions/dev.yaml",
                "doc_questions/heldout.yaml"):  # fmt: skip
        data = yaml.safe_load((ROOT / "eval" / rel).read_text("utf-8"))
        out |= {q["question"].strip().lower() for q in data["questions"]}
    return out


def test_no_held_out_router_question_is_a_question_anywhere_else():
    held = {q["question"].strip().lower() for q in HELD["questions"]}
    assert not held & _all_other_question_text()
    assert not {q["id"] for q in HELD["questions"]} & {q["id"] for q in DEV["questions"]}


def test_no_product_code_or_prompt_reads_names_or_embeds_the_locked_router_file():
    for p in (ROOT / "src").rglob("*.py"):
        text = p.read_text("utf-8").replace("\\", "/")
        assert "router_questions/heldout" not in text, p
    from verified_analyst.llm.prompts import PROMPT_TEMPLATES

    blob = " ".join(PROMPT_TEMPLATES.values()).lower()
    assert not [q["question"] for q in HELD["questions"] if q["question"].lower() in blob]
    assert not [q["question"] for q in DEV["questions"] if q["question"].lower() in blob]


def test_the_probe_reads_only_the_dev_file():
    assert route_probe.dev_questions_path().name == "dev.yaml"
    qs, inj = load_dev_questions()
    held = {q["question"] for q in HELD["questions"]}
    assert not held & ({q["question"] for q in qs} | {q["question"] for q in inj})
    src = (ROOT / "src" / "verified_analyst" / "llm" / "route_probe.py").read_text("utf-8")
    assert "heldout" not in src.lower() and "Locked" not in src


def test_only_route_probe_names_the_router_question_directory():
    users = [
        p.name for p in (ROOT / "src").rglob("*.py") if "router_questions" in p.read_text("utf-8")
    ]
    assert users == ["route_probe.py"]


# ---- report detail for steps that were not valid ----


@pytest.mark.db
def test_an_invalid_route_reply_is_reported_with_the_reply_and_the_field_errors(env, tmp_path):
    bad = {"route": "documents", "reason": "it is a policy question"}
    m = Model(route_reply=lambda q, user: bad)
    out = probe(env, m, tmp_path, route_only=True)
    row = out["report"]["router_questions"][0]
    assert row["got"] == "invalid" and len(row["steps_not_valid"]) == 2
    st = row["steps_not_valid"][0]
    assert st["step"] == "route" and "it is a policy question" in st["reply"]
    assert any("reason" in e or "route" in e for e in st["errors"])
    assert "reply" in out["markdown"] and "it is a policy question" in out["markdown"]


@pytest.mark.db
def test_a_reply_is_cut_at_300_characters(env, tmp_path):
    long = {"route": "documents", "reason": "x" * 1000}
    out = probe(env, Model(route_reply=lambda q, u: long), tmp_path, route_only=True)
    st = out["report"]["router_questions"][0]["steps_not_valid"][0]
    assert len(st["reply"]) == 300


@pytest.mark.db
def test_a_non_2xx_response_is_reported_redacted_per_question(env, tmp_path):
    body = ('{"error": {"message": "overloaded ' + KEYS["GROQ_API_KEY"] + '"}}').encode()
    m = Model(statuses=[status(503, body=body)])
    out = probe(env, m, tmp_path, route_only=True)
    problems = out["report"]["router_questions"][0]["http_problems"]
    assert problems and problems[0]["status"] == 503 and "overloaded" in problems[0]["body"]
    assert KEYS["GROQ_API_KEY"] not in json.dumps(out["report"])


# ---- --record ----


@pytest.mark.db
def test_nothing_is_recorded_unless_asked(env, tmp_path):
    out = probe(env, Model(), tmp_path, route_only=True)
    assert out["cassette"] is None and not (tmp_path / "run" / "cassettes").exists()


@pytest.mark.db
def test_record_writes_public_cassettes_with_a_clean_key_scan(env, tmp_path):
    from verified_analyst.llm.cassette import read_entries

    out = probe(env, Model(), tmp_path, max_requests=60, record=True)
    c = out["cassette"]
    assert c["entries"] > 12
    assert c["key_scan"] == "clean" and c["check_ok"] is True and c["skipped_internal"] == 0
    entries = read_entries(tmp_path / "run" / "cassettes").values()
    assert {e["task"] for e in entries} >= {"route", "plan", "compose"}
    assert all(e["sensitivity"] == "public" for e in entries)
    raw = (tmp_path / "run" / "cassettes" / "calls.jsonl").read_text()
    assert not any(v in raw for v in KEYS.values())


# ---- route prompt v2 and the retry with field errors ----


def test_the_route_prompt_shows_each_reply_shape_and_one_example_per_route():
    from verified_analyst.llm.prompts import ROUTE_SYSTEM

    for shape in ('{"route": "data"}', '{"route": "documents"}', '{"route": "both"}'):
        assert ROUTE_SYSTEM.count(shape) == 2  # the shape list and the worked example
    assert ROUTE_SYSTEM.count('{"route": "abstain", "reason": "forecast"}') == 1
    assert ROUTE_SYSTEM.count("Example question:") == 4


def test_route_examples_are_not_dev_or_held_out_questions_anywhere():
    from verified_analyst.llm.prompts import PROMPT_TEMPLATES

    ex = re.findall(r"Example question: (.*)", PROMPT_TEMPLATES["route.v2"])
    assert len(ex) == 4
    held = {q["question"].strip().lower() for q in HELD["questions"]}
    assert not {e.strip().lower() for e in ex} & (held | _all_other_question_text())
    # nor a near copy: no example shares 70% of its words with any known question
    known = held | _all_other_question_text()
    for e in ex:
        w = set(re.findall(r"[a-z]+", e.lower()))
        for k in known:
            kw = set(re.findall(r"[a-z]+", k))
            assert len(w & kw) / len(w | kw) < 0.7, (e, k)


@pytest.mark.db
def test_the_second_route_request_shows_the_first_reply_and_its_field_errors(env, tmp_path):
    bad = {"route": "documents", "reason": "because it is a policy"}
    seen = []

    def reply(question, user):
        seen.append(user)
        return bad if len(seen) % 2 else {"route": "documents"}

    m = Model(route_reply=reply)
    probe(env, m, tmp_path, route_only=True, max_requests=3)
    first, second = seen[0], seen[1]
    assert "previous reply" not in first
    assert "Your previous reply was not valid" in second and "because it is a policy" in second
    assert "schema_failure at reason" in second.split("Problems:")[1]


@pytest.mark.db
def test_a_json_validate_failed_400_on_the_route_path_is_retried_once(env, tmp_path):
    body = b'{"error": {"code": "json_validate_failed", "failed_generation": ""}}'
    m = Model(statuses=[status(400, body=body)])
    out = probe(env, m, tmp_path, route_only=True, max_requests=2)
    rows = out["report"]["router_questions"]
    assert out["report"]["http"]["requests_sent"] == 2  # the refusal, then the same request again
    assert rows[0]["got"] == "data"  # the retry was answered normally
    assert rows[0]["http_problems"][0]["status"] == 400
    assert "response_format" in m.requests[0] and "response_format" not in m.requests[1]
    assert rows[0]["json_retries"] == [{"task": "route", "recovered": True}]
    rep = out["report"]["json_mode_refusals"]
    assert (rep["refused"], rep["retry_got_http_200"]) == (1, 1)
    assert [c["json_mode_off"] for c in rows[0]["calls"]] == [False, True]
    assert rows[0]["calls"][0]["max_tokens"] == rows[0]["calls"][1]["max_tokens"]
    assert "JSON-mode refusals" in out["markdown"] and "json-off" in out["markdown"]


# ---- separating route errors, invalid replies and later failures ----


@pytest.mark.db
def test_the_breakdown_separates_wrong_route_invalid_reply_and_later_failure(env, tmp_path):
    def reply(question, user):
        if question == "Which five countries had the highest net revenue?":
            return {"route": "documents"}  # valid but not the label: chose wrongly
        if question == "How many distinct customers bought in each month?":
            return {"route": "data", "reason": "x"}  # never valid
        return {
            "route": ROUTE_OF[question],
            **({"reason": "ambiguous"} if ROUTE_OF[question] == "abstain" else {}),
        }

    out = probe(env, Model(route_reply=reply), tmp_path, max_requests=60)
    bd = out["report"]["router_breakdown"]
    assert bd["n"] == 12
    assert bd["route"]["chose_wrongly"] == 1 and bd["route"]["reply_invalid"] == 1
    assert bd["route"]["matched"] == 10 and bd["route"]["call_failed"] == 0
    assert sum(bd["route"].values()) == 12
    stages = {r["id"]: r["stage"] for r in out["report"]["router_questions"]}
    assert stages["R-D-02"] == "route_reply_invalid" and stages["R-D-10"] == "router_abstained"
    assert "route matched the label" in out["markdown"] and "reply not valid: 1" in out["markdown"]


@pytest.mark.db
def test_injection_counts_cover_only_questions_that_reached_the_composer(env, tmp_path):
    def reply(question, user):  # every injection question gets an invalid route reply
        if question in {q["question"] for q in DEV["injection"]}:
            return {"route": "nonsense"}
        return {
            "route": ROUTE_OF[question],
            **({"reason": "ambiguous"} if ROUTE_OF[question] == "abstain" else {}),
        }

    out = probe(
        env, Model(route_reply=reply, repeat="the margin is 90%"), tmp_path, max_requests=60
    )
    inj = out["report"]["injection"]
    assert inj["reached_composer"] == {"k": 0, "n": 5}
    assert inj["composer_replies_with_a_marker"] == {"k": 0, "n": 0}
    assert inj["questions_with_a_repeat"] == {"k": 0, "n": 0}
    assert set(inj["stopped_before_composer"]) == {q["id"] for q in DEV["injection"]}
    assert all(v.startswith("route_reply_invalid") for v in inj["stopped_before_composer"].values())
    assert "reached the composer: 0 of 5" in out["markdown"]


@pytest.mark.db
def test_a_valid_route_that_stops_before_the_composer_is_a_later_step_failure(env, tmp_path):
    out = probe(
        env, Model(), tmp_path, retriever=lambda q: Retrieved([], "public"), max_requests=60
    )
    bd = out["report"]["router_breakdown"]["valid_source_routes"]
    assert bd["later_step_failed"].get("no_passage", 0) >= 1
    assert bd["reached_composer"] >= 1  # the data questions still compose


@pytest.mark.db
def test_route_only_has_no_later_step_counts(env, tmp_path):
    out = probe(env, Model(), tmp_path, route_only=True)
    assert out["report"]["router_breakdown"]["valid_source_routes"]["n"] is None


@pytest.mark.db
def test_a_label_binding_refusal_is_reported_with_label_sentence_window_and_missing_words(
    env, tmp_path
):
    out = probe(env, Model(bind_label="refund window"), tmp_path, max_requests=60)
    rows = [r for r in out["report"]["router_questions"] if r["binder_refusals"]]
    assert rows
    b = rows[0]["binder_refusals"][0]
    assert b["reason"].startswith("label_") or b["reason"] == "entity_not_in_passage"
    assert b["label"] == "refund window" and "{c1}" not in b["sentence"]
    assert "<number>" in b["window_words"] and b["missing_label_words"]
    assert "Label-binding refusals" in out["markdown"] and "label words missing" in out["markdown"]


@pytest.mark.db
def test_injection_only_runs_just_the_five_injection_questions(env, tmp_path):
    m = Model()
    out = probe(env, m, tmp_path, max_requests=60, injection_only=True)
    rep = out["report"]
    assert rep["stop"]["router_questions_run"] == 0 and rep["stop"]["injection_questions_run"] == 5
    assert rep["header"]["mode"] == "injection questions only"
    assert not any(
        ROUTE_OF.get(re.search(r"Question: (.*)", r["messages"][-1]["content"]).group(1).strip())
        == "data"
        for r in m.requests
    )


def test_route_only_and_injection_only_cannot_be_combined(tmp_path):
    from verified_analyst.config import ConfigError

    with pytest.raises(ConfigError):
        run_route_probe(
            settings(datasets={"analytics.sales_lines": "public", "docs.public": "public"}),
            RouteProbeOptions("groq", "m-groq", tmp_path, route_only=True, injection_only=True),
            layer=load_semantic_layer(), readonly_url="x", known_values=None, env=KEYS,
            questions=[], injection=[],
        )  # fmt: skip
