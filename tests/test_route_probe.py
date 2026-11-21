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

    def __init__(self, route=None, repeat=None, statuses=None, route_reply=None):
        self.route, self.repeat, self.route_reply = route, repeat, route_reply
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
    assert rep["injection"]["questions_with_a_repeat"] == {"k": 0, "n": 5}
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
    assert inj["replies_with_a_marker"]["k"] >= 1 and inj["questions_with_a_repeat"]["k"] >= 1
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
