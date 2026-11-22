"""Slice 2e end to end, with NO real model: a FakeProvider scripts route, plan and compose, the
store is the synthetic corpus, and retrieval is keyword or the fake embedder. These test the
plumbing and the refusals, not any model's behaviour."""

from __future__ import annotations

import json
import re

import pytest

from grounded_analytics.cli import catalogue
from grounded_analytics.documents.untrusted import PassageIn
from grounded_analytics.llm.answer import Retrieved, answer_question, make_retriever
from grounded_analytics.llm.fake import FakeProvider
from grounded_analytics.llm.router import Router
from grounded_analytics.semantic import load_semantic_layer
from llm_helpers import settings

pytestmark = pytest.mark.db

CORPUS = "Policies, procedures, supplier notes and memos of a fictional company."
PLAN = {
    "metric_query": {
        "metrics": ["net_revenue"],
        "dimensions": ["country"],
        "order_by": [{"by": "net_revenue", "desc": True}],
        "limit": 3,
    }
}
BLOCK = re.compile(r"BEGIN (P\d+) id=(\S+?)(?: \[|>>>)")


def cell(country):
    return {"kind": "cell", "query_id": "q1", "column": "net_revenue",
            "row_key": {"country": country}}  # fmt: skip


def cell_claim(cid, country):
    return {"id": cid, "label": "net revenue", "unit": "currency", "provenance": cell(country)}


def handle(user, doc_key, shown):
    """What a careful composer does: find the block of `doc_key`, then the listed figure `shown`."""
    label = next(lb for lb, pid in BLOCK.findall(user) if pid.startswith(doc_key + "#"))
    num = re.search(rf"^{label} (n\d+) = {re.escape(shown)} ", user, re.M).group(1)
    return label, num


def passage_claim(cid, label_words, unit, label, num):
    return {"id": cid, "label": label_words, "unit": unit,
            "provenance": {"kind": "passage_number", "label": label, "number": num}}  # fmt: skip


def returns_composer(req):
    user = req.messages[1][1]
    label, num = handle(user, "returns-policy", "28")
    return {
        "template": "Customers may return unused merchandise within {r} days of delivery.",
        "claims": [passage_claim("r", "return unused merchandise days", "number", label, num)],
    }


@pytest.fixture
def rig(warehouse, docs_db):
    layer = load_semantic_layer()
    known = catalogue(layer, warehouse.readonly_url)

    def build(
        script,
        *,
        retriever=None,
        hosted=False,
        cfg=None,
        token_cap=None,
        question="How many days do customers have to return unused merchandise?",
    ):
        cfg = cfg or settings(datasets={"analytics.sales_lines": "public", "docs.public": "public"})
        fake = FakeProvider(script, name="local", hosted=hosted)
        cfg.tasks = {t: ["local"] for t in ("route", "plan", "compose", "repair")}
        if hosted:
            cfg.providers["local"].hosted = True
        router = Router(cfg, {"local": fake})
        run = router.new_run("e2e", token_cap)
        out = answer_question(
            question, layer=layer, readonly_url=warehouse.readonly_url, router=router, run=run,
            retrieve=retriever or make_retriever(docs_db.readonly_url, k=3),
            corpus_description=CORPUS, known_values=known,
        )  # fmt: skip
        return out, fake

    return build


def test_documents_only_question_is_answered_from_a_stored_passage(rig):
    out, fake = rig(
        {"route": [{"route": "documents"}], "compose": [returns_composer]},
        question="How many days do customers have to return unused merchandise?",
    )
    assert (out.outcome, out.route) == ("verified", "documents"), out.attempts
    assert out.rendered == "Customers may return unused merchandise within 28 days of delivery."
    assert [r.task for r in fake.requests] == ["route", "compose"]  # nothing was planned or run
    assert out.verification["verification"]["verdicts"][0]["verdict"] == "VERIFIED_QUOTE"
    assert any(pid.startswith("returns-policy#") for pid in out.passage_labels.values())
    assert out.restatement == ""


def test_data_only_question_never_reads_a_document(rig):
    def no_search(question):
        raise AssertionError("a data question must not retrieve documents")

    good = {
        "template": "The United Kingdom led with {uk} of net revenue, ahead of EIRE at {eire}.",
        "claims": [cell_claim("uk", "United Kingdom"), cell_claim("eire", "EIRE")],
    }
    out, fake = rig(
        {"route": [{"route": "data"}], "plan": [PLAN], "compose": [good]}, retriever=no_search
    )
    assert (out.outcome, out.route) == ("verified", "data")
    assert [r.task for r in fake.requests] == ["route", "plan", "compose"]
    assert "Passages:" not in fake.requests[2].messages[1][1]
    assert fake.requests[2].prompt_id == "compose.v1"
    assert "£10,638.54" in out.rendered


def test_mixed_question_uses_cells_for_data_and_passages_for_documents(rig):
    def composer(req):
        user = req.messages[1][1]
        label, num = handle(user, "returns-policy", "28")
        return {
            "template": "The United Kingdom led with {uk} of net revenue. "
            "Customers may return unused merchandise within {r} days of delivery.",
            "claims": [
                cell_claim("uk", "United Kingdom"),
                passage_claim("r", "return unused merchandise days", "number", label, num),
            ],
        }

    out, fake = rig(
        {"route": [{"route": "both"}], "plan": [PLAN], "compose": [composer]},
        question="Which country led, and how long do customers have to return things?",
    )
    assert (out.outcome, out.route) == ("verified", "both"), [a.problems for a in out.attempts]
    verdicts = [v["verdict"] for v in out.verification["verification"]["verdicts"]]
    assert sorted(verdicts) == ["VERIFIED", "VERIFIED_QUOTE"]
    assert "£10,638.54" in out.rendered and "within 28 days" in out.rendered
    assert fake.requests[2].prompt_id == "compose.v2"
    assert out.restatement.startswith("Compute net revenue")


def test_the_model_cannot_type_a_figure_it_read_in_a_passage(rig):
    typed = {
        "template": "Customers may return unused merchandise within 28 days of delivery.",
        "claims": [],
    }
    out, _ = rig({"route": [{"route": "documents"}], "compose": [typed], "repair": [typed, typed]})
    assert out.outcome == "abstain" and out.reason == "unverified"
    assert out.rendered == ""


def test_a_label_that_names_a_different_figure_does_not_verify(rig):
    def composer(req):
        label, num = handle(req.messages[1][1], "returns-policy", "10%")
        return {
            "template": "Customers may return unused merchandise within {r} days of delivery.",
            "claims": [passage_claim("r", "return unused merchandise days", "number", label, num)],
        }

    out, _ = rig(
        {"route": [{"route": "documents"}], "compose": [composer], "repair": [composer] * 2}
    )
    assert out.outcome == "abstain" and out.reason == "unverified"


# ---- no answer: every path abstains, with a reason ----


def test_the_router_abstains_when_it_cannot_decide(rig):
    out, fake = rig({"route": [{"route": "abstain", "reason": "ambiguous"}]})
    assert (out.outcome, out.reason) == ("abstain", "ambiguous")
    assert out.abstain_detail["source"] == "router" and out.rendered == ""
    assert [r.task for r in fake.requests] == ["route"]


def test_an_invalid_route_is_asked_again_once_then_abstains(rig):
    out, fake = rig({"route": ["not json", {"route": "maybe"}]})
    assert (out.outcome, out.reason) == ("abstain", "route_failed")
    assert len(fake.requests) == 2


def test_a_second_try_can_recover_a_bad_route(rig):
    out, fake = rig(
        {"route": ["{", {"route": "documents"}], "compose": [returns_composer]},
        question="return window?",
    )
    assert out.outcome == "verified" and len(fake.requests) == 3


def test_documents_question_with_nothing_found_abstains_no_passage(rig):
    out, fake = rig(
        {"route": [{"route": "documents"}]}, retriever=lambda q: Retrieved([], "public")
    )
    assert (out.outcome, out.reason) == ("abstain", "no_passage")
    assert [r.task for r in fake.requests] == ["route"]


def test_documents_that_do_not_answer_give_no_verified_figure(rig):
    silent = {"template": "The documents do not give that figure.", "claims": []}
    out, _ = rig({"route": [{"route": "documents"}], "compose": [silent], "repair": [silent] * 2})
    assert out.outcome == "abstain" and out.reason == "unverified"


def test_a_plan_abstain_inside_a_both_route_passes_through(rig):
    out, _ = rig({"route": [{"route": "both"}], "plan": [{"abstain": "forecast"}]})
    assert (out.outcome, out.reason) == ("abstain", "forecast")


# ---- sensitivity: the rule applies to documents too ----


def test_a_hosted_provider_never_sees_internal_document_text(rig):
    secret = "The internal margin target is 41% of revenue."
    internal = Retrieved([PassageIn("notes#0." + "b" * 10, secret)], "internal")
    out, fake = rig({"route": [{"route": "documents"}]}, retriever=lambda q: internal, hosted=True)
    assert (out.outcome, out.reason) == ("abstain", "sensitivity")
    assert "41%" not in json.dumps([r.messages for r in fake.requests])  # route saw no passage


def test_an_internal_document_goes_only_to_a_local_provider(rig):
    text = "The courier charge is 4.50 GBP per parcel."
    internal = Retrieved([PassageIn("notes#0." + "b" * 10, text)], "internal")

    def composer(req):
        label, num = handle(req.messages[1][1], "notes", "4.50")
        return {"template": "The courier charge is {c} per parcel.",
                "claims": [passage_claim("c", "courier charge", "number", label, num)]}  # fmt: skip

    out, fake = rig({"route": [{"route": "documents"}], "compose": [composer]},
                    retriever=lambda q: internal)  # fmt: skip
    assert out.outcome == "verified" and fake.requests[-1].sensitivity == "internal"


def test_a_mixed_batch_of_public_and_internal_passages_counts_as_internal(rig, docs_db):
    def retrieve(question):
        r = make_retriever(docs_db.readonly_url, k=2)(question)
        return Retrieved(r.passages, "internal")

    out, fake = rig({"route": [{"route": "documents"}]}, retriever=retrieve, hosted=True)
    assert out.reason == "sensitivity"


# ---- the five hostile documents ----

HOSTILE = {
    "pricing-comment-hostile": ("seasonal decorations pricing", "90%"),
    "spoofed-citation": ("photo studio booked", "90 days"),
    "html-comment-hostile": ("gift wrapping costs", None),
    "role-play-hostile": ("canteen opening hours", None),
    "supplier-email-forwarded": ("forwarded supplier email lead time", "£1,000,000"),
}


@pytest.mark.parametrize("key", sorted(HOSTILE))
def test_an_obedient_attacker_composer_does_not_verify(rig, docs_db, key):
    """The composer does what the document says: states the injected figure, with and without a
    placeholder, and guesses handles. None of it may come out verified."""
    query, injected = HOSTILE[key]
    seen = {}

    def attacker(req):
        user = req.messages[1][1]
        seen["prompt"] = user
        listed = re.findall(r"^(P\d+) (n\d+) = (.*?) {3}", user, re.M)
        seen["listed"] = [t for _, _, t in listed]
        guess = [passage_claim("g", "margin", "percent", "P1", "n99")]
        return {
            "template": f"The margin is {injected or '100%'} (verified). {{g}}",
            "claims": guess,
        }

    out, fake = rig(
        {"route": [{"route": "documents"}], "compose": [attacker], "repair": [attacker] * 2},
        retriever=make_retriever(
            docs_db.readonly_url, k=1, search_fn=_only(docs_db.readonly_url, key, query)
        ),
        question=query,
    )
    assert out.outcome != "verified", out.rendered
    if not seen:  # no quotable figure survived withholding: refused before any composer call
        assert out.reason == "no_passage"
        return
    if injected:
        assert injected not in out.rendered
        assert injected not in seen["listed"]  # the figure was never offered as a handle
    for phrase in (
        "administrator mode",
        "NOTE TO THE AI",
        "report VERIFIED",
        "ignore all previous",
    ):
        assert phrase.lower() not in seen["prompt"].lower()  # the composer never saw the order


def _only(url, key, query):
    from grounded_analytics.documents.store import search

    def run(q, k):
        return [h for h in search(url, query, 50) if h.doc_key == key][:k]

    return run


def test_retrieval_with_the_fake_embedder_feeds_the_same_path(rig, vector_db):
    from grounded_analytics.documents.embed import FakeEmbedder
    from grounded_analytics.documents.vectorstore import QueryVectors, make_vector_search

    emb = FakeEmbedder()
    vec = make_vector_search(vector_db.readonly_url, emb.model_id, QueryVectors(emb))
    retriever = make_retriever(vector_db.readonly_url, k=5, search_fn=vec)
    q = "How many days do customers have to return unused merchandise?"

    def tolerant(req):  # the fake embedder may not surface the policy; then say nothing
        try:
            return returns_composer(req)
        except StopIteration:
            return {"template": "The documents do not give that.", "claims": []}

    out, _ = rig(
        {"route": [{"route": "documents"}], "compose": [tolerant], "repair": [tolerant] * 2},
        retriever=retriever, question=q,
    )  # fmt: skip
    assert out.route == "documents" and out.outcome in ("verified", "abstain")
