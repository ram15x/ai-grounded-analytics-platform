"""route -> (plan/execute) and/or (retrieve passages) -> compose -> verify (slice 2e).

The router chooses data, documents or both, or abstains. Everything after it is the existing
pipeline plus passages: data figures come from verified cells, document figures from handles the
system lists from stored text (documents/numbers.py), and both are written by the renderer, never
by the model. The route only decides which sources are consulted. It cannot loosen any check:
a flagged passage is withheld from the prompt and cannot be cited, and the sensitivity rule is
applied to every request that carries passage text. Not tuned for a real model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..documents.fusion import Search
from ..documents.numbers import PassageNumbers
from ..documents.store import document_sensitivities, search
from ..documents.untrusted import PassageIn, render_passages
from ..semantic import SemanticLayer
from .accounting import RunContext
from .pipeline import REJECTED_TEXT, REPLY_CHARS, QuestionRun, _json, run_question, schema_problems
from .prompts import route_request
from .router import Router
from .schemas import RouteOutput
from .types import (
    JSON_REJECTED,
    AllProvidersFailed,
    BudgetExceeded,
    CassetteMiss,
    SensitivityRefusal,
)

ROUTE_ATTEMPTS = 2


@dataclass(frozen=True)
class Retrieved:
    """What a retriever returns: passages, and the sensitivity of the documents they came from
    ("public" only if every one is public; anything else is internal)."""

    passages: list[PassageIn]
    sensitivity: str = "internal"
    pages: dict[str, str] = field(default_factory=dict)  # passage id -> "p. 3" (PDFs only)


Retriever = Callable[[str], Retrieved]


def docs_dataset(sensitivity: str) -> str:
    """The dataset tag the router resolves: docs.public is tagged public in router.yaml and
    docs.internal is not tagged, so hosted providers refuse it."""
    return "docs.public" if sensitivity == "public" else "docs.internal"


def page_label(first: int | None, last: int | None) -> str:
    if not first:
        return ""
    return f"p. {first}" if not last or last == first else f"pp. {first}-{last}"


def make_retriever(readonly_url: str, k: int = 3, search_fn: Search | None = None) -> Retriever:
    """Top-k passages for a question (keyword search unless `search_fn` is given), with the
    sensitivity of their documents read from the store. A document with no recorded sensitivity
    counts as internal."""

    def retrieve(question: str) -> Retrieved:
        hits = search_fn(question, k) if search_fn else search(readonly_url, question, k)
        sens = document_sensitivities(readonly_url, [h.doc_key for h in hits])
        public = bool(hits) and all(sens.get(h.doc_key) == "public" for h in hits)
        return Retrieved(
            [PassageIn(h.passage_id, h.text) for h in hits],
            "public" if public else "internal",
            {h.passage_id: page_label(h.page_first, h.page_last) for h in hits if h.page_first},
        )

    return retrieve


def decide_route(
    question: str,
    *,
    layer: SemanticLayer,
    router: Router,
    run: RunContext,
    corpus_description: str,
    max_tokens: int | None = None,
) -> tuple[RouteOutput | None, list[dict[str, Any]], str]:
    """Ask the router model, with one more try if its reply is not a valid route; the second
    request shows the first reply and its field-level errors, as the repair prompt does. Returns the
    route (None when it gave no valid one), one event per reply, and the abstain reason to use
    when the route is None ("route_failed", or "budget", "sensitivity", "provider_error")."""
    events: list[dict[str, Any]] = []
    previous: str | None = None
    problems: list[str] = []
    for _ in range(ROUTE_ATTEMPTS):
        req = route_request(layer, question, corpus_description, previous, problems)
        if max_tokens:
            req = dataclasses.replace(req, max_tokens=max_tokens)
        try:
            got = router.call(req, run)
        except BudgetExceeded:
            return None, events, "budget"
        except SensitivityRefusal:
            return None, events, "sensitivity"
        except (AllProvidersFailed, CassetteMiss):
            return None, events, "provider_error"
        ev: dict[str, Any] = {"step": "route", "json_ok": False, "schema_ok": False}
        events.append(ev)
        try:
            if got.finish_reason == JSON_REJECTED:
                raise ValueError(REJECTED_TEXT)
            raw = _json(got.text)
            ev["json_ok"] = True
            route = RouteOutput.model_validate(raw)
            ev["schema_ok"] = True
            ev["route"] = route.route
            return route, events, ""
        except (ValueError, ValidationError) as exc:
            problems = schema_problems(exc)
            previous = got.text[:REPLY_CHARS]
            ev["reply"], ev["error"] = previous, problems
            continue
    return None, events, "route_failed"


def answer_question(
    question: str,
    *,
    layer: SemanticLayer,
    readonly_url: str,
    router: Router,
    run: RunContext,
    retrieve: Retriever,
    corpus_description: str,
    known_values: dict[str, set[str]] | None = None,
    max_repairs: int = 2,
    max_tokens: int | None = None,
    plan_prompt: str = "plan.v1",
) -> QuestionRun:
    result = QuestionRun(question, "abstain", run=run)

    def stop(reason: str, source: str = "pipeline") -> QuestionRun:
        result.outcome, result.reason = "abstain", reason
        last = result.events[-1]["step"] if result.events else None
        result.abstain_detail = {"source": source, "reason": reason, "last_model_step": last,
                                 "planner_reply": None}  # fmt: skip
        return result

    route, events, why = decide_route(
        question, layer=layer, router=router, run=run, corpus_description=corpus_description,
        max_tokens=max_tokens,
    )  # fmt: skip
    result.events.extend(events)
    if route is None:
        return stop(why)
    if route.route == "abstain":
        return stop(route.reason or "ambiguous", source="router")
    result.route = route.route

    kwargs: dict[str, Any] = {}
    if route.route in ("documents", "both"):
        found = retrieve(question)
        passages = list(found.passages)
        if not passages:
            return stop("no_passage")
        rendered = render_passages(passages)
        pn = PassageNumbers(passages)
        if not any(pn.by_label.values()):
            # nothing the composer could quote: no figure, or every figure sits in withheld text
            return stop("no_passage")
        result.passage_labels = dict(rendered.labels)
        kwargs = {
            "passages": pn,
            "passage_prompt": (rendered.text, pn.listing()),
            "doc_datasets": (docs_dataset(found.sensitivity),),
        }
    done = run_question(
        question,
        layer=layer,
        readonly_url=readonly_url,
        router=router,
        run=run,
        known_values=known_values,
        max_repairs=max_repairs,
        max_tokens=max_tokens,
        skip_plan=route.route == "documents",
        prior_events=result.events,
        plan_prompt=plan_prompt,
        **kwargs,
    )
    done.route, done.passage_labels = result.route, result.passage_labels
    if kwargs:
        done.evidence["passages"] = {
            label: {"id": pid, "text": kwargs["passages"].texts.get(pid, "")}
            | ({"page": found.pages[pid]} if found.pages.get(pid) else {})
            for label, pid in result.passage_labels.items()
        }
    return done
