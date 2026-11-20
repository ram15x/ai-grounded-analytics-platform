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
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..documents.fusion import Search
from ..documents.numbers import PassageNumbers
from ..documents.store import document_sensitivities, search
from ..documents.untrusted import PassageIn, render_passages
from ..semantic import SemanticLayer
from .accounting import RunContext
from .pipeline import REJECTED_TEXT, QuestionRun, _json, run_question
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


Retriever = Callable[[str], Retrieved]


def docs_dataset(sensitivity: str) -> str:
    """The dataset tag the router resolves: docs.public is tagged public in router.yaml and
    docs.internal is not tagged, so hosted providers refuse it."""
    return "docs.public" if sensitivity == "public" else "docs.internal"


def make_retriever(readonly_url: str, k: int = 3, search_fn: Search | None = None) -> Retriever:
    """Top-k passages for a question (keyword search unless `search_fn` is given), with the
    sensitivity of their documents read from the store. A document with no recorded sensitivity
    counts as internal."""

    def retrieve(question: str) -> Retrieved:
        hits = search_fn(question, k) if search_fn else search(readonly_url, question, k)
        sens = document_sensitivities(readonly_url, [h.doc_key for h in hits])
        public = bool(hits) and all(sens.get(h.doc_key) == "public" for h in hits)
        return Retrieved(
            [PassageIn(h.passage_id, h.text) for h in hits], "public" if public else "internal"
        )

    return retrieve


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
) -> QuestionRun:
    result = QuestionRun(question, "abstain", run=run)

    def stop(reason: str, source: str = "pipeline") -> QuestionRun:
        result.outcome, result.reason = "abstain", reason
        last = result.events[-1]["step"] if result.events else None
        result.abstain_detail = {"source": source, "reason": reason, "last_model_step": last,
                                 "planner_reply": None}  # fmt: skip
        return result

    route: RouteOutput | None = None
    for _ in range(ROUTE_ATTEMPTS):
        req = route_request(layer, question, corpus_description)
        if max_tokens:
            req = dataclasses.replace(req, max_tokens=max_tokens)
        try:
            got = router.call(req, run)
        except BudgetExceeded:
            return stop("budget")
        except SensitivityRefusal:
            return stop("sensitivity")
        except (AllProvidersFailed, CassetteMiss):
            return stop("provider_error")
        ev: dict[str, Any] = {"step": "route", "json_ok": False, "schema_ok": False}
        result.events.append(ev)
        try:
            if got.finish_reason == JSON_REJECTED:
                raise ValueError(REJECTED_TEXT)
            raw = _json(got.text)
            ev["json_ok"] = True
            route = RouteOutput.model_validate(raw)
            ev["schema_ok"] = True
            ev["route"] = route.route
            break
        except (ValueError, ValidationError):
            continue
    if route is None:
        return stop("route_failed")
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
        **kwargs,
    )
    done.route, done.passage_labels = result.route, result.passage_labels
    return done
