"""MINIMAL prompt templates, enough to drive the scripted (fake) pipeline test.

These are NOT tuned and have never been sent to a real model. Any change to a template changes
its hash and therefore invalidates every cassette recorded with it."""

from __future__ import annotations

import json
from typing import Any

from ..semantic import SemanticLayer
from .cassette import sha256_text
from .schemas import ABSTAIN_REASONS
from .types import LLMRequest

PLAN_SYSTEM = (
    "You turn one business question into JSON. Reply with a single JSON object and nothing else.\n"
    'Either {"metric_query": {"metrics": [...], "dimensions": [...], "filters": [...], '
    '"order_by": [{"by": "...", "desc": true}], "limit": N}}\n'
    'or {"abstain": "<reason>"} where reason is one of: ' + ", ".join(ABSTAIN_REASONS) + ".\n"
    "Use only the metrics and dimensions in the catalogue."
)
COMPOSE_SYSTEM = (
    "You write a short answer as JSON: "
    '{"template": "...", "claims": [{"id": "...", "label": "...", "unit": "...", '
    '"provenance": {...}}]}.\n'
    "Never type a number. Write {id} placeholders in the template, one per figure, and describe "
    "the right row or period in the same sentence. Each claim names a cell of the result "
    '(query_id "q1", column, row_key), a derived value, or a row count.'
)
REPAIR_SYSTEM = (
    COMPOSE_SYSTEM + "\nYour previous answer had the problems listed. Return a corrected object."
)

PROMPT_TEMPLATES = {
    "plan.v0": PLAN_SYSTEM,
    "compose.v0": COMPOSE_SYSTEM,
    "repair.v0": REPAIR_SYSTEM,
}


def prompt_hashes() -> dict[str, str]:
    return {pid: sha256_text(text) for pid, text in PROMPT_TEMPLATES.items()}


def catalogue_text(layer: SemanticLayer) -> str:
    metrics = {n: {"unit": m.unit, "dimensions": list(m.allowed_dimensions)}
               for n, m in layer.metrics.items()}  # fmt: skip
    return json.dumps({"metrics": metrics, "dimensions": sorted(layer.dimensions)}, sort_keys=True)


def plan_request(layer: SemanticLayer, question: str, error: str | None = None) -> LLMRequest:
    user = f"Catalogue: {catalogue_text(layer)}\nQuestion: {question}"
    if error:
        user += f"\nYour previous plan failed: {error}\nReturn a corrected plan."
    return LLMRequest(
        task="plan",
        messages=(("system", PLAN_SYSTEM), ("user", user)),
        datasets=(),  # the catalogue holds names, not data
        prompt_id="plan.v0",
        prompt_hash=prompt_hashes()["plan.v0"],
        schema_name="PlanOutput",
    )


def tables_text(executed: dict[str, Any], limit_rows: int = 50) -> str:
    out = {}
    for qid, q in executed.items():
        out[qid] = {
            "columns": q.columns,
            "rows": [[None if v is None else str(v) for v in row] for row in q.rows[:limit_rows]],
        }
    return json.dumps(out, sort_keys=True)


def compose_request(
    question: str,
    restatement: str,
    executed: dict[str, Any],
    datasets: tuple[str, ...],
    previous: str | None = None,
    problems: list[str] | None = None,
) -> LLMRequest:
    user = (
        f"Question: {question}\nQuery, restated: {restatement}\n"
        f"Result tables: {tables_text(executed)}"
    )
    repair = previous is not None
    if repair:
        user += f"\nPrevious answer: {previous}\nProblems: {json.dumps(problems or [])}"
    pid = "repair.v0" if repair else "compose.v0"
    return LLMRequest(
        task="repair" if repair else "compose",
        messages=(("system", REPAIR_SYSTEM if repair else COMPOSE_SYSTEM), ("user", user)),
        datasets=datasets,  # the result tables are data from these relations
        prompt_id=pid,
        prompt_hash=prompt_hashes()[pid],
        schema_name="ComposeOutput",
    )
