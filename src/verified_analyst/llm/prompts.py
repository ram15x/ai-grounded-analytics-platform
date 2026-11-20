"""Prompt templates. plan.v0 is the untuned minimum, kept so cassettes recorded with it still
check. plan.v1 adds metric and dimension descriptions and synonyms to the catalogue, taken from
the semantic layer, after the second Groq probe's "ambiguous" abstain on a merchandise-units
question. compose.v1 and repair.v1 replaced v0 after the first Groq probe (2026-11-20), where
every v0 compose/repair answer failed the schema: they state the exact claim and provenance
shapes, the allowed units and the row_key format, and carry one worked example on a made-up
dataset. All are first drafts shaped by single runs, not tuned or proven. Any change to a
template changes its hash and so invalidates every cassette recorded with it."""

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
COMPOSE_SYSTEM = """\
You write a short answer to a business question as one JSON object and nothing else.
Shape: {"template": "...", "claims": [CLAIM, ...]}

The system writes every figure, not you. Never type a digit or a number word for a figure in \
the template. Put a placeholder in braces where each figure belongs, such as {c1}, {c2}. The \
text inside the braces is the claim's id. An id starts with a letter and then has letters, \
digits or underscores only (c1, total_sales). In the template write the id inside braces; in \
the claim write the id without braces. The placeholder is replaced by the figure together with \
its unit sign (a currency symbol, a % sign), so do not write GBP, a currency symbol or % next \
to it. In the same sentence say which row, group or period the figure is for. When one \
sentence has figures for different rows, put a comma or a full stop between them; "and" alone \
is not enough to tell the system which name goes with which figure.

CLAIM = {"id": "c1", "label": "what the number measures, in words", "unit": UNIT, \
"provenance": PROVENANCE}
Optional claim fields: "decimals" (0 to 6) and "scale" ("none", "thousand", "million", \
"billion"; scale only for currency and count). No other fields are accepted.

UNIT is exactly one of these words: "currency" (money), "count" (a whole number of things), \
"percent" (shown with a % sign), "percentage_points" (a gap between two percentages), "ratio" \
(a fraction shown as is, such as 0.1234), "number" (any other number). A currency code such as \
GBP is not a unit.

PROVENANCE is exactly one of:
{"kind": "cell", "query_id": "q1", "column": "<a column of the result table>", "row_key": {...}}
{"kind": "row_count", "query_id": "q1"}
{"kind": "derived", "op": OP, "args": [ARG, ...]}
For derived, each ARG is a cell or a row_count provenance written as above, never a typed \
number. OP is one of: sum, mean, min, max, count (any number of args); difference (first minus \
second); ratio (first divided by second); percent_of (first as a percent of second); \
percent_change (change from first to second, as a percent of the first).
row_key chooses the row. It is an object whose keys are column names of the result table and \
whose values are copied exactly from that row, for example {"country": "France"}. It is never a \
row position: "row_0" and "0" are wrong. When the result has only one row, use {}. Name enough \
columns to match exactly one row. Use only query_id "q1" and only columns that appear in the \
result tables.

Example, on a made-up dataset unrelated to this one (do not copy its names or figures).
Question: How many support tickets did each region close, and how do the two compare?
Result tables: {"q1": {"columns": ["region", "tickets_closed"], \
"rows": [["North", "40"], ["South", "10"]]}}
Answer: {"template": "The North region closed {c1} tickets, and the South region closed {c2}. \
The North's closed tickets were {c3} of the South's.", "claims": [\
{"id": "c1", "label": "tickets closed by the North region", "unit": "count", \
"provenance": {"kind": "cell", "query_id": "q1", "column": "tickets_closed", \
"row_key": {"region": "North"}}}, \
{"id": "c2", "label": "tickets closed by the South region", "unit": "count", \
"provenance": {"kind": "cell", "query_id": "q1", "column": "tickets_closed", \
"row_key": {"region": "South"}}}, \
{"id": "c3", "label": "North tickets closed as a percent of South", "unit": "percent", \
"provenance": {"kind": "derived", "op": "percent_of", "args": [\
{"kind": "cell", "query_id": "q1", "column": "tickets_closed", "row_key": {"region": "North"}}, \
{"kind": "cell", "query_id": "q1", "column": "tickets_closed", "row_key": {"region": "South"}}\
]}}]}"""
PLAN_SYSTEM_V1 = (
    PLAN_SYSTEM
    + "\nEach metric and dimension in the catalogue has a description and a list of synonyms: "
    "match the words of the question against the names, descriptions and synonyms."
)
REPAIR_SYSTEM = (
    COMPOSE_SYSTEM
    + """

Your previous answer had the problems listed after "Problems:". Each one names the field path \
that is wrong (for example claims.0.unit) and what is wrong with it. Return the whole corrected \
JSON object, in the same shape, changing only what the problems require."""
)

# plan.v0 stays so cassettes recorded with it still check; the pipeline now uses plan.v1.
PROMPT_TEMPLATES = {
    "plan.v0": PLAN_SYSTEM,
    "plan.v1": PLAN_SYSTEM_V1,
    "compose.v1": COMPOSE_SYSTEM,
    "repair.v1": REPAIR_SYSTEM,
}
ACTIVE_PROMPT_IDS = ("plan.v1", "compose.v1", "repair.v1")


def prompt_hashes() -> dict[str, str]:
    return {pid: sha256_text(text) for pid, text in PROMPT_TEMPLATES.items()}


def active_prompt_hashes() -> dict[str, str]:
    """The prompts the pipeline sends today (what a report should name)."""
    return {pid: sha256_text(PROMPT_TEMPLATES[pid]) for pid in ACTIVE_PROMPT_IDS}


def catalogue_text_v0(layer: SemanticLayer) -> str:
    """The plan.v0 catalogue: names, units and allowed dimensions only. Frozen."""
    metrics = {n: {"unit": m.unit, "dimensions": list(m.allowed_dimensions)}
               for n, m in layer.metrics.items()}  # fmt: skip
    return json.dumps({"metrics": metrics, "dimensions": sorted(layer.dimensions)}, sort_keys=True)


def catalogue_text(layer: SemanticLayer) -> str:
    """The plan.v1 catalogue: every metric and dimension with its description and synonyms,
    taken word for word from the semantic layer (nothing is written for the prompt). SQL
    expressions and column names are not included. Built from the layer object only."""
    metrics = {
        n: {
            "unit": m.unit,
            "description": m.description,
            "synonyms": list(m.synonyms),
            "dimensions": list(m.allowed_dimensions),
        }
        for n, m in layer.metrics.items()
    }
    dims = {
        n: {"description": d.description, "synonyms": list(d.synonyms)}
        for n, d in layer.dimensions.items()
    }
    return json.dumps({"metrics": metrics, "dimensions": dims}, sort_keys=True)


def plan_request(layer: SemanticLayer, question: str, error: str | None = None) -> LLMRequest:
    user = f"Catalogue: {catalogue_text(layer)}\nQuestion: {question}"
    if error:
        user += f"\nYour previous plan failed: {error}\nReturn a corrected plan."
    return LLMRequest(
        task="plan",
        messages=(("system", PLAN_SYSTEM_V1), ("user", user)),
        datasets=(),  # the catalogue holds names and definitions, not data
        prompt_id="plan.v1",
        prompt_hash=prompt_hashes()["plan.v1"],
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
    pid = "repair.v1" if repair else "compose.v1"
    return LLMRequest(
        task="repair" if repair else "compose",
        messages=(("system", REPAIR_SYSTEM if repair else COMPOSE_SYSTEM), ("user", user)),
        datasets=datasets,  # the result tables are data from these relations
        prompt_id=pid,
        prompt_hash=prompt_hashes()[pid],
        schema_name="ComposeOutput",
    )
