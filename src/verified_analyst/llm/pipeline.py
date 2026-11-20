"""plan -> compile -> guard -> execute -> compose -> verify -> render, with at most two repairs.

Every model call goes through the Router. The model never writes SQL and never types a number:
the plan is a MetricQuery (or an abstention) and the answer is a template with placeholders
(ADR-0008). Nothing here is tuned for a real model; it is driven by a FakeProvider in tests.

Not built yet (PHASE_1C.md section a): routing a METRIC_MISMATCH verdict back to the planner
(repairs here always go to the composer), the route node, and the free-form SQL fallback.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..claims import QuerySpec, TemplateAnswer
from ..compiler import CompileError, MetricQuery, compile_query
from ..guard import GuardError, QueryTimeout, execute_guarded
from ..restate import restate
from ..semantic import SemanticLayer
from ..template import TemplateReport, verify_template
from ..verifier import GOOD, ExecutedQuery
from .accounting import RunContext, render_markdown
from .prompts import compose_request, plan_request
from .router import Router
from .schemas import ComposeOutput, PlanOutput
from .types import (
    AllProvidersFailed,
    BudgetExceeded,
    CassetteMiss,
    LLMResponse,
    SensitivityRefusal,
)

QUERY_ID = "q1"
MAX_REPAIRS = 2


@dataclass
class Attempt:
    kind: str  # compose | repair
    status: str  # verified | partially_verified | schema_failure
    verified_numbers: int = 0
    total_numbers: int = 0
    rendered: str = ""
    problems: list[str] = field(default_factory=list)
    report: TemplateReport | None = None


@dataclass
class QuestionRun:
    question: str
    outcome: str  # verified | partially_verified | abstain
    reason: str = ""  # abstain reason code, or "" otherwise
    rendered: str = ""
    restatement: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    plan_failures: int = 0
    repairs_used: int = 0
    # one entry per model response: step, json_ok, schema_ok, and step-specific flags
    events: list[dict[str, Any]] = field(default_factory=list)
    run: RunContext | None = None
    verification: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "outcome": self.outcome,
            "reason": self.reason,
            "rendered": self.rendered,
            "restatement": self.restatement,
            "plan_failures": self.plan_failures,
            "repairs_used": self.repairs_used,
            "attempts": [
                {
                    "kind": a.kind,
                    "status": a.status,
                    "verified_numbers": a.verified_numbers,
                    "total_numbers": a.total_numbers,
                    "problems": a.problems,
                }
                for a in self.attempts
            ],
            "verification": self.verification,
            "accounting": self.run.summary() if self.run else None,
        }

    def render_md(self) -> str:
        lines = [
            f"**Question:** {self.question}",
            f"**Outcome:** {self.outcome}" + (f" (reason: {self.reason})" if self.reason else ""),
        ]
        if self.restatement:
            lines.append(f"**Query, restated:** {self.restatement}")
        if self.rendered:
            lines.append(f"**Answer:** {self.rendered}")
        lines.append(
            f"**Attempts:** {len(self.attempts)} compose/repair, "
            f"{self.plan_failures} plan failures, {self.repairs_used} repairs used"
        )
        if self.run:
            lines += ["", render_markdown(self.run.summary())]
        lines += ["", "No accuracy figure is computed here."]
        return "\n".join(lines)


def _execute(layer: SemanticLayer, readonly_url: str, mq: MetricQuery) -> ExecutedQuery:
    cq = compile_query(layer, mq)
    res = execute_guarded(readonly_url, cq.sql)
    return ExecutedQuery(QUERY_ID, cq.sql, res.columns, res.rows, metric_query=mq)


def _json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):  # tolerate a code fence around JSON
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    return json.loads(text)


TYPED_DIGIT_CODES = {"digit_outside_placeholder", "digits_in_unrecognised_form", "non_ascii_digit"}


def typed_digit(rep: TemplateReport) -> bool:
    """True if the model put a digit or number outside a placeholder (placeholder adherence)."""
    if any(i.code in TYPED_DIGIT_CODES for i in rep.issues):
        return True
    return any(v.reason == "number_outside_placeholder" for v in rep.report.verdicts)


def schema_problems(exc: Exception, max_items: int = 8, msg_chars: int = 140) -> list[str]:
    """Field-level problems for the repair prompt: the path and the message of each pydantic
    error, never the offending input. Other errors (for example unparseable JSON) give one
    line. Each message and the list are cut, so a bad answer cannot make the prompt grow."""
    if not isinstance(exc, ValidationError):
        return [f"schema_failure: {str(exc).splitlines()[0][:200]}"]
    errs = exc.errors(include_url=False, include_context=False, include_input=False)
    out = []
    for e in errs[:max_items]:
        path = ".".join(str(x) for x in e["loc"]) or "(top level)"
        out.append(f"schema_failure at {path}: {e['msg'][:msg_chars]}")
    if len(errs) > max_items:
        out.append(f"schema_failure: and {len(errs) - max_items} more errors")
    return out


def _problems(report: TemplateReport) -> list[str]:
    out = [f"{i.code}: {i.detail}" for i in report.issues]
    for v in report.report.verdicts:
        if v.verdict not in GOOD:
            out.append(f"{v.verdict} on {v.text!r}: {v.reason}")
    return out


def run_question(
    question: str,
    *,
    layer: SemanticLayer,
    readonly_url: str,
    router: Router,
    run: RunContext,
    known_values: dict[str, set[str]] | None = None,
    max_repairs: int = MAX_REPAIRS,
    max_tokens: int | None = None,
) -> QuestionRun:
    result = QuestionRun(question, "abstain", run=run)
    dataset = layer.source.relation
    mq: MetricQuery | None = None
    executed: dict[str, ExecutedQuery] = {}
    plan_error: str | None = None
    previous: str | None = None
    problems: list[str] = []

    def shadow(q: MetricQuery) -> ExecutedQuery:
        return _execute(layer, readonly_url, q)

    def abstain(reason: str) -> QuestionRun:
        result.outcome, result.reason = "abstain", reason
        return result

    def call(req) -> LLMResponse | QuestionRun:
        if max_tokens:
            req = dataclasses.replace(req, max_tokens=max_tokens)
        try:
            return router.call(req, run)
        except BudgetExceeded:
            return abstain("budget")
        except SensitivityRefusal:
            return abstain("sensitivity")
        except (AllProvidersFailed, CassetteMiss):
            return abstain("provider_error")

    while True:
        if mq is None:  # ---- plan, compile, guard, execute ----
            got = call(plan_request(layer, question, plan_error))
            if isinstance(got, QuestionRun):
                return got
            ev: dict[str, Any] = {
                "step": "plan", "json_ok": False, "schema_ok": False,
                "abstained": False, "exec_ok": None,
            }  # fmt: skip
            result.events.append(ev)
            try:
                raw = _json(got.text)
                ev["json_ok"] = True
                plan = PlanOutput.model_validate(raw)
                ev["schema_ok"] = True
            except (ValueError, ValidationError) as exc:
                plan, err = None, f"plan is not valid ({str(exc).splitlines()[0][:200]})"
            else:
                err = None
            if plan is not None and plan.abstain:
                ev["abstained"] = True
                return abstain(plan.abstain)
            if plan is not None:
                try:
                    mq = plan.metric_query
                    executed = {QUERY_ID: _execute(layer, readonly_url, mq)}
                    result.restatement = restate(layer, mq)
                    ev["exec_ok"] = True
                except (CompileError, GuardError, QueryTimeout) as exc:
                    mq, err = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                    ev["exec_ok"] = False
            if err is not None:
                result.plan_failures += 1
                result.repairs_used += 1
                plan_error = err
                if result.repairs_used > max_repairs:
                    return abstain("plan_failed")
                continue
        # ---- compose / repair ----
        kind = "repair" if previous is not None else "compose"
        got = call(
            compose_request(
                question,
                result.restatement,
                executed,
                (dataset,),
                previous,
                problems if previous is not None else None,
            )
        )
        if isinstance(got, QuestionRun):
            return got
        previous = got.text
        cev: dict[str, Any] = {
            "step": kind, "json_ok": False, "schema_ok": False, "typed_digit": None,
        }  # fmt: skip
        result.events.append(cev)
        try:
            raw = _json(got.text)
            cev["json_ok"] = True
            out = ComposeOutput.model_validate(raw)
            answer = TemplateAnswer(
                question=question,
                template=out.template,
                queries=[QuerySpec(id=QUERY_ID, metric_query=mq)],
                claims=out.claims,
            )
        except (ValueError, ValidationError) as exc:
            problems = schema_problems(exc)
            result.attempts.append(Attempt(kind, "schema_failure", problems=problems))
        else:
            cev["schema_ok"] = True
            rep = verify_template(layer, answer, executed, shadow, known_values)
            nums = rep.report.verdicts
            n_good = sum(v.verdict in GOOD for v in nums)
            cev["typed_digit"] = typed_digit(rep)
            status, probs = rep.status, _problems(rep)
            if status == "verified" and n_good == 0:
                # a vacuous pass: the text states no figure from the result, so nothing was
                # verified. It is never shown as verified (see docs/VERIFIER_LIMITS.md).
                status = "no_figures"
                probs = ["no_figures: the answer uses no placeholder, so no figure is verified"]
            att = Attempt(
                kind, status,
                verified_numbers=n_good,
                total_numbers=len(nums),
                rendered=rep.rendered,
                problems=probs,
                report=rep,
            )  # fmt: skip
            result.attempts.append(att)
            problems = att.problems
            if status == "verified":
                result.outcome, result.rendered = "verified", rep.rendered
                result.verification = rep.as_dict()
                return result
        if result.repairs_used >= max_repairs:
            break
        result.repairs_used += 1
    best = max(result.attempts, key=lambda a: a.verified_numbers, default=None)
    if best is not None and best.verified_numbers > 0 and best.report is not None:
        result.outcome, result.rendered = "partially_verified", best.rendered
        result.verification = best.report.as_dict()
        return result
    return abstain("unverified")
