"""Deterministic verifier (ARCHITECTURE 4.1 steps A, C-G). No LLM, no network, no database.

Inputs are the final prose, the composer's claims, and the EXECUTED query results. The verifier
never believes the composer: every number in the prose is found by the extractor (step B), and
each must be backed by a claim whose provenance recomputes, with Decimal arithmetic, to the
displayed value within its displayed precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any

import sqlglot
from sqlglot import exp

from .claims import (
    AnswerDoc,
    CellRef,
    Claim,
    DerivedRef,
    LiteralArg,
    ParameterRef,
    PassageRef,
    RowCountRef,
    to_decimal,
)
from .compiler import CompileError, MetricQuery, compile_query
from .context import ContextFail, check_cell_context
from .extract import ExtractedNumber, extract_numbers
from .semantic import SemanticLayer, norm_term

VERIFIED = "VERIFIED"
VERIFIED_DERIVED = "VERIFIED_DERIVED"
VERIFIED_QUOTE = "VERIFIED_QUOTE"
METRIC_MISMATCH = "METRIC_MISMATCH"
DERIVATION_MISMATCH = "DERIVATION_MISMATCH"
UNSUPPORTED = "UNSUPPORTED"
AMBIGUOUS = "AMBIGUOUS"
GOOD = (VERIFIED, VERIFIED_DERIVED, VERIFIED_QUOTE)


@dataclass
class ExecutedQuery:
    """An executed, immutable result. `rows` hold Decimal/str/None cells in `columns` order."""

    id: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    metric_query: MetricQuery | None = None
    declared_metric_query: MetricQuery | None = None


@dataclass
class Verdict:
    text: str
    start: int
    end: int
    verdict: str
    reason: str
    claim_id: str | None = None
    expected: Decimal | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in GOOD

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "span": [self.start, self.end],
            "verdict": self.verdict,
            "reason": self.reason,
            "claim_id": self.claim_id,
            "expected": None if self.expected is None else str(self.expected),
            "detail": self.detail,
        }


@dataclass
class Report:
    verdicts: list[Verdict] = field(default_factory=list)
    excluded: list[ExtractedNumber] = field(default_factory=list)
    dropped_claims: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "verified" if all(v.ok for v in self.verdicts) else "partially_verified"

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.verdicts:
            out[v.verdict] = out.get(v.verdict, 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "counts": self.counts(),
            "verdicts": [v.as_dict() for v in self.verdicts],
            "excluded": [{"text": e.text, "reason": e.reason} for e in self.excluded],
            "dropped_claims": self.dropped_claims,
        }


class _Fail(Exception):
    def __init__(self, verdict: str, reason: str, detail: str = ""):
        super().__init__(reason)
        self.verdict, self.reason, self.detail = verdict, reason, detail


@dataclass
class _Cell:
    value: Decimal
    query: ExecutedQuery
    column: str
    row: tuple[Any, ...]


ResolveFailure = _Fail  # public name: raised by Verifier.resolve when a provenance cannot resolve


@dataclass
class Resolved:
    """A provenance resolved to a value, with the metrics (if any) of the cells it came from."""

    value: Decimal
    metrics: list[str]


ShadowFn = Callable[[MetricQuery], ExecutedQuery]
CONFLICTING_DIRECTIONS = {"up": 1, "down": -1}


class Verifier:
    def __init__(
        self,
        layer: SemanticLayer,
        queries: dict[str, ExecutedQuery],
        passages: dict[str, str] | None = None,
        shadow: ShadowFn | None = None,
        known_values: dict[str, set[str]] | None = None,
    ):
        self.known_values = known_values or {}
        self._prose = ""
        self._spans: list[tuple[int, int]] = []
        self.layer = layer
        self.queries = queries
        self.passages = passages or {}
        self.shadow = shadow

    # ------------------------------------------------------------------ public

    def verify(self, prose: str, claims: list[Claim]) -> Report:
        report = Report()
        self._prose = prose
        extraction = extract_numbers(prose)
        self._spans = [(i.start, i.end) for i in extraction.items if i.status != "excluded"]
        report.excluded = list(extraction.excluded)
        prose_values = {i.value for i in extraction.ok if i.value is not None}
        remaining = list(claims)

        for item in sorted(extraction.items, key=lambda i: i.start):
            if item.status == "excluded":
                continue
            if item.status == "ambiguous":
                report.verdicts.append(
                    Verdict(item.text, item.start, item.end, AMBIGUOUS, item.reason or "ambiguous")
                )
                continue
            claim = self._take_claim(item, remaining)
            if claim is None:
                report.verdicts.append(
                    Verdict(item.text, item.start, item.end, UNSUPPORTED, "no_claim")
                )
                continue
            report.verdicts.append(self._check(item, claim, prose_values))
        report.dropped_claims = [c.id for c in remaining]
        return report

    def resolve(self, prov: CellRef | RowCountRef | DerivedRef) -> Resolved:
        """Resolve a cell, row count or derived provenance to its recomputed value (Decimal).

        Used by the placeholder renderer: the value shown to the reader comes from here, never
        from the composer. Derived claims may not take literal arguments in this mode.
        """
        if isinstance(prov, DerivedRef):
            if any(isinstance(a, LiteralArg) for a in prov.args):
                raise _Fail(UNSUPPORTED, "literal_argument_not_allowed_in_template")
            value = self._derive(prov, set())
            args = prov.args
        else:
            value = self._arg(prov, set())
            args = [prov]
        metrics: list[str] = []
        for a in args:
            if isinstance(a, CellRef) and a.column in self.layer.metrics:
                metrics.append(a.column)
        return Resolved(value, metrics)

    # ------------------------------------------------------------- reconciliation

    @staticmethod
    def _take_claim(item: ExtractedNumber, remaining: list[Claim]) -> Claim | None:
        def matches(c: Claim, exact: bool) -> bool:
            t, ct = item.text.strip(), c.text.strip()
            if exact:
                return t == ct
            return ct in t and (t.startswith(ct) or t.endswith(ct))

        for exact in (True, False):
            for c in remaining:
                if matches(c, exact):
                    remaining.remove(c)
                    return c
        return None

    # --------------------------------------------------------------------- checks

    def _check(self, item: ExtractedNumber, claim: Claim, prose_values: set[Decimal]) -> Verdict:
        def v(
            verdict: str, reason: str, expected: Decimal | None = None, detail: str = ""
        ) -> Verdict:
            return Verdict(
                item.text, item.start, item.end, verdict, reason, claim.id, expected, detail
            )

        try:
            shown = item.value
            assert shown is not None
            if claim.value not in (shown, -shown if item.direction else shown):
                raise _Fail(UNSUPPORTED, "claim_value_differs_from_text")
            self._unit_compatible(item, claim)
            prov = claim.provenance
            if isinstance(prov, PassageRef):
                return v(VERIFIED_QUOTE, "in_passage", None, self._passage(item, prov))
            if isinstance(prov, ParameterRef):
                self._parameter(item, prov)
                return v(VERIFIED, "sql_parameter")
            if isinstance(prov, RowCountRef):
                expected = Decimal(len(self._query(prov.query_id).rows))
                self._compare(item, claim, expected, UNSUPPORTED, derived=False)
                return v(VERIFIED, "row_count", expected)
            if isinstance(prov, CellRef):
                cell = self._cell(prov)
                expected, note, metric = self._bind(item, claim, cell)
                self._compare(item, claim, expected, UNSUPPORTED, derived=False)
                self._context(item, prov, cell, metric)
                return v(VERIFIED, note, expected)
            assert isinstance(prov, DerivedRef)
            expected = self._derive(prov, prose_values)
            self._compare(item, claim, expected, DERIVATION_MISMATCH, derived=True)
            return v(VERIFIED_DERIVED, prov.op, expected)
        except _Fail as f:
            expected = None
            return v(f.verdict, f.reason, expected, f.detail)

    # ---- unit compatibility (step E, second half)

    def _unit_compatible(self, item: ExtractedNumber, claim: Claim) -> None:
        ok = {
            "currency": {"currency", "number"},
            "percent": {"percent", "ratio"},
            "percentage_points": {"percentage_points"},
            "basis_points": {"percentage_points", "number"},
            "number": {"count", "number", "ratio", "currency"},
        }[item.unit]
        if claim.unit not in ok:
            raise _Fail(UNSUPPORTED, "unit_incompatible", f"text={item.unit} claim={claim.unit}")

    # ---- comparison: displayed precision, direction, qualifiers

    def _compare(
        self, item: ExtractedNumber, claim: Claim, expected: Decimal, fail: str, derived: bool
    ) -> None:
        shown = item.value
        assert shown is not None
        if item.direction in CONFLICTING_DIRECTIONS and shown >= 0:
            if expected == 0 or (expected > 0) != (item.direction == "up"):
                raise _Fail(fail, "direction_mismatch", f"expected {expected}")
            expected = abs(expected)
        diff = abs(expected - shown)
        tol = item.quantum / 2
        if item.qualifier == "lower_bound":
            good = expected >= shown
        elif item.qualifier == "upper_bound":
            good = expected <= shown
        else:
            good = diff <= tol
        if not good:
            raise _Fail(fail, "value_mismatch", f"expected {expected}, shown {shown}")

    # ---- cells and label binding

    def _query(self, qid: str) -> ExecutedQuery:
        if qid not in self.queries:
            raise _Fail(UNSUPPORTED, "unknown_query", qid)
        return self.queries[qid]

    def _cell(self, ref: CellRef) -> _Cell:
        q = self._query(ref.query_id)
        if ref.column not in q.columns:
            raise _Fail(UNSUPPORTED, "unknown_column", ref.column)
        for k in ref.row_key:
            if k not in q.columns:
                raise _Fail(UNSUPPORTED, "unknown_row_key_column", k)
        hits = [
            r
            for r in q.rows
            if all(_same(r[q.columns.index(k)], x) for k, x in ref.row_key.items())
        ]
        if not hits:
            raise _Fail(UNSUPPORTED, "row_not_found")
        if len(hits) > 1:
            raise _Fail(AMBIGUOUS, "multiple_rows_match")
        raw = hits[0][q.columns.index(ref.column)]
        if raw is None:
            raise _Fail(UNSUPPORTED, "cell_is_null")
        try:
            return _Cell(to_decimal(raw), q, ref.column, hits[0])
        except ValueError as exc:
            raise _Fail(UNSUPPORTED, "cell_not_numeric") from exc

    def _bind(
        self, item: ExtractedNumber, claim: Claim, cell: _Cell
    ) -> tuple[Decimal, str, str | None]:
        layer = self.layer
        col_metric = cell.column if cell.column in layer.metrics else None
        label_metric = layer.resolve_metric(claim.label)
        if label_metric and col_metric and label_metric != col_metric:
            raise _Fail(METRIC_MISMATCH, "label_column_mismatch", f"{label_metric} vs {col_metric}")
        metric = col_metric or label_metric
        if metric is None:
            if (
                norm_term(claim.label) != norm_term(cell.column)
                and cell.column not in layer.dimensions
            ):
                raise _Fail(UNSUPPORTED, "label_unbound", claim.label)
            return cell.value, "cell", None
        m = layer.metrics[metric]
        kind = m.unit_kind
        if (
            kind == "currency"
            and item.currency
            and ":" in m.unit
            and item.currency != m.unit.split(":")[1]
        ):
            raise _Fail(UNSUPPORTED, "currency_mismatch", f"{item.currency} vs {m.unit}")
        if kind == "currency" and item.unit not in ("currency", "number"):
            raise _Fail(UNSUPPORTED, "unit_incompatible", f"{item.unit} vs {m.unit}")
        if kind == "count" and item.unit != "number":
            raise _Fail(UNSUPPORTED, "unit_incompatible", f"{item.unit} vs {m.unit}")
        if kind == "ratio" and item.unit not in ("percent", "number"):
            raise _Fail(UNSUPPORTED, "unit_incompatible", f"{item.unit} vs {m.unit}")
        self._definition_check(cell, metric)
        value = cell.value * 100 if kind == "ratio" and item.unit == "percent" else cell.value
        return value, "cell_bound", metric

    def _definition_check(self, cell: _Cell, metric: str) -> None:
        """Step F: the executed query must be the metric's definition (compiled or shadowed)."""
        q = cell.query
        if q.metric_query is not None:
            if metric not in q.metric_query.metrics:
                raise _Fail(METRIC_MISMATCH, "metric_not_in_compiled_query")
            try:
                canonical = compile_query(self.layer, q.metric_query).sql
            except CompileError as exc:
                raise _Fail(METRIC_MISMATCH, "compiled_query_invalid", str(exc)) from exc
            if _squash(canonical) != _squash(q.sql):
                raise _Fail(METRIC_MISMATCH, "executed_sql_is_not_the_compiled_sql")
            return
        declared = q.declared_metric_query
        if declared is None or metric not in declared.metrics:
            raise _Fail(
                AMBIGUOUS, "no_canonical_query", "free-form SQL without a declared metric query"
            )
        if self.shadow is None:
            raise _Fail(AMBIGUOUS, "no_shadow_executor")
        canon = self.shadow(declared)
        dims = list(declared.dimensions)
        if any(d not in q.columns for d in dims):
            raise _Fail(METRIC_MISMATCH, "free_form_query_lacks_declared_dimensions")
        key = [cell.row[q.columns.index(d)] for d in dims]
        match = [
            r
            for r in canon.rows
            if all(_same(r[canon.columns.index(d)], k) for d, k in zip(dims, key, strict=True))
        ]
        if len(match) != 1:
            raise _Fail(METRIC_MISMATCH, "canonical_row_missing_or_ambiguous")
        canon_val = to_decimal(match[0][canon.columns.index(metric)])
        tol = Decimal(1).scaleb(-self.layer.metrics[metric].decimals) / 2
        if abs(canon_val - cell.value) > tol:
            raise _Fail(
                METRIC_MISMATCH,
                "disagrees_with_metric_definition",
                f"canonical {canon_val}, query {cell.value}",
            )

    def _context(
        self, item: ExtractedNumber, ref: CellRef, cell: _Cell, metric: str | None
    ) -> None:
        q = cell.query
        try:
            check_cell_context(
                prose=self._prose, start=item.start, end=item.end, layer=self.layer,
                columns=q.columns, rows=q.rows, row=cell.row, row_key=ref.row_key,
                metric_query=q.metric_query or q.declared_metric_query, metric=metric,
                known_values=self.known_values, number_spans=self._spans,
            )  # fmt: skip
        except ContextFail as f:
            raise _Fail(f.verdict, f.reason, f.detail) from f

    # ---- derived

    def _arg(self, arg: Any, prose_values: set[Decimal]) -> Decimal:
        if isinstance(arg, LiteralArg):
            if arg.value not in prose_values:
                raise _Fail(UNSUPPORTED, "literal_not_in_prose", str(arg.value))
            return arg.value
        if isinstance(arg, RowCountRef):
            return Decimal(len(self._query(arg.query_id).rows))
        return self._cell(arg).value

    def _derive(self, ref: DerivedRef, prose_values: set[Decimal]) -> Decimal:
        xs = [self._arg(a, prose_values) for a in ref.args]
        op = ref.op

        def need(n: int) -> None:
            if len(xs) != n:
                raise _Fail(DERIVATION_MISMATCH, "wrong_arity", f"{op} takes {n} args")

        try:
            if op == "sum":
                return sum(xs, Decimal(0))
            if op == "mean":
                return sum(xs, Decimal(0)) / len(xs)
            if op == "min":
                return min(xs)
            if op == "max":
                return max(xs)
            if op == "count":
                return Decimal(len(xs))
            need(2)
            a, b = xs
            if op == "difference":
                return a - b
            if op == "ratio":
                return a / b
            if op == "percent_of":
                return a / b * 100
            if op == "percent_change":
                return (b - a) / abs(a) * 100
        except (DivisionByZero, InvalidOperation) as exc:
            raise _Fail(DERIVATION_MISMATCH, "division_by_zero") from exc
        raise _Fail(DERIVATION_MISMATCH, "unknown_op", op)

    # ---- passages and parameters

    def _passage(self, item: ExtractedNumber, ref: PassageRef) -> str:
        text = self.passages.get(ref.passage_id)
        if text is None:
            raise _Fail(UNSUPPORTED, "unknown_passage", ref.passage_id)
        s, e = ref.span
        if not (0 <= s < e <= len(text)):
            raise _Fail(UNSUPPORTED, "span_out_of_range")
        shown = item.value
        assert shown is not None
        for p in extract_numbers(text[s:e]).ok:
            if p.value is None or p.unit != item.unit:
                continue
            tol = item.quantum / 2 if p.quantum <= item.quantum else Decimal(0)
            if abs(p.value - shown) <= tol:
                return f"matched {p.text!r}"
        raise _Fail(UNSUPPORTED, "number_not_in_cited_span")

    def _parameter(self, item: ExtractedNumber, ref: ParameterRef) -> None:
        q = self._query(ref.query_id)
        try:
            tree = sqlglot.parse_one(q.sql, read="postgres")
        except sqlglot.errors.SqlglotError as exc:
            raise _Fail(UNSUPPORTED, "unparseable_sql") from exc
        shown = item.value
        for lit in tree.find_all(exp.Literal):
            if not lit.is_string:
                try:
                    if Decimal(lit.name) == shown:
                        return
                except InvalidOperation:
                    continue
        raise _Fail(UNSUPPORTED, "not_a_sql_literal")


def _same(cell: Any, wanted: Any) -> bool:
    if cell is None or wanted is None:
        return cell is wanted
    try:
        return to_decimal(cell) == to_decimal(wanted)
    except ValueError:
        return str(cell) == str(wanted)


def _squash(sql: str) -> str:
    return " ".join(sql.split())


def verify_answer(
    layer: SemanticLayer,
    doc: AnswerDoc,
    executed: dict[str, ExecutedQuery],
    shadow: ShadowFn | None = None,
    known_values: dict[str, set[str]] | None = None,
) -> Report:
    passages = {p.id: p.text for p in doc.passages}
    return Verifier(layer, executed, passages, shadow, known_values).verify(doc.prose, doc.claims)
