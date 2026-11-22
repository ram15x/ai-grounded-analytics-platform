"""Deterministic plain-English restatement of a MetricQuery. No LLM; a template over the query.

Shown with every answer so a reader can see exactly what was asked of the warehouse, in words
that come from the semantic layer and the query object only. Every value shown is the value
the compiler would put in the SQL (same normalisation), so the text cannot drift from the query.
"""

from __future__ import annotations

import json

from .compiler import CompileError, Filter, MetricQuery, _literal, compile_query
from .semantic import SemanticLayer

_OPS = {
    "eq": "is",
    "ne": "is not",
    "gt": "is greater than",
    "gte": "is at least",
    "lt": "is less than",
    "lte": "is at most",
}


def _label(name: str) -> str:
    return name.replace("_", " ")


def _show(literal_sql: str) -> str:
    """Turn a compiled SQL literal back into display text."""
    if literal_sql.startswith("DATE '"):
        return literal_sql[6:-1]
    return json.dumps(literal_sql[1:-1].replace("''", "'"), ensure_ascii=False)


def _filter_text(layer: SemanticLayer, f: Filter) -> str:
    dim = layer.dimensions[f.dimension]
    name = _label(f.dimension)

    def one(v: object) -> str:
        return _show(_literal(v, dim, f.dimension))

    if f.op in _OPS:
        return f"{name} {_OPS[f.op]} {one(f.value)}"
    if f.op in ("in", "not_in"):
        vals = ", ".join(one(v) for v in f.value)
        return f"{name} is {'not ' if f.op == 'not_in' else ''}one of [{vals}]"
    lo, hi = f.value
    return f"{name} is between {one(lo)} and {one(hi)} (inclusive)"


def restate(layer: SemanticLayer, query: MetricQuery | dict) -> str:
    """Return the restatement. Raises CompileError for anything the compiler would refuse."""
    q = query if isinstance(query, MetricQuery) else MetricQuery.model_validate(query)
    compile_query(layer, q)  # same validation as execution: never describe an invalid query
    metrics = [_label(m) for m in q.metrics]
    head = "Compute " + _join(metrics)
    if q.dimensions:
        head += ", grouped by " + _join([_label(d) for d in q.dimensions])
    else:
        head += ", as a single total"
    parts = [head]
    if q.filters:
        parts.append("only where " + "; and ".join(_filter_text(layer, f) for f in q.filters))
    if q.order_by:
        parts.append(
            "sorted by "
            + ", ".join(
                f"{_label(o.by)} {'descending' if o.desc else 'ascending'}" for o in q.order_by
            )
        )
    if q.limit is not None:
        parts.append(f"limited to the first {q.limit} rows")
    text = ", ".join(parts) + "."
    defs = [f"{_label(m)}: {layer.metrics[m].description.rstrip('.')}" for m in q.metrics]
    unit_notes = [f"{_label(m)} in {_unit(layer.metrics[m].unit)}" for m in q.metrics]
    return (
        f"{text} Definitions: {'; '.join(defs)}. Units: {'; '.join(unit_notes)}. "
        f"Source: {layer.source.relation}."
    )


def _unit(unit: str) -> str:
    kind, _, code = unit.partition(":")
    return f"{code} (currency)" if kind == "currency" and code else kind


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


__all__ = ["CompileError", "restate"]
