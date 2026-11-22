"""Metric compiler: metric + dimensions + filters -> a single SELECT statement.

Values are rendered as validated literals (never string-formatted from raw input),
and callers still pass the result through the SQL guard before executing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from grounded_analytics.semantic import Dimension, SemanticLayer

MAX_IN_VALUES = 100
MAX_TEXT_LEN = 200
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CompileError(ValueError):
    pass


Op = Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between"]
_SIMPLE_OPS = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dimension: str
    op: Op
    value: Any


class OrderBy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    by: str
    desc: bool = False


class MetricQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics: list[str] = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[Filter] = Field(default_factory=list)
    order_by: list[OrderBy] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=10_000)


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    columns: list[str]
    dimensions: list[str]
    metrics: list[str]


def _text_literal(value: Any) -> str:
    if not isinstance(value, str):
        raise CompileError(f"expected a string, got {type(value).__name__}")
    if len(value) > MAX_TEXT_LEN or _CONTROL.search(value):
        raise CompileError("text value too long or contains control characters")
    return "'" + value.replace("'", "''") + "'"


def _date_literal(value: Any, dim_name: str) -> str:
    if not isinstance(value, str):
        raise CompileError(f"{dim_name}: dates must be ISO strings")
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}", text) and dim_name == "month":
        text += "-01"
    try:
        d = date.fromisoformat(text)
    except ValueError as exc:
        raise CompileError(f"{dim_name}: bad date {value!r}") from exc
    return f"DATE '{d.isoformat()}'"


def _literal(value: Any, dim: Dimension, name: str) -> str:
    if isinstance(value, bool):
        raise CompileError(f"{name}: boolean values are not supported")
    if dim.type == "date":
        return _date_literal(value, name)
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            d = Decimal(str(value))
        except InvalidOperation as exc:
            raise CompileError(f"{name}: bad number {value!r}") from exc
        if not d.is_finite():
            raise CompileError(f"{name}: number must be finite")
        return _text_literal(str(d))  # text dimensions compare as text
    return _text_literal(value)


def _filter_sql(f: Filter, layer: SemanticLayer) -> str:
    dim = layer.dimensions.get(f.dimension)
    if dim is None:
        raise CompileError(f"unknown dimension {f.dimension!r}")
    col = dim.column
    if f.op in _SIMPLE_OPS:
        if isinstance(f.value, (list, tuple, dict)):
            raise CompileError(f"{f.op} needs a scalar value")
        return f"{col} {_SIMPLE_OPS[f.op]} {_literal(f.value, dim, f.dimension)}"
    if f.op in ("in", "not_in"):
        if not isinstance(f.value, list) or not f.value:
            raise CompileError(f"{f.op} needs a non-empty list")
        if len(f.value) > MAX_IN_VALUES:
            raise CompileError(f"{f.op} accepts at most {MAX_IN_VALUES} values")
        items = ", ".join(_literal(v, dim, f.dimension) for v in f.value)
        return f"{col} {'IN' if f.op == 'in' else 'NOT IN'} ({items})"
    if f.op == "between":
        if not isinstance(f.value, list) or len(f.value) != 2:
            raise CompileError("between needs a [low, high] list")
        lo, hi = (_literal(v, dim, f.dimension) for v in f.value)
        return f"{col} BETWEEN {lo} AND {hi}"
    raise CompileError(f"unsupported operator {f.op!r}")  # pragma: no cover


def metric_expression(layer: SemanticLayer, name: str) -> str:
    m = layer.metrics[name]
    if m.kind == "simple":
        assert m.expr is not None
        return m.expr
    num = metric_expression(layer, m.numerator)  # type: ignore[arg-type]
    den = metric_expression(layer, m.denominator)  # type: ignore[arg-type]
    return f"({num}) / NULLIF(({den}), 0)"


def compile_query(layer: SemanticLayer, query: MetricQuery | dict) -> CompiledQuery:
    q = query if isinstance(query, MetricQuery) else MetricQuery.model_validate(query)
    for name in q.metrics:
        if name not in layer.metrics:
            raise CompileError(f"unknown metric {name!r}")
    if len(set(q.metrics)) != len(q.metrics) or len(set(q.dimensions)) != len(q.dimensions):
        raise CompileError("duplicate metric or dimension")
    for d in q.dimensions:
        if d not in layer.dimensions:
            raise CompileError(f"unknown dimension {d!r}")
        for name in q.metrics:
            if d not in layer.metrics[name].allowed_dimensions:
                raise CompileError(f"metric {name} cannot be grouped by {d}")
    eqs: dict[str, Any] = {}
    for f in q.filters:
        if f.op == "eq":
            if f.dimension in eqs and str(eqs[f.dimension]) != str(f.value):
                raise CompileError(
                    f"{f.dimension} cannot equal both {eqs[f.dimension]!r} and {f.value!r}; "
                    "use one filter with op 'in' for several values"
                )
            eqs[f.dimension] = f.value
    columns = [*q.dimensions, *q.metrics]
    for ob in q.order_by:
        if ob.by not in columns:
            raise CompileError(f"order_by {ob.by!r} is not a selected column")

    select_items = [f"{layer.dimensions[d].column} AS {d}" for d in q.dimensions]
    select_items += [f"{metric_expression(layer, m)} AS {m}" for m in q.metrics]
    parts = [
        "SELECT " + ", ".join(select_items),
        f"FROM {layer.source.relation} AS {layer.source.alias}",
    ]
    if q.filters:
        parts.append("WHERE " + " AND ".join(_filter_sql(f, layer) for f in q.filters))
    if q.dimensions:
        parts.append("GROUP BY " + ", ".join(layer.dimensions[d].column for d in q.dimensions))
    if q.order_by:
        parts.append(
            "ORDER BY " + ", ".join(f"{o.by}{' DESC' if o.desc else ''}" for o in q.order_by)
        )
    elif q.dimensions:
        parts.append("ORDER BY " + ", ".join(q.dimensions))
    if q.limit is not None:
        parts.append(f"LIMIT {q.limit}")
    return CompiledQuery("\n".join(parts), columns, list(q.dimensions), list(q.metrics))
