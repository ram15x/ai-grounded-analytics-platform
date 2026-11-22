"""The boxes a person fills in about an uploaded table, and the semantic layer built from them.

Nothing here accepts free SQL. A metric is an aggregate over one column with optional conditions,
or a ratio of two other metrics; the SQL is generated from that closed grammar, and every literal
is checked against the column's type. Only metrics marked `approved` reach the semantic layer.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..semantic import IDENT, SemanticLayer
from .profile import DatasetProfile

Role = Literal["date", "dimension", "identifier", "measure", "ignore"]
Op = Literal["eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "is_null", "not_null"]
Aggregate = Literal["sum", "count", "count_distinct", "avg", "min", "max"]
NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
UNIT = re.compile(r"^(currency:[A-Z]{3}|count|ratio)$")
_SIMPLE = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


class ManifestError(ValueError):
    pass


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    op: Op
    value: Any = None


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Role
    label: str = ""
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    enumerable: bool = False  # a small closed set of values (the person confirms this)
    value_aliases: dict[str, list[str]] = Field(default_factory=dict)
    sensitivity: Literal["public", "internal", "personal"] = "internal"
    # for a date column whose day/month order the values do not settle: the person says which
    date_order: Literal["dmy", "mdy"] | None = None


class Known(BaseModel):
    """A figure from the person's own report, used to check a metric before it is approved."""

    model_config = ConfigDict(extra="forbid")
    value: str
    where: list[Condition] = Field(default_factory=list)
    note: str = ""
    tolerance: str = "0.005"


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["draft", "approved"] = "draft"
    description: str
    aggregate: Aggregate | None = None
    column: str | None = None
    where: list[Condition] = Field(default_factory=list)
    numerator: str | None = None
    denominator: str | None = None
    unit: str
    decimals: int = Field(ge=0, le=8)
    synonyms: list[str] = Field(default_factory=list)
    known: list[Known] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> MetricSpec:
        ratio = bool(self.numerator or self.denominator)
        if ratio and not (self.numerator and self.denominator):
            raise ValueError("a ratio needs both numerator and denominator")
        if ratio and (self.aggregate or self.column or self.where):
            raise ValueError("a ratio metric names two metrics and nothing else")
        if not ratio and not self.aggregate:
            raise ValueError("a metric needs an aggregate (or numerator and denominator)")
        if self.aggregate and self.aggregate != "count" and not self.column:
            raise ValueError(f"{self.aggregate} needs a column")
        if self.aggregate == "count" and self.column:
            raise ValueError(
                "count counts rows and takes no column; use count_distinct for a column"
            )
        if not UNIT.match(self.unit):
            raise ValueError("unit must be currency:XXX, count or ratio")
        return self


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    expected: str


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    grain: str  # what one row is
    currency: str = ""
    # who may see this data: "public" lets hosted model providers receive its query results;
    # the default, "internal", keeps it on local models only (see ADR-0005 sensitivity rule)
    visibility: Literal["public", "internal"] = "internal"
    columns: dict[str, Column]
    exclusions: list[dict[str, Any]] = Field(default_factory=list)  # {name, description, where}
    glossary: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, MetricSpec] = Field(default_factory=dict)
    examples: list[Example] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        if not NAME.match(self.name):
            raise ValueError("name must be lower-case letters, digits and underscores")
        for field_name in ("description", "grain"):
            text = getattr(self, field_name).strip()
            if not text or "TODO" in text:
                raise ValueError(f"{field_name} is required: replace the TODO with your own words")
        for m in self.metrics:
            if not IDENT.match(m):
                raise ValueError(f"bad metric name {m!r}")
        return self

    @property
    def relation(self) -> str:
        return f"ds_{self.name}.data"


def skeleton(profile: DatasetProfile, name: str) -> dict[str, Any]:
    """A manifest to start from, filled with GUESSES from the profile.

    Every guess is the person's to confirm. Description and grain stay TODO, and the loader
    refuses them until they are replaced."""
    cols: dict[str, Any] = {}
    for c in profile.columns:
        if any(h.startswith("possible_personal_data") for h in c.hints):
            cols[c.name] = {"role": "ignore", "sensitivity": "personal"}
        elif c.type in ("date", "datetime"):
            cols[c.name] = {"role": "date", "label": c.source_name}
        elif "identifier" in c.hints:
            cols[c.name] = {"role": "identifier", "label": c.source_name}
        elif c.type in ("integer", "decimal") and "category" not in c.hints:
            cols[c.name] = {"role": "measure", "label": c.source_name}
        elif c.type in ("text", "boolean", "integer"):
            cols[c.name] = {"role": "dimension", "label": c.source_name,
                            "enumerable": "category" in c.hints}  # fmt: skip
        else:
            cols[c.name] = {"role": "ignore"}
    return {"name": name, "description": "TODO what is this data and whose?",
            "grain": "TODO what does one row stand for?", "currency": "", "columns": cols,
            "exclusions": [], "glossary": {}, "metrics": {}, "examples": []}  # fmt: skip


def load_manifest(path) -> Manifest:
    try:
        return Manifest.model_validate(yaml.safe_load(path.read_text("utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ManifestError(f"cannot load manifest {path}: {exc}") from exc


def _types(profile: DatasetProfile) -> dict[str, str]:
    return {c.name: c.type for c in profile.columns}


def literal(value: Any, ctype: str) -> str:
    """A SQL literal for `value`, checked against the column type. Never interpolates raw text."""
    if ctype in ("integer", "decimal"):
        try:
            d = Decimal(str(value))
        except InvalidOperation:
            raise ManifestError(f"{value!r} is not a number") from None
        if not d.is_finite() or isinstance(value, bool):
            raise ManifestError(f"{value!r} is not a number")
        return format(d, "f")
    if ctype in ("date", "datetime"):
        try:
            return "'" + date.fromisoformat(str(value)[:10]).isoformat() + "'"
        except ValueError:
            raise ManifestError(f"{value!r} is not an ISO date (YYYY-MM-DD)") from None
    if ctype == "boolean":
        if str(value).lower() not in ("true", "false"):
            raise ManifestError(f"{value!r} is not true or false")
        return str(value).lower()
    s = str(value)
    if len(s) > 200 or re.search(r"[\x00-\x1f]", s):
        raise ManifestError("text value too long or has control characters")
    return "'" + s.replace("'", "''") + "'"


def condition_sql(c: Condition, types: dict[str, str], alias: str = "l") -> str:
    if c.column not in types:
        raise ManifestError(f"unknown column {c.column!r}")
    t, col = types[c.column], f"{alias}.{c.column}"
    if c.op in ("is_null", "not_null"):
        return f"{col} IS {'NOT ' if c.op == 'not_null' else ''}NULL"
    if c.op in ("in", "not_in"):
        if not isinstance(c.value, list) or not c.value or len(c.value) > 100:
            raise ManifestError(f"{c.op} needs a list of 1 to 100 values")
        items = ", ".join(literal(v, t) for v in c.value)
        return f"{col} {'NOT ' if c.op == 'not_in' else ''}IN ({items})"
    return f"{col} {_SIMPLE[c.op]} {literal(c.value, t)}"


def where_sql(conds: list[Condition], types: dict[str, str]) -> str:
    return " AND ".join(condition_sql(c, types) for c in conds)


def metric_expr(spec: MetricSpec, types: dict[str, str], extra: list[Condition]) -> str:
    """The aggregate expression for a simple metric; `extra` are the dataset-wide exclusions."""
    conds = [*extra, *spec.where]
    flt = f" FILTER (WHERE {where_sql(conds, types)})" if conds else ""
    agg, col = spec.aggregate, f"l.{spec.column}" if spec.column else ""
    if agg in ("sum", "count"):
        inner = f"SUM({col})" if agg == "sum" else "COUNT(*)"
        return f"COALESCE({inner}{flt}, 0)"
    inner = {"count_distinct": f"COUNT(DISTINCT {col})", "avg": f"AVG({col})",
             "min": f"MIN({col})", "max": f"MAX({col})"}[agg]  # fmt: skip
    return f"{inner}{flt}"


def validate(m: Manifest, profile: DatasetProfile) -> list[str]:
    """Problems that make the manifest unsafe or wrong for this table. Empty means usable."""
    types, out = _types(profile), []
    for name, col in m.columns.items():
        t = types.get(name)
        if t is None:
            out.append(f"column {name!r} is not in the table")
        elif col.role == "measure" and t not in ("integer", "decimal"):
            out.append(f"{name}: a measure must be numeric, but it is {t}")
        elif col.role == "date" and t not in ("date", "datetime"):
            out.append(f"{name}: a date column must hold dates, but it is {t}")
    for ex in m.exclusions:
        try:
            where_sql([Condition.model_validate(c) for c in ex.get("where", [])], types)
        except (ManifestError, ValueError) as exc:
            out.append(f"exclusion {ex.get('name', '?')}: {exc}")
    for mname, s in m.metrics.items():
        try:
            if s.aggregate:
                if s.column and s.column not in m.columns:
                    out.append(f"{mname}: column {s.column!r} has no entry in columns")
                elif s.column and s.aggregate in ("sum", "avg") and types.get(s.column) not in (
                    "integer", "decimal",
                ):  # fmt: skip
                    out.append(f"{mname}: {s.aggregate} needs a numeric column")
                metric_expr(s, types, [])
            else:
                for ref in (s.numerator, s.denominator):
                    if ref not in m.metrics:
                        out.append(f"{mname}: unknown metric {ref!r}")
            for k in s.known:
                where_sql(k.where, types)
                Decimal(k.value)
        except (ManifestError, InvalidOperation, ValueError) as exc:
            out.append(f"{mname}: {exc}")
    return out


def exclusion_conditions(m: Manifest) -> list[Condition]:
    return [Condition.model_validate(c) for ex in m.exclusions for c in ex.get("where", [])]


def build_layer(m: Manifest, profile: DatasetProfile, approved_only: bool = True) -> SemanticLayer:
    """The semantic layer the existing pipeline already understands. Raises on any problem."""
    problems = validate(m, profile)
    if problems:
        raise ManifestError("; ".join(problems))
    types, excl = _types(profile), exclusion_conditions(m)
    dims: dict[str, Any] = {}
    date_cols = [n for n, c in m.columns.items() if c.role == "date"]
    for n in date_cols:
        pre = "" if len(date_cols) == 1 else f"{n}_"
        dims[f"{pre}month"] = {"column": f"l.{n}_month", "type": "date",
            "description": f"Calendar month of {m.columns[n].label or n} (first day of month)",
            "synonyms": ["monthly", "by month"]}  # fmt: skip
        dims[f"{pre}day"] = {"column": f"l.{n}", "type": "date",
            "description": m.columns[n].label or n, "synonyms": ["daily", "by day"]}  # fmt: skip
    for n, c in m.columns.items():
        if c.role in ("dimension", "identifier"):
            dims[n] = {"column": f"l.{n}", "type": "text", "description": c.description or c.label,
                "synonyms": c.synonyms, "enumerable": c.enumerable and c.role == "dimension",
                "value_aliases": c.value_aliases}  # fmt: skip
    metrics: dict[str, Any] = {}
    chosen = {k: s for k, s in m.metrics.items() if _usable(s, approved_only)}
    for k, s in chosen.items():
        if not s.aggregate:
            continue
        allowed = [d for d in dims if not (s.aggregate == "count_distinct" and d == s.column)]
        scopes: dict[str, list[str]] = {}
        for cnd in [*excl, *s.where]:
            d = dims.get(cnd.column)
            if cnd.op == "eq" and d and d["type"] == "text" and d["enumerable"]:
                scopes.setdefault(cnd.column, []).append(str(cnd.value))
                allowed = [x for x in allowed if x != cnd.column]
        metrics[k] = {"kind": "simple", "description": s.description,
            "expr": metric_expr(s, types, excl), "unit": s.unit, "decimals": s.decimals,
            "synonyms": s.synonyms, "allowed_dimensions": allowed,
            "implicit_scopes": scopes}  # fmt: skip
    for k, s in chosen.items():
        if s.aggregate:
            continue
        if s.numerator in metrics and s.denominator in metrics:  # both approved simple metrics
            n_dims = metrics[s.numerator]["allowed_dimensions"]
            d_dims = metrics[s.denominator]["allowed_dimensions"]
            metrics[k] = {"kind": "ratio", "description": s.description, "numerator": s.numerator,
                "denominator": s.denominator, "unit": s.unit, "decimals": s.decimals,
                "synonyms": s.synonyms,
                "allowed_dimensions": [d for d in n_dims if d in d_dims]}  # fmt: skip
    source = {"relation": m.relation, "alias": "l"}
    return SemanticLayer.model_validate(
        {"version": 1, "source": source, "dimensions": dims, "metrics": metrics}
    )


def _usable(s: MetricSpec, approved_only: bool) -> bool:
    return s.status == "approved" or not approved_only
