"""Pydantic schema for the structured answer a composer must produce (ARCHITECTURE 4.1, step A).

The composer is untrusted. Everything here is validated, and nothing here is believed until the
verifier has checked it against executed query results.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .compiler import MetricQuery

DERIVED_OPS = (
    "sum",
    "difference",
    "ratio",
    "percent_change",
    "percent_of",
    "mean",
    "min",
    "max",
    "count",
)
Op = Literal[
    "sum", "difference", "ratio", "percent_change", "percent_of", "mean", "min", "max", "count"
]  # noqa: E501


def to_decimal(value: Any) -> Decimal:
    """Exact conversion. Floats are taken via their shortest repr, never trusted for arithmetic."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a number: {value!r}") from exc


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CellRef(_Strict):
    kind: Literal["cell"] = "cell"
    query_id: str
    row_key: dict[str, Any] = Field(default_factory=dict)
    column: str


class RowCountRef(_Strict):
    kind: Literal["row_count"] = "row_count"
    query_id: str


class LiteralArg(_Strict):
    kind: Literal["literal"] = "literal"
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def _dec(cls, v: Any) -> Decimal:
        return to_decimal(v)


Arg = Annotated[CellRef | RowCountRef | LiteralArg, Field(discriminator="kind")]


class DerivedRef(_Strict):
    kind: Literal["derived"] = "derived"
    op: Op
    args: list[Arg] = Field(min_length=1, max_length=50)


class PassageRef(_Strict):
    kind: Literal["passage"] = "passage"
    passage_id: str
    span: tuple[int, int]


class ParameterRef(_Strict):
    """A chosen number (top 5, last 30 days) that appears as a literal in the SQL."""

    kind: Literal["parameter"] = "parameter"
    query_id: str


Provenance = Annotated[
    CellRef | RowCountRef | DerivedRef | PassageRef | ParameterRef, Field(discriminator="kind")
]


class Claim(_Strict):
    id: str
    text: str = Field(min_length=1, description="The number exactly as written in the prose")
    value: Decimal = Field(description="The value as displayed (12.5 for '12.5%', not 0.125)")
    unit: Literal["currency", "count", "percent", "percentage_points", "ratio", "number"]
    label: str = Field(min_length=1, description="What the number measures, in business words")
    provenance: Provenance

    @field_validator("value", mode="before")
    @classmethod
    def _dec(cls, v: Any) -> Decimal:
        return to_decimal(v)


class TemplateClaim(_Strict):
    """A claim in placeholder mode. There is NO text and NO value: the system renders the number.

    The composer chooses which cell or derivation backs the placeholder, the unit it is shown in,
    and optionally decimals and a scale. It never writes the digits.
    """

    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
    label: str = Field(min_length=1, description="What the number measures, in business words")
    unit: Literal["currency", "count", "percent", "percentage_points", "ratio", "number"]
    provenance: Annotated[CellRef | RowCountRef | DerivedRef, Field(discriminator="kind")]
    decimals: int | None = Field(default=None, ge=0, le=6)
    scale: Literal["none", "thousand", "million", "billion"] = "none"

    @field_validator("provenance")
    @classmethod
    def _no_literals(cls, v: Any) -> Any:
        if isinstance(v, DerivedRef) and any(isinstance(a, LiteralArg) for a in v.args):
            raise ValueError("derived claims in a template may not take literal arguments")
        return v


class QuerySpec(_Strict):
    """How a query in the answer was produced.

    Exactly one of `metric_query` (compiled path) or `sql` (free-form, guarded). For free-form SQL,
    `declared_metric_query` states the canonical metric query it is meant to equal, enabling
    shadow recomputation.
    """

    id: str
    metric_query: MetricQuery | None = None
    sql: str | None = None
    declared_metric_query: MetricQuery | None = None

    @field_validator("sql")
    @classmethod
    def _nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("empty sql")
        return v

    def model_post_init(self, __context: Any) -> None:
        if (self.metric_query is None) == (self.sql is None):
            raise ValueError("give exactly one of metric_query or sql")
        if self.metric_query is not None and self.declared_metric_query is not None:
            raise ValueError("declared_metric_query is only for free-form sql")


class Passage(_Strict):
    id: str
    text: str


class AnswerDoc(_Strict):
    question: str = ""
    prose: str
    queries: list[QuerySpec] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


class TemplateAnswer(_Strict):
    """Placeholder-mode answer: prose with `{claim_id}` placeholders instead of numbers."""

    question: str = ""
    template: str
    queries: list[QuerySpec] = Field(default_factory=list)
    claims: list[TemplateClaim] = Field(default_factory=list)
