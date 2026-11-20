"""Semantic layer: pydantic models and loader for semantic/metrics.yaml."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


class SemanticError(ValueError):
    pass


def norm_term(text: str) -> str:
    """Normalise a label for matching: lower-case, alphanumerics only, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


class Dimension(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    type: Literal["text", "date"]
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    enumerable: bool = False  # small closed set: the verifier may load all its values


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["simple", "ratio"]
    description: str
    expr: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    unit: str
    decimals: int = Field(ge=0, le=8)
    synonyms: list[str] = Field(default_factory=list)
    allowed_dimensions: list[str]

    @model_validator(mode="after")
    def _shape(self) -> Metric:
        if self.kind == "simple" and not self.expr:
            raise ValueError("simple metric needs expr")
        if self.kind == "ratio" and not (self.numerator and self.denominator):
            raise ValueError("ratio metric needs numerator and denominator")
        if self.kind == "simple" and (self.numerator or self.denominator):
            raise ValueError("simple metric must not set numerator/denominator")
        if not re.fullmatch(r"currency:[A-Z]{3}|count|ratio", self.unit):
            raise ValueError(f"bad unit {self.unit!r}")
        return self

    @property
    def unit_kind(self) -> str:
        return self.unit.split(":")[0]


class Source(BaseModel):
    relation: str
    alias: str


class SemanticLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    source: Source
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]

    @model_validator(mode="after")
    def _check(self) -> SemanticLayer:
        for name in [*self.dimensions, *self.metrics]:
            if not IDENT.match(name):
                raise ValueError(f"bad identifier {name!r}")
        owners: dict[str, str] = {}
        for name, m in self.metrics.items():
            for term in [name, *m.synonyms]:
                key = norm_term(term)
                if key in owners and owners[key] != name:
                    raise ValueError(f"term {term!r} used by both {owners[key]} and {name}")
                owners[key] = name
            for d in m.allowed_dimensions:
                if d not in self.dimensions:
                    raise ValueError(f"{name}: unknown dimension {d!r}")
            if m.kind == "ratio":
                for ref in (m.numerator, m.denominator):
                    if ref not in self.metrics:
                        raise ValueError(f"{name}: unknown metric {ref!r}")
                    if self.metrics[ref].kind != "simple":
                        raise ValueError(f"{name}: ratio of ratios is not supported")
        return self

    def resolve_metric(self, label: str) -> str | None:
        """Map a free-text label to a metric name via name/synonym match, else None."""
        key = norm_term(label)
        for name, m in self.metrics.items():
            if key in {norm_term(name), *(norm_term(s) for s in m.synonyms)}:
                return name
        return None


def default_path() -> Path:
    env = os.environ.get("VA_SEMANTIC_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "semantic" / "metrics.yaml"


def load_semantic_layer(path: Path | None = None) -> SemanticLayer:
    path = path or default_path()
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
        return SemanticLayer.model_validate(data)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise SemanticError(f"cannot load semantic layer {path}: {exc}") from exc
