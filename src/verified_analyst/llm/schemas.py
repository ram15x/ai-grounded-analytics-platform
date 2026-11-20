"""JSON shapes the models are asked to produce. The plan is a MetricQuery or an abstention; the
composer supplies only prose with placeholders and claims (ADR-0008). The system adds `queries`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..claims import TemplateClaim
from ..compiler import MetricQuery

ABSTAIN_REASONS = (
    "unsupported_metric",  # not in the data or the catalogue (for example no cost column)
    "forecast",
    "causal",
    "out_of_period",
    "ambiguous",
    "personal_data",
    "out_of_scope",
)


class PlanOutput(BaseModel):
    """Exactly one of `metric_query` or `abstain`."""

    model_config = ConfigDict(extra="forbid")
    metric_query: MetricQuery | None = None
    abstain: (
        Literal[
            "unsupported_metric",
            "forecast",
            "causal",
            "out_of_period",
            "ambiguous",
            "personal_data",
            "out_of_scope",
        ]
        | None
    ) = None

    def model_post_init(self, __context: object) -> None:
        if (self.metric_query is None) == (self.abstain is None):
            raise ValueError("give exactly one of metric_query or abstain")


class ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: str = Field(min_length=1)
    claims: list[TemplateClaim] = Field(default_factory=list)
