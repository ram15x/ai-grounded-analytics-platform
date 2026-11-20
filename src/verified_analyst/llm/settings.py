"""Router configuration, loaded from config/router.yaml. Model IDs live only there.

Every limit in the file is UNVERIFIED (see the comments in the file and
docs/PHASE_1C1_ASSUMPTIONS.md)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import ConfigError


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderSettings(_M):
    kind: Literal["groq", "gemini", "openrouter", "fake"]
    hosted: bool
    env_key: str | None = None
    endpoint: str | None = None
    timeout_s: float = Field(default=30, gt=0, le=300)
    rpm: float | None = Field(default=None, gt=0)  # documented requests per minute (UNVERIFIED)
    tpd: float | None = Field(default=None, gt=0)  # documented tokens per day (UNVERIFIED)
    models: dict[str, str] = Field(default_factory=dict)  # task or "default" -> model id

    def model_for(self, task: str) -> str:
        model = self.models.get(task) or self.models.get("default")
        if not model:
            raise ConfigError(f"no model configured for task {task!r}")
        return model


class RetrySettings(_M):
    max_retries: int = Field(default=3, ge=0, le=10)
    base_s: float = Field(default=1.0, ge=0)
    cap_s: float = Field(default=20.0, ge=0)
    jitter: float = Field(default=0.5, ge=0, le=1)  # fraction of the delay that is random
    max_retry_after_s: float = Field(default=30.0, ge=0)  # longer waits fall through instead


class BreakerSettings(_M):
    consecutive_429: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=60.0, ge=0)


class BucketSettings(_M):
    fraction: float = Field(default=0.5, gt=0, le=1)  # share of the documented limit to use
    max_wait_s: float = Field(default=10.0, ge=0)  # longer waits fall through to the next provider


class BudgetSettings(_M):
    per_run_tokens: int = Field(default=30_000, gt=0)


class ProbeSettings(_M):
    max_requests: int = Field(default=25, ge=1, le=25)  # hard ceiling of 25 per probe run
    token_cap: int = Field(default=60_000, gt=0)  # across the whole probe run


class RouterSettings(_M):
    providers: dict[str, ProviderSettings]
    tasks: dict[str, list[str]]
    datasets: dict[str, Literal["public", "internal"]] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    breaker: BreakerSettings = Field(default_factory=BreakerSettings)
    bucket: BucketSettings = Field(default_factory=BucketSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    probe: ProbeSettings = Field(default_factory=ProbeSettings)

    @model_validator(mode="after")
    def _check(self) -> RouterSettings:
        for task, order in self.tasks.items():
            if not order:
                raise ValueError(f"task {task!r} has no providers")
            for name in order:
                if name not in self.providers:
                    raise ValueError(f"task {task!r} names unknown provider {name!r}")
        for name, p in self.providers.items():
            if p.kind != "fake" and not p.hosted and not p.endpoint:
                raise ValueError(f"{name}: a non-hosted provider needs an endpoint")
            if p.hosted and p.kind != "fake" and not p.env_key:
                raise ValueError(f"{name}: a hosted provider needs env_key")
        return self

    def sensitivity_of(self, datasets: tuple[str, ...]) -> str:
        """Internal if any dataset is internal or untagged (fail closed); public otherwise."""
        for ds in datasets:
            if self.datasets.get(ds) != "public":
                return "internal"
        return "public"


def default_router_path() -> Path:
    env = os.environ.get("VA_ROUTER_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "config" / "router.yaml"


def load_router_settings(path: Path | None = None) -> RouterSettings:
    path = path or default_router_path()
    try:
        return RouterSettings.model_validate(yaml.safe_load(path.read_text("utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"cannot load router config {path}: {exc}") from exc
