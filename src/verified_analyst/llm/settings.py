"""Router configuration, loaded from config/router.yaml. Model IDs live only there.

Every limit in the file is UNVERIFIED unless its comment in the file says where it was seen
(see the comments in the file and docs/PHASE_1C1_ASSUMPTIONS.md)."""

from __future__ import annotations

import json
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
    tpm: float | None = Field(default=None, gt=0)  # documented tokens per minute (UNVERIFIED)
    # Tokens per rolling 24 h (prompt + completion) MEASURED for a specific model id, from a 429
    # body that names the limit. `tpd` is the fallback for any other model and stays UNVERIFIED.
    daily_tokens: dict[str, int] = Field(default_factory=dict)
    # Sent as `reasoning_effort` on every request to this provider, only when set. UNVERIFIED:
    # whether a given Groq model accepts it, and which values, is not known. Groq only for now.
    reasoning_effort: str | None = Field(default=None, pattern=r"^[a-z]{1,16}$")
    # For a model that reports reasoning tokens: compose and repair requests get at least this
    # completion allowance (a request with a fixed allowance, such as `--max-tokens`, keeps it).
    reasoning_min_tokens: int | None = Field(default=None, gt=0)
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


class RateLimitSettings(_M):
    """HTTP 429: wait the provider's Retry-After (or, if it sent none, until the token bucket can
    cover the request again) and retry this many times; a wait longer than max_wait_s falls
    through to the next provider instead of being shortened (a shortened wait would just be a
    blind retry)."""

    max_wait_s: float = Field(default=60.0, ge=0)
    retries: int = Field(default=1, ge=0, le=3)


class BreakerSettings(_M):
    consecutive_429: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=60.0, ge=0)


class BucketSettings(_M):
    fraction: float = Field(default=0.5, gt=0, le=1)  # share of the documented limit to use
    max_wait_s: float = Field(default=10.0, ge=0)  # longer waits fall through to the next provider


class BudgetSettings(_M):
    per_run_tokens: int = Field(default=30_000, gt=0)


class ProbeSettings(_M):
    max_requests: int = Field(default=25, ge=1, le=60)  # default cap; hard ceiling 60 per run
    token_cap: int = Field(default=60_000, gt=0)  # across the whole probe run


class RouterSettings(_M):
    providers: dict[str, ProviderSettings]
    tasks: dict[str, list[str]]
    datasets: dict[str, Literal["public", "internal"]] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
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
            if p.reasoning_effort and p.kind != "groq":
                raise ValueError(
                    f"{name}: reasoning_effort is only wired for groq; no parameter name has "
                    f"been checked for {p.kind}"
                )
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


def apply_dataset_visibility(cfg: RouterSettings, layer_path: Path) -> RouterSettings:
    """Read `visibility.json` next to an uploaded dataset's layer file and register it. Anything
    missing or malformed leaves the dataset unregistered, which the router treats as internal."""
    f = layer_path.parent / "visibility.json"
    try:
        info = json.loads(f.read_text("utf-8"))
        relation, vis = info["relation"], info["visibility"]
    except (OSError, ValueError, KeyError, TypeError):
        return cfg
    if isinstance(relation, str) and vis in ("public", "internal"):
        cfg.datasets[relation] = vis
    return cfg


def load_router_settings(path: Path | None = None) -> RouterSettings:
    path = path or default_router_path()
    try:
        cfg = RouterSettings.model_validate(yaml.safe_load(path.read_text("utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"cannot load router config {path}: {exc}") from exc
    layer = os.environ.get("VA_SEMANTIC_PATH")
    return apply_dataset_visibility(cfg, Path(layer)) if layer else cfg
