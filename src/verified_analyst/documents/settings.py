"""Embedder and reranker candidates, loaded from config/retrieval.yaml.

Model names live only in that file. Each is a candidate to be measured by `va docs probe`; none
has been called yet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from verified_analyst.config import ConfigError


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbedderSettings(_M):
    kind: Literal["openai_compat", "gemini", "ollama", "local", "fake"]
    hosted: bool
    model: str = "fake"
    env_key: str | None = None
    base_url: str | None = None
    dims: int | None = Field(default=None, ge=8, le=8192)
    document_prefix: str = ""
    query_prefix: str = ""
    batch_size: int = Field(default=16, ge=1, le=128)
    timeout_s: float = Field(default=60, gt=0, le=300)
    # gemini only: "prompt" puts the task in the text (gemini-embedding-2), "task_type" sends
    # taskType (gemini-embedding-001).
    task_style: Literal["prompt", "task_type"] = "task_type"
    normalize: bool = False  # L2-normalise what comes back (needed for gemini-embedding-001 < 3072)
    send_dimensions: bool = False  # openai_compat: send `dimensions`; UNVERIFIED for OpenRouter
    cache_dir: str | None = None  # local: where model files are kept (default: the library's own)

    @model_validator(mode="after")
    def _check(self) -> EmbedderSettings:
        if self.kind not in ("fake", "ollama", "local") and not self.env_key:
            raise ValueError("a hosted embedder needs env_key")
        if self.kind == "ollama" and self.hosted:
            raise ValueError("an ollama embedder runs on this computer: hosted must be false")
        if self.kind == "local" and (self.hosted or not self.dims):
            raise ValueError("a local embedder runs on this computer: hosted false, and dims set")
        if self.kind in ("openai_compat", "ollama") and not self.base_url:
            raise ValueError(f"{self.kind} needs base_url")
        if self.kind in ("openai_compat", "gemini") and not self.hosted:
            raise ValueError(f"{self.kind} is a hosted service: hosted must be true")
        return self


class RerankerSettings(_M):
    kind: Literal["rerank_api", "cross_encoder", "fake"]
    hosted: bool
    model: str = "fake"
    url: str | None = None
    env_key: str | None = None
    timeout_s: float = Field(default=60, gt=0, le=300)
    depth: int = Field(default=20, ge=2, le=100)  # candidates handed to the reranker
    min_interval_s: float = Field(default=0.0, ge=0, le=30)  # pause between requests (rate limits)
    cache_dir: str | None = None  # cross_encoder: where model files are kept

    @model_validator(mode="after")
    def _check(self) -> RerankerSettings:
        if self.kind == "rerank_api" and not (self.url and self.env_key and self.hosted):
            raise ValueError("rerank_api needs url, env_key and hosted: true")
        if self.kind == "cross_encoder" and self.hosted:
            raise ValueError("cross_encoder runs on this computer: hosted must be false")
        return self


class ProbeSettings(_M):
    max_requests: int = Field(default=60, ge=1, le=150)  # default cap; hard ceiling 150 per run


class RetrievalSettings(_M):
    embedders: dict[str, EmbedderSettings]
    rerankers: dict[str, RerankerSettings] = Field(default_factory=dict)
    probe: ProbeSettings = Field(default_factory=ProbeSettings)


def default_retrieval_path() -> Path:
    env = os.environ.get("VA_RETRIEVAL_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "config" / "retrieval.yaml"


def load_retrieval_settings(path: Path | None = None) -> RetrievalSettings:
    path = path or default_retrieval_path()
    try:
        return RetrievalSettings.model_validate(yaml.safe_load(path.read_text("utf-8")))
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigError(f"cannot load retrieval config {path}: {exc}") from exc
