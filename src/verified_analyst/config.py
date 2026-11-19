"""Configuration from environment variables only. No keys or passwords live in files."""

from __future__ import annotations

import os


class ConfigError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"environment variable {name} is not set")
    return value


def admin_database_url() -> str:
    """Connection used by `va load` (owner of the warehouse schemas)."""
    return require_env("VA_DATABASE_URL")


def readonly_database_url() -> str:
    """Connection used for all guarded queries (SELECT-only role)."""
    return require_env("VA_READONLY_DATABASE_URL")
