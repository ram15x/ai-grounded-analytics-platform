"""Configuration from environment variables only. No keys or passwords live in files."""

from __future__ import annotations

import os

from psycopg.conninfo import conninfo_to_dict

DEFAULT_CONNECT_TIMEOUT_S = 10
MAX_CONNECT_TIMEOUT_S = 300


class ConfigError(RuntimeError):
    pass


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"environment variable {name} is not set")
    return value


def admin_database_url() -> str:
    """Connection used by `grounded load` (owner of the warehouse schemas)."""
    return require_env("GA_DATABASE_URL")


def readonly_database_url() -> str:
    """Connection used for all guarded queries (SELECT-only role)."""
    return require_env("GA_READONLY_DATABASE_URL")


def connect_timeout_s() -> int:
    """Seconds to wait for a database connection to open (GA_CONNECT_TIMEOUT_S, default 10).

    Without a limit a connect to an unreachable `localhost` can wait minutes (on Windows `localhost`
    tries ::1 first, and a Docker port mapped only on IPv4 does not answer there)."""
    raw = os.environ.get("GA_CONNECT_TIMEOUT_S")
    if raw is None or not raw.strip():
        return DEFAULT_CONNECT_TIMEOUT_S
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_CONNECT_TIMEOUT_S:
        raise ConfigError(
            "GA_CONNECT_TIMEOUT_S must be a whole number of seconds from 1 to "
            f"{MAX_CONNECT_TIMEOUT_S}"
        )
    return value


def connect_kwargs(conninfo: str) -> dict:
    """Keyword arguments for `psycopg.connect`: the timeout, unless the URL already sets one."""
    if "connect_timeout" in conninfo_to_dict(conninfo):
        return {}
    return {"connect_timeout": connect_timeout_s()}
