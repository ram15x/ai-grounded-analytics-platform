"""Shared fixtures. Database tests run only when VA_TEST_DATABASE_URL points at a
Postgres server where the test user may create databases and roles, e.g.

    VA_TEST_DATABASE_URL="host=localhost port=5432 user=postgres dbname=postgres password=..."

The URL comes from the environment; nothing here stores a credential.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from verified_analyst.warehouse.load import load_file, read_raw
from verified_analyst.warehouse.normalize import Normalized

FIXTURE = Path(__file__).parent / "fixtures" / "mini_retail.csv"


@dataclass
class Warehouse:
    admin_url: str
    readonly_url: str
    role: str
    normalized: Normalized
    raw: pd.DataFrame


def _db_skip_reason() -> str | None:
    """None when database tests can run; otherwise the reason they are skipped."""
    base = os.environ.get("VA_TEST_DATABASE_URL")
    if not base:
        return (
            "database tests skipped: set VA_TEST_DATABASE_URL to a Postgres server where the "
            "user may create databases and roles (see tests/conftest.py)"
        )
    try:
        with psycopg.connect(base, connect_timeout=5):
            pass
    except psycopg.Error as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return f"database tests skipped: VA_TEST_DATABASE_URL is set but not reachable ({first})"
    return None


def pytest_collection_modifyitems(config, items):
    db_items = [i for i in items if "db" in i.keywords]
    if not db_items:
        return
    reason = _db_skip_reason()
    if reason is None:
        return
    if os.environ.get("VA_REQUIRE_DB") == "1":
        raise pytest.UsageError(
            reason.replace(
                "database tests skipped", "VA_REQUIRE_DB=1 but the test database is unusable"
            )
            + f" [{len(db_items)} database tests would have been skipped]"
        )
    skip = pytest.mark.skip(reason=reason)
    for item in db_items:
        item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_outbound_network(monkeypatch):
    """Python-level sockets may only reach loopback or unix sockets. Tests never use the network;
    Postgres goes through libpq, which does not use these sockets."""
    import socket

    real_connect = socket.socket.connect

    def guarded(self, address):
        # Windows has no AF_UNIX: look it up here, and treat its absence as "no unix sockets"
        unix = getattr(socket, "AF_UNIX", None)
        if (unix is not None and self.family == unix) or (
            isinstance(address, tuple) and str(address[0]) in ("127.0.0.1", "::1", "localhost")
        ):
            return real_connect(self, address)
        raise AssertionError(f"tests must not use the network (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda self, address: guarded(self, address))


@pytest.fixture(scope="session")
def raw_fixture() -> pd.DataFrame:
    return read_raw(FIXTURE)


def _make_warehouse(base: str, csv_path: Path) -> tuple[Warehouse, callable]:
    suffix = secrets.token_hex(4)
    dbname, role = f"va_test_{suffix}", f"va_ro_{suffix}"
    password = secrets.token_urlsafe(16)
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{dbname}"')
    admin_url = make_conninfo(base, dbname=dbname)
    normalized, _ = load_file(csv_path, admin_url, readonly_password=password, readonly_role=role)
    readonly_url = make_conninfo(base, dbname=dbname, user=role, password=password)
    wh = Warehouse(admin_url, readonly_url, role, normalized, read_raw(csv_path))

    def cleanup() -> None:
        with psycopg.connect(admin_url, autocommit=True) as c:
            c.execute(f'DROP OWNED BY "{role}"')
        with psycopg.connect(base, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
            admin.execute(f'DROP ROLE "{role}"')

    return wh, cleanup


@pytest.fixture(scope="session")
def warehouse():
    base = os.environ.get("VA_TEST_DATABASE_URL")
    if not base:
        pytest.skip("VA_TEST_DATABASE_URL not set")
    wh, cleanup = _make_warehouse(base, FIXTURE)
    yield wh
    cleanup()


@pytest.fixture(scope="session")
def warehouse_factory(tmp_path_factory):
    """Build extra warehouses from hand-written raw frames (for hand-checked tests)."""
    base = os.environ.get("VA_TEST_DATABASE_URL")
    if not base:
        pytest.skip("VA_TEST_DATABASE_URL not set")
    cleanups = []

    def make(frame: pd.DataFrame) -> Warehouse:
        path = tmp_path_factory.mktemp("wh") / "raw.csv"
        frame.to_csv(path, index=False)
        wh, cleanup = _make_warehouse(base, path)
        cleanups.append(cleanup)
        return wh

    yield make
    for fn in cleanups:
        fn()
