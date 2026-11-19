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


def pytest_collection_modifyitems(config, items):
    if os.environ.get("VA_TEST_DATABASE_URL"):
        return
    skip = pytest.mark.skip(reason="VA_TEST_DATABASE_URL not set")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


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
