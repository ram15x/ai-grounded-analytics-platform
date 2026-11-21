import os
import secrets

import pandas as pd
import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from test_dataset_manifest import DF, PROFILE, manifest  # noqa: I001
from verified_analyst.compiler import MetricQuery, compile_query
from verified_analyst.datasets import load as L
from verified_analyst.datasets import manifest as M
from verified_analyst.datasets.profile import profile_frame
from verified_analyst.guard import allowed_for, execute_guarded

pytestmark = pytest.mark.db


@pytest.fixture(scope="module")
def scratch():
    base = os.environ.get("VA_TEST_DATABASE_URL")
    if not base:
        pytest.skip("VA_TEST_DATABASE_URL not set")
    suffix = secrets.token_hex(4)
    dbname, role, pw = f"va_ds_{suffix}", f"va_dsro_{suffix}", secrets.token_urlsafe(12)
    with psycopg.connect(base, autocommit=True) as a:
        a.execute(f'CREATE DATABASE "{dbname}"')
        a.execute(f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'")
    admin = make_conninfo(base, dbname=dbname)
    ro = make_conninfo(base, dbname=dbname, user=role, password=pw)
    yield admin, ro, role
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'DROP OWNED BY "{role}"')
    with psycopg.connect(base, autocommit=True) as a:
        a.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
        a.execute(f'DROP ROLE "{role}"')


def test_load_then_ask_through_the_existing_compiler_and_guard(scratch):
    admin, ro, role = scratch
    m = manifest(
        columns={**manifest().columns, "customer": M.Column(role="ignore", sensitivity="personal")},
        metrics={
            k: v for k, v in manifest().model_dump().get("metrics", {}).items() if k != "customers"
        },
    )
    info = L.load_dataset(DF, PROFILE, m, admin, role)
    assert info["rows"] == 3 and "customer" not in info["loaded_columns"]
    assert "customer" in info["not_loaded"]
    layer = M.build_layer(m, PROFILE)
    cq = compile_query(
        layer, MetricQuery.model_validate({"metrics": ["revenue"], "dimensions": ["region"]})
    )
    res = execute_guarded(ro, cq.sql, allowed_relations=allowed_for(layer))
    assert {r[0]: float(r[1]) for r in res.rows} == {
        "North": 10.5,
        "South": 20.0,
    }  # refunded row out
    with psycopg.connect(ro) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("DELETE FROM ds_shop.data")


def test_reconcile_catches_a_definition_that_does_not_match_the_persons_figure(scratch):
    admin, ro, role = scratch
    m = manifest()
    L.load_dataset(DF, PROFILE, m, admin, role)
    good = {
        "value": "30.5",
        "where": [{"column": "region", "op": "in", "value": ["North", "South"]}],
    }
    m.metrics["revenue"].known = [M.Known.model_validate(good)]
    res = L.reconcile(m, PROFILE, ro)
    assert res[0]["ok"] is True and res[0]["got"] == "30.5"
    L.check_reconciliation(res)
    m.metrics["revenue"].known = [M.Known.model_validate({"value": "35.5"})]  # counts the refund
    with pytest.raises(M.ManifestError, match="does not reproduce"):
        L.check_reconciliation(L.reconcile(m, PROFILE, ro))


def test_values_that_do_not_fit_are_refused_not_nulled(scratch):
    admin, _, role = scratch
    bad = DF.copy()
    bad.loc[1, "amount"] = "twenty"
    with pytest.raises(L.LoadError, match="amount: 1 values are not numbers"):
        L.load_dataset(bad, profile_frame(bad), manifest(), admin, role)
    amb = pd.DataFrame(
        {**DF.to_dict("list"), "order_date": ["03/01/2024", "04/02/2024", "05/03/2024"]}
    )
    with pytest.raises(L.LoadError, match="date_order"):
        L.load_dataset(amb, profile_frame(amb), manifest(), admin, role)
    m = manifest()
    m.columns["order_date"].date_order = "dmy"
    prof = profile_frame(amb)
    L.load_dataset(amb, prof, m, admin, role)
    with psycopg.connect(admin) as c:
        assert (
            str(c.execute("SELECT MIN(order_date) FROM ds_shop.data").fetchone()[0]) == "2024-01-03"
        )
