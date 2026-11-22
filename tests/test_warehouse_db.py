"""Warehouse tests against a real Postgres (skipped without GA_TEST_DATABASE_URL)."""

import psycopg
import pytest

pytestmark = pytest.mark.db


def q(url, sql, params=None):
    with psycopg.connect(url) as c:
        return c.execute(sql, params).fetchall()


def test_row_counts(warehouse):
    dq = warehouse.normalized.dq
    got = q(
        warehouse.admin_url,
        "SELECT (SELECT count(*) FROM warehouse.fact_invoice_lines),"
        " (SELECT count(*) FROM warehouse.fact_invoice),"
        " (SELECT count(*) FROM warehouse.dim_customer),"
        " (SELECT count(*) FROM warehouse.dim_product)",
    )[0]
    assert got == (dq["lines"], dq["invoices"], dq["customers"], dq["products"])


def test_line_amount_is_exact_product(warehouse):
    bad = q(
        warehouse.admin_url,
        "SELECT count(*) FROM warehouse.fact_invoice_lines"
        " WHERE line_amount <> quantity * unit_price",
    )
    assert bad[0][0] == 0


def test_missing_customer_invoices_have_null_key(warehouse):
    n = q(
        warehouse.admin_url,
        "SELECT count(*) FROM warehouse.fact_invoice WHERE customer_key IS NULL",
    )
    assert n[0][0] == warehouse.normalized.dq["invoices_without_customer"] == 19


def test_view_matches_lines(warehouse):
    n = q(warehouse.readonly_url, "SELECT count(*) FROM analytics.sales_lines")[0][0]
    assert n == 368


def test_line_type_counts_in_db(warehouse):
    rows = dict(
        q(
            warehouse.readonly_url,
            "SELECT line_type, count(*) FROM analytics.sales_lines GROUP BY 1",
        )
    )
    assert rows == warehouse.normalized.dq["line_type_counts"]


# ---- the read-only role -------------------------------------------------------------


@pytest.mark.parametrize(
    "stmt",
    [
        "INSERT INTO analytics.products VALUES ('X', 'y', 'merchandise')",
        "UPDATE analytics.invoices SET country = 'x'",
        "DELETE FROM analytics.customers",
        "CREATE TABLE analytics.t (a int)",
        "CREATE TABLE public.t (a int)",
        "DROP VIEW analytics.products",
        "TRUNCATE warehouse.fact_invoice_lines",
        "SELECT * FROM warehouse.fact_invoice_lines",
        "SELECT * FROM warehouse.dim_customer",
    ],
)
def test_readonly_role_cannot_write_or_see_base_tables(warehouse, stmt):
    with psycopg.connect(warehouse.readonly_url) as c, pytest.raises(psycopg.Error):
        c.execute(stmt)


def test_readonly_role_defaults(warehouse):
    with psycopg.connect(warehouse.readonly_url) as c:
        assert c.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
        assert c.execute("SHOW statement_timeout").fetchone()[0] == "10s"
        assert c.execute("SHOW search_path").fetchone()[0] == "analytics"


def test_readonly_role_is_not_superuser(warehouse):
    row = q(
        warehouse.admin_url,
        "SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname = %s",
        (warehouse.role,),
    )
    assert row == [(False, False, False)]
