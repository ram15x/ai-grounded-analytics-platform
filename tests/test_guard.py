"""SQL guard: adversarial rejections, accepted shapes, rewriting, fuzz invariants.

Pure-parser tests run everywhere; execution tests need Postgres (db marker)."""

import random

import pytest
import sqlglot
from sqlglot import exp

from verified_analyst.compiler import compile_query
from verified_analyst.guard import (
    GuardError,
    QueryTimeout,
    execute_guarded,
    validate_sql,
    wrap_limit,
)
from verified_analyst.semantic import load_semantic_layer

# (sql, expected reason code)
REJECTED = [
    # multiple statements and statement smuggling
    ("SELECT 1; SELECT 2", "multiple_statements"),
    ("SELECT 1; DROP TABLE analytics.products", "multiple_statements"),
    ("SELECT 1 /* x */; DROP TABLE y", "multiple_statements"),
    ("SELECT 1; ; SELECT 2", "multiple_statements"),
    (
        "SELECT 1; -- a comment after the semicolon counts as a second statement",
        "multiple_statements",
    ),
    ("BEGIN; SELECT 1", "multiple_statements"),
    (
        "SELECT * FROM analytics.sales_lines WHERE country = 'a' ; DELETE FROM x",
        "multiple_statements",
    ),
    ("SELECT 1 /* /* nested */ ; DROP TABLE x; /* */", "parse_error"),
    # DML / DDL / utility
    ("DROP TABLE analytics.products", "not_select"),
    ("DELETE FROM analytics.products", "not_select"),
    ("UPDATE analytics.products SET description = 'x'", "not_select"),
    ("INSERT INTO analytics.products VALUES ('a', 'b', 'merchandise')", "not_select"),
    ("CREATE TABLE t AS SELECT 1", "not_select"),
    ("ALTER TABLE analytics.products ADD COLUMN x int", "not_select"),
    ("TRUNCATE analytics.products", "not_select"),
    ("COPY analytics.products TO '/tmp/x'", "not_select"),
    ("COPY (SELECT 1) TO PROGRAM 'id'", "not_select"),
    ("SET statement_timeout = 0", "not_select"),
    ("RESET ALL", "not_select"),
    ("SHOW ALL", "not_select"),
    ("EXPLAIN ANALYZE SELECT 1", "not_select"),
    ("GRANT ALL ON analytics.products TO public", "not_select"),
    ("COMMIT", "not_select"),
    ("VACUUM", "not_select"),
    ("DO $$ BEGIN PERFORM 1; END $$", "not_select"),
    ("CALL f()", "not_select"),
    ("LISTEN x", "not_select"),
    ("NOTIFY x", "not_select"),
    ("PREPARE x AS SELECT 1", "not_select"),
    ("EXECUTE x", "not_select"),
    ("DISCARD ALL", "not_select"),
    ("CREATE FUNCTION f() RETURNS int AS 'select 1' LANGUAGE sql", "not_select"),
    ("VALUES (1)", "not_select"),
    # SELECT variants that write or lock
    ("SELECT * INTO newt FROM analytics.products", "forbidden_construct"),
    ("SELECT * FROM analytics.products FOR UPDATE", "forbidden_construct"),
    (
        "WITH d AS (DELETE FROM analytics.products RETURNING *) SELECT * FROM d",
        "forbidden_construct",
    ),
    ("SELECT $1", "forbidden_construct"),
    ("SELECT current_user", "forbidden_construct"),
    ("SELECT session_user", "forbidden_construct"),
    ("SELECT version()", "forbidden_construct"),
    # dangerous functions (case, quoting and qualification tricks included)
    ("SELECT pg_sleep(100)", "function_denied"),
    ("SeLeCt PG_SLEEP(1)", "function_denied"),
    ('SELECT "pg_sleep"(1)', "function_denied"),
    ("SELECT pg_catalog.pg_sleep(1)", "function_denied"),
    ("SELECT pg_read_file('/etc/passwd')", "function_denied"),
    ("SELECT pg_ls_dir('.')", "function_denied"),
    ("SELECT pg_terminate_backend(1)", "function_denied"),
    ("SELECT pg_advisory_lock(1)", "function_denied"),
    ("SELECT lo_import('/etc/passwd')", "function_denied"),
    ("SELECT lo_unlink(1)", "function_denied"),
    ("SELECT dblink('host=x', 'select 1')", "function_denied"),
    ("SELECT set_config('statement_timeout', '0', false)", "function_denied"),
    ("SELECT current_setting('server_version')", "function_denied"),
    ("SELECT nextval('s')", "function_denied"),
    ("SELECT current_database()", "function_denied"),
    ("SELECT query_to_xml('select 1', true, true, '')", "function_denied"),
    ("SELECT read_csv_auto('x')", "function_denied"),
    ("SELECT 1 UNION SELECT pg_read_file('x')", "function_denied"),
    ("SELECT * FROM read_csv('x')", "function_denied"),
    ("SELECT * FROM generate_series(1, 10)", "table_function"),
    # relations outside the allow-list
    ("SELECT * FROM pg_catalog.pg_roles", "relation_denied"),
    ("SELECT * FROM pg_roles", "relation_denied"),
    ("SELECT * FROM information_schema.tables", "relation_denied"),
    ("SELECT * FROM warehouse.fact_invoice", "relation_denied"),
    ("SELECT * FROM public.t", "relation_denied"),
    ("SELECT * FROM otherdb.analytics.products", "relation_denied"),
    ("SELECT * FROM analytics.products, pg_authid", "relation_denied"),
    ("SELECT (SELECT rolpassword FROM pg_authid)", "relation_denied"),
    (
        "SELECT * FROM analytics.products WHERE EXISTS (SELECT 1 FROM warehouse.dim_customer)",
        "relation_denied",
    ),
    ("WITH products AS (SELECT 1) SELECT * FROM warehouse.fact_invoice", "relation_denied"),
    ("SELECT * FROM analytics.products UNION ALL SELECT * FROM pg_user", "relation_denied"),
    # malformed input
    ("", "empty"),
    ("   ;  ", "empty"),
    ("SELECT", "parse_error"),
    ("SELECT * FROM analytics.products WHERE", "parse_error"),
    ("SELECT 1\x00", "null_byte"),
    ("SELECT '" + "x" * 25_000 + "'", "too_long"),
]


@pytest.mark.parametrize(("sql", "code"), REJECTED)
def test_rejected(sql, code):
    with pytest.raises(GuardError) as err:
        validate_sql(sql)
    assert err.value.code == code


def test_at_least_forty_adversarial_cases():
    assert len(REJECTED) >= 40


def test_non_string_rejected():
    with pytest.raises(GuardError, match="not_text"):
        validate_sql(None)  # type: ignore[arg-type]


ACCEPTED = [
    "SELECT 1",
    "SELECT count(*) FROM analytics.sales_lines",
    "SELECT count(*) FROM sales_lines",
    "SELECT count(*) FROM ANALYTICS.SALES_LINES",
    "SELECT 1;",
    "SELECT 1 -- trailing comment",
    "/* lead */ SELECT country, sum(line_amount) FROM analytics.sales_lines GROUP BY 1",
    "WITH m AS (SELECT country, count(*) n FROM analytics.sales_lines GROUP BY 1) SELECT * FROM m",
    "SELECT * FROM analytics.products p"
    " JOIN analytics.sales_lines s ON s.stock_code = p.stock_code",
    "SELECT country FROM analytics.customers UNION SELECT country FROM analytics.invoices",
    "SELECT * FROM (SELECT stock_code FROM analytics.products) AS q",
    "SELECT stock_code, rank() OVER (ORDER BY line_amount DESC) FROM analytics.sales_lines",
    "SELECT count(*) FROM analytics.sales_lines WHERE country = 'a;b'",
    "SELECT 'DROP TABLE x; pg_sleep(5)' AS s",
    "SELECT * FROM analytics.sales_lines LIMIT 5 OFFSET 2",
    "(SELECT 1)",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_accepted_and_regenerated_as_one_select(sql):
    g = validate_sql(sql)
    parsed = sqlglot.parse(g.validated, dialect="postgres")
    assert len(parsed) == 1
    assert isinstance(parsed[0], exp.Select | exp.SetOperation)
    assert ";" not in g.validated.replace("a;b", "").replace("x; pg_sleep", "")


def test_comments_do_not_survive_regeneration():
    g = validate_sql("SELECT 1 /* hidden */ -- more\n")
    assert g.validated == "SELECT 1"


def test_table_command_is_normalised_to_a_select():
    assert validate_sql("TABLE analytics.products").validated == "SELECT * FROM analytics.products"


def test_relations_reported():
    g = validate_sql("SELECT * FROM products p JOIN analytics.sales_lines s ON true")
    assert sorted(g.relations) == ["analytics.products", "analytics.sales_lines"]


def test_cte_names_do_not_widen_access():
    with pytest.raises(GuardError, match="relation_denied"):
        validate_sql("WITH x AS (SELECT * FROM pg_authid) SELECT * FROM x")


def test_custom_allow_list():
    allowed = frozenset({"analytics.products"})
    validate_sql("SELECT * FROM analytics.products", allowed)
    with pytest.raises(GuardError, match="relation_denied"):
        validate_sql("SELECT * FROM analytics.customers", allowed)


@pytest.mark.parametrize("limit", [0, -1, 100_001, True, "5", 2.5])
def test_bad_row_limit(limit):
    with pytest.raises(GuardError, match="bad_limit"):
        wrap_limit("SELECT 1", limit)


def test_wrap_limit_asks_for_one_extra_row():
    assert wrap_limit("SELECT 1", 10) == "SELECT * FROM (SELECT 1) AS _va_q LIMIT 11"


def test_every_compiled_metric_query_passes_the_guard():
    layer = load_semantic_layer()
    for name, m in layer.metrics.items():
        sql = compile_query(layer, {"metrics": [name], "dimensions": m.allowed_dimensions[:2]}).sql
        validate_sql(sql)


def test_fuzz_invariant_validated_sql_is_one_select_over_allowed_relations():
    """Random token injection: whatever the guard accepts must be a single SELECT that
    touches only allow-listed relations. Seeded, so it is deterministic."""
    rng = random.Random(7)
    base = [
        "SELECT country, count(*) FROM analytics.sales_lines WHERE line_type = 'sale' GROUP BY 1",
        "SELECT stock_code FROM analytics.products ORDER BY 1 LIMIT 3",
        "WITH t AS (SELECT * FROM analytics.invoices) SELECT count(*) FROM t",
    ]
    bits = [
        ";", "; DROP TABLE x;", "--", "/*", "*/", "UNION SELECT 1", "FROM pg_authid", "INTO x",
        "FOR UPDATE", "pg_sleep(1)", "'", '"', "$$", "\\", ", pg_roles", "warehouse.fact_invoice",
        "(SELECT 1 FROM pg_user)", "DELETE FROM analytics.products", "current_setting('x')",
    ]  # fmt: skip
    accepted = 0
    for _ in range(600):
        words = rng.choice(base).split(" ")
        for _ in range(rng.randint(1, 3)):
            words.insert(rng.randint(0, len(words)), rng.choice(bits))
        sql = " ".join(words)
        try:
            g = validate_sql(sql)
        except GuardError:
            continue
        accepted += 1
        parsed = sqlglot.parse(g.validated, dialect="postgres")
        assert len(parsed) == 1 and isinstance(parsed[0], exp.Select | exp.SetOperation), sql
        assert set(g.relations) <= {
            "analytics.sales_lines",
            "analytics.products",
            "analytics.customers",
            "analytics.invoices",
        }, sql
    assert accepted > 0  # the harness is not vacuous


# ---- execution (needs Postgres) -----------------------------------------------------


@pytest.mark.db
def test_executes_and_returns_columns_and_rows(warehouse):
    r = execute_guarded(warehouse.readonly_url, "SELECT count(*) AS n FROM analytics.sales_lines")
    assert r.columns == ["n"] and r.rows == [(368,)] and not r.truncated


@pytest.mark.db
def test_row_limit_truncates_and_flags(warehouse):
    r = execute_guarded(warehouse.readonly_url, "SELECT * FROM analytics.sales_lines", row_limit=10)
    assert len(r.rows) == 10 and r.truncated
    assert "LIMIT 11" in r.executed_sql


@pytest.mark.db
def test_row_limit_not_flagged_when_result_fits(warehouse):
    r = execute_guarded(warehouse.readonly_url, "SELECT * FROM analytics.products", row_limit=100)
    assert not r.truncated and len(r.rows) == 15


@pytest.mark.db
def test_inner_limit_larger_than_cap_is_still_capped(warehouse):
    r = execute_guarded(
        warehouse.readonly_url, "SELECT * FROM analytics.sales_lines LIMIT 100000", row_limit=5
    )
    assert len(r.rows) == 5 and r.truncated


@pytest.mark.db
def test_order_by_survives_the_wrapper(warehouse):
    r = execute_guarded(
        warehouse.readonly_url,
        "SELECT stock_code FROM analytics.products ORDER BY stock_code DESC",
        row_limit=3,
    )
    assert [x[0] for x in r.rows] == sorted((x[0] for x in r.rows), reverse=True)
    assert r.rows[0][0] == "POST"


@pytest.mark.db
def test_statement_timeout_raises(warehouse):
    heavy = (
        "SELECT count(*) FROM analytics.sales_lines a, analytics.sales_lines b, "
        "analytics.sales_lines c, analytics.sales_lines d"
    )
    with pytest.raises(QueryTimeout):
        execute_guarded(warehouse.readonly_url, heavy, timeout_ms=150)


@pytest.mark.db
@pytest.mark.parametrize("timeout", [0, -5, 60_001, True])
def test_bad_timeout_rejected(warehouse, timeout):
    with pytest.raises(GuardError, match="bad_timeout"):
        execute_guarded(warehouse.readonly_url, "SELECT 1", timeout_ms=timeout)


@pytest.mark.db
def test_guard_blocks_attempt_to_disable_the_timeout(warehouse):
    with pytest.raises(GuardError):
        execute_guarded(warehouse.readonly_url, "SET statement_timeout = 0")


@pytest.mark.db
def test_regenerated_sql_gives_same_results_as_compiled_sql(warehouse):
    layer = load_semantic_layer()
    for name, m in layer.metrics.items():
        c = compile_query(layer, {"metrics": [name], "dimensions": m.allowed_dimensions[:1]})
        import psycopg

        with psycopg.connect(warehouse.readonly_url) as conn:
            direct = [tuple(r) for r in conn.execute(c.sql).fetchall()]
        via_guard = execute_guarded(warehouse.readonly_url, c.sql, row_limit=10_000)
        assert via_guard.rows == direct, name


@pytest.mark.db
def test_even_if_the_guard_were_bypassed_the_role_cannot_write(warehouse):
    import psycopg

    with psycopg.connect(warehouse.readonly_url) as conn, pytest.raises(psycopg.Error):
        conn.execute("DELETE FROM analytics.products")
