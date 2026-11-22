"""Metric compiler: SQL shape, validation and literal safety (no database)."""

import pytest
import sqlglot
from sqlglot import exp

from grounded_analytics.compiler import CompileError, compile_query
from grounded_analytics.semantic import load_semantic_layer

LAYER = load_semantic_layer()


def sql(**kw):
    return compile_query(LAYER, kw).sql


def test_simple_metric_no_dimensions():
    c = compile_query(LAYER, {"metrics": ["net_revenue"]})
    assert c.columns == ["net_revenue"]
    assert "GROUP BY" not in c.sql
    assert c.sql.startswith("SELECT ")
    assert "FROM analytics.sales_lines AS l" in c.sql


def test_dimension_adds_group_and_order():
    s = sql(metrics=["units_sold"], dimensions=["country"])
    assert "GROUP BY l.country" in s
    assert "ORDER BY country" in s


def test_ratio_expands_with_nullif():
    s = sql(metrics=["aov"])
    assert "NULLIF(" in s
    assert "COUNT(DISTINCT l.invoice_no)" in s


def test_multiple_metrics_and_dimensions_column_order():
    c = compile_query(
        LAYER, {"metrics": ["net_revenue", "order_count"], "dimensions": ["month", "country"]}
    )
    assert c.columns == ["month", "country", "net_revenue", "order_count"]


def test_filter_operators_render():
    s = sql(
        metrics=["net_revenue"],
        filters=[
            {"dimension": "country", "op": "eq", "value": "France"},
            {"dimension": "country", "op": "ne", "value": "EIRE"},
            {"dimension": "stock_code", "op": "in", "value": ["85123A", "71053"]},
            {"dimension": "stock_code", "op": "not_in", "value": ["POST"]},
            {"dimension": "day", "op": "between", "value": ["2011-01-01", "2011-01-31"]},
            {"dimension": "day", "op": "gte", "value": "2011-01-01"},
            {"dimension": "day", "op": "lt", "value": "2011-02-01"},
        ],
    )
    assert "l.country = 'France'" in s
    assert "l.country <> 'EIRE'" in s
    assert "l.stock_code IN ('85123A', '71053')" in s
    assert "NOT IN ('POST')" in s
    assert "BETWEEN DATE '2011-01-01' AND DATE '2011-01-31'" in s
    assert "l.invoice_date >= DATE '2011-01-01'" in s
    assert "l.invoice_date < DATE '2011-02-01'" in s


def test_month_accepts_year_month_strings():
    s = sql(
        metrics=["net_revenue"], filters=[{"dimension": "month", "op": "eq", "value": "2011-01"}]
    )
    assert "DATE '2011-01-01'" in s


def test_order_by_and_limit():
    s = sql(
        metrics=["net_revenue"],
        dimensions=["stock_code"],
        order_by=[{"by": "net_revenue", "desc": True}],
        limit=5,
    )
    assert s.endswith("ORDER BY net_revenue DESC\nLIMIT 5")


@pytest.mark.parametrize(
    ("query", "msg"),
    [
        ({"metrics": ["bitcoin"]}, "unknown metric"),
        ({"metrics": ["net_revenue"], "dimensions": ["planet"]}, "unknown dimension"),
        ({"metrics": ["active_customers"], "dimensions": ["customer_id"]}, "cannot be grouped"),
        ({"metrics": ["aov"], "dimensions": ["stock_code"]}, "cannot be grouped"),
        ({"metrics": ["net_revenue", "net_revenue"]}, "duplicate"),
        ({"metrics": ["net_revenue"], "dimensions": ["month", "month"]}, "duplicate"),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "planet", "op": "eq", "value": "x"}],
            },
            "unknown dimension",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "in", "value": []}],
            },
            "non-empty",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "in", "value": "x"}],
            },
            "non-empty",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "day", "op": "eq", "value": "yesterday"}],
            },
            "bad date",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "day", "op": "eq", "value": 20110101}],
            },
            "ISO strings",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "day", "op": "between", "value": ["2011-01-01"]}],
            },
            "low, high",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "eq", "value": ["a"]}],
            },
            "scalar",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "eq", "value": "a\x00b"}],
            },
            "control characters",
        ),
        (
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "eq", "value": True}],
            },
            "boolean",
        ),
        ({"metrics": ["net_revenue"], "order_by": [{"by": "aov"}]}, "not a selected column"),
    ],
)
def test_rejections(query, msg):
    with pytest.raises(CompileError, match=msg):
        compile_query(LAYER, query)


def test_too_many_in_values():
    vals = [f"c{i}" for i in range(101)]
    with pytest.raises(CompileError, match="at most"):
        sql(metrics=["net_revenue"], filters=[{"dimension": "country", "op": "in", "value": vals}])


@pytest.mark.parametrize("bad", [{"metrics": []}, {"metrics": ["net_revenue"], "limit": 0}])
def test_schema_validation(bad):
    with pytest.raises(ValueError):
        compile_query(LAYER, bad)


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        compile_query(LAYER, {"metrics": ["net_revenue"], "raw_sql": "DROP TABLE x"})


@pytest.mark.parametrize(
    "evil",
    [
        "'; DROP TABLE warehouse.fact_invoice; --",
        "France' OR '1'='1",
        "x') UNION SELECT password FROM pg_authid --",
        "\\'; select 1; --",
        "$$ ; select 1 $$",
    ],
)
def test_hostile_values_stay_inside_one_string_literal(evil):
    s = sql(metrics=["net_revenue"], filters=[{"dimension": "country", "op": "eq", "value": evil}])
    tree = sqlglot.parse(s, dialect="postgres")
    assert len(tree) == 1 and isinstance(tree[0], exp.Select)
    literals = [n for n in tree[0].args["where"].find_all(exp.Literal) if n.is_string]
    assert [n.this for n in literals] == [evil]
    assert not list(tree[0].find_all(exp.Drop, exp.Union))


def test_compiled_sql_parses_for_every_metric():
    for name, m in LAYER.metrics.items():
        dims = m.allowed_dimensions[:2]
        c = compile_query(LAYER, {"metrics": [name], "dimensions": dims})
        parsed = sqlglot.parse(c.sql, dialect="postgres")
        assert len(parsed) == 1 and isinstance(parsed[0], exp.Select), name
