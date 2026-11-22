"""Every metric, computed by the compiled SQL in Postgres, must equal an independent
pandas derivation from the raw table (tests/pandas_oracle.py)."""

from decimal import Decimal

import pandas as pd
import psycopg
import pytest

from grounded_analytics.compiler import compile_query
from grounded_analytics.semantic import load_semantic_layer
from pandas_oracle import oracle

pytestmark = pytest.mark.db
LAYER = load_semantic_layer()
ALL = list(LAYER.metrics)


def run(wh, query):
    c = compile_query(LAYER, query)
    with psycopg.connect(wh.readonly_url) as conn:
        rows = conn.execute(c.sql).fetchall()
    nd = len(c.dimensions)
    return {tuple(r[:nd]): dict(zip(c.metrics, r[nd:], strict=True)) for r in rows}


def assert_same(sql_result, expected):
    assert set(sql_result) == set(expected), set(sql_result) ^ set(expected)
    for key, metrics in expected.items():
        for name, want in metrics.items():
            got = sql_result[key][name]
            if want is None:
                assert got is None, (key, name, got)
            else:
                assert got is not None, (key, name)
                assert float(got) == pytest.approx(float(want), abs=1e-6), (key, name, got, want)


@pytest.mark.parametrize("metric", ALL)
def test_ungrouped_metric_matches_pandas(warehouse, metric):
    got = run(warehouse, {"metrics": [metric]})
    assert_same(got, oracle(warehouse.raw, [metric]))


@pytest.mark.parametrize(
    "dims",
    [["month"], ["country"], ["month", "country"], ["stock_code"], ["product_kind"], ["day"]],
)
@pytest.mark.parametrize("metric", ALL)
def test_grouped_metric_matches_pandas(warehouse, metric, dims):
    if not set(dims) <= set(LAYER.metrics[metric].allowed_dimensions):
        pytest.skip("dimension not allowed for this metric")
    got = run(warehouse, {"metrics": [metric], "dimensions": dims})
    assert_same(got, oracle(warehouse.raw, [metric], dims))


@pytest.mark.parametrize(
    "metric", ["gross_revenue", "net_revenue", "cancelled_revenue", "order_count"]
)
def test_grouped_by_customer_includes_null_group(warehouse, metric):
    got = run(warehouse, {"metrics": [metric], "dimensions": ["customer_id"]})
    assert (None,) in got
    assert_same(got, oracle(warehouse.raw, [metric], ["customer_id"]))


FILTER_CASES = [
    [{"dimension": "country", "op": "in", "value": ["France", "Germany"]}],
    [{"dimension": "country", "op": "ne", "value": "United Kingdom"}],
    [{"dimension": "day", "op": "between", "value": ["2011-01-01", "2011-01-31"]}],
    [{"dimension": "month", "op": "between", "value": ["2010-12", "2011-01"]}],
    [
        {"dimension": "country", "op": "eq", "value": "United Kingdom"},
        {"dimension": "day", "op": "gte", "value": "2011-02-01"},
    ],
    [{"dimension": "stock_code", "op": "not_in", "value": ["POST", "M", "85123A"]}],
]


@pytest.mark.parametrize("filters", FILTER_CASES)
@pytest.mark.parametrize("metric", ALL)
def test_filtered_metric_matches_pandas(warehouse, metric, filters):
    got = run(warehouse, {"metrics": [metric], "filters": filters})
    assert_same(got, oracle(warehouse.raw, [metric], None, filters))


def test_filter_and_group_together(warehouse):
    filters = [{"dimension": "country", "op": "in", "value": ["France", "EIRE", "Germany"]}]
    metrics = ["net_revenue", "aov", "active_customers"]
    got = run(warehouse, {"metrics": metrics, "dimensions": ["month"], "filters": filters})
    assert_same(got, oracle(warehouse.raw, metrics, ["month"], filters))


def test_net_equals_gross_minus_cancelled_minus_discounts_in_every_group(warehouse):
    got = run(
        warehouse,
        {
            "metrics": ["net_revenue", "gross_revenue", "cancelled_revenue", "discount_amount"],
            "dimensions": ["month", "country"],
        },
    )
    assert got
    for key, m in got.items():
        assert (
            m["net_revenue"] == m["gross_revenue"] - m["cancelled_revenue"] - m["discount_amount"]
        ), key


# ---- hand-checked tiny dataset: expected values computed by hand, no oracle ----------


def _tiny():
    cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]
    t = "2011-01-10 09:00:00"
    rows = [
        ["1", "10001", "P1", 2, t, 10.0, "1", "United Kingdom"],
        ["1", "10002", "P2", 1, t, 5.0, "1", "United Kingdom"],
        ["2", "10001", "P1", 3, t, 10.0, "2", "France"],
        ["C3", "10001", "P1", -1, t, 10.0, "1", "United Kingdom"],
        ["4", "POST", "POSTAGE", 1, t, 4.0, "", "United Kingdom"],
        ["5", "10002", "P2", 1, t, 0.0, "", "United Kingdom"],
    ]
    return pd.DataFrame(rows, columns=cols)


def test_hand_checked_tiny_dataset(warehouse_factory):
    wh = warehouse_factory(_tiny())
    got = run(wh, {"metrics": ALL})[()]
    expect = {
        "gross_revenue": Decimal("59"),  # 20 + 5 + 30 + 4
        "cancelled_revenue": Decimal("10"),
        "discount_amount": Decimal("0"),
        "net_revenue": Decimal("49"),
        "units_sold": 6,  # 2 + 1 + 3 (postage and the free item excluded)
        "order_count": 3,  # invoices 1, 2, 4
        "active_customers": 2,  # customers 1 and 2; invoice 4 has no CustomerID
        "identified_gross_revenue": Decimal("55"),
    }
    for k, v in expect.items():
        assert got[k] == v, k
    assert float(got["aov"]) == pytest.approx(59 / 3)
    assert float(got["revenue_per_customer"]) == pytest.approx(27.5)
    assert float(got["cancellation_rate"]) == pytest.approx(10 / 59)


def test_empty_filter_result_is_no_rows(warehouse):
    got = run(
        warehouse,
        {
            "metrics": ["net_revenue"],
            "filters": [{"dimension": "country", "op": "eq", "value": "Narnia"}],
        },
    )
    assert got == {(): {"net_revenue": Decimal("0")}}
