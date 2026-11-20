import pytest

from verified_analyst.compiler import CompileError, MetricQuery
from verified_analyst.restate import restate
from verified_analyst.semantic import load_semantic_layer

L = load_semantic_layer()


def r(**kw):
    return restate(L, MetricQuery(**kw))


def test_single_total():
    out = r(metrics=["net_revenue"])
    assert out.startswith("Compute net revenue, as a single total.")
    assert "Source: analytics.sales_lines." in out and "GBP (currency)" in out


def test_group_filter_order_limit_exact_head():
    out = r(
        metrics=["net_revenue", "order_count"],
        dimensions=["country"],
        filters=[{"dimension": "month", "op": "between", "value": ["2011-01", "2011-03"]}],
        order_by=[{"by": "net_revenue", "desc": True}],
        limit=5,
    )
    assert out.startswith(
        "Compute net revenue and order count, grouped by country, only where month is between "
        "2011-01-01 and 2011-03-01 (inclusive), sorted by net revenue descending, "
        "limited to the first 5 rows."
    )


@pytest.mark.parametrize(
    ("op", "value", "text"),
    [
        ("eq", "France", 'country is "France"'),
        ("ne", "France", 'country is not "France"'),
        ("in", ["France", "EIRE"], 'country is one of ["France", "EIRE"]'),
        ("not_in", ["France"], 'country is not one of ["France"]'),
        ("gt", "A", 'country is greater than "A"'),
        ("gte", "A", 'country is at least "A"'),
        ("lt", "A", 'country is less than "A"'),
        ("lte", "A", 'country is at most "A"'),
        ("between", ["A", "C"], 'country is between "A" and "C" (inclusive)'),
    ],
)
def test_every_filter_op(op, value, text):
    assert text in r(
        metrics=["net_revenue"], filters=[{"dimension": "country", "op": op, "value": value}]
    )


def test_values_match_compiler_normalisation():
    out = r(
        metrics=["order_count"],
        filters=[{"dimension": "country", "op": "eq", "value": "Côte d'Ivoire"}],
    )
    assert """country is "Côte d'Ivoire\"""" in out
    out2 = r(
        metrics=["order_count"], filters=[{"dimension": "day", "op": "eq", "value": " 2011-03-04 "}]
    )
    assert "day is 2011-03-04" in out2


def test_hostile_value_is_quoted_not_interpreted():
    out = r(
        metrics=["order_count"],
        filters=[{"dimension": "country", "op": "eq", "value": "x'; DROP TABLE t; --"}],
    )
    assert '"x\'; DROP TABLE t; --"' in out


def test_every_metric_and_dimension_restates():
    for m, metric in L.metrics.items():
        dims = list(metric.allowed_dimensions)
        out = r(metrics=[m], dimensions=dims[:1])
        assert m.replace("_", " ") in out and metric.description.rstrip(".") in out


def test_deterministic_and_pure():
    q = MetricQuery(metrics=["aov"], dimensions=["country"], limit=3)
    assert restate(L, q) == restate(L, q.model_copy(deep=True)) == restate(L, q.model_dump())


def test_invalid_queries_are_refused_like_the_compiler():
    with pytest.raises(CompileError):
        restate(L, {"metrics": ["nope"]})
    with pytest.raises(CompileError):
        restate(
            L,
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "month", "op": "eq", "value": "bad"}],
            },
        )
    with pytest.raises(CompileError):
        restate(
            L,
            {
                "metrics": ["net_revenue"],
                "filters": [{"dimension": "country", "op": "eq", "value": "a\x00b"}],
            },
        )


def test_three_metrics_oxford_join_and_ratio_unit():
    out = r(metrics=["net_revenue", "order_count", "cancellation_rate"])
    assert "net revenue, order count, and cancellation rate" in out
    assert "cancellation rate in ratio" in out
