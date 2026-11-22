import pandas as pd
import pytest

from grounded_analytics.datasets import manifest as M
from grounded_analytics.datasets.profile import profile_frame

DF = pd.DataFrame(
    {
        "order_date": ["2024-01-05", "2024-02-06", "2024-02-07"],
        "region": ["North", "South", "North"],
        "status": ["paid", "paid", "refunded"],
        "amount": ["10.5", "20", "5"],
        "customer": ["c1", "c2", "c1"],
    }
)
PROFILE = profile_frame(DF, "orders.csv")


def manifest(**over):
    base = {
        "name": "shop",
        "description": "Orders of a small online shop.",
        "grain": "one row per order",
        "columns": {
            "order_date": {"role": "date", "label": "Order date"},
            "region": {"role": "dimension", "enumerable": True},
            "status": {"role": "dimension", "enumerable": True},
            "amount": {"role": "measure"},
            "customer": {"role": "identifier"},
        },
        "metrics": {
            "revenue": {
                "status": "approved",
                "description": "Paid order value",
                "aggregate": "sum",
                "column": "amount",
                "where": [{"column": "status", "op": "eq", "value": "paid"}],
                "unit": "currency:GBP",
                "decimals": 2,
            },
            "orders": {
                "status": "approved",
                "description": "Number of orders",
                "aggregate": "count",
                "unit": "count",
                "decimals": 0,
            },
            "customers": {
                "status": "approved",
                "description": "Distinct customers",
                "aggregate": "count_distinct",
                "column": "customer",
                "unit": "count",
                "decimals": 0,
            },
            "aov": {
                "status": "approved",
                "description": "Revenue per order",
                "numerator": "revenue",
                "denominator": "orders",
                "unit": "currency:GBP",
                "decimals": 2,
            },
            "draft_metric": {
                "description": "Not approved yet",
                "aggregate": "sum",
                "column": "amount",
                "unit": "currency:GBP",
                "decimals": 2,
            },
        },
    }
    base.update(over)
    return M.Manifest.model_validate(base)


def test_layer_has_only_approved_metrics_and_the_expected_sql():
    layer = M.build_layer(manifest(), PROFILE)
    assert set(layer.metrics) == {"revenue", "orders", "customers", "aov"}
    assert layer.source.relation == "ds_shop.data"
    assert layer.metrics["revenue"].expr == (
        "COALESCE(SUM(l.amount) FILTER (WHERE l.status = 'paid'), 0)"
    )
    assert layer.metrics["customers"].expr == "COUNT(DISTINCT l.customer)"
    assert "draft_metric" in M.build_layer(manifest(), PROFILE, approved_only=False).metrics


def test_dimensions_scopes_and_ratio_dimensions():
    layer = M.build_layer(manifest(), PROFILE)
    assert {"month", "day", "region", "status", "customer"} <= set(layer.dimensions)
    rev = layer.metrics["revenue"]
    assert rev.implicit_scopes == {"status": ["paid"]} and "status" not in rev.allowed_dimensions
    assert "customer" not in layer.metrics["customers"].allowed_dimensions
    assert (
        layer.metrics["aov"].kind == "ratio"
        and "status" not in layer.metrics["aov"].allowed_dimensions
    )


def test_a_hostile_value_is_quoted_not_executed():
    c = M.Condition(column="status", op="eq", value="x'; DROP TABLE data; --")
    sql = M.condition_sql(c, {"status": "text"})
    assert sql == "l.status = 'x''; DROP TABLE data; --'"
    with pytest.raises(M.ManifestError):
        M.literal("1; DROP", "integer")
    with pytest.raises(M.ManifestError):
        M.literal("yesterday", "date")
    with pytest.raises(M.ManifestError):
        M.literal("maybe", "boolean")


def test_unknown_column_and_bad_roles_are_reported():
    m = manifest()
    m.columns["region"] = M.Column(role="measure")
    m.metrics["bad"] = M.MetricSpec(
        description="x", aggregate="sum", column="nope", unit="count", decimals=0
    )
    problems = M.validate(m, PROFILE)
    assert any("region: a measure must be numeric" in p for p in problems)
    assert any("bad:" in p for p in problems)
    with pytest.raises(M.ManifestError):
        M.build_layer(m, PROFILE)


def test_metric_shape_rules():
    with pytest.raises(ValueError):
        M.MetricSpec(description="x", unit="count", decimals=0)  # no aggregate
    with pytest.raises(ValueError):
        M.MetricSpec(description="x", aggregate="sum", unit="count", decimals=0)  # no column
    with pytest.raises(ValueError):
        M.MetricSpec(description="x", aggregate="sum", column="a", unit="GBP", decimals=0)
    with pytest.raises(ValueError):
        M.MetricSpec(description="x", numerator="a", unit="ratio", decimals=2)


def test_manifest_needs_a_grain_and_a_safe_name():
    with pytest.raises(ValueError):
        manifest(grain=" ")
    with pytest.raises(ValueError):
        manifest(name="Bad Name; drop")


def test_exclusions_apply_to_every_metric():
    m = manifest(
        exclusions=[{"name": "test", "where": [{"column": "customer", "op": "ne", "value": "c9"}]}]
    )
    layer = M.build_layer(m, PROFILE)
    assert "l.customer <> 'c9'" in layer.metrics["orders"].expr


def test_a_dataset_layer_may_read_only_its_own_table():
    from grounded_analytics.guard import ALLOWED_RELATIONS, GuardError, allowed_for, validate_sql

    layer = M.build_layer(manifest(), PROFILE)
    assert allowed_for(layer) == frozenset({"ds_shop.data"})
    validate_sql("SELECT COUNT(*) FROM ds_shop.data AS l", allowed_for(layer))
    for other in ("analytics.sales_lines", "ds_other.data"):
        with pytest.raises(GuardError):
            validate_sql(f"SELECT 1 FROM {other}", allowed_for(layer))
    assert "analytics.sales_lines" in ALLOWED_RELATIONS
