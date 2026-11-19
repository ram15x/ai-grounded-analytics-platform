"""Semantic layer file: structure and invariants (no database)."""

import copy

import pytest
import yaml

from verified_analyst.semantic import (
    SemanticError,
    SemanticLayer,
    default_path,
    load_semantic_layer,
    norm_term,
)

REQUIRED = {
    "net_revenue",
    "gross_revenue",
    "cancelled_revenue",
    "units_sold",
    "order_count",
    "aov",
    "active_customers",
    "revenue_per_customer",
}


@pytest.fixture(scope="module")
def layer():
    return load_semantic_layer()


def test_required_metrics_present(layer):
    assert set(layer.metrics) >= REQUIRED
    assert len(layer.metrics) >= 8


def test_every_metric_has_unit_synonyms_and_dimensions(layer):
    for name, m in layer.metrics.items():
        assert m.unit, name
        assert m.allowed_dimensions, name
        assert m.description, name
    for name in REQUIRED - {"aov"}:
        assert layer.metrics[name].synonyms or name == "order_count", name


def test_synonym_resolution(layer):
    assert layer.resolve_metric("Total Revenue") == "net_revenue"
    assert layer.resolve_metric("average-order-value") == "aov"
    assert layer.resolve_metric("gross sales") == "gross_revenue"
    assert layer.resolve_metric("number of orders") == "order_count"
    assert layer.resolve_metric("population of mars") is None


def test_norm_term():
    assert norm_term("  Average_Order-Value ") == "average order value"


def _raw():
    return yaml.safe_load(default_path().read_text("utf-8"))


def test_duplicate_synonym_rejected():
    data = _raw()
    data["metrics"]["units_sold"]["synonyms"].append("revenue")  # belongs to net_revenue
    with pytest.raises(ValueError, match="used by both"):
        SemanticLayer.model_validate(data)


def test_unknown_dimension_rejected():
    data = copy.deepcopy(_raw())
    data["metrics"]["units_sold"]["allowed_dimensions"].append("planet")
    with pytest.raises(ValueError, match="unknown dimension"):
        SemanticLayer.model_validate(data)


def test_ratio_of_ratio_rejected():
    data = copy.deepcopy(_raw())
    data["metrics"]["aov"]["numerator"] = "cancellation_rate"
    with pytest.raises(ValueError, match="ratio of ratios"):
        SemanticLayer.model_validate(data)


def test_bad_unit_rejected():
    data = copy.deepcopy(_raw())
    data["metrics"]["units_sold"]["unit"] = "bananas"
    with pytest.raises(ValueError, match="bad unit"):
        SemanticLayer.model_validate(data)


def test_missing_file_gives_semantic_error(tmp_path):
    with pytest.raises(SemanticError):
        load_semantic_layer(tmp_path / "nope.yaml")
