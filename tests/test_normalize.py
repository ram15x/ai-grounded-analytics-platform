"""Normalisation rules, tested without a database."""

import pandas as pd
import pytest

from grounded_analytics.warehouse.normalize import RAW_COLUMNS, classify_line, normalize


@pytest.mark.parametrize(
    ("code", "inv", "qty", "price", "expected"),
    [
        ("85123A", "536365", 6, 2.55, "sale"),
        ("85123A", "C536379", -2, 2.55, "cancellation"),
        ("85123A", "C536379", 2, 2.55, "adjustment"),  # C invoice with positive qty
        ("85123A", "536365", -5, 2.55, "adjustment"),  # negative qty without C prefix
        ("85123A", "536365", 0, 2.55, "adjustment"),
        ("85123A", "536365", 3, 0.0, "adjustment"),  # free item
        ("B", "A563185", 1, -11062.06, "adjustment"),  # bad-debt adjustment
        ("D", "C536379", -1, 27.5, "discount"),
        ("D", "C536379", -1, 0.0, "adjustment"),
        ("POST", "536365", 1, 18.0, "sale"),  # postage is revenue
        ("POST", "C536379", -1, 18.0, "cancellation"),
    ],
)
def test_classify_line(code, inv, qty, price, expected):
    assert classify_line(code, inv, qty, price) == expected


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        normalize(pd.DataFrame({"InvoiceNo": ["1"]}))


def test_fixture_counts(raw_fixture):
    n = normalize(raw_fixture)
    dq = n.dq
    assert dq["raw_rows"] == len(raw_fixture) == 368
    assert dq["lines"] == 368
    assert dq["exact_duplicate_rows"] == 3
    assert dq["line_type_counts"] == {
        "sale": 352,
        "cancellation": 11,
        "discount": 1,
        "adjustment": 4,
    }
    assert dq["c_invoice_with_positive_qty"] == 1
    assert dq["negative_qty_without_c_prefix"] == 1
    assert dq["zero_price_rows"] == 2
    assert dq["negative_price_rows"] == 1
    assert dq["nonstandard_invoice_no_samples"] == ["A536499"]
    assert dq["missing_customer_rows"] == 40
    assert dq["invoices_without_customer"] == 19
    assert dq["products_with_multiple_descriptions"] == 1


def test_lowercase_stock_code_is_uppercased(raw_fixture):
    n = normalize(raw_fixture)
    assert "M" in set(n.products["stock_code"])
    assert "m" not in set(n.products["stock_code"])
    assert n.dq["stock_codes_changed_by_uppercasing"] == 1


def test_missing_customer_is_null_not_sentinel(raw_fixture):
    n = normalize(raw_fixture)
    assert n.invoices["customer_key"].isna().sum() == 19
    assert (n.customers["customer_id"] != "").all()
    assert n.customers["customer_id"].str.fullmatch(r"\d+").all()  # "12346.0" -> "12346"


def test_description_is_trimmed_and_empty_becomes_null(raw_fixture):
    n = normalize(raw_fixture)
    desc = n.products.set_index("stock_code")["description"]
    assert desc["20725"] == "LUNCH BAG RED RETROSPOT"
    assert desc["84029G"] == "KNITTED UNION FLAG HOT WATER BOTTLE"  # empty desc ignored


def test_keys_are_unique_and_deterministic(raw_fixture):
    a, b = normalize(raw_fixture), normalize(raw_fixture)
    assert a.lines["line_key"].is_unique
    pd.testing.assert_frame_equal(a.lines, b.lines)
    pd.testing.assert_frame_equal(a.invoices, b.invoices)


def test_raw_columns_constant():
    assert RAW_COLUMNS[0] == "InvoiceNo"
    assert len(RAW_COLUMNS) == 8
