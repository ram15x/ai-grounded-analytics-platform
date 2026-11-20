"""Gold answers for tier T1 (single metric, no dimension), straight from the RAW rows.

INDEPENDENCE RULE: this file imports only the standard library and pandas. It must not import
anything from verified_analyst (semantic layer, compiler, normaliser) or the pandas oracle, and
it does not read semantic/metrics.yaml. Each question's rules are restated here in code and in
plain English in eval/questions/dev.yaml. tests/test_gold.py enforces the imports.

Gold questions and rules are AUTHOR-WRITTEN (by the project author). They are not an
independent benchmark, and the same author wrote the semantic layer: treat agreement as weak
evidence and every disagreement as a finding.

Money is computed in integer 1/10000 units (price ticks) so sums are exact.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

TICKS = 10_000
MERCH = re.compile(r"^[0-9]{5}[A-Z]{0,3}$")


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw file with plain pandas and add only the columns the rules below need."""
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"unsupported file type {path.suffix}")
    out = pd.DataFrame()
    out["invoice"] = df["InvoiceNo"].astype(str).str.strip()
    out["code"] = df["StockCode"].astype(str).str.strip().str.upper()
    out["qty"] = pd.to_numeric(df["Quantity"]).astype("int64")
    out["ticks"] = (pd.to_numeric(df["UnitPrice"]) * TICKS).round().astype("int64")

    def clean(v: object) -> str | None:
        t = str(v).strip()
        if t == "" or t.lower() in ("nan", "none"):
            return None
        return str(int(float(t)))

    out["customer"] = pd.Series([clean(v) for v in df["CustomerID"]], dtype="object")
    out["amount"] = out["qty"] * out["ticks"]  # in 1/10000 currency units
    return out


def _masks(d: pd.DataFrame) -> dict[str, pd.Series]:
    is_c = d["invoice"].str.upper().str.startswith("C")
    priced = d["ticks"] > 0
    not_d = d["code"] != "D"
    return {
        # a normal sale: not a C invoice, positive quantity and price, not the discount code
        "sale": ~is_c & (d["qty"] > 0) & priced & not_d,
        # a customer cancellation: C invoice, negative quantity, positive price, not code D
        "cancel": is_c & (d["qty"] < 0) & priced & not_d,
        # a discount line: stock code D with a positive price
        "discount": (d["code"] == "D") & priced,
    }


def _money(ticks_sum: int) -> Decimal:
    return Decimal(int(ticks_sum)) / TICKS


def gross_revenue(d: pd.DataFrame) -> Decimal:
    return _money(d.loc[_masks(d)["sale"], "amount"].sum())


def cancelled_revenue(d: pd.DataFrame) -> Decimal:
    return -_money(d.loc[_masks(d)["cancel"], "amount"].sum())


def discount_amount(d: pd.DataFrame) -> Decimal:
    return -_money(d.loc[_masks(d)["discount"], "amount"].sum())


def net_revenue(d: pd.DataFrame) -> Decimal:
    m = _masks(d)
    return _money(d.loc[m["sale"] | m["cancel"] | m["discount"], "amount"].sum())


def units_sold(d: pd.DataFrame) -> Decimal:
    """ALL units on sale rows, whatever the stock code (postage, fees and manual lines included)."""
    return Decimal(int(d.loc[_masks(d)["sale"], "qty"].sum()))


def order_count(d: pd.DataFrame) -> Decimal:
    return Decimal(int(d.loc[_masks(d)["sale"], "invoice"].nunique()))


def active_customers(d: pd.DataFrame) -> Decimal:
    s = d.loc[_masks(d)["sale"], "customer"].dropna()
    return Decimal(int(s.nunique()))


def average_order_value(d: pd.DataFrame) -> Decimal:
    return gross_revenue(d) / order_count(d)


def revenue_per_customer(d: pd.DataFrame) -> Decimal:
    sale = d[_masks(d)["sale"] & d["customer"].notna()]
    return _money(sale["amount"].sum()) / Decimal(int(sale["customer"].nunique()))


def cancellation_rate(d: pd.DataFrame) -> Decimal:
    return cancelled_revenue(d) / gross_revenue(d)


def merchandise_gross_revenue(d: pd.DataFrame) -> Decimal:
    sale = d[_masks(d)["sale"] & d["code"].map(lambda c: bool(MERCH.match(c)))]
    return _money(sale["amount"].sum())


def sale_line_count(d: pd.DataFrame) -> Decimal:
    return Decimal(int(_masks(d)["sale"].sum()))


def distinct_products_sold(d: pd.DataFrame) -> Decimal:
    return Decimal(int(d.loc[_masks(d)["sale"], "code"].nunique()))


def cancelled_invoice_count(d: pd.DataFrame) -> Decimal:
    return Decimal(int(d.loc[_masks(d)["cancel"], "invoice"].nunique()))


def identified_net_revenue(d: pd.DataFrame) -> Decimal:
    m = _masks(d)
    keep = (m["sale"] | m["cancel"] | m["discount"]) & d["customer"].notna()
    return _money(d.loc[keep, "amount"].sum())


GOLD_FUNCTIONS = {
    f.__name__: f
    for f in (
        gross_revenue, cancelled_revenue, discount_amount, net_revenue, units_sold, order_count,
        active_customers, average_order_value, revenue_per_customer, cancellation_rate,
        merchandise_gross_revenue, sale_line_count, distinct_products_sold,
        cancelled_invoice_count, identified_net_revenue,
    )
}  # fmt: skip
