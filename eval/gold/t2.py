"""Gold answers for the HELD-OUT questions (eval/questions/heldout.yaml), tiers T2 and T3,
straight from the RAW rows.

HELD-OUT: these questions and their rules must never be shown to a prompt-tuning step, a model
prompt, or an example in a prompt. The gold VALUES are not stored in the repository; they are
computed on demand from the raw file by eval/gold/run_heldout.py.

INDEPENDENCE RULE: this file imports only the standard library and pandas. It does not import
anything from grounded_analytics, it does not read semantic/metrics.yaml, and it does not import
t1: the row rules are written out again here on purpose, so a mistake in one gold file is not
silently copied into the other (tests/test_gold.py checks that both agree on the overall totals).

Questions and rules are AUTHOR-WRITTEN by the project author, who also wrote the semantic layer.
Agreement with the layer is weak evidence and every disagreement is a finding.

Money is computed in integer 1/10000 units (price ticks), so sums are exact. A "month" is the
calendar month of InvoiceDate (the file runs from 2010-12-01 to 2011-12-09, so 2011-12 is partial).

Every function takes the frame from `load_raw` and keyword arguments from the question's
`gold_args`, and returns {"kind": "scalar", "value": Decimal} or
{"kind": "table", "columns": [...], "rows": [[str, ...], ...]}.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

TICKS = 10_000
MERCH_CODE = re.compile(r"^[0-9]{5}[A-Z]{0,3}$")  # 5 digits and up to 3 capital letters


def load_raw(path: Path) -> pd.DataFrame:
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
    out["month"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%Y-%m")
    out["country"] = df["Country"].astype(str).str.strip()

    def customer(v: object) -> str | None:
        t = str(v).strip()
        return None if t == "" or t.lower() in ("nan", "none") else str(int(float(t)))

    out["customer"] = pd.Series([customer(v) for v in df["CustomerID"]], dtype="object")
    out["amount"] = out["qty"] * out["ticks"]
    out["is_merch"] = out["code"].map(lambda c: bool(MERCH_CODE.match(c)))
    c_invoice = out["invoice"].str.upper().str.startswith("C")
    priced = out["ticks"] > 0
    out["sale"] = ~c_invoice & (out["qty"] > 0) & priced & (out["code"] != "D")
    out["cancel"] = c_invoice & (out["qty"] < 0) & priced & (out["code"] != "D")
    out["discount"] = (out["code"] == "D") & priced
    return out


def _money(ticks: int) -> Decimal:
    return Decimal(int(ticks)) / TICKS


def _scalar(v: Decimal) -> dict[str, Any]:
    return {"kind": "scalar", "value": v}


def _table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"kind": "table", "columns": columns, "rows": [[str(x) for x in r] for r in rows]}


def _gross(d: pd.DataFrame) -> int:
    return int(d.loc[d["sale"], "amount"].sum())


def _net(d: pd.DataFrame) -> int:
    return int(d.loc[d["sale"] | d["cancel"] | d["discount"], "amount"].sum())


def _top(values: dict[str, int], n: int) -> list[tuple[str, int]]:
    """The n largest by value, ties broken by key. Refuses a question whose cut-off is a tie."""
    ranked = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > n and ranked[n - 1][1] == ranked[n][1]:
        raise ValueError(f"tie at the top-{n} cut-off: the question would be ambiguous")
    return ranked[:n]


# ---- T2: one filter, or one grouping ----------------------------------------------------------


def gross_revenue_in_country(d: pd.DataFrame, country: str) -> dict[str, Any]:
    return _scalar(_money(_gross(d[d["country"] == country])))


def net_revenue_in_month(d: pd.DataFrame, month: str) -> dict[str, Any]:
    return _scalar(_money(_net(d[d["month"] == month])))


def top_countries_by_gross_revenue(d: pd.DataFrame, n: int) -> dict[str, Any]:
    by = {c: _gross(g) for c, g in d.groupby("country")}
    return _table(["country", "gross_revenue"], [[c, _money(v)] for c, v in _top(by, n)])


def orders_per_month(d: pd.DataFrame, months: list[str]) -> dict[str, Any]:
    s = d[d["sale"]]
    rows = [[m, int(s.loc[s["month"] == m, "invoice"].nunique())] for m in sorted(months)]
    return _table(["month", "order_count"], rows)


def active_customers_in_country(d: pd.DataFrame, country: str) -> dict[str, Any]:
    s = d[d["sale"] & (d["country"] == country)]["customer"].dropna()
    return _scalar(Decimal(int(s.nunique())))


def month_with_highest_cancelled_revenue(d: pd.DataFrame) -> dict[str, Any]:
    c = d[d["cancel"]]
    by = {m: -int(g["amount"].sum()) for m, g in c.groupby("month")}
    ((m, v),) = _top(by, 1)
    return _table(["month", "cancelled_revenue"], [[m, _money(v)]])


def average_order_value_in_country(d: pd.DataFrame, country: str) -> dict[str, Any]:
    s = d[d["sale"] & (d["country"] == country)]
    return _scalar(_money(int(s["amount"].sum())) / Decimal(int(s["invoice"].nunique())))


# ---- T3: two periods, several filters, ratios, things the catalogue may not cover -------------


def percent_change_gross_revenue(d: pd.DataFrame, month_from: str, month_to: str) -> dict[str, Any]:
    a = _money(_gross(d[d["month"] == month_from]))
    b = _money(_gross(d[d["month"] == month_to]))
    return _scalar((b - a) / a * 100)


def net_revenue_in_two_months(d: pd.DataFrame, months: list[str]) -> dict[str, Any]:
    rows = [[m, _money(_net(d[d["month"] == m]))] for m in months]
    return _table(["month", "net_revenue"], rows)


def net_revenue_in_country_between_months(
    d: pd.DataFrame, country: str, first: str, last: str
) -> dict[str, Any]:
    s = d[(d["country"] == country) & (d["month"] >= first) & (d["month"] <= last)]
    return _scalar(_money(_net(s)))


def top_products_by_merchandise_units_in_country(
    d: pd.DataFrame, country: str, n: int
) -> dict[str, Any]:
    s = d[d["sale"] & d["is_merch"] & (d["country"] == country)]
    by = {c: int(g["qty"].sum()) for c, g in s.groupby("code")}
    return _table(["stock_code", "units_sold"], [[c, v] for c, v in _top(by, n)])


def merchandise_units_per_order(d: pd.DataFrame) -> dict[str, Any]:
    units = int(d.loc[d["sale"] & d["is_merch"], "qty"].sum())
    orders = int(d.loc[d["sale"], "invoice"].nunique())
    return _scalar(Decimal(units) / Decimal(orders))


def gross_revenue_without_customer_id(d: pd.DataFrame) -> dict[str, Any]:
    return _scalar(_money(_gross(d[d["customer"].isna()])))


def share_of_gross_revenue_from_top_countries(d: pd.DataFrame, n: int) -> dict[str, Any]:
    by = {c: _gross(g) for c, g in d.groupby("country")}
    top = sum(v for _, v in _top(by, n))
    return _scalar(Decimal(top) / Decimal(_gross(d)) * 100)


def median_order_value(d: pd.DataFrame) -> dict[str, Any]:
    """An order's value is the sum of Quantity x UnitPrice over that invoice's sale lines. The
    median is over invoices that have at least one sale line; for an even count it is the mean
    of the two middle values."""
    s = d[d["sale"]]
    totals = sorted(int(v) for v in s.groupby("invoice")["amount"].sum())
    n = len(totals)
    mid = totals[n // 2] * 2 if n % 2 else totals[n // 2 - 1] + totals[n // 2]
    return _scalar(Decimal(mid) / (2 * TICKS))


GOLD_FUNCTIONS = {
    f.__name__: f
    for f in (
        gross_revenue_in_country, net_revenue_in_month, top_countries_by_gross_revenue,
        orders_per_month, active_customers_in_country, month_with_highest_cancelled_revenue,
        average_order_value_in_country, percent_change_gross_revenue, net_revenue_in_two_months,
        net_revenue_in_country_between_months, top_products_by_merchandise_units_in_country,
        merchandise_units_per_order, gross_revenue_without_customer_id,
        share_of_gross_revenue_from_top_countries, median_order_value,
    )
}  # fmt: skip
