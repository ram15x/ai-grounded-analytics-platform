"""Independent derivation of every metric, straight from the RAW table in pandas.

It deliberately shares no code with the SQL, the compiler, or the warehouse
normaliser: it re-reads raw columns and applies the metric definitions with boolean
masks. It is independent of the implementation, not of the business rules (those
are the specification in semantic/metrics.yaml and docs/DATA_PROFILE.md).
"""

from __future__ import annotations

import pandas as pd


def _clean_cust(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    t = str(v).strip()
    if t == "" or t.lower() == "nan":
        return None
    return str(int(float(t)))


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "invoice_no": raw["InvoiceNo"].astype(str).str.strip().str.upper(),
            "code": raw["StockCode"].astype(str).str.strip().str.upper(),
            "qty": pd.to_numeric(raw["Quantity"]).astype(int),
            "price": pd.to_numeric(raw["UnitPrice"]).astype(float),
            "ts": pd.to_datetime(raw["InvoiceDate"]),
            "cust": raw["CustomerID"].map(_clean_cust),
            "country": raw["Country"].astype(str).str.strip(),
        }
    )
    df["amount"] = df["qty"] * df["price"]
    is_c = df["invoice_no"].str.startswith("C")
    pos = df["price"] > 0
    df["merch"] = df["code"].str.match(r"^[0-9]{5}[A-Z]{0,3}$")
    df["sale"] = pos & ~is_c & (df["code"] != "D") & (df["qty"] > 0)
    df["canc"] = pos & is_c & (df["code"] != "D") & (df["qty"] < 0)
    df["disc"] = pos & (df["code"] == "D")
    df["d_month"] = df["ts"].dt.to_period("M").dt.to_timestamp().dt.date
    df["d_day"] = df["ts"].dt.date
    df["d_country"] = df["country"]
    df["d_stock_code"] = df["code"]
    df["d_product_kind"] = df["merch"].map({True: "merchandise", False: "non_merchandise"})
    df["d_customer_id"] = df["cust"]
    return df


def _ratio(a, b):
    return None if not b else a / b


def metrics_for(g: pd.DataFrame) -> dict:
    known = g["cust"].notna()
    gross = g.loc[g["sale"], "amount"].sum()
    canc = -g.loc[g["canc"], "amount"].sum()
    disc = -g.loc[g["disc"], "amount"].sum()
    ident = g.loc[g["sale"] & known, "amount"].sum()
    orders = g.loc[g["sale"], "invoice_no"].nunique()
    active = g.loc[g["sale"] & known, "cust"].nunique()
    return {
        "gross_revenue": gross,
        "cancelled_revenue": canc,
        "discount_amount": disc,
        "net_revenue": g.loc[g["sale"] | g["canc"] | g["disc"], "amount"].sum(),
        "units_sold": g.loc[g["sale"] & g["merch"], "qty"].sum(),
        "order_count": orders,
        "active_customers": active,
        "identified_gross_revenue": ident,
        "aov": _ratio(gross, orders),
        "revenue_per_customer": _ratio(ident, active),
        "cancellation_rate": _ratio(canc, gross),
    }


def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    for f in filters:
        col, op, val = f"d_{f['dimension']}", f["op"], f["value"]
        if f["dimension"] in ("month", "day"):

            def to_date(v):
                return pd.Timestamp(v + "-01" if len(v) == 7 else v).date()

            val = [to_date(v) for v in val] if isinstance(val, list) else to_date(val)
        s = df[col]
        if op == "eq":
            df = df[s == val]
        elif op == "ne":
            df = df[s != val]
        elif op == "in":
            df = df[s.isin(val)]
        elif op == "not_in":
            df = df[~s.isin(val)]
        elif op == "between":
            df = df[(s >= val[0]) & (s <= val[1])]
        elif op == "gte":
            df = df[s >= val]
        elif op == "lte":
            df = df[s <= val]
        else:
            raise ValueError(op)
    return df


def oracle(raw, metrics: list[str], dimensions: list[str] | None = None, filters=None) -> dict:
    """Return {group_key_tuple: {metric: value}}."""
    df = apply_filters(prepare(raw), filters or [])
    dims = [f"d_{d}" for d in (dimensions or [])]
    out: dict = {}
    if not dims:
        m = metrics_for(df)
        return {(): {k: m[k] for k in metrics}} if len(df) else {}
    for key, g in df.groupby(dims, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        key = tuple(
            None if (k is None or (isinstance(k, float) and pd.isna(k))) else k for k in key
        )
        m = metrics_for(g)
        out[key] = {k: m[k] for k in metrics}
    return out
