"""Pure-pandas normalisation of the raw Online Retail table.

No database access here. Every data-quality decision is made in this file and
counted in the returned ``dq`` dictionary, which `grounded profile` renders into
docs/DATA_PROFILE.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

MERCHANDISE_RE = re.compile(r"^[0-9]{5}[A-Z]{0,3}$")
LINE_TYPES = ("sale", "cancellation", "discount", "adjustment")


@dataclass
class Normalized:
    customers: pd.DataFrame
    products: pd.DataFrame
    invoices: pd.DataFrame
    lines: pd.DataFrame
    dq: dict


def _clean_customer_id(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _mode(values: pd.Series) -> str:
    """Most frequent value, ties broken alphabetically, so output is deterministic."""
    counts = values.value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def classify_line(stock_code: str, invoice_no: str, quantity: int, unit_price: float) -> str:
    """Line type rules, applied in this order (documented in docs/DATA_PROFILE.md)."""
    if stock_code == "D":
        return "discount" if unit_price > 0 else "adjustment"
    if unit_price <= 0:
        return "adjustment"
    if invoice_no.startswith("C"):
        return "cancellation" if quantity < 0 else "adjustment"
    if quantity <= 0:
        return "adjustment"
    return "sale"


def normalize(raw: pd.DataFrame) -> Normalized:  # noqa: PLR0915
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")

    df = raw[RAW_COLUMNS].copy().reset_index(drop=True)
    dq: dict = {"raw_rows": len(df)}
    dq["null_counts"] = {c: int(df[c].isna().sum()) for c in RAW_COLUMNS}

    # Exact duplicates are flagged, not removed: repeated identical lines can be real.
    df["is_exact_duplicate"] = df.duplicated(subset=RAW_COLUMNS, keep="first")
    dq["exact_duplicate_rows"] = int(df["is_exact_duplicate"].sum())

    df["source_row"] = df.index + 1
    df["invoice_no"] = df["InvoiceNo"].astype(str).str.strip().str.upper()
    raw_codes = df["StockCode"].astype(str).str.strip()
    df["stock_code"] = raw_codes.str.upper()
    dq["stock_codes_changed_by_uppercasing"] = int((raw_codes != df["stock_code"]).sum())
    dq["stock_code_case_collisions"] = int(
        raw_codes.groupby(df["stock_code"]).nunique().gt(1).sum()
    )

    desc = df["Description"].astype("string").str.strip()
    desc = desc.mask(desc == "", pd.NA)
    dq["descriptions_trimmed"] = int(
        (df["Description"].astype("string").fillna("") != desc.fillna("")).sum()
    )
    df["description"] = desc

    df["quantity"] = pd.to_numeric(df["Quantity"], errors="raise").astype("int64")
    df["unit_price"] = pd.to_numeric(df["UnitPrice"], errors="raise").round(4)
    df["invoice_ts"] = pd.to_datetime(df["InvoiceDate"], errors="raise")
    df["customer_id"] = df["CustomerID"].map(_clean_customer_id)
    country = df["Country"].astype("string").str.strip()
    dq["country_missing_filled"] = int(country.isna().sum() + (country == "").sum())
    df["country"] = country.mask(country.isna() | (country == ""), "Unknown")

    standard = df["invoice_no"].str.fullmatch(r"C?[0-9]{6}")
    dq["nonstandard_invoice_no_rows"] = int((~standard).sum())
    dq["nonstandard_invoice_no_samples"] = sorted(df.loc[~standard, "invoice_no"].unique())[:10]

    df["product_kind"] = df["stock_code"].map(
        lambda s: "merchandise" if MERCHANDISE_RE.match(s) else "non_merchandise"
    )
    df["line_type"] = [
        classify_line(s, i, q, p)
        for s, i, q, p in zip(
            df["stock_code"], df["invoice_no"], df["quantity"], df["unit_price"], strict=True
        )
    ]

    dq["rows_invoice_prefix_C"] = int(df["invoice_no"].str.startswith("C").sum())
    dq["c_invoice_with_positive_qty"] = int(
        (df["invoice_no"].str.startswith("C") & (df["quantity"] > 0)).sum()
    )
    dq["negative_qty_without_c_prefix"] = int(
        ((df["quantity"] < 0) & ~df["invoice_no"].str.startswith("C")).sum()
    )
    dq["zero_quantity_rows"] = int((df["quantity"] == 0).sum())
    dq["zero_price_rows"] = int((df["unit_price"] == 0).sum())
    dq["negative_price_rows"] = int((df["unit_price"] < 0).sum())
    dq["missing_customer_rows"] = int(df["customer_id"].isna().sum())
    dq["line_type_counts"] = {t: int((df["line_type"] == t).sum()) for t in LINE_TYPES}
    dq["product_kind_counts"] = df["product_kind"].value_counts().sort_index().to_dict()
    dq["invoice_ts_min"] = str(df["invoice_ts"].min())
    dq["invoice_ts_max"] = str(df["invoice_ts"].max())

    # ---- dimensions -------------------------------------------------------------
    descs = df.dropna(subset=["description"])
    per_code = descs.groupby("stock_code")["description"]
    dq["products_with_multiple_descriptions"] = int(per_code.nunique().gt(1).sum())
    best_desc = per_code.agg(_mode)
    codes = sorted(df["stock_code"].unique())
    products = pd.DataFrame({"stock_code": codes})
    products["product_key"] = range(1, len(products) + 1)
    products["description"] = products["stock_code"].map(best_desc)
    products["product_kind"] = products["stock_code"].map(
        lambda s: "merchandise" if MERCHANDISE_RE.match(s) else "non_merchandise"
    )
    dq["products_without_description"] = int(products["description"].isna().sum())

    known = df.dropna(subset=["customer_id"])
    cust_country = known.groupby("customer_id")["country"]
    dq["customers_in_multiple_countries"] = int(cust_country.nunique().gt(1).sum())
    customers = pd.DataFrame({"customer_id": sorted(known["customer_id"].unique())})
    customers["customer_key"] = range(1, len(customers) + 1)
    customers["country"] = customers["customer_id"].map(cust_country.agg(_mode))

    # ---- invoices ---------------------------------------------------------------
    g = df.groupby("invoice_no", sort=False)
    dq["invoices_with_multiple_timestamps"] = int(g["invoice_ts"].nunique().gt(1).sum())
    dq["invoices_with_multiple_countries"] = int(g["country"].nunique().gt(1).sum())
    dq["invoices_with_multiple_customers"] = int(g["customer_id"].nunique().gt(1).sum())
    invoices = g.agg(
        invoice_ts=("invoice_ts", "min"),
        country=("country", _mode),
        n_lines=("source_row", "size"),
        customer_id=("customer_id", lambda s: _mode(s.dropna()) if s.notna().any() else None),
    ).reset_index()
    invoices = invoices.sort_values(["invoice_ts", "invoice_no"]).reset_index(drop=True)
    invoices["invoice_key"] = invoices.index + 1
    invoices["is_cancellation"] = invoices["invoice_no"].str.startswith("C")
    invoices = invoices.merge(
        customers[["customer_id", "customer_key"]], on="customer_id", how="left"
    )
    dq["invoices"] = len(invoices)
    dq["invoices_without_customer"] = int(invoices["customer_key"].isna().sum())

    # ---- lines ------------------------------------------------------------------
    lines = df.merge(invoices[["invoice_no", "invoice_key"]], on="invoice_no", how="left")
    lines = lines.merge(products[["stock_code", "product_key"]], on="stock_code", how="left")
    lines["line_no"] = lines.groupby("invoice_key").cumcount() + 1
    lines = lines[
        [
            "source_row",
            "invoice_key",
            "product_key",
            "line_no",
            "quantity",
            "unit_price",
            "line_type",
            "is_exact_duplicate",
        ]
    ].rename(columns={"source_row": "line_key"})

    dq["customers"] = len(customers)
    dq["products"] = len(products)
    dq["lines"] = len(lines)
    return Normalized(
        customers=customers[["customer_key", "customer_id", "country"]],
        products=products[["product_key", "stock_code", "description", "product_kind"]],
        invoices=invoices[
            [
                "invoice_key",
                "invoice_no",
                "invoice_ts",
                "customer_key",
                "country",
                "is_cancellation",
                "n_lines",
            ]
        ],
        lines=lines,
        dq=dq,
    )
