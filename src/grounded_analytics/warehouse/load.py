"""Read the raw file, normalise it, and load it into Postgres."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from grounded_analytics.config import connect_kwargs
from grounded_analytics.warehouse.normalize import RAW_COLUMNS, Normalized, normalize

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
READONLY_ROLE_DEFAULT = "ga_readonly"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_uci(dest_dir: Path) -> Path:
    """Download the official zip from UCI and extract the xlsx. Needs network access."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
        if not names:
            raise RuntimeError(f"no xlsx/csv inside the UCI zip: {zf.namelist()}")
        target = dest_dir / Path(names[0]).name
        target.write_bytes(zf.read(names[0]))
    return target


def read_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ValueError(f"unsupported file type: {path.suffix}")


def create_schema(conn: psycopg.Connection) -> None:
    ddl = resources.files("grounded_analytics.warehouse").joinpath("schema.sql").read_text("utf-8")
    conn.execute(ddl)


def _copy(conn: psycopg.Connection, table: str, columns: list[str], frame: pd.DataFrame) -> None:
    stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.SQL(table), sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with conn.cursor().copy(stmt) as cp:
        for row in frame[columns].itertuples(index=False, name=None):
            cp.write_row([None if pd.isna(v) else v for v in row])


def load_tables(conn: psycopg.Connection, n: Normalized) -> None:
    _copy(conn, "warehouse.dim_customer", ["customer_key", "customer_id", "country"], n.customers)
    _copy(
        conn,
        "warehouse.dim_product",
        ["product_key", "stock_code", "description", "product_kind"],
        n.products,
    )
    inv = n.invoices.copy()
    inv["customer_key"] = inv["customer_key"].astype("Int64")
    _copy(
        conn,
        "warehouse.fact_invoice",
        [
            "invoice_key",
            "invoice_no",
            "invoice_ts",
            "customer_key",
            "country",
            "is_cancellation",
            "n_lines",
        ],
        inv,
    )
    _copy(
        conn,
        "warehouse.fact_invoice_lines",
        [
            "line_key",
            "invoice_key",
            "product_key",
            "line_no",
            "quantity",
            "unit_price",
            "line_type",
            "is_exact_duplicate",
        ],
        n.lines,
    )


def setup_readonly_role(
    conn: psycopg.Connection, role: str, password: str, database: str, timeout_ms: int = 10_000
) -> None:
    """Create (or reset) a SELECT-only role that can see the `analytics` schema only."""
    r = sql.Identifier(role)
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    verb = "ALTER" if exists else "CREATE"
    conn.execute(
        sql.SQL("{} ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE").format(
            sql.SQL(verb), r, sql.Literal(password)
        )
    )
    conn.execute(sql.SQL("REVOKE ALL ON SCHEMA public FROM PUBLIC"))
    conn.execute(sql.SQL("REVOKE ALL ON SCHEMA warehouse FROM PUBLIC"))
    conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(database), r))
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA analytics TO {}").format(r))
    conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO {}").format(r))
    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(r))
    conn.execute(
        sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(r, sql.Literal(timeout_ms))
    )
    conn.execute(sql.SQL("ALTER ROLE {} SET search_path = analytics").format(r))
    conn.execute(sql.SQL("ALTER ROLE {} SET idle_in_transaction_session_timeout = 30000").format(r))


def reconcile(conn: psycopg.Connection, n: Normalized, raw: pd.DataFrame) -> dict:
    """Check the database against the in-memory data. Raises on any mismatch."""
    counts = {
        "lines": conn.execute("SELECT count(*) FROM warehouse.fact_invoice_lines").fetchone()[0],
        "invoices": conn.execute("SELECT count(*) FROM warehouse.fact_invoice").fetchone()[0],
        "customers": conn.execute("SELECT count(*) FROM warehouse.dim_customer").fetchone()[0],
        "products": conn.execute("SELECT count(*) FROM warehouse.dim_product").fetchone()[0],
    }
    expected = {
        "lines": len(raw),
        "invoices": n.dq["invoices"],
        "customers": n.dq["customers"],
        "products": n.dq["products"],
    }
    if counts != expected:
        raise RuntimeError(f"row-count mismatch: db={counts} expected={expected}")
    db_total = conn.execute("SELECT sum(line_amount) FROM warehouse.fact_invoice_lines").fetchone()[
        0
    ]
    raw_total = (
        pd.to_numeric(raw["Quantity"]).astype("int64") * pd.to_numeric(raw["UnitPrice"]).round(4)
    ).sum()
    if abs(float(db_total) - float(raw_total)) > 0.01:
        raise RuntimeError(f"line-amount total mismatch: db={db_total} raw={raw_total}")
    return {"counts": counts, "line_amount_total": str(db_total)}


def load_file(
    path: Path,
    admin_url: str,
    readonly_password: str | None = None,
    readonly_role: str = READONLY_ROLE_DEFAULT,
) -> tuple[Normalized, dict]:
    raw = read_raw(path)
    n = normalize(raw)
    with psycopg.connect(admin_url, autocommit=False, **connect_kwargs(admin_url)) as conn:
        create_schema(conn)
        load_tables(conn, n)
        info = reconcile(conn, n, raw)
        if readonly_password:
            db = conn.info.dbname
            setup_readonly_role(conn, readonly_role, readonly_password, db)
        conn.commit()
    info["source_file"] = path.name
    info["source_sha256"] = sha256_of(path)
    return n, info


__all__ = ["RAW_COLUMNS", "download_uci", "load_file", "read_raw"]
