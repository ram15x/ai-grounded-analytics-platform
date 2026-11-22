"""Load an uploaded table into its own schema, and check metrics against the person's own figures.

Only columns the manifest names (and does not mark `ignore`) are loaded, so a column the person
flags as personal data never reaches the database. Values that do not fit their column are refused
with a count and examples; nothing is silently turned into NULL.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

from ..config import connect_kwargs
from .manifest import Manifest, ManifestError, condition_sql, metric_expr
from .profile import DatasetProfile, _numeric

SQL_TYPES = {"integer": "bigint", "decimal": "numeric", "date": "date", "datetime": "timestamp",
             "boolean": "boolean", "text": "text"}  # fmt: skip
_TRUE = {"true", "yes", "y", "t", "1"}


class LoadError(ValueError):
    pass


def _bad(name: str, orig: pd.Series, ok: pd.Series, what: str) -> LoadError:
    n = int((~ok).sum())
    ex = orig[~ok].astype(str).head(3).tolist()
    return LoadError(
        f"{name}: {n} values are not {what} (for example {ex}); fix the file or the role"
    )


def prepare(df: pd.DataFrame, profile: DatasetProfile, m: Manifest) -> tuple[pd.DataFrame, list]:
    """The frame to load (safe names, typed, derived month columns) and [(name, sql_type)]."""
    renamed = df.copy()
    renamed.columns = [c.name for c in profile.columns]
    ptype = {c.name: c for c in profile.columns}
    out: dict[str, pd.Series] = {}
    cols: list[tuple[str, str]] = []
    for name, spec in m.columns.items():
        if spec.role == "ignore":
            continue
        src = renamed[name]
        blank = src.map(lambda v: v is None or (isinstance(v, str) and not v.strip()))
        src = src.where(~blank, None)
        if spec.role == "date":
            issues = " ".join(ptype[name].issues)
            order = spec.date_order
            if ("ambiguous" in issues or "conflict" in issues) and order is None:
                raise LoadError(
                    f"{name}: the day/month order is not clear; set date_order: dmy or mdy"
                )
            strings = src.astype(object).where(src.notna(), None)
            dt = pd.to_datetime(strings, errors="coerce", format="mixed", dayfirst=order == "dmy")
            if not (dt.notna() | src.isna()).all():
                raise _bad(name, src, dt.notna() | src.isna(), "dates")
            has_time = (dt.dropna().dt.normalize() != dt.dropna()).any()
            out[name] = dt if has_time else dt.dt.date
            out[f"{name}_month"] = dt.dt.to_period("M").dt.to_timestamp().dt.date
            cols += [(name, "timestamp" if has_time else "date"), (f"{name}_month", "date")]
        elif spec.role == "measure":
            num = _numeric(src.dropna().astype(str)).reindex(src.index)
            ok = num.notna() | src.isna()
            if not ok.all():
                raise _bad(name, src, ok, "numbers")
            whole = (num.dropna() % 1 == 0).all()
            out[name] = num.astype("Int64") if whole else num
            cols.append((name, "bigint" if whole else "numeric"))
        else:  # dimension or identifier: kept as text
            s = (
                src.astype(object)
                .where(src.notna(), None)
                .map(lambda v: None if v is None else str(v).strip())
            )
            s = s.map(
                lambda v: (
                    v[:-2] if isinstance(v, str) and v.endswith(".0") and v[:-2].isdigit() else v
                )
            )
            out[name] = s
            cols.append((name, "text"))
    return pd.DataFrame(out), cols


def load_dataset(df: pd.DataFrame, profile: DatasetProfile, m: Manifest, admin_url: str,
                 readonly_role: str | None = None) -> dict[str, Any]:  # fmt: skip
    frame, cols = prepare(df, profile, m)
    schema = f"ds_{m.name}"
    with psycopg.connect(admin_url, autocommit=False, **connect_kwargs(admin_url)) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        ddl = sql.SQL("CREATE TABLE {}.data ({})").format(
            sql.Identifier(schema),
            sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(n), sql.SQL(t)) for n, t in cols
            ),
        )
        conn.execute(ddl)
        names = [n for n, _ in cols]
        with conn.cursor().copy(
            sql.SQL("COPY {}.data ({}) FROM STDIN").format(
                sql.Identifier(schema), sql.SQL(", ").join(map(sql.Identifier, names))
            )
        ) as cp:
            for row in frame[names].itertuples(index=False, name=None):
                cp.write_row([None if pd.isna(v) else v for v in row])
        if readonly_role:
            r = sql.Identifier(readonly_role)
            conn.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), r)
            )
            conn.execute(sql.SQL("GRANT SELECT ON {}.data TO {}").format(sql.Identifier(schema), r))
        n = conn.execute(
            sql.SQL("SELECT COUNT(*) FROM {}.data").format(sql.Identifier(schema))
        ).fetchone()[0]
        conn.commit()
    skipped = [
        c.name
        for c in profile.columns
        if c.name not in m.columns or m.columns[c.name].role == "ignore"
    ]
    return {"relation": m.relation, "rows": n, "loaded_columns": names, "not_loaded": skipped}


def reconcile(m: Manifest, profile: DatasetProfile, url: str) -> list[dict[str, Any]]:
    """Recompute each metric's `known` figures straight from the table and compare.

    This does not use the compiler, so it checks the definition itself, by a separate path."""
    types = {c.name: c.type for c in profile.columns}
    from .manifest import exclusion_conditions  # noqa: PLC0415

    excl = exclusion_conditions(m)
    results = []
    with psycopg.connect(url, **connect_kwargs(url)) as conn:
        for name, spec in m.metrics.items():
            for k in spec.known:
                where = " AND ".join(condition_sql(c, types) for c in k.where)
                tail = f" FROM {m.relation} AS l" + (f" WHERE {where}" if where else "")
                want, tol = Decimal(k.value), Decimal(k.tolerance)
                if spec.aggregate:
                    got = conn.execute(f"SELECT {metric_expr(spec, types, excl)}{tail}").fetchone()[
                        0
                    ]
                    got = None if got is None else Decimal(str(got))
                else:
                    # a ratio: both parts recomputed by this separate path, then divided and
                    # compared at the ratio's own decimal places (half a unit in the last place)
                    num, den = m.metrics[spec.numerator], m.metrics[spec.denominator]
                    n_expr, d_expr = metric_expr(num, types, excl), metric_expr(den, types, excl)
                    n, d = conn.execute(f"SELECT {n_expr}, {d_expr}{tail}").fetchone()
                    got = None if n is None or d in (None, 0) else Decimal(str(n)) / Decimal(str(d))
                    if "tolerance" not in k.model_fields_set:  # the default is for amounts
                        tol = Decimal(1).scaleb(-spec.decimals) / 2
                ok = got is not None and abs(got - want) <= tol
                if got is not None and not spec.aggregate:  # a ratio: two places past its own
                    got = got.quantize(Decimal(1).scaleb(-(spec.decimals + 2)))
                shown = None if got is None else str(got)
                results.append({"metric": name, "expected": k.value, "got": shown,
                                "ok": ok, "note": k.note})  # fmt: skip
    return results


def check_reconciliation(results: list[dict[str, Any]]) -> None:
    failed = [r for r in results if r["ok"] is False]
    if failed:
        lines = [
            f"{r['metric']}: expected {r['expected']}, the data gives {r['got']}" for r in failed
        ]
        raise ManifestError("the definition does not reproduce your figure: " + "; ".join(lines))
