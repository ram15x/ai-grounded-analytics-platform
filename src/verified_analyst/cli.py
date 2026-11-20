"""Command line: `va load`, `va profile`, `va verify answer.json`.

Environment (never files):
  VA_DATABASE_URL           admin/loader connection (load, profile)
  VA_READONLY_DATABASE_URL  read-only role connection (verify)
  VA_READONLY_PASSWORD      optional: create/refresh the read-only role during load
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from pydantic import ValidationError

from .claims import AnswerDoc
from .compiler import CompileError, compile_query
from .config import ConfigError, admin_database_url, readonly_database_url
from .guard import GuardError, QueryTimeout, execute_guarded
from .semantic import SemanticError, load_semantic_layer
from .verifier import ExecutedQuery, Report, verify_answer


def _execute(layer, readonly_url: str, spec) -> ExecutedQuery:
    if spec.metric_query is not None:
        cq = compile_query(layer, spec.metric_query)
        res = execute_guarded(readonly_url, cq.sql)
        return ExecutedQuery(spec.id, cq.sql, res.columns, res.rows, metric_query=spec.metric_query)
    res = execute_guarded(readonly_url, spec.sql)
    return ExecutedQuery(
        spec.id, spec.sql, res.columns, res.rows, declared_metric_query=spec.declared_metric_query
    )


def catalogue(layer, readonly_url: str) -> dict[str, set[str]]:
    """All values of small closed dimensions, so the verifier can spot a wrong scope in prose."""
    out: dict[str, set[str]] = {}
    for name, dim in layer.dimensions.items():
        if dim.enumerable and dim.type == "text":
            sql = f"SELECT DISTINCT {dim.column} AS v FROM {layer.source.relation} AS {layer.source.alias}"
            out[name] = {
                str(r[0]) for r in execute_guarded(readonly_url, sql).rows if r[0] is not None
            }
    return out


def run_verify(path: Path, readonly_url: str) -> Report:
    layer = load_semantic_layer()
    doc = AnswerDoc.model_validate_json(path.read_text(encoding="utf-8"))
    executed = {q.id: _execute(layer, readonly_url, q) for q in doc.queries}

    def shadow(mq):
        cq = compile_query(layer, mq)
        res = execute_guarded(readonly_url, cq.sql)
        return ExecutedQuery("shadow", cq.sql, res.columns, res.rows, metric_query=mq)

    return verify_answer(layer, doc, executed, shadow, catalogue(layer, readonly_url))


def render(report: Report) -> str:
    lines = [f"STATUS: {report.status}"]
    for v in report.verdicts:
        extra = f" ({v.detail})" if v.detail else ""
        lines.append(f"  {v.verdict:<19} {v.text!r:<22} {v.reason}{extra}")
    if report.dropped_claims:
        lines.append(f"  dropped claims (text not in prose): {', '.join(report.dropped_claims)}")
    lines.append(f"  excluded (not claims): {len(report.excluded)}")
    return "\n".join(lines)


def cmd_verify(args: argparse.Namespace) -> int:
    report = run_verify(Path(args.answer), readonly_database_url())
    print(json.dumps(report.as_dict(), indent=2) if args.json else render(report))
    return 0 if report.status == "verified" else 1


def cmd_load(args: argparse.Namespace) -> int:
    from .warehouse.load import download_uci, load_file  # noqa: PLC0415

    if args.download:
        path = download_uci(Path(args.dest))
    elif args.source:
        path = Path(args.source)
    else:
        print("give --source FILE or --download", file=sys.stderr)
        return 2
    n, info = load_file(path, admin_database_url(), os.environ.get("VA_READONLY_PASSWORD"))
    print(json.dumps({"reconcile": info, "dq": n.dq}, indent=2, default=str))
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    import pandas as pd  # noqa: PLC0415

    from .warehouse.load import read_raw  # noqa: PLC0415
    from .warehouse.normalize import normalize  # noqa: PLC0415

    n = normalize(read_raw(Path(args.source)))
    print(json.dumps(n.dq, indent=2, default=str))
    print(
        json.dumps(
            {
                "customers": len(n.customers),
                "products": len(n.products),
                "invoices": len(n.invoices),
                "lines": len(n.lines),
                "date_range": [
                    str(pd.to_datetime(n.invoices["invoice_ts"]).min()),
                    str(pd.to_datetime(n.invoices["invoice_ts"]).max()),
                ],
            },
            indent=2,
        )
    )
    return 0


PROFILE_METRICS = [
    "gross_revenue", "cancelled_revenue", "discount_amount", "net_revenue",
    "units_sold", "order_count", "active_customers", "aov", "revenue_per_customer",
]  # fmt: skip


def profile_db(readonly_url: str) -> dict:
    """Metric values over analytics.sales_lines, all rows vs excluding flagged exact duplicates.

    Uses the semantic layer's own expressions, so these are the numbers the product would show.
    """
    from .compiler import metric_expression  # noqa: PLC0415

    layer = load_semantic_layer()
    src = f"{layer.source.relation} AS {layer.source.alias}"
    out: dict = {}
    a = layer.source.alias
    cols = ", ".join(f"{metric_expression(layer, m)} AS {m}" for m in PROFILE_METRICS)
    for label, where in (
        ("all_rows", ""),
        ("excluding_exact_duplicates", f" WHERE NOT {a}.is_exact_duplicate"),
    ):
        res = execute_guarded(readonly_url, f"SELECT {cols} FROM {src}{where}")
        out[label] = {k: str(v) for k, v in zip(res.columns, res.rows[0], strict=True)}
    counts = execute_guarded(
        readonly_url,
        f"SELECT {a}.line_type, {a}.is_exact_duplicate, COUNT(*) AS n, "
        f"SUM({a}.line_amount) AS amount FROM {src} GROUP BY 1, 2 ORDER BY 1, 2",
    )
    out["rows_by_line_type_and_duplicate_flag"] = [
        dict(zip(counts.columns, (str(x) for x in r), strict=True)) for r in counts.rows
    ]
    return out


def cmd_profile_db(args: argparse.Namespace) -> int:
    print(json.dumps(profile_db(readonly_database_url()), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="va", description="Verified Analyst (deterministic core)")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify a hand-written answer.json against the warehouse")
    v.add_argument("answer")
    v.add_argument("--json", action="store_true", help="machine-readable report")
    v.set_defaults(fn=cmd_verify)
    ld = sub.add_parser("load", help="load the UCI Online Retail file into Postgres")
    ld.add_argument("--source", help="path to Online Retail .xlsx/.csv")
    ld.add_argument("--download", action="store_true", help="download from UCI first")
    ld.add_argument("--dest", default="data", help="download directory")
    ld.set_defaults(fn=cmd_load)
    pr = sub.add_parser("profile", help="data-quality counts from a source file (no database)")
    pr.add_argument("source")
    pr.set_defaults(fn=cmd_profile)
    pd_ = sub.add_parser(
        "profile-db", help="metric totals from the loaded warehouse, with/without duplicates"
    )
    pd_.set_defaults(fn=cmd_profile_db)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except psycopg.Error as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        print(f"error: database: {first}", file=sys.stderr)
        return 2
    except (ValidationError, SemanticError, CompileError, GuardError, QueryTimeout) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
