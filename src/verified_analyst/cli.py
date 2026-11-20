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

from .claims import AnswerDoc, TemplateAnswer
from .compiler import CompileError, compile_query
from .config import ConfigError, admin_database_url, readonly_database_url
from .guard import GuardError, QueryTimeout, execute_guarded
from .llm.live import redact
from .llm.types import LLMError
from .semantic import SemanticError, load_semantic_layer
from .template import TemplateReport, is_template_doc, verify_template
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
            src = f"{layer.source.relation} AS {layer.source.alias}"
            sql = f"SELECT DISTINCT {dim.column} AS v FROM {src}"
            out[name] = {
                str(r[0]) for r in execute_guarded(readonly_url, sql).rows if r[0] is not None
            }
    return out


def run_verify(path: Path, readonly_url: str) -> Report | TemplateReport:
    layer = load_semantic_layer()
    raw = json.loads(path.read_text(encoding="utf-8"))

    def shadow(mq):
        cq = compile_query(layer, mq)
        res = execute_guarded(readonly_url, cq.sql)
        return ExecutedQuery("shadow", cq.sql, res.columns, res.rows, metric_query=mq)

    known = catalogue(layer, readonly_url)
    if isinstance(raw, dict) and is_template_doc(raw):
        tdoc = TemplateAnswer.model_validate(raw)
        executed = {q.id: _execute(layer, readonly_url, q) for q in tdoc.queries}
        return verify_template(layer, tdoc, executed, shadow, known)
    doc = AnswerDoc.model_validate(raw)
    executed = {q.id: _execute(layer, readonly_url, q) for q in doc.queries}
    return verify_answer(layer, doc, executed, shadow, known)


def render(report: Report | TemplateReport) -> str:
    lines = [f"STATUS: {report.status}"]
    if isinstance(report, TemplateReport):
        lines.append(f"  rendered: {report.rendered}")
        lines += [f"  ISSUE {i.code}: {i.detail}" for i in report.issues]
        report = report.report
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


def cmd_cassettes_check(args: argparse.Namespace) -> int:
    from .llm.cassette import (  # noqa: PLC0415
        check_cassettes,
        layer_hash,
        render_check,
        schema_hash,
        warehouse_content_hash,
    )
    from .llm.prompts import prompt_hashes  # noqa: PLC0415

    wh = args.warehouse_hash
    if wh is None and os.environ.get("VA_READONLY_DATABASE_URL"):
        wh = warehouse_content_hash(readonly_database_url())
    usage = json.loads(Path(args.usage).read_text("utf-8")) if args.usage else None
    result = check_cassettes(
        Path(args.dir),
        {"layer": layer_hash(), "schema": schema_hash(), "warehouse": wh},
        prompt_hashes(),
        usage=usage,
        max_age_days=args.max_age_days,
    )
    print(json.dumps(result, indent=2) if args.json else render_check(result))
    return 0 if result["ok"] else 1


def cmd_live_models(args: argparse.Namespace) -> int:
    from .llm import live  # noqa: PLC0415
    from .llm.settings import load_router_settings  # noqa: PLC0415

    settings = load_router_settings()
    names = args.provider or [
        n for n, p in settings.providers.items() if p.kind != "fake" and p.hosted
    ]
    unknown = [n for n in names if n not in settings.providers]
    if unknown:
        raise ConfigError(f"unknown provider(s): {', '.join(unknown)}")
    results = [live.list_provider_models(settings, n, os.environ) for n in names]
    text = (
        json.dumps(results, indent=2, sort_keys=True)
        if args.json
        else live.render_models(results, show_all=args.all)
    )
    print(live.redact(text, os.environ))
    any_key = any(r["key_present"] for r in results)
    return 0 if any_key and not any(r["error"] for r in results) else 1


def cmd_live_probe(args: argparse.Namespace) -> int:
    from .llm import live  # noqa: PLC0415
    from .llm.settings import load_router_settings  # noqa: PLC0415

    settings = load_router_settings()
    layer = load_semantic_layer()
    url = readonly_database_url()
    opts = live.ProbeOptions(
        provider=args.provider,
        model=args.model,
        out_dir=Path(args.out_dir)
        if args.out_dir
        else live.default_out_dir(args.provider, args.model),
        max_requests=args.max_requests,
        token_cap=args.token_cap,
        max_tokens=args.max_tokens,
        quota_check=args.quota_check,
        record=not args.no_record,
    )
    out = live.run_probe(
        settings,
        opts,
        layer=layer,
        readonly_url=url,
        questions=live.load_probe_questions(),
        known_values=catalogue(layer, url),
        env=os.environ,
    )
    print(live.redact(out["markdown"], os.environ))
    print(f"files written to: {out['out_dir']}")
    if out["cassette"]:
        c = out["cassette"]
        print(
            f"cassettes: {c['entries']} entries in {c['dir']}; key scan: {c['key_scan']}; "
            f"check ok: {c['check_ok']}; internal payloads skipped: {c['skipped_internal']}"
        )
        print("Review the cassette files before committing anything. Nothing was committed.")
    return 3 if out["stopped_on_auth"] else 4 if out["stopped_early"] else 0


def _request_cap(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if not 1 <= n <= 60:
        raise argparse.ArgumentTypeError("must be between 1 and 60")
    return n


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
    cs = sub.add_parser("cassettes", help="inspect recorded model-call cassettes")
    cs_sub = cs.add_subparsers(dest="cassette_cmd", required=True)
    cc = cs_sub.add_parser(
        "check", help="list stale, unused, missing and old cassette entries (no network)"
    )
    cc.add_argument("dir", help="cassette directory, for example tests/cassettes/<suite>")
    cc.add_argument("--usage", help="usage.json written by a replay run (for unused/missing)")
    cc.add_argument("--max-age-days", type=float, default=None)
    cc.add_argument("--warehouse-hash", default=None, help="else computed from the database")
    cc.add_argument("--json", action="store_true")
    cc.set_defaults(fn=cmd_cassettes_check)
    lv = sub.add_parser("live", help="first real provider calls (needs keys in the environment)")
    lv_sub = lv.add_subparsers(dest="live_cmd", required=True)
    lm = lv_sub.add_parser("models", help="list the model IDs available to each key")
    lm.add_argument("--provider", action="append", help="groq, gemini or openrouter (repeatable)")
    lm.add_argument(
        "--all", action="store_true", help="print every OpenRouter model, not only free"
    )
    lm.add_argument("--json", action="store_true")
    lm.set_defaults(fn=cmd_live_models)
    lp = lv_sub.add_parser("probe", help="fixed battery against ONE model, public data only")
    lp.add_argument("--provider", required=True)
    lp.add_argument("--model", required=True)
    lp.add_argument("--out-dir", help="default data/live_runs/<time>_<provider>_<model>")
    lp.add_argument(
        "--max-requests", type=_request_cap, default=None, metavar="N",
        help="HTTP request cap for the run, 1 to 60 (default: the config value, 25)",
    )  # fmt: skip
    lp.add_argument("--token-cap", type=int, default=None, help="lowers the config cap only")
    lp.add_argument("--max-tokens", type=int, default=None, help="completion allowance per call")
    lp.add_argument("--quota-check", action="store_true", help="3 extra requests, see the report")
    lp.add_argument("--no-record", action="store_true", help="do not record cassettes")
    lp.set_defaults(fn=cmd_live_probe)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, LLMError) as exc:
        print(f"error: {redact(str(exc), os.environ)}", file=sys.stderr)
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
