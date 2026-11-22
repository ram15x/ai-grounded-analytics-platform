"""Command line: `va load`, `va profile`, `va verify answer.json`.

Environment (never files):
  VA_DATABASE_URL           admin/loader connection (load, profile)
  VA_READONLY_DATABASE_URL  read-only role connection (verify)
  VA_READONLY_PASSWORD      optional: create/refresh the read-only role during load
  VA_CONNECT_TIMEOUT_S      optional: seconds to wait for a database connection (default 10)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from pydantic import ValidationError

from .claims import AnswerDoc, PassageRef, TemplateAnswer
from .compiler import CompileError, compile_query
from .config import ConfigError, admin_database_url, connect_kwargs, readonly_database_url
from .documents.store import DocumentError
from .guard import GuardError, QueryTimeout, allowed_for, execute_guarded
from .llm.live import redact
from .llm.types import LLMError
from .semantic import SemanticError, load_semantic_layer
from .template import TemplateReport, is_template_doc, verify_template
from .verifier import ExecutedQuery, Report, verify_answer


def _execute(layer, readonly_url: str, spec) -> ExecutedQuery:
    if spec.metric_query is not None:
        cq = compile_query(layer, spec.metric_query)
        res = execute_guarded(readonly_url, cq.sql, allowed_relations=allowed_for(layer))
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
            rows = execute_guarded(readonly_url, sql, allowed_relations=allowed_for(layer)).rows
            out[name] = {str(r[0]) for r in rows if r[0] is not None}
    return out


def run_verify(path: Path, readonly_url: str) -> Report | TemplateReport:
    layer = load_semantic_layer()
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Validate the file BEFORE any database connection: a malformed answer must fail at once and
    # never wait on a connection (on Windows a connect to `localhost` can hang for minutes).
    tdoc = doc = None
    if isinstance(raw, dict) and is_template_doc(raw):
        tdoc = TemplateAnswer.model_validate(raw)
    else:
        doc = AnswerDoc.model_validate(raw)

    def shadow(mq):
        cq = compile_query(layer, mq)
        res = execute_guarded(readonly_url, cq.sql, allowed_relations=allowed_for(layer))
        return ExecutedQuery("shadow", cq.sql, res.columns, res.rows, metric_query=mq)

    known = catalogue(layer, readonly_url)
    if tdoc is not None:
        executed = {q.id: _execute(layer, readonly_url, q) for q in tdoc.queries}
        return verify_template(layer, tdoc, executed, shadow, known)
    assert doc is not None
    executed = {q.id: _execute(layer, readonly_url, q) for q in doc.queries}
    return verify_answer(layer, doc, executed, shadow, known, _stored_passages(doc, readonly_url))


def _stored_passages(doc: AnswerDoc, readonly_url: str) -> dict[str, str] | None:
    """Passage text comes from the document store, never from the answer file. None: no store."""
    from .documents.store import resolve_passages  # noqa: PLC0415

    if doc.passages:
        print(
            f"note: ignored {len(doc.passages)} passage text(s) written in the answer file; "
            "cited passages are read from the document store",
            file=sys.stderr,
        )
    ids = [c.provenance.passage_id for c in doc.claims if isinstance(c.provenance, PassageRef)]
    if not ids:
        return {}
    try:
        return resolve_passages(readonly_url, ids)
    except (psycopg.errors.UndefinedTable, psycopg.errors.InsufficientPrivilege):
        return None  # no `docs` schema (or not readable): passage claims are UNSUPPORTED


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


def cmd_docs_ingest(args: argparse.Namespace) -> int:
    from .documents.store import ingest, read_document  # noqa: PLC0415
    from .documents.untrusted import scan  # noqa: PLC0415
    from .warehouse.load import READONLY_ROLE_DEFAULT  # noqa: PLC0415

    files: list[Path] = []
    for item in args.paths:
        p = Path(item)
        files += (
            sorted([*p.glob("*.md"), *p.glob("*.txt"), *p.glob("*.pdf")]) if p.is_dir() else [p]
        )
    if not files:
        print("error: no .md, .txt or .pdf files found", file=sys.stderr)
        return 2
    docs = [(f, *read_document(f)) for f in files]  # every file is checked before connecting
    admin_url = admin_database_url()
    role = args.reader_role or READONLY_ROLE_DEFAULT
    with psycopg.connect(admin_url, **connect_kwargs(admin_url)) as conn:
        for f, key, title, text in docs:
            r = ingest(
                conn,
                key,
                title,
                f.name,
                text,
                args.max_chars,
                reader_role=role,
                sensitivity=args.sensitivity,
            )
            state = "stored" if r.changed else "unchanged"
            print(f"{state:<9} {r.doc_key} ({r.chunks} chunks)")
            kinds = sorted({fl.kind for fl in scan(text)})
            if kinds:
                print(
                    f"warning: {r.doc_key} has text flagged as {', '.join(kinds)}; it is stored "
                    "as written, withheld from model prompts and cannot be cited where flagged",
                    file=sys.stderr,
                )
    return 0


def cmd_docs_search(args: argparse.Namespace) -> int:
    from .documents.store import search  # noqa: PLC0415
    from .documents.untrusted import printable, scan  # noqa: PLC0415

    hits = search(readonly_database_url(), args.query, args.k, args.config, args.mode)
    if args.json:
        rows = [{**h.__dict__, "flags": sorted({f.kind for f in scan(h.text)})} for h in hits]
        print(json.dumps(rows, indent=2))  # ASCII-escaped: invisible characters stay visible
        return 0
    for h in hits:
        title = printable(h.title, 80)
        print(f"{h.score:6.3f}  {h.passage_id}  [{h.char_start}:{h.char_end}] {title}")
        print(f"        {h.text[:160]!r}")
        flags = sorted({f.kind for f in scan(h.text)})
        if flags:
            print(f"        flagged: {', '.join(flags)} (cannot be cited where flagged)")
    if not hits:
        print("no passage matched")
    return 0


DEV_DOC_QUESTIONS = Path(__file__).resolve().parents[2] / "eval" / "doc_questions" / "dev.yaml"


def cmd_docs_eval(args: argparse.Namespace) -> int:
    from .documents.metrics import Retrieved, evaluate, load_questions  # noqa: PLC0415
    from .documents.store import all_chunks, search  # noqa: PLC0415

    url = readonly_database_url()
    questions = load_questions(DEV_DOC_QUESTIONS)  # dev questions only; there is no option
    corpus = [Retrieved(k, t) for k, _, t in all_chunks(url)]

    def run(query: str, k: int) -> list[Retrieved]:
        return [Retrieved(h.doc_key, h.text) for h in search(url, query, k, args.config, args.mode)]

    out = evaluate(run, questions, corpus)
    out["config"] = {"ts_config": args.config, "mode": args.mode}
    print(json.dumps(out, indent=2))
    return 0 if not out["unreachable_anchors"] else 1


def cmd_docs_probe(args: argparse.Namespace) -> int:
    import time  # noqa: PLC0415

    from .documents.metrics import load_questions  # noqa: PLC0415
    from .documents.probe import render_markdown, run_probe, write_report  # noqa: PLC0415
    from .documents.settings import load_retrieval_settings  # noqa: PLC0415

    settings = load_retrieval_settings(Path(args.config) if args.config else None)
    res = run_probe(
        settings=settings,
        embedder_name=args.embedder,
        reranker_name=args.reranker,
        admin_url=admin_database_url(),
        readonly_url=readonly_database_url(),
        questions=load_questions(DEV_DOC_QUESTIONS),  # dev questions only; there is no option
        env=os.environ,
        max_requests=args.max_requests,
        reader_role=args.reader_role,
        study_fusion=args.fusion_study,
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tail = args.embedder + (f"_{args.reranker}" if args.reranker else "")
    out_dir = (
        Path(args.out_dir) if args.out_dir else Path("data") / "live_runs" / f"{stamp}_docs_{tail}"
    )
    write_report(res.report, out_dir)
    print(render_markdown(res.report))
    print(f"report written to {out_dir}")
    return res.exit_code


def cmd_docs_heldout(args: argparse.Namespace) -> int:
    import time  # noqa: PLC0415

    from .documents.heldout_run import render_markdown, run_heldout  # noqa: PLC0415
    from .documents.probe import write_report  # noqa: PLC0415
    from .documents.settings import load_retrieval_settings  # noqa: PLC0415

    res = run_heldout(
        settings=load_retrieval_settings(),
        embedder_name=args.embedder,
        candidates_arg=args.candidates,
        admin_url=admin_database_url(),
        readonly_url=readonly_database_url(),
        env=os.environ,
        max_requests=args.max_requests,
    )
    out_dir = (
        Path("data")
        / "live_runs"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_docs_heldout_{args.embedder}"
    )
    write_report(res.report, out_dir, render_markdown)
    print(render_markdown(res.report))
    print(f"report written to {out_dir}")
    return res.exit_code


def cmd_docs_binder_study(args: argparse.Namespace) -> int:
    from .documents import binder_study as bs  # noqa: PLC0415

    cases = bs.dev_cases()
    print(bs.render(bs.dev_table(cases), "dev", cases))
    print(bs.render_changes(bs.verdict_changes(cases, bs.CURRENT, bs.DEFAULT)))
    return 0


def cmd_docs_binder_locked(args: argparse.Namespace) -> int:
    import time  # noqa: PLC0415

    from .documents import binder_locked as bl  # noqa: PLC0415

    try:
        rep = bl.run_locked(candidates_arg=args.candidates)
    except DocumentError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(bl.render_report(rep))
    out = Path("data") / "live_runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_binder_locked"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"report written to {out}")
    return 0


def cmd_docs_binder_model(args: argparse.Namespace) -> int:
    from .documents import binder_model as bm  # noqa: PLC0415

    try:
        if args.what == "labels":
            from .llm import usage  # noqa: PLC0415
            from .llm.cassette import (  # noqa: PLC0415
                CassetteContext,
                layer_hash,
                schema_hash,
                warehouse_content_hash,
            )  # fmt: skip
            from .llm.live import redact  # noqa: PLC0415
            from .llm.settings import load_router_settings  # noqa: PLC0415

            cfg_now = load_router_settings()
            planned, left = bm.plan_generation(cfg_now, args.provider, args.model,
                                               Path(args.out_dir or bm.WORK_DIR))  # fmt: skip
            usage.start(cfg_now, args.provider, args.model, mode="binder-model-labels",
                        questions=left, ceiling=planned, planned_override=planned,
                        force=args.force)  # fmt: skip
            ctx = CassetteContext(layer_hash(), schema_hash(),
                                  warehouse_content_hash(readonly_database_url()))  # fmt: skip
            out = bm.run_generation(
                load_router_settings(), provider=args.provider, model=args.model,
                # one fixed folder, so an unfinished run is resumed from its cassettes
                out_dir=Path(args.out_dir or bm.WORK_DIR),
                env=os.environ, seed_report=Path(args.seed_run)
                if args.seed_run else None, max_tokens=args.max_tokens, cassette_ctx=ctx,
            )  # fmt: skip
            print(redact(out["markdown"], os.environ))
            print(f"files written to: {out['out_dir']}")
            m = out["meta"]
            print(f"usable labels: {out['usable_labels']} of {m['sentences_asked']} sentences "
                  f"asked (the locked scoring needs at least {bm.MIN_CLAIMS})")  # fmt: skip
            if not out["complete"]:
                print("INCOMPLETE: run the same command again to resume; nothing was spent.",
                      file=sys.stderr)  # fmt: skip
                return 3
            print("Labels written once; commit eval/binder_model_labels/labels.yaml, then run "
                  "`va docs binder-model-dev`. Do not open the locked-half claims.")  # fmt: skip
            return 0
        if args.what == "dev":
            print(bm.render_half(bm.half_report("dev")))
            return 0
        rep = bm.run_locked(candidates_arg=args.candidates)
    except (DocumentError, ConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(bm.render_half(rep))
    import time  # noqa: PLC0415

    out_dir = Path("data") / "live_runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_binder_model_locked"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"report written to {out_dir}")
    return 0


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
    from .llm import live, usage  # noqa: PLC0415
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
    qs = live.load_probe_questions()
    cap = min(args.token_cap or 10**9, settings.probe.token_cap)
    usage.start(settings, args.provider, args.model, mode="live probe", questions=len(qs),
                ceiling=cap, force=args.force)  # fmt: skip
    out = live.run_probe(
        settings,
        opts,
        layer=layer,
        readonly_url=url,
        questions=qs,
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


def cmd_route_probe(args: argparse.Namespace) -> int:
    from .llm import route_probe, usage  # noqa: PLC0415
    from .llm.live import default_out_dir, redact  # noqa: PLC0415
    from .llm.settings import load_router_settings  # noqa: PLC0415

    layer = load_semantic_layer()
    url = readonly_database_url()
    opts = route_probe.RouteProbeOptions(
        provider=args.provider,
        model=args.model,
        out_dir=Path(args.out_dir) if args.out_dir else default_out_dir(args.provider, args.model),
        max_requests=args.max_requests,
        token_cap=args.token_cap,
        max_tokens=args.max_tokens,
        route_only=args.route_only,
        record=args.record,
        injection_only=args.injection_only,
        reasoning_effort=args.reasoning_effort,
    )
    kw = {"layer": layer, "readonly_url": url, "known_values": catalogue(layer, url),
          "env": os.environ}  # fmt: skip
    router_qs, injection_qs = route_probe.load_dev_questions()
    nq = (0 if args.injection_only else len(router_qs)) + (
        0 if args.route_only else len(injection_qs)
    )
    mode = ("route only" if args.route_only else "injection questions only"
            if args.injection_only else "route plus full answer")  # fmt: skip
    cfg_now = load_router_settings()
    usage.start(cfg_now, args.provider, args.model, mode=mode, questions=nq,
                ceiling=min(args.token_cap or 10**9, cfg_now.probe.token_cap), force=args.force,
                multiplier=2 if args.compare_reasoning else 1)  # fmt: skip
    if args.compare_reasoning:
        if args.reasoning_effort:
            raise SystemExit("--compare-reasoning sets the effort itself; drop --reasoning-effort")
        efforts = tuple(e.strip() for e in args.compare_reasoning.split(",") if e.strip())
        if len(efforts) != 2:
            raise SystemExit("--compare-reasoning takes exactly two settings, e.g. low,medium")
        cmp = route_probe.run_reasoning_compare(load_router_settings(), opts, efforts, **kw)
        print(redact(cmp["markdown"], os.environ))
        print(f"files written to: {cmp['out_dir']}")
        return 0
    out = route_probe.run_route_probe(load_router_settings(), opts, **kw)
    print(redact(out["markdown"], os.environ))
    print(f"files written to: {out['out_dir']}")
    c = out["cassette"]
    if c:
        print(f"cassettes: {c['entries']} entries in {c['dir']}; key scan {c['key_scan']}; "
              f"check ok {c['check_ok']}; skipped internal {c['skipped_internal']}")  # fmt: skip
    print(("Cassettes were recorded locally. " if c else "Nothing was recorded. ")
          + "Nothing was committed.")  # fmt: skip
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
    dc = sub.add_parser("docs", help="documents: ingest, keyword search, retrieval metrics")
    dc_sub = dc.add_subparsers(dest="docs_cmd", required=True)
    di = dc_sub.add_parser("ingest", help="store .md/.txt files (or folders of them)")
    di.add_argument("paths", nargs="+")
    di.add_argument("--max-chars", type=int, default=800, help="largest chunk, in characters")
    di.add_argument("--reader-role", default=None, help="read-only role to grant access to")
    di.add_argument(
        "--sensitivity",
        choices=["public", "internal"],
        default="internal",
        help="only public documents may be sent to a hosted model (default: internal)",
    )
    di.set_defaults(fn=cmd_docs_ingest)
    ds = dc_sub.add_parser("search", help="keyword search; prints passage ids and spans")
    ds.add_argument("query")
    ds.add_argument("-k", type=int, default=5)
    ds.add_argument("--config", choices=["english", "simple"], default="english")
    ds.add_argument("--mode", choices=["any", "all"], default="any")
    ds.add_argument("--json", action="store_true")
    ds.set_defaults(fn=cmd_docs_search)
    de = dc_sub.add_parser("eval", help="retrieval metrics on the dev questions")
    de.add_argument("--config", choices=["english", "simple"], default="english")
    de.add_argument("--mode", choices=["any", "all"], default="any")
    de.set_defaults(fn=cmd_docs_eval)
    dp = dc_sub.add_parser(
        "probe",
        help="compare keyword, vector, hybrid and reranked search on the dev questions "
        "(calls the embedder you name; hosted ones only see public documents)",
    )
    dp.add_argument("--embedder", required=True, help="a name from config/retrieval.yaml")
    dp.add_argument("--reranker", default=None, help="optional name from config/retrieval.yaml")
    dp.add_argument("--max-requests", type=int, default=None, help="per transport; at most 150")
    dp.add_argument(
        "--reader-role", default=None, help="read-only role to grant access to (default: as ingest)"
    )
    dp.add_argument(
        "--fusion-study",
        action="store_true",
        help="also score a grid of fusion settings on the dev questions (no extra requests)",
    )
    dp.add_argument("--config", default=None, help="another retrieval config file")
    dp.add_argument("--out-dir", default=None, help="default data/live_runs/<time>_docs_<names>")
    dp.set_defaults(fn=cmd_docs_probe)
    dh = dc_sub.add_parser(
        "heldout",
        help="score the three pre-registered fusion candidates on the locked held-out questions, "
        "once per embedder (needs docs/adr/0009 committed and unchanged)",
    )
    dh.add_argument("--candidates", required=True, help="must be exactly A,B,baseline")
    dh.add_argument("--embedder", required=True, help="a name from config/retrieval.yaml")
    dh.add_argument("--max-requests", type=int, default=None, help="at most 150")
    dh.set_defaults(fn=cmd_docs_heldout)
    bsd = dc_sub.add_parser(
        "binder-study",
        help="measure the passage label binder variants on the dev half of the synthetic "
        "documents (no model, no network)",
    )
    bsd.set_defaults(fn=cmd_docs_binder_study)
    bl = dc_sub.add_parser(
        "binder-locked",
        help="score the pre-registered binder variants on the locked half, once (needs "
        "docs/adr/0011 committed and unchanged)",
    )
    bl.add_argument("--candidates", required=True, help="must be exactly current,C1,C2,S1,S2,SL")
    bl.set_defaults(fn=cmd_docs_binder_locked)
    bm_ = dc_sub.add_parser(
        "binder-model-labels",
        help="ONE live run: a model writes labels for number-carrying sentences (ADR-0013; "
        "public documents only, recorded in cassettes)",
    )
    bm_.add_argument("--provider", required=True)
    bm_.add_argument("--model", required=True)
    bm_.add_argument("--out-dir")
    bm_.add_argument("--seed-run", help="report.json of a probe, for its real binder refusals")
    bm_.add_argument("--max-tokens", type=int, default=None)
    bm_.add_argument(
        "--force",
        action="store_true",
        help="start although the daily token tally plus this run is above the cap",
    )
    bm_.set_defaults(fn=cmd_docs_binder_model, what="labels")
    bmd = dc_sub.add_parser(
        "binder-model-dev", help="score current, S2 and SL on the dev half of the labels file"
    )
    bmd.set_defaults(fn=cmd_docs_binder_model, what="dev")
    bml = dc_sub.add_parser(
        "binder-model-locked",
        help="score the locked half once (needs docs/adr/0013 and the labels file committed)",
    )
    bml.add_argument("--candidates", required=True, help="must be exactly current,S2,SL")
    bml.set_defaults(fn=cmd_docs_binder_model, what="locked")
    sv = sub.add_parser(
        "serve", help="web view of the recorded question runs (no key, no database)"
    )
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument(
        "--replay",
        nargs="?",
        const="demo/recordings.jsonl",
        metavar="FILE",
        help="answer new questions from recorded replies (needs the database)",
    )
    sv.add_argument(
        "--provider", help="answer new questions live (needs the key in the environment)"
    )
    sv.add_argument("--model")
    sv.add_argument("--trace-dir", default="data/traces")
    sv.add_argument(
        "--workspace",
        action="store_true",
        help="local only: upload your own data, fill in the boxes, load it and ask about it",
    )
    sv.add_argument(
        "--byok",
        action="store_true",
        help="public tryout: visitors bring their own API key (used once, never stored)",
    )
    sv.add_argument("--build-static", metavar="DIR", help="write DIR/index.html and exit")
    sv.set_defaults(fn=cmd_serve)

    ds = sub.add_parser("dataset", help="bring your own data: profile an uploaded table")
    ds_sub = ds.add_subparsers(dest="dataset_cmd", required=True)
    dpf = ds_sub.add_parser("profile", help="profile a CSV, TSV or XLSX (no model, no database)")
    dpf.add_argument("file")
    dpf.add_argument("--sheet", default="0", help="XLSX sheet name or number")
    dpf.add_argument("--out", help="write profile.json and profile.md into this folder")
    dpf.set_defaults(fn=cmd_dataset_profile)
    dtp = ds_sub.add_parser("template", help="write a manifest to fill in, from the profile")
    dtp.add_argument("file")
    dtp.add_argument("--name", required=True, help="dataset name: lower-case letters, digits, _")
    dtp.add_argument("--out", required=True, help="manifest.yaml to write")
    dtp.add_argument("--sheet", default="0")
    dtp.set_defaults(fn=cmd_dataset_template)
    dld = ds_sub.add_parser(
        "load", help="load the table, check your known figures, write the layer"
    )
    dld.add_argument("file")
    dld.add_argument("--manifest", required=True)
    dld.add_argument("--sheet", default="0")
    dld.add_argument("--readonly-role", default="va_readonly")
    dld.add_argument("--out-dir", default="data/datasets")
    dld.set_defaults(fn=cmd_dataset_load)

    ak = sub.add_parser("ask", help="answer one question (replay of recorded replies, or live)")
    ak.add_argument("question")
    ak.add_argument(
        "--replay",
        nargs="?",
        const="demo/recordings.jsonl",
        metavar="FILE",
        help="use recorded model replies (no key, no network); the database is needed",
    )
    ak.add_argument("--provider", help="live: provider name from config/router.yaml")
    ak.add_argument("--model", help="live: model id")
    ak.add_argument("--trace-dir", default="data/traces", help="a JSON trace is written here")
    ak.set_defaults(fn=cmd_ask)

    ev = sub.add_parser("eval", help="end-to-end scoring against gold values (dev questions)")
    ev_sub = ev.add_subparsers(dest="eval_cmd", required=True)
    e2 = ev_sub.add_parser(
        "e2e", help="run the 15 dev questions through the pipeline and score them"
    )
    e2.add_argument(
        "--raw", default="data/Online Retail.xlsx", help="the raw UCI file (gold source)"
    )
    e2.add_argument(
        "--split",
        choices=["dev", "locked"],
        default="dev",
        help="locked = the 15 held-out questions: one run per model, ever (ADR-0014)",
    )
    e2.add_argument(
        "--confirm-locked",
        action="store_true",
        help="required with --split locked; the run is recorded in a one-shot ledger",
    )
    e2.add_argument(
        "--plan-prompt",
        default="plan.v1",
        choices=["plan.v1", "plan.v2"],
        help="plan.v2 adds the filter shape to the prompt (dev comparison only)",
    )
    e2.add_argument("--only", metavar="IDS", help="dev only: comma-separated question ids to run")
    e2.add_argument("--out-dir", default="data/e2e_runs")
    e2.add_argument("--replay", nargs="?", const="demo/recordings.jsonl", metavar="FILE")
    e2.add_argument("--provider")
    e2.add_argument("--model")
    e2.add_argument("--trace-dir", default="data/traces")
    e2.set_defaults(fn=cmd_eval_e2e)

    rp_ = sub.add_parser("report", help="draft a report, then approve it to write the file")
    rp_sub = rp_.add_subparsers(dest="report_cmd", required=True)
    rd = rp_sub.add_parser("draft", help="collect verified answers into a pending report")
    rd.add_argument("--title", required=True)
    rd.add_argument("--runs", default="demo/results.json", help="JSON list of question runs")
    rd.add_argument("--out", default="data/reports/draft.json")
    rd.set_defaults(fn=cmd_report_draft)
    rr = rp_sub.add_parser("run", help="answer a question list, write a pending draft (for cron)")
    rr.add_argument("--title", required=True)
    rr.add_argument("--questions", default="demo/weekly_questions.json")
    rr.add_argument("--out", default="data/reports/draft.json")
    rr.add_argument("--replay", nargs="?", const="demo/recordings.jsonl", metavar="FILE")
    rr.add_argument("--provider")
    rr.add_argument("--model")
    rr.add_argument("--trace-dir", default="data/traces")
    rr.set_defaults(fn=cmd_report_run)
    ra = rp_sub.add_parser("approve", help="approve a draft by its digest; writes the .md file")
    ra.add_argument("--draft", default="data/reports/draft.json")
    ra.add_argument("--who", required=True)
    ra.add_argument("--digest", required=True)
    ra.add_argument("--out-dir", default="data/reports")
    ra.set_defaults(fn=cmd_report_approve)

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
    lp.add_argument(
        "--force",
        action="store_true",
        help="start although the daily token tally plus this run is above the cap",
    )
    lp.set_defaults(fn=cmd_live_probe)
    rt = sub.add_parser("route", help="router and injection probe (needs a key in the environment)")
    rt_sub = rt.add_subparsers(dest="route_cmd", required=True)
    rp = rt_sub.add_parser("probe", help="dev router and injection questions against ONE model")
    rp.add_argument("--provider", required=True)
    rp.add_argument("--model", required=True)
    rp.add_argument("--out-dir", help="default data/live_runs/<time>_<provider>_<model>")
    rp.add_argument(
        "--max-requests", type=_request_cap, default=None, metavar="N",
        help="HTTP request cap for the run, 1 to 60 (default: the config value, 25)",
    )  # fmt: skip
    rp.add_argument("--token-cap", type=int, default=None, help="lowers the config cap only")
    rp.add_argument("--max-tokens", type=int, default=None, help="completion allowance per call")
    rp.add_argument(
        "--route-only", action="store_true",
        help="only the route call for the 12 router questions (12 requests, no injection set)",
    )  # fmt: skip
    rp.add_argument(
        "--injection-only", action="store_true",
        help="only the 5 injection questions (fits under the token cap; not with --route-only)",
    )  # fmt: skip
    rp.add_argument(
        "--force", action="store_true",
        help="start although the daily token tally plus this run's budget is above the daily cap",
    )  # fmt: skip
    rp.add_argument(
        "--reasoning-effort", default=None, metavar="LEVEL",
        help="overrides reasoning_effort in config/router.yaml for this run (groq gpt-oss only)",
    )  # fmt: skip
    rp.add_argument(
        "--compare-reasoning", default=None, metavar="A,B",
        help="the 5 injection questions once per setting, e.g. low,medium (two runs, each with "
        "its own request cap)",
    )  # fmt: skip
    rp.add_argument(
        "--record", action="store_true",
        help="record cassettes next to the report (public only; key scan and check as live probe)",
    )  # fmt: skip
    rp.set_defaults(fn=cmd_route_probe)
    return p


def _ask_setup(args: argparse.Namespace):
    """Shared by `va ask` and `va report run`: layer, url, router, retriever, known values."""
    from .llm import route_probe  # noqa: PLC0415
    from .llm.answer import make_retriever  # noqa: PLC0415
    from .llm.providers import build_one  # noqa: PLC0415
    from .llm.replay import ReplayProvider  # noqa: PLC0415
    from .llm.router import Router  # noqa: PLC0415
    from .llm.settings import load_router_settings  # noqa: PLC0415

    layer = load_semantic_layer()
    url = readonly_database_url()
    base = load_router_settings()
    if args.replay:
        name, provider = "groq", ReplayProvider.from_file(Path(args.replay))
        s = base
        s.tasks = {t: [name] for t in ("route", "plan", "compose", "repair")}

        def make_router():
            return Router(s, {name: provider}, sleep=lambda _x: None)
    else:
        if not (args.provider and args.model):
            raise SystemExit("give --replay, or --provider and --model for a live call")
        opts = route_probe.RouteProbeOptions(args.provider, args.model, Path(args.trace_dir))
        s = route_probe.route_settings(base, opts)
        for rel in (layer.source.relation, "docs.public"):
            if s.sensitivity_of((rel,)) != "public":
                raise SystemExit(f"refusing: {rel} is not tagged public in router.yaml")

        def make_router():
            return Router(s, {args.provider: build_one(s, args.provider, env=os.environ)})

    retrieve = route_probe.public_only(make_retriever(url, k=3))
    return layer, url, s, make_router, retrieve, catalogue(layer, url)


def _byok_ask(
    question: str, provider: str, model: str | None, key: str, tier: str = "own", cache_dir=None
) -> dict:
    """One question with the visitor's own key. The key exists only in `env` below, which is
    built for this call and dropped with it: it is not written to disk, a trace or a log, and any
    error text is scrubbed of it before it is returned.

    Menu questions (not "own") also use a cache of model replies: a stored reply is used with no
    model call and no key, and the checks are still re-run against the warehouse now. Replies are
    stored only when the whole run ended verified, and a visitor's own question is never stored."""
    from .llm import route_probe  # noqa: PLC0415
    from .llm.answer import make_retriever  # noqa: PLC0415
    from .llm.cassette import (  # noqa: PLC0415
        Cassette,
        CassetteContext,
        layer_hash,
        schema_hash,
        warehouse_content_hash,
    )
    from .llm.live import redact  # noqa: PLC0415
    from .llm.providers import build_one  # noqa: PLC0415
    from .llm.router import Router  # noqa: PLC0415
    from .llm.settings import load_router_settings  # noqa: PLC0415

    base = load_router_settings()
    env = {base.providers[provider].env_key: key} if key else {}
    layer = load_semantic_layer()
    url = readonly_database_url()
    cassette = before = None
    try:
        opts = route_probe.RouteProbeOptions(provider, model or "", Path(os.devnull))
        s = route_probe.route_settings(base, opts)
        for rel in (layer.source.relation, "docs.public"):
            if s.sensitivity_of((rel,)) != "public":
                return {"outcome": "abstain", "reason": "sensitivity", "rendered": ""}
        if tier != "own" and cache_dir:
            ctx = CassetteContext(layer_hash(), schema_hash(), warehouse_content_hash(url))
            cassette = Cassette(Path(cache_dir), "resume", ctx, env=env)
            before = set(cassette.entries)
        setup = (
            layer, url, s,
            lambda: Router(s, {provider: build_one(s, provider, env=env)}, cassette=cassette),
            route_probe.public_only(make_retriever(url, k=3)), catalogue(layer, url),
        )  # fmt: skip
        res = _ask_one(question, "byok", setup)
        d = res.as_dict()
        if cassette is not None and res.outcome != "verified":
            cassette.discard_since(before)  # keep only replies from verified runs
    except (ConfigError, LLMError) as exc:
        return {"outcome": "abstain", "reason": "provider_error", "rendered": "",
                "error": redact(str(exc), env)}  # fmt: skip
    d["id"] = "TRY"
    acct = d.get("accounting") or {}
    d["stored_replies_only"] = bool(cassette is not None and acct.get("tokens_used") == 0)
    return json.loads(redact(json.dumps(d, default=str), env))


def _readonly_role() -> str | None:
    from psycopg.conninfo import conninfo_to_dict  # noqa: PLC0415

    return conninfo_to_dict(readonly_database_url()).get("user")


def _ws_ask(ws, dataset: str, question: str, provider: str, model: str | None, key: str) -> dict:
    """A question about an uploaded dataset with the visitor's key (same handling as _byok_ask).
    Internal datasets are refused for hosted providers by the router; nothing is stored."""
    from .documents.store import search  # noqa: PLC0415
    from .llm import route_probe  # noqa: PLC0415
    from .llm.answer import make_retriever  # noqa: PLC0415
    from .llm.live import redact  # noqa: PLC0415
    from .llm.providers import build_one  # noqa: PLC0415
    from .llm.router import Router  # noqa: PLC0415
    from .llm.settings import apply_dataset_visibility, load_router_settings  # noqa: PLC0415

    lp = ws.layer_path(dataset)
    base = apply_dataset_visibility(load_router_settings(), lp)
    if provider not in base.providers:
        return {"outcome": "abstain", "reason": "provider_error", "rendered": "",
                "error": "unknown provider"}  # fmt: skip
    local = base.providers[provider].kind == "ollama"
    env = {base.providers[provider].env_key: key} if key and not local else {}
    layer = load_semantic_layer(lp)
    url = readonly_database_url()
    try:
        if local:
            if not model:
                raise ConfigError("name the model you have pulled in Ollama")
            s = base.model_copy(deep=True)
            s.providers[provider].models = {"default": model}
            s.tasks = {t: [provider] for t in ("route", "plan", "compose", "repair")}
        else:
            s = route_probe.route_settings(
                base, route_probe.RouteProbeOptions(provider, model or "", Path(os.devnull))
            )
        if s.sensitivity_of((layer.source.relation,)) != "public":
            return {"outcome": "abstain", "reason": "sensitivity", "rendered": "",
                    "error": "This dataset is marked internal, so it is not sent to a hosted "
                    "model. Mark it public in its manifest, or use a local model."}  # fmt: skip
        setup = (
            layer, url, s,
            lambda: Router(s, {provider: build_one(s, provider, env=env)}),
            make_retriever(
                url, k=3, search_fn=lambda q, k: search(url, q, k, doc_prefix=f"ws-{dataset}-")
            ),
            catalogue(layer, url),
        )  # fmt: skip
        docs = ws.documents(dataset)
        corpus = (
            "Documents uploaded for this dataset: " + "; ".join(d["title"] for d in docs) + "."
            if docs
            else "There are no documents for this dataset."
        )
        d = _ask_one(question, "ws", setup, corpus=corpus).as_dict()
    except (ConfigError, LLMError) as exc:
        return {"outcome": "abstain", "reason": "provider_error", "rendered": "",
                "error": redact(str(exc), env)}  # fmt: skip
    d["id"], d["dataset"] = "WS", dataset
    return json.loads(redact(json.dumps(d, default=str), env))


def _byok_models(provider: str, key: str, model: str | None = None) -> dict:
    from .llm.live import (  # noqa: PLC0415
        has_remaining_tokens,
        list_provider_models,
        measure_headroom,
        redact,
    )
    from .llm.settings import load_router_settings  # noqa: PLC0415

    base = load_router_settings()
    env = {base.providers[provider].env_key: key}
    res = list_provider_models(base, provider, env)
    if res.get("http_status") == 200 and not has_remaining_tokens(res["rate_headers"]):
        # the free list call did not carry the limits: spend a few tokens to read them
        probe = measure_headroom(base, provider, model, env)
        res["rate_headers"].update(probe["rate_headers"])
        res["headroom_probe"] = {k: v for k, v in probe.items() if k != "rate_headers"}
    return json.loads(redact(json.dumps(res, default=str), env))


def _ask_one(
    question: str, qid: str, setup, plan_prompt: str = "plan.v1", corpus: str | None = None
):
    from .llm import route_probe  # noqa: PLC0415
    from .llm.answer import answer_question  # noqa: PLC0415

    layer, url, s, make_router, retrieve, known = setup
    router = make_router()  # a fresh router per question: one question's failures do not carry over
    run = router.new_run(qid, token_cap=s.probe.token_cap)
    return answer_question(
        question, layer=layer, readonly_url=url, router=router, run=run, retrieve=retrieve,
        corpus_description=corpus or route_probe.CORPUS_DESCRIPTION, known_values=known,
        plan_prompt=plan_prompt,
    )  # fmt: skip


def cmd_ask(args: argparse.Namespace) -> int:
    import json as _json  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    res = _ask_one(args.question, "ask", _ask_setup(args))
    rec = {"asked_at": datetime.now(UTC).isoformat(), "mode": "replay" if args.replay else "live",
           **res.as_dict()}  # fmt: skip
    out = Path(args.trace_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(_json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(res.render_md())
    print(f"trace: {path}")
    return 0 if res.outcome != "abstain" else 1


def cmd_eval_e2e(args: argparse.Namespace) -> int:
    """Dev: dev.yaml, gold from t1. Locked: see e2e_locked, gold from t2, one shot."""
    import json as _json  # noqa: PLC0415
    import sys as _sys  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    from . import e2e, e2e_locked  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    locked = args.split == "locked"
    ledger = Path("data") / "e2e_locked_ledger.jsonl"
    who = "replay" if args.replay else f"{args.provider}/{args.model}"
    if locked:
        if not args.confirm_locked:
            raise SystemExit(
                "--split locked needs --confirm-locked: the held-out set is run once per model"
            )
        if args.replay:
            raise SystemExit("the locked split is not run on replay")
        if ledger.exists() and any(
            _json.loads(line).get("model") == who
            for line in ledger.read_text().splitlines()
            if line
        ):
            raise SystemExit(f"refusing: the locked split was already run for {who} (see {ledger})")
    _sys.path.insert(0, str(root / "eval" / "gold"))
    import t1  # noqa: PLC0415
    import t2  # noqa: PLC0415

    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"raw file not found: {raw}")
    qfile = e2e_locked.QUESTIONS if locked else root / "eval" / "questions" / "dev.yaml"
    qs = yaml.safe_load(qfile.read_text(encoding="utf-8"))["questions"]
    if args.only:
        if locked:
            raise SystemExit("--only is for the dev split; the locked split always runs whole")
        want = {i.strip() for i in args.only.split(",")}
        qs = [q for q in qs if q["id"] in want]
        if not qs:
            raise SystemExit(f"no dev question matches --only {args.only}")
    df = (t2 if locked else t1).load_raw(raw)
    setup = _ask_setup(args)
    rows = []
    for q in qs:
        if locked:
            res = t2.GOLD_FUNCTIONS[q["gold_fn"]](df, **q.get("gold_args", {}))
            entry = (
                {"kind": "scalar", "value": str(res["value"])} if res["kind"] == "scalar" else res
            )
        else:
            entry = {"kind": "scalar", "value": str(t1.GOLD_FUNCTIONS[q["gold_fn"]](df))}
        d = _ask_one(q["question"], q["id"], setup, args.plan_prompt).as_dict()
        d["id"] = q["id"]
        if d["reason"] == "provider_error" and args.replay:
            d["outcome"], d["reason"] = "not_recorded", "no_recording"
        sc = e2e.score_entry(d, entry, q["decimals"])
        rows.append({"id": q["id"], "question": q["question"], "score": sc,
                     "gold": entry if not locked else "(withheld from the report until review)",
                     "expressible": q["expressible_as_metric_query"], "run": d})  # fmt: skip
    summary = e2e.summarise(rows)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    out = Path(args.out_dir) / f"{stamp}_{args.split}_{'replay' if args.replay else args.provider}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(_json.dumps({"split": args.split, "summary": summary,
        "rows": rows}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")  # fmt: skip
    md = e2e.render_md(summary, rows, f"End-to-end {args.split} run")
    (out / "report.md").write_text(md, encoding="utf-8")
    if locked:  # written only after the whole run completed
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({"model": who, "at": stamp, "summary": summary}) + "\n")
    print(md)
    print(f"files written to: {out}")
    return 1 if summary["verified_wrong"] else 0


def cmd_report_run(args: argparse.Namespace) -> int:
    """Answer a fixed list of questions and write a PENDING draft. A person approves it later."""
    import json as _json  # noqa: PLC0415

    from . import reports  # noqa: PLC0415

    qs = _json.loads(Path(args.questions).read_text(encoding="utf-8"))
    setup = _ask_setup(args)
    runs = []
    for i, q in enumerate(qs, 1):
        d = _ask_one(q, f"report-{i}", setup).as_dict()
        d["id"] = f"Q{i:02d}"
        if d["reason"] == "provider_error" and args.replay:
            d["outcome"], d["reason"] = "not_recorded", "no_recording"
        runs.append(d)
    rep = reports.draft(args.title, runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rep['sections'])} verified, {len(rep['withheld'])} withheld. Draft: {out}")
    print(f"pending approval. digest {rep['digest']}")
    return 0


def cmd_report_draft(args: argparse.Namespace) -> int:
    import json as _json

    from . import reports

    runs = _json.loads(Path(args.runs).read_text(encoding="utf-8"))
    rep = reports.draft(args.title, runs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(reports.to_markdown(rep))
    print(f"pending approval. digest {rep['digest']}; approve with --digest {rep['digest']}")
    return 0


def cmd_report_approve(args: argparse.Namespace) -> int:
    import json as _json

    from . import reports

    rep = _json.loads(Path(args.draft).read_text(encoding="utf-8"))
    try:
        done = reports.approve(rep, args.who, args.digest)
    except reports.ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {reports.publish(done, Path(args.out_dir))} (nothing was sent anywhere)")
    return 0


def cmd_dataset_profile(args: argparse.Namespace) -> int:
    from .datasets import profile as dp  # noqa: PLC0415

    sheet: str | int = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    try:
        prof = dp.profile_file(Path(args.file), sheet)
    except dp.ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    md = dp.render_markdown(prof)
    print(md)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "profile.json").write_text(
            json.dumps(prof.as_dict(), indent=1, default=str), "utf-8"
        )
        (out / "profile.md").write_text(md, encoding="utf-8")
        print(f"wrote {out}")
    return 0


def cmd_dataset_template(args: argparse.Namespace) -> int:
    import yaml  # noqa: PLC0415

    from .datasets import manifest as dm  # noqa: PLC0415
    from .datasets import profile as dp  # noqa: PLC0415

    sheet: str | int = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    try:
        prof = dp.profile_file(Path(args.file), sheet)
        data = dm.skeleton(prof, args.name)
        dm.Manifest.model_construct(**data)  # shape only; the TODO text is refused at load time
    except (dp.ProfileError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {out}. Replace the TODOs, check every role, add metrics, then va dataset load.")
    return 0


def cmd_dataset_load(args: argparse.Namespace) -> int:

    from .datasets import load as dl  # noqa: PLC0415
    from .datasets import manifest as dm  # noqa: PLC0415
    from .datasets import profile as dp  # noqa: PLC0415

    sheet: str | int = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    try:
        man = dm.load_manifest(Path(args.manifest))
        df = dp.read_table(Path(args.file), sheet)
        prof = dp.profile_frame(df, Path(args.file).name)
        layer = dm.build_layer(man, prof)  # refuses an unsafe or inconsistent manifest
        info = dl.load_dataset(df, prof, man, admin_database_url(), args.readonly_role)
        rec = dl.reconcile(man, prof, admin_database_url())
        dl.check_reconciliation(rec)
    except (dm.ManifestError, dp.ProfileError, dl.LoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from .datasets.workspace import write_outputs  # noqa: PLC0415

    out = Path(args.out_dir) / man.name
    write_outputs(out, man, prof, layer, info)
    unchecked = [k for k, s in man.metrics.items() if s.status == "approved" and not s.known]
    print(f"loaded {info['rows']} rows into {info['relation']}; not loaded: {info['not_loaded']}")
    print(f"approved metrics: {sorted(layer.metrics)}")
    print(f"known-figure checks passed: {sum(1 for r in rec if r['ok'])} of {len(rec)}")
    if unchecked:
        print(f"WARNING approved without a known figure to check against: {unchecked}")
    where = "may go to hosted models" if man.visibility == "public" else "stays on local models"
    print(f"visibility: {man.visibility} ({where})")
    print(f"layer written to {out / 'layer.yaml'} (use it with VA_SEMANTIC_PATH)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web.app import FONTS, load_payload, serve, static_page

    if args.build_static:
        out = Path(args.build_static)
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text(static_page(load_payload()), encoding="utf-8")
        fonts = Path(__file__).parent / "web" / "fonts"
        (out / "fonts").mkdir(exist_ok=True)
        for url, name in FONTS.items():
            (out / "fonts" / Path(url).name).write_bytes((fonts / name).read_bytes())
        for lic in fonts.glob("LICENSE-*.txt"):
            (out / "fonts" / lic.name).write_bytes(lic.read_bytes())
        (out / ".nojekyll").write_text("", encoding="utf-8")
        print(f"wrote {out / 'index.html'} and {out / 'fonts'}")
        return 0
    if getattr(args, "workspace", False):
        from .datasets.workspace import Workspace  # noqa: PLC0415

        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print("error: --workspace accepts uploads, so it only runs on this computer "
                  "(host 127.0.0.1)", file=sys.stderr)  # fmt: skip
            return 2
        ws = Workspace(Path("data/workspace"), admin_database_url(), _readonly_role())
        serve(
            args.port,
            args.host,
            workspace=ws,
            ws_ask_fn=lambda d, q, p, m, k: _ws_ask(ws, d, q, p, m, k),
            models_fn=_byok_models,
            tryout_cfg=None,
        )
        return 0
    if getattr(args, "byok", False):
        from .tryout import load_config as load_tryout  # noqa: PLC0415

        cfg = load_tryout()
        serve(
            args.port,
            args.host,
            byok_fn=lambda q, p, m, k, tier: _byok_ask(q, p, m, k, tier, cfg.get("cache_dir")),
            models_fn=_byok_models,
            tryout_cfg=cfg,
        )
        return 0
    ask_fn, mode = None, "static"
    if args.replay or args.provider:
        import itertools  # noqa: PLC0415
        import json as _json  # noqa: PLC0415
        from datetime import UTC, datetime  # noqa: PLC0415

        setup = _ask_setup(args)
        counter = itertools.count(1)
        trace = Path(args.trace_dir)
        trace.mkdir(parents=True, exist_ok=True)

        def ask_fn(question: str) -> dict:
            n = next(counter)
            d = _ask_one(question, f"web-{n}", setup).as_dict()
            d["id"] = f"ASK-{n:03d}"
            if d["reason"] == "provider_error" and args.replay:
                d["outcome"], d["reason"], d["abstain_detail"] = (
                    "not_recorded",
                    "no_recording",
                    None,
                )
            stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{n:03d}.json"
            (trace / stamp).write_text(
                _json.dumps(d, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
            )
            return d

        mode = "replay" if args.replay else "live"
    serve(args.port, args.host, ask_fn, mode)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, LLMError, DocumentError) as exc:
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
