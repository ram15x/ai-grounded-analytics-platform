"""`va docs probe`: run the retrieval methods on the DEV questions with one embedder (and one
optional reranker) and write what happened.

It measures mechanics: how each method ranks a set of questions that the corpus author labelled.
It says nothing about accuracy on other documents. The held-out questions are not reachable from
here (the question file is a fixed dev path passed in by the caller). Hosted models only ever see
documents marked public; the probe refuses before any request if any document is not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from verified_analyst.config import connect_kwargs
from verified_analyst.documents.embed import Embedder, EmbeddingError
from verified_analyst.documents.embed_http import build_embedder
from verified_analyst.documents.fusion import Search, hybrid, reranked
from verified_analyst.documents.metrics import KS, Question, Retrieved, evaluate
from verified_analyst.documents.rerank import Reranker, build_reranker
from verified_analyst.documents.settings import RetrievalSettings
from verified_analyst.documents.store import DocumentError, Hit, search
from verified_analyst.documents.vectorstore import (
    QueryVectors,
    ensure_vector_schema,
    make_vector_search,
    pending_chunks,
    store_embeddings,
)
from verified_analyst.llm.live import (
    AuthStop,
    CountingTransport,
    ModelGone,
    ProviderUnavailable,
    RequestCapReached,
)
from verified_analyst.llm.transport import Transport, loopback_transport, urllib_transport

HARD_MAX_REQUESTS = 150
NOTE = (
    "Mechanics only. The questions and the documents were written by the same author, so these "
    "numbers show how the methods behave on this synthetic set, not how they would do on other "
    "documents. Nothing here is an accuracy claim."
)


class Paced:
    """Waits so requests are at least `interval` seconds apart. Clock and sleep are injectable."""

    def __init__(self, inner: Transport, interval: float, sleep=time.sleep, clock=time.monotonic):
        self.inner, self.interval, self.sleep, self.clock = inner, interval, sleep, clock
        self._last: float | None = None

    def __call__(self, url, headers, body, timeout):
        if self._last is not None and self.interval > 0:
            wait = self.interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        try:
            return self.inner(url, headers, body, timeout)
        finally:
            self._last = self.clock()


@dataclass
class ProbeResult:
    report: dict[str, Any]
    exit_code: int  # 0 ok, 2 refused or bad setup, 3 auth, 4 provider unavailable


def _cap(n: int) -> int:
    if not 1 <= n <= HARD_MAX_REQUESTS:
        raise DocumentError(f"--max-requests must be between 1 and {HARD_MAX_REQUESTS}")
    return n


def _counting(inner: Transport, cap: int, env: Mapping[str, str]) -> CountingTransport:
    ct = CountingTransport(inner, 1, env=env)
    ct.max_requests = (
        cap  # CountingTransport clamps to the chat probe's ceiling; this run has its own
    )
    return ct


def _summary(ct: CountingTransport | None) -> dict[str, Any]:
    if ct is None:
        return {"requests": 0}
    lat = [e.latency_ms for e in ct.events]
    codes: dict[str, int] = {}
    for e in ct.events:
        key = str(e.status) if e.status is not None else (e.error or "error")
        codes[key] = codes.get(key, 0) + 1
    return {
        "requests": ct.count,
        "statuses": codes,
        "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
        "latency_ms_max": round(max(lat), 1) if lat else None,
    }


def _gate_sensitivity(conn: psycopg.Connection, hosted: bool) -> int:
    rows = conn.execute("SELECT sensitivity, count(*) FROM docs.documents GROUP BY 1").fetchall()
    n = sum(c for _, c in rows)
    if n == 0:
        raise DocumentError("no documents are ingested; run `va docs ingest` first")
    if hosted and any(s != "public" for s, _ in rows):
        raise DocumentError(
            "a hosted model may only see public documents, and some ingested documents are not "
            "marked public; use a local model or ingest with --sensitivity public"
        )
    return n


def run_probe(
    *,
    settings: RetrievalSettings,
    embedder_name: str,
    reranker_name: str | None,
    admin_url: str,
    readonly_url: str,
    questions: Sequence[Question],
    env: Mapping[str, str],
    max_requests: int | None = None,
    reader_role: str | None = None,
    embed_transport: Transport | None = None,
    rerank_transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> ProbeResult:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if embedder_name not in settings.embedders:
        raise DocumentError(
            f"unknown embedder {embedder_name!r}; known: {sorted(settings.embedders)}"
        )
    if reranker_name and reranker_name not in settings.rerankers:
        raise DocumentError(
            f"unknown reranker {reranker_name!r}; known: {sorted(settings.rerankers)}"
        )
    cap = _cap(max_requests if max_requests is not None else settings.probe.max_requests)
    es = settings.embedders[embedder_name]
    rs = settings.rerankers[reranker_name] if reranker_name else None

    emb_ct: CountingTransport | None = None
    rr_ct: CountingTransport | None = None
    embedder: Embedder
    if es.kind == "fake":
        embedder = build_embedder(embedder_name, es, env)
    else:
        inner = embed_transport or (loopback_transport if es.kind == "ollama" else urllib_transport)
        emb_ct = _counting(inner, cap, env)
        embedder = build_embedder(embedder_name, es, env, emb_ct)
    reranker: Reranker | None = None
    if rs is not None:
        if rs.kind == "rerank_api":
            inner = rerank_transport or urllib_transport
            if rs.min_interval_s:
                inner = Paced(inner, rs.min_interval_s, sleep)
            rr_ct = _counting(inner, cap, env)
            reranker = build_reranker(reranker_name, rs, env, rr_ct)
        else:
            reranker = build_reranker(reranker_name, rs, env)

    report: dict[str, Any] = {
        "started": started,
        "scope": "dev questions only",
        "note": NOTE,
        "embedder": {
            "name": embedder_name,
            "model": es.model,
            "dims": es.dims,
            "hosted": es.hosted,
            "model_id": embedder.model_id,
        },
        "reranker": None
        if rs is None
        else {"name": reranker_name, "model": rs.model, "hosted": rs.hosted, "depth": rs.depth},
        "max_requests": cap,
    }
    hosted = es.hosted or bool(rs and rs.hosted)
    stop: dict[str, Any] | None = None
    code = 0
    try:
        with psycopg.connect(admin_url, **connect_kwargs(admin_url)) as conn:
            report["documents"] = _gate_sensitivity(conn, hosted)
            ensure_vector_schema(conn, reader_role)
            conn.commit()
            todo = pending_chunks(conn, embedder.model_id)
            report["chunks_embedded_this_run"] = len(todo)
            step = es.batch_size
            for i in range(0, len(todo), step):
                part = todo[i : i + step]
                store_embeddings(
                    conn,
                    embedder.model_id,
                    part,
                    embedder.embed([c.text for c in part], "document"),
                )
                conn.commit()
            n_chunks = conn.execute("SELECT count(*) FROM docs.chunks").fetchone()[0]
        report["chunks"] = n_chunks

        qv = QueryVectors(embedder)
        qv.prefill(
            [q.question for q in questions], lambda texts: embedder.embed(list(texts), "query")
        )
        from verified_analyst.documents.store import all_chunks  # noqa: PLC0415

        corpus = [Retrieved(k, t) for k, _, t in all_chunks(readonly_url)]

        def keyword(query: str, k: int) -> list[Hit]:
            return search(readonly_url, query, k, "english", "any")

        vector: Search = make_vector_search(readonly_url, embedder.model_id, qv)
        methods: dict[str, Search] = {
            "keyword": keyword,
            "vector": vector,
            "hybrid": hybrid(keyword, vector),
        }
        if reranker is not None:
            rk = reranker
            methods["hybrid+rerank"] = reranked(
                methods["hybrid"],
                lambda q, hits: rk.score(q, [h.text for h in hits]),
                depth=rk.depth,
            )
        report["methods"] = {}
        for name, fn in methods.items():
            t0 = clock()
            out = evaluate(
                lambda q, k, f=fn: [Retrieved(h.doc_key, h.text) for h in f(q, k)],
                questions,
                corpus,
                KS,
            )
            out["seconds"] = round(clock() - t0, 2)
            report["methods"][name] = out
    except AuthStop as exc:
        stop, code = {"kind": "auth", "detail": f"HTTP {exc.status}"}, 3
    except (ProviderUnavailable, ModelGone) as exc:
        stop, code = {"kind": "provider_unavailable", "detail": str(exc)}, 4
    except RequestCapReached as exc:
        stop, code = {"kind": "request_cap", "detail": str(exc)}, 4
    except EmbeddingError as exc:
        stop = {"kind": exc.kind, "detail": str(exc)}
        code = 3 if exc.kind == "auth" else 2 if exc.kind in ("config", "no_key") else 4
    report["stop"] = stop
    report["http"] = {"embedder": _summary(emb_ct), "reranker": _summary(rr_ct)}
    return ProbeResult(report, code)


def render_markdown(rep: dict[str, Any]) -> str:
    e = rep["embedder"]
    lines = [
        f"# Document retrieval probe: {e['name']}",
        "",
        f"Started {rep['started']}. Scope: {rep['scope']}.",
        "",
        f"> {rep['note']}",
        "",
        f"- Embedder: {e['model']} (dims {e['dims']}, hosted {e['hosted']}, id `{e['model_id']}`)",
        f"- Reranker: {rep['reranker']}",
        f"- Documents {rep.get('documents')}, chunks {rep.get('chunks')}, "
        f"embedded this run {rep.get('chunks_embedded_this_run')}",
        f"- HTTP: {json.dumps(rep['http'])}",
    ]
    if rep.get("stop"):
        lines += ["", f"**Stopped early:** {rep['stop']['kind']}: {rep['stop']['detail']}"]
    cols = [f"recall@{k}" for k in KS] + ["mrr", "ndcg@5"]
    if rep.get("methods"):
        lines += [
            "",
            "| method | " + " | ".join(cols) + " | no-answer returned something | seconds |",
        ]
        lines += ["|---|" + "---|" * (len(cols) + 2)]
        for name, m in rep["methods"].items():
            u = m["unanswerable"]
            cells = " | ".join(f"{m['overall'].get(c, 0):.2f}" for c in cols)
            lines.append(
                f"| {name} | {cells} | {u['returned_something']} of {u['n']} | {m['seconds']} |"
            )
    return "\n".join(lines) + "\n"


def write_report(rep: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2), "utf-8")
    (out_dir / "report.md").write_text(render_markdown(rep), "utf-8")
    return out_dir
