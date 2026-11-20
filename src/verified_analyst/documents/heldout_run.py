"""`va docs heldout`: score the three pre-registered fusion candidates on the held-out document
questions, once per embedder.

This is the only module that names the held-out question file. It accepts no settings: the three
candidates are fixed in docs/adr/0009-document-retrieval-fusion.md and in `PRE_REGISTERED` below,
and the run refuses unless the ADR is committed, unchanged since, and agrees with the code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import yaml

from verified_analyst.config import connect_kwargs
from verified_analyst.documents.embed import EmbeddingError
from verified_analyst.documents.fusion import Search
from verified_analyst.documents.fusion_study import BASELINE, CANDIDATE_B, Setting, fused
from verified_analyst.documents.metrics import KS, Question, Retrieved, evaluate, load_questions
from verified_analyst.documents.probe import (
    Models,
    ProbeResult,
    _cap,
    _gate_sensitivity,
    _summary,
    build_models,
    embed_pending,
)
from verified_analyst.documents.settings import RetrievalSettings
from verified_analyst.documents.store import DocumentError, Hit, all_chunks, search
from verified_analyst.documents.vectorstore import (
    QueryVectors,
    ensure_vector_schema,
    make_vector_search,
)
from verified_analyst.llm.live import (
    AuthStop,
    ModelGone,
    ProviderUnavailable,
    RequestCapReached,
)
from verified_analyst.llm.transport import Transport

ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "adr" / "0009-document-retrieval-fusion.md"
HELDOUT_QUESTIONS = ROOT / "eval" / "doc_questions" / "heldout.yaml"
LEDGER_PATH = ROOT / "data" / "heldout_ledger.jsonl"

NAMES = ("A", "B", "baseline")
# What the ADR must say, and the only settings this command will ever run.
PRE_REGISTERED: dict[str, dict[str, Any]] = {
    "A": {"method": "vector"},
    "B": {
        "method": "hybrid",
        "rrf_k": CANDIDATE_B.rrf_k,
        "depth": CANDIDATE_B.depth,
        "keyword_weight": CANDIDATE_B.keyword_weight,
    },
    "baseline": {
        "method": "hybrid",
        "rrf_k": BASELINE.rrf_k,
        "depth": BASELINE.depth,
        "keyword_weight": BASELINE.keyword_weight,
    },
}
NOTE = (
    "The questions and documents have one author, and there are only 12 held-out questions (one "
    "with no answer), so small differences mean nothing. This is not an independent benchmark "
    "and nothing here is an accuracy claim. No decision rule is applied; the owner decides."
)

Git = Callable[[Sequence[str]], tuple[int, str]]


@dataclass(frozen=True)
class AdrInfo:
    path: str
    sha256: str
    commit: str


def _git(root: Path) -> Git:
    def run(args: Sequence[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(  # noqa: S603
                ["git", "-C", str(root), *args],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, str(exc)
        return p.returncode, p.stdout.strip()

    return run


def parse_candidates(text: str) -> dict[str, Any]:
    start, end = "<!-- candidates:begin -->", "<!-- candidates:end -->"
    if start not in text or end not in text:
        raise DocumentError("the ADR has no machine-readable candidates block")
    block = text.split(start, 1)[1].split(end, 1)[0]
    body = block.replace("```yaml", "").replace("```", "")
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise DocumentError("the ADR candidates block is not valid YAML") from exc
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), dict):
        raise DocumentError("the ADR candidates block has no `candidates:` mapping")
    return data["candidates"]


def check_adr(adr_path: Path = ADR_PATH, root: Path = ROOT, git: Git | None = None) -> AdrInfo:
    """Refuse unless the ADR exists, is committed, is unchanged since, and matches the code."""
    git = git or _git(root)
    rel = adr_path.relative_to(root).as_posix() if adr_path.is_absolute() else str(adr_path)
    if not adr_path.is_file():
        raise DocumentError(f"refusing to run: the ADR {rel} is missing")
    code, out = git(["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out != "true":
        raise DocumentError(
            "refusing to run: cannot confirm the ADR is committed (no git checkout)"
        )
    if git(["ls-files", "--error-unmatch", "--", rel])[0] != 0:
        raise DocumentError(f"refusing to run: the ADR {rel} has not been committed")
    code, out = git(["status", "--porcelain", "--", rel])
    if code != 0 or out:
        raise DocumentError(f"refusing to run: the ADR {rel} has changed since it was committed")
    code, commit = git(["log", "-1", "--format=%H", "--", rel])
    if code != 0 or not commit:
        raise DocumentError(f"refusing to run: no commit found for the ADR {rel}")
    text = adr_path.read_text("utf-8")
    if parse_candidates(text) != PRE_REGISTERED:
        raise DocumentError(
            "refusing to run: the candidates in the ADR do not match the ones built into the code"
        )
    return AdrInfo(rel, hashlib.sha256(text.encode("utf-8")).hexdigest(), commit)


def check_candidates_arg(arg: str) -> None:
    given = [c.strip() for c in arg.split(",")]
    if sorted(given) != sorted(NAMES):
        raise DocumentError(
            "only the three pre-registered candidates can be scored, all together: "
            "--candidates A,B,baseline"
        )


def _ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _ledger_add(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _setting(name: str) -> Setting:
    spec = PRE_REGISTERED[name]
    return Setting(spec["rrf_k"], spec["depth"], spec["keyword_weight"])


def _rank(mrr: float) -> int | None:
    return None if mrr == 0 else round(1 / mrr)


def compare(results: dict[str, dict[str, Any]], questions: Sequence[Question]) -> dict[str, Any]:
    """Per-question first-relevant rank and recall@5 for each candidate, and who differs."""
    ids = sorted(results["A"]["per_question"])
    kinds = {q.id: q.kind for q in questions}
    rows = []
    for i in ids:
        row: dict[str, Any] = {"id": i, "kind": kinds[i]}
        for n in NAMES:
            pq = results[n]["per_question"][i]
            row[n] = {"rank": _rank(pq["mrr"]), "recall@5": pq["recall@5"]}
        row["differs"] = len({row[n]["rank"] for n in NAMES}) > 1
        rows.append(row)
    pairs = {}
    for a, b in (("A", "baseline"), ("B", "baseline"), ("A", "B")):
        better, worse = [], []
        for i in ids:
            d = results[a]["per_question"][i]["mrr"] - results[b]["per_question"][i]["mrr"]
            (better if d > 1e-12 else worse if d < -1e-12 else []).append(i)
        pairs[f"{a} vs {b}"] = {"higher_mrr": better, "lower_mrr": worse}
    return {"rows": rows, "pairs": pairs}


def run_heldout(
    *,
    settings: RetrievalSettings,
    embedder_name: str,
    candidates_arg: str,
    admin_url: str,
    readonly_url: str,
    env: Mapping[str, str],
    max_requests: int | None = None,
    adr_path: Path = ADR_PATH,
    root: Path = ROOT,
    git: Git | None = None,
    ledger_path: Path = LEDGER_PATH,
    questions_path: Path = HELDOUT_QUESTIONS,
    embed_transport: Transport | None = None,
    local_library: Callable | None = None,
    notice: Callable[[str], None] | None = None,
) -> ProbeResult:
    check_candidates_arg(candidates_arg)
    adr = check_adr(adr_path, root, git)
    if embedder_name not in settings.embedders:
        raise DocumentError(
            f"unknown embedder {embedder_name!r}; known: {sorted(settings.embedders)}"
        )
    es = settings.embedders[embedder_name]
    cap = _cap(max_requests if max_requests is not None else settings.probe.max_requests)
    models: Models = build_models(
        settings, embedder_name, None, env, cap,
        embed_transport=embed_transport, local_library=local_library, notice=notice,
    )  # fmt: skip
    embedder = models.embedder
    for e in _ledger_entries(ledger_path):
        if e.get("embedder") == embedder_name or e.get("model_id") == embedder.model_id:
            raise DocumentError(
                f"refusing to run: the held-out questions were already scored with {embedder_name} "
                f"({e.get('started')}); each embedder is scored once"
            )
    questions = load_questions(questions_path)
    report: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "held-out document questions, once per embedder",
        "note": NOTE,
        "adr": {"path": adr.path, "sha256": adr.sha256, "commit": adr.commit},
        "candidates": PRE_REGISTERED,
        "embedder": {
            "name": embedder_name,
            "model": es.model,
            "dims": es.dims,
            "hosted": es.hosted,
            "model_id": embedder.model_id,
        },
        "max_requests": cap,
    }
    stop: dict[str, Any] | None = None
    code = 0
    try:
        with psycopg.connect(admin_url, **connect_kwargs(admin_url)) as conn:
            report["documents"] = _gate_sensitivity(conn, es.hosted)
            ensure_vector_schema(conn, _reader_role())
            conn.commit()
            report["chunks_embedded_this_run"] = embed_pending(conn, embedder, es.batch_size)
        qv = QueryVectors(embedder)
        qv.prefill([q.question for q in questions], lambda t: embedder.embed(list(t), "query"))
        corpus = [Retrieved(k, t) for k, _, t in all_chunks(readonly_url)]

        def keyword(query: str, k: int) -> list[Hit]:
            return search(readonly_url, query, k, "english", "any")

        vector: Search = make_vector_search(readonly_url, embedder.model_id, qv)
        fns: dict[str, Search] = {
            "A": vector,
            "B": fused(keyword, vector, _setting("B")),
            "baseline": fused(keyword, vector, _setting("baseline")),
        }
        results = {
            n: evaluate(
                lambda q, k, f=fn: [Retrieved(h.doc_key, h.text) for h in f(q, k)],
                questions,
                corpus,
                KS,
            )
            for n, fn in fns.items()
        }
        report["results"] = {
            n: {
                "overall": r["overall"],
                "unanswerable": r["unanswerable"],
                "questions_scored": r["questions_scored"],
            }
            for n, r in results.items()
        }
        report["unreachable_anchors"] = results["A"]["unreachable_anchors"]
        report["comparison"] = compare(results, questions)
        _ledger_add(
            ledger_path,
            {
                "embedder": embedder_name,
                "model_id": embedder.model_id,
                "started": report["started"],
                "adr_commit": adr.commit,
                "adr_sha256": adr.sha256,
            },
        )
    except AuthStop as exc:
        stop, code = {"kind": "auth", "detail": f"HTTP {exc.status}"}, 3
    except (ProviderUnavailable, ModelGone) as exc:
        stop, code = {"kind": "provider_unavailable", "detail": str(exc)}, 4
    except RequestCapReached as exc:
        stop, code = {"kind": "request_cap", "detail": str(exc)}, 4
    except EmbeddingError as exc:
        stop = {"kind": exc.kind, "detail": str(exc)}
        code = (
            3 if exc.kind == "auth" else 2 if exc.kind in ("config", "no_key", "load_failed") else 4
        )
    report["stop"] = stop
    report["http"] = {"embedder": _summary(models.emb_ct)}
    return ProbeResult(report, code)


def _reader_role() -> str:
    from verified_analyst.warehouse.load import READONLY_ROLE_DEFAULT  # noqa: PLC0415

    return READONLY_ROLE_DEFAULT


def render_markdown(rep: dict[str, Any]) -> str:
    e = rep["embedder"]
    lines = [
        f"# Held-out document questions: {e['name']}",
        "",
        f"Started {rep['started']}. Scope: {rep['scope']}.",
        "",
        f"> {rep['note']}",
        "",
        f"- Embedder: {e['model']} (dims {e['dims']}, hosted {e['hosted']}, id `{e['model_id']}`)",
        f"- ADR {rep['adr']['path']} at commit {rep['adr']['commit'][:12]} "
        f"(sha256 {rep['adr']['sha256'][:16]}...)",
        f"- Candidates: {json.dumps(rep['candidates'])}",
        f"- HTTP: {json.dumps(rep['http'])}",
    ]
    if rep.get("stop"):
        lines += ["", f"**Stopped early:** {rep['stop']['kind']}: {rep['stop']['detail']}"]
    if rep.get("results"):
        cols = [f"recall@{k}" for k in KS] + ["mrr", "ndcg@5"]
        lines += ["", "| candidate | " + " | ".join(cols) + " | no-answer returned something |"]
        lines += ["|---|" + "---|" * (len(cols) + 1)]
        for n in NAMES:
            r = rep["results"][n]
            u = r["unanswerable"]
            cells = " | ".join(f"{r['overall'].get(c, 0):.2f}" for c in cols)
            lines.append(f"| {n} | {cells} | {u['returned_something']} of {u['n']} |")
        cmp_ = rep["comparison"]
        lines += [
            "",
            "Rank of the first relevant passage per question (`-` = not in the top 10).",
            "",
        ]
        lines += ["| question | kind | A | B | baseline | differs |", "|---|---|---|---|---|---|"]
        for row in cmp_["rows"]:
            ranks = [("-" if row[n]["rank"] is None else str(row[n]["rank"])) for n in NAMES]
            lines.append(
                f"| {row['id']} | {row['kind']} | "
                + " | ".join(ranks)
                + f" | {'yes' if row['differs'] else ''} |"
            )
        lines += ["", "Questions where the first relevant passage ranks higher or lower:", ""]
        for pair, d in cmp_["pairs"].items():
            lines.append(
                f"- {pair}: higher {len(d['higher_mrr'])} {d['higher_mrr']}, "
                f"lower {len(d['lower_mrr'])} {d['lower_mrr']}"
            )
    return "\n".join(lines) + "\n"
