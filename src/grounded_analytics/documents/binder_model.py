"""Model-written labels for the passage label binder test (va_next24, ADR-0013).

The earlier binder study generates its labels with no model. This one asks a real model (one Groq
run, recorded in cassettes, public documents only) to write the label for number-carrying
sentences of the synthetic corpus, the way a composer would, and then asks the binder to judge
those labels. A matched swapped case reuses a model-written label on another number of the same
passage or of another document.

Procedure (fixed in ADR-0013 before any label was written):
  1. `select_sentences`: every number-carrying sentence of the hostile documents (the seed), then
     other documents' sentences in a fixed hash order, one sentence per number, at most
     MAX_PER_DOC per document, up to TARGET_CLAIMS. Real refusals from an earlier run can be
     added with their own label as seeds (`seed_from_report`).
  2. `generate_labels`: one model call per BATCH sentences. The labels are written to a file the
     owner commits. The ledger allows one generation.
  3. Documents split dev/locked with `binder_study.half_of`. The dev half may be scored freely;
     the locked half only by `run_locked` (ADR committed and unchanged, labels file committed and
     unchanged, one run).

Nothing here changes a binder default. The corpus is synthetic and written by the project author;
this is a screen, not a benchmark of real answers.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..llm.live import wilson
from ..llm.prompts import LABEL_SYSTEM, prompt_hashes
from ..llm.types import PUBLIC, LLMRequest
from .binder_locked import Git, _git, parse_candidates
from .binder_study import CORPUS_DIR, SEED, Case, Number, _fair, half_of, judge, load_numbers
from .binding import CURRENT, DEFAULT, BinderConfig, content
from .store import DocumentError

ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "adr" / "0013-model-written-labels.md"
LABELS_PATH = ROOT / "eval" / "binder_model_labels" / "labels.yaml"
LEDGER_PATH = ROOT / "data" / "binder_model_ledger.jsonl"
WORK_DIR = ROOT / "data" / "binder_model_labels_work"  # cassettes of the one generation
# Set to the ADR that used the locked half once it has been scored; nothing else may use it.
SPENT_BY: str | None = None
TARGET_CLAIMS = 80
MIN_CLAIMS = 40
BATCH = 10
MAX_PER_DOC = 3
SWAP_KINDS = ("same_passage", "other_document")

PRE_REGISTERED: dict[str, dict[str, Any]] = {
    name: {
        "frac": c.frac, "window": c.window, "ignore_repeated": c.ignore_repeated,
        "head_noun": c.head_noun, "skip_units": c.skip_units, "scope": c.scope,
        "nearest": c.nearest,
    }
    for name, c in {
        "current": CURRENT,
        "S2": DEFAULT,
        "SL": BinderConfig(1.0, 6, False, False, False, "sentence", False),
    }.items()
}  # fmt: skip
RULE = (
    "No candidate is put forward and no default changes. For each of current, S2 and SL the "
    "report gives false rejects (model-written labels on their own number, refused) and false "
    "accepts (the same labels on another number, accepted) with k, n and a 95% Wilson interval, "
    "for the dev half and, once, the locked half. The owner decides after reading both."
)
NOTE = (
    "Synthetic corpus written by the project author; labels written by one model in one run; "
    "n is small and the interval says so. Nothing here is an accuracy claim."
)


# ----------------------------------------------------------------------------- selection


HOSTILE_DOCS = (
    "html-comment-hostile",
    "pricing-comment-hostile",
    "role-play-hostile",
    "spoofed-citation",
    "supplier-email-forwarded",
)  # the five hostile documents of the corpus (the injection seed)


def hostile_docs() -> list[str]:
    return list(HOSTILE_DOCS)


def _first_per_sentence(numbers: list[Number]) -> list[Number]:
    seen: set[tuple[str, int, int, int]] = set()
    out = []
    for n in numbers:
        key = (n.doc_key, n.chunk, n.s_start, n.s_end)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def select_sentences(
    corpus_dir: Path = CORPUS_DIR,
    hostile: Sequence[str] | None = None,
    target: int = TARGET_CLAIMS,
    max_per_doc: int = MAX_PER_DOC,
) -> list[Number]:
    """The hostile documents first (the seed), then other documents in hash order."""
    hostile = list(hostile if hostile is not None else hostile_docs())
    pool = _first_per_sentence(load_numbers("dev", corpus_dir) + load_numbers("locked", corpus_dir))
    by_doc: dict[str, list[Number]] = {}
    for n in pool:
        by_doc.setdefault(n.doc_key, []).append(n)
    chosen: list[Number] = []
    for key in sorted(by_doc):
        if key in hostile:
            chosen += by_doc[key][:max_per_doc]
    rest = sorted(
        (k for k in by_doc if k not in hostile),
        key=lambda k: hashlib.sha256(f"{SEED}:{k}".encode()).hexdigest(),
    )
    for rnd in range(max_per_doc):  # round robin, so the claims spread over many documents
        for key in rest:
            if len(chosen) >= target:
                break
            if rnd < len(by_doc[key]):
                chosen.append(by_doc[key][rnd])
    return chosen[:target] if len(chosen) > target else chosen


def seed_from_report(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Real binder refusals (label, sentence) from a probe's report.json."""
    rows = [  # every per-question list of the report (the router and the injection questions)
        r
        for key, v in [*report.items(), *report.get("injection", {}).items()]
        if key.endswith("questions") and isinstance(v, list)
        for r in v
        if isinstance(r, dict)
    ]
    out = []
    for r in rows:
        for b in r.get("binder_refusals", []):
            if b.get("label") and b.get("sentence"):
                out.append({"label": b["label"], "sentence": b["sentence"]})
    return out


def locate(seed: dict[str, str], numbers: list[Number]) -> Number | None:
    """The number whose sentence starts with the (possibly truncated) seed sentence, if exactly
    one sentence does."""
    want = " ".join(seed["sentence"].split())
    hits = [n for n in numbers if " ".join(n.sentence.split()).startswith(want)]
    ids = {(n.doc_key, n.chunk, n.s_start) for n in hits}
    return hits[0] if len(ids) == 1 else None


# ----------------------------------------------------------------------------- generation


def marked(n: Number) -> str:
    s = n.text[n.s_start : n.start] + "<<" + n.shown + ">>" + n.text[n.end : n.s_end]
    return " ".join(s.split())


def label_request(batch: list[Number], max_tokens: int | None = None) -> LLMRequest:
    lines = "\n".join(f"s{i}: {marked(n)}" for i, n in enumerate(batch, 1))
    return LLMRequest(
        task="compose",
        messages=(("system", LABEL_SYSTEM), ("user", f"Sentences:\n{lines}")),
        datasets=("docs.public",),
        prompt_id="label.v0",
        prompt_hash=prompt_hashes()["label.v0"],
        max_tokens=max_tokens or 2048,
        max_tokens_fixed=bool(max_tokens),
    )


def parse_labels(text: str, batch: list[Number]) -> dict[int, str]:
    """index in the batch -> label; entries that are missing, not text or without a content
    word are left out."""
    try:
        data = json.loads(text)
        rows = data["labels"]
    except (ValueError, KeyError, TypeError):
        return {}
    out: dict[int, str] = {}
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict) or not isinstance(r.get("label"), str):
            continue
        sid = str(r.get("id", ""))
        if not (sid.startswith("s") and sid[1:].isdigit()):
            continue
        i = int(sid[1:])
        label = " ".join(r["label"].split())
        if 1 <= i <= len(batch) and content(label) and i - 1 not in out:
            out[i - 1] = label
    return out


@dataclass
class Generated:
    claims: list[dict[str, Any]]
    sentences_asked: int
    requests: int  # batches answered
    dropped: int
    batches_total: int = 0
    complete: bool = True
    stopped: str = ""  # why it stopped before the last batch (a rate limit, the request cap)
    sentences_answered: int = 0  # sentences in the batches that were answered


def generate_labels(
    call: Callable[[LLMRequest], str],
    sentences: list[Number],
    seeds: Sequence[tuple[Number, str]] = (),
    max_tokens: int | None = None,
    stop_on: tuple[type[BaseException], ...] = (),
) -> Generated:
    """One model call per BATCH sentences; `call` returns the reply text. An exception in
    `stop_on` (the router giving up, the request cap) ends the run at that batch: the batches
    before it are kept, none after it is tried, and nothing is retried by hand. Any other
    exception propagates."""
    claims: list[dict[str, Any]] = [_row(n, label, "real_refusal") for n, label in seeds]
    dropped = answered = done = 0
    total = -(-len(sentences) // BATCH)
    stopped = ""
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i : i + BATCH]
        try:
            text = call(label_request(batch, max_tokens))
        except stop_on as exc:
            stopped = str(exc) or type(exc).__name__
            break
        done += 1
        answered += len(batch)
        got = parse_labels(text, batch)
        dropped += len(batch) - len(got)
        claims += [_row(batch[j], lab, "model") for j, lab in sorted(got.items())]
    return Generated(claims, len(sentences), done, dropped, total, done == total, stopped, answered)


def _row(n: Number, label: str, source: str) -> dict[str, Any]:
    return {
        "id": n.id, "doc": n.doc_key, "shown": n.shown, "label": label, "source": source,
        "sentence": " ".join(n.sentence.split()),
    }  # fmt: skip


def write_labels(path: Path, gen: Generated, meta: dict[str, Any]) -> str:
    """Write the labels file; return its sha256."""
    doc = {"meta": meta, "claims": sorted(gen.claims, key=lambda c: (c["doc"], c["id"]))}
    text = yaml.safe_dump(doc, sort_keys=True, allow_unicode=True, width=100)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")  # the same bytes on every OS
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ledger_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def record(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def check_generation_allowed(ledger: Path = LEDGER_PATH) -> None:
    if any(e["event"] == "generated" for e in ledger_entries(ledger)):
        raise DocumentError("refusing to run: the ledger shows the labels were already generated")


# ----------------------------------------------------------------------------- cases


def load_claims(path: Path = LABELS_PATH, corpus_dir: Path = CORPUS_DIR):
    """(claims with their Number) from the labels file; an id that no longer matches the corpus
    is an error, not a silent drop."""
    doc = yaml.safe_load(path.read_text("utf-8"))
    by_id = {n.id: n for h in ("dev", "locked") for n in load_numbers(h, corpus_dir)}
    out = []
    for c in doc["claims"]:
        n = by_id.get(c["id"])
        if n is None:
            raise DocumentError(f"labels file: {c['id']} is not a number of the corpus any more")
        out.append((c, n))
    return out


def build_model_cases(claims, numbers: list[Number]) -> list[Case]:
    """`numbers` is the pool of the half being scored, for the swapped cases."""
    by_chunk: dict[tuple[str, int], list[Number]] = {}
    for m in numbers:
        by_chunk.setdefault((m.doc_key, m.chunk), []).append(m)
    cases: list[Case] = []
    for c, n in claims:
        label = c["label"]
        cases.append(Case("correct", c["source"], label, f"The {label} is {n.shown}.", n, n))
        rng = random.Random(f"{SEED}:{n.id}:model")
        span = (n.s_start, n.s_end)
        same = [
            m for m in by_chunk.get((n.doc_key, n.chunk), [])
            if (m.s_start, m.s_end) != span and not _fair(label, m)
        ]  # fmt: skip
        other = [m for m in numbers if m.doc_key != n.doc_key and not _fair(label, m)]
        for kind, pool in (("same_passage", same), ("other_document", other)):
            if pool:
                m = rng.choice(sorted(pool, key=lambda x: x.id))
                cases.append(Case(kind, c["source"], label, f"The {label} is {m.shown}.", m, n))
    return cases


def usable_labels(claims) -> int:
    """Claims whose label has at least one content word (the model's own labels and the real
    refusals alike)."""
    return sum(bool(content(c["label"])) for c, _ in claims)


def half_claims(claims, half: str):
    return [(c, n) for c, n in claims if half_of(n.doc_key) == half]


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "wilson95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)]}


def config_of(name: str) -> BinderConfig:
    return BinderConfig(**PRE_REGISTERED[name])


def evaluate_model(name: str, cfg: BinderConfig, cases: list[Case]) -> dict[str, Any]:
    correct = [c for c in cases if c.kind == "correct"]
    swapped = [c for c in cases if c.kind != "correct"]
    rej = [c for c in correct if not judge(c, cfg)]
    acc = [c for c in swapped if judge(c, cfg)]
    return {
        "variant": name,
        "false_reject": _rate(len(rej), len(correct)),
        "false_accept": _rate(len(acc), len(swapped)),
        "false_accept_by_kind": {
            k: _rate(sum(c.kind == k for c in acc), sum(c.kind == k for c in swapped))
            for k in SWAP_KINDS
        },
        "false_rejected_labels": [
            {"label": c.label, "sentence": " ".join(c.number.sentence.split())} for c in rej
        ],
    }  # fmt: skip


def half_report(half: str, labels: Path = LABELS_PATH, corpus_dir: Path = CORPUS_DIR,
                names: Sequence[str] = tuple(PRE_REGISTERED)) -> dict[str, Any]:  # fmt: skip
    claims = half_claims(load_claims(labels, corpus_dir), half)
    pool = load_numbers(half, corpus_dir)
    cases = build_model_cases(claims, pool)
    return {
        "half": half,
        "documents": len({n.doc_key for _, n in claims}),
        "claims": len(claims),
        "claims_real_refusal": sum(c["source"] == "real_refusal" for c, _ in claims),
        "cases": {k: sum(c.kind == k for c in cases) for k in ("correct", *SWAP_KINDS)},
        "rows": {nm: evaluate_model(nm, config_of(nm), cases) for nm in names},
    }


def render_half(rep: dict[str, Any]) -> str:
    lines = [
        f"Model-written labels, {rep['half']} half: {rep['documents']} documents, "
        f"{rep['claims']} claims ({rep['claims_real_refusal']} real refusals), "
        f"cases {rep['cases']}",
        NOTE,
        "",
        f"{'variant':9} {'false reject (correct)':>34} {'false accept (swapped)':>34}",
    ]

    def cell(r: dict[str, Any]) -> str:
        ci = r["wilson95"]
        span = "no cases" if ci is None else f"{ci[0]:.1%} to {ci[1]:.1%}"
        return f"{r['k']} of {r['n']} (n={r['n']}), 95% Wilson {span}"

    for nm, r in rep["rows"].items():
        lines.append(f"{nm:9} {cell(r['false_reject'])}")
        lines.append(f"{'':9} FA: {cell(r['false_accept'])}")
        for k, v in r["false_accept_by_kind"].items():
            lines.append(f"{'':12} {k}: {v['k']} of {v['n']}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- gated runs


def _committed_unchanged(git: Git, root: Path, path: Path, what: str) -> None:
    rel = path.relative_to(root).as_posix() if path.is_absolute() else str(path)
    if not path.is_file():
        raise DocumentError(f"refusing to run: {what} {rel} is missing")
    code, out = git(["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out != "true":
        raise DocumentError(
            f"refusing to run: cannot confirm {what} is committed (no git checkout)"
        )
    if git(["ls-files", "--error-unmatch", "--", rel])[0] != 0:
        raise DocumentError(f"refusing to run: {what} {rel} has not been committed")
    code, out = git(["status", "--porcelain", "--", rel])
    if code != 0 or out:
        raise DocumentError(f"refusing to run: {what} {rel} has changed since it was committed")


def check_gate(adr_path: Path = ADR_PATH, labels: Path = LABELS_PATH, root: Path = ROOT,
               git: Git | None = None) -> dict[str, str]:  # fmt: skip
    git = git or _git(root)
    _committed_unchanged(git, root, adr_path, "the ADR")
    _committed_unchanged(git, root, labels, "the labels file")
    text = adr_path.read_text("utf-8")
    if parse_candidates(text) != PRE_REGISTERED:
        raise DocumentError("refusing to run: the variants in the ADR do not match the code")
    return {
        "adr_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
    }


def run_locked(*, candidates_arg: str, adr_path: Path = ADR_PATH, labels: Path = LABELS_PATH,
               root: Path = ROOT, git: Git | None = None, ledger: Path = LEDGER_PATH,
               corpus_dir: Path = CORPUS_DIR) -> dict[str, Any]:  # fmt: skip
    if sorted(c.strip() for c in candidates_arg.split(",")) != sorted(PRE_REGISTERED):
        raise DocumentError(
            "only the pre-registered variants can be scored, all together: "
            "--candidates " + ",".join(PRE_REGISTERED)
        )
    if SPENT_BY is not None:
        raise DocumentError(f"refusing to run: the locked half is spent ({SPENT_BY})")
    hashes = check_gate(adr_path, labels, root, git)
    if any(e["event"] == "locked_scored" for e in ledger_entries(ledger)):
        raise DocumentError("refusing to run: the ledger shows the locked half was already scored")
    usable = usable_labels(load_claims(labels, corpus_dir))
    if usable < MIN_CLAIMS:
        raise DocumentError(
            f"refusing to run: only {usable} usable labels in the labels file; ADR-0013 needs at "
            f"least {MIN_CLAIMS} before anything is scored"
        )
    rep = half_report("locked", labels, corpus_dir)
    if rep["claims"] < 1:
        raise DocumentError("refusing to run: the locked half has no claims")
    record(ledger, "locked_scored", **hashes)
    return {**rep, **hashes, "rule": RULE, "note": NOTE, "shipped_default": DEFAULT.name}


__all__ = ["PRE_REGISTERED", "PUBLIC", "run_locked", "half_report", "generate_labels"]


# ----------------------------------------------------------------------------- the live run


def plan_generation(settings: Any, provider: str, model: str, out_dir: Path,
                    corpus_dir: Path = CORPUS_DIR) -> tuple[int, int]:  # fmt: skip
    """(tokens planned, batches still to ask). Batches already recorded in the work folder with
    the effort that would be sent now are answered from cassettes and cost nothing. A batch is
    planned at the mean of the recorded ones, else at the router's own estimate (characters / 4
    plus the allowance), which is an upper bound."""
    from ..llm.cassette import read_entries  # noqa: PLC0415
    from ..llm.router import estimate_tokens  # noqa: PLC0415

    sentences = select_sentences(corpus_dir)
    total = -(-len(sentences) // BATCH)
    effort = settings.providers[provider].reasoning_effort if "gpt-oss" in model.lower() else None
    entries = [
        e for e in (read_entries(out_dir / "cassettes") if (out_dir / "cassettes").is_dir()
                    else {}).values()
        if (e["provider"], e["model"]) == (provider, model)
        and e["request"].get("prompt_id") == "label.v0"
        and e["request"].get("reasoning_effort") == effort
    ]  # fmt: skip
    left = max(0, total - len(entries))
    if entries:
        per = sum(e["tokens"]["prompt"] + e["tokens"]["completion"] for e in entries) / len(entries)
    else:
        per = estimate_tokens(label_request(sentences[:BATCH]))
    return round(per * left), left


def run_generation(
    settings: Any,
    *,
    provider: str,
    model: str,
    out_dir: Path,
    env: Mapping[str, str],
    seed_report: Path | None = None,
    labels: Path = LABELS_PATH,
    ledger: Path = LEDGER_PATH,
    corpus_dir: Path = CORPUS_DIR,
    max_tokens: int | None = None,
    cassette_ctx: Any = None,
    transport: Any = None,
    sleep: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """The one generation run. Public documents only; the router refuses anything else."""
    from ..llm import (
        live,  # noqa: PLC0415
        route_probe,  # noqa: PLC0415
        usage,  # noqa: PLC0415
    )
    from ..llm.cassette import Cassette  # noqa: PLC0415
    from ..llm.providers import build_one  # noqa: PLC0415
    from ..llm.router import Router  # noqa: PLC0415
    from ..llm.transport import urllib_transport  # noqa: PLC0415
    from ..llm.types import AllProvidersFailed  # noqa: PLC0415

    check_generation_allowed(ledger)
    if settings.sensitivity_of(("docs.public",)) != PUBLIC:
        raise DocumentError("refusing to run: docs.public is not tagged public in router.yaml")
    s = live.probe_settings(settings, live.ProbeOptions(provider, model, out_dir, record=True,
                                                         max_tokens=max_tokens))  # fmt: skip
    s.tasks = {"compose": [provider]}
    cfg = s.providers[provider]
    if not env.get(cfg.env_key or ""):
        raise DocumentError(f"environment variable {cfg.env_key} is not set")
    sentences = select_sentences(corpus_dir)
    seeds: list[tuple[Number, str]] = []
    unlocated = 0
    if seed_report is not None:
        pool = load_numbers("dev", corpus_dir) + load_numbers("locked", corpus_dir)
        for sd in seed_from_report(json.loads(seed_report.read_text("utf-8"))):
            n = locate(sd, pool)
            if n is None:
                unlocated += 1
            else:
                seeds.append((n, sd["label"]))
    counting = live.CountingTransport(transport or urllib_transport, s.probe.max_requests, env=env)
    # "resume": a batch that is already in the cassettes of an earlier, unfinished run is answered
    # from them without a request; only the missing batches go to the provider
    cassette = Cassette(out_dir / "cassettes", "resume", cassette_ctx, env=env) \
        if cassette_ctx is not None else None  # fmt: skip
    kw: dict[str, Any] = {"sleep": sleep or live._sleep}
    router = Router(s, {provider: build_one(s, provider, transport=counting, env=env)},
                    cassette=cassette, **kw)  # fmt: skip
    run = router.new_run("binder-model-labels", token_cap=s.probe.token_cap)

    batch_efforts: list[dict[str, Any]] = []

    def call(req: LLMRequest) -> str:
        before = sum(r.outcome == "cassette" for r in run.records)
        effort = router.prepared(provider, req).reasoning_effort  # what is (or was) sent
        text = router.call(req, run).text
        batch_efforts.append({
            "batch": len(batch_efforts) + 1, "reasoning_effort": effort or "none",
            "from_cassette": sum(r.outcome == "cassette" for r in run.records) > before,
        })  # fmt: skip
        return text

    gen = generate_labels(call, sentences, seeds, max_tokens, (AllProvidersFailed,
                                                               live.RequestCapReached))  # fmt: skip
    from_cassette = sum(r.outcome == "cassette" for r in run.records)
    tg = route_probe.token_guard([run], counting)
    usable = sum(bool(content(c["label"])) for c in gen.claims)
    meta = {
        "provider": provider, "model": model, "prompt": "label.v0",
        "prompt_hash": prompt_hashes()["label.v0"], "sentences_asked": gen.sentences_asked,
        "batches_total": gen.batches_total, "batches_answered": gen.requests,
        "batches_answered_from_cassettes": from_cassette,
        "http_requests_sent": counting.count, "complete": gen.complete,
        "sentences_answered": gen.sentences_answered,
        "labels_written": sum(c["source"] == "model" for c in gen.claims),
        "labels_dropped_no_usable_label": gen.dropped, "seeds_from_real_refusals": len(seeds),
        "seed_refusals_not_located": unlocated, "claims": len(gen.claims),
        "documents": len({c["doc"] for c in gen.claims}), "stopped": gen.stopped,
        "usable_labels": usable, "at": datetime.now(UTC).isoformat(timespec="seconds"),
        # the effort each batch was ASKED with (a cassette hit keeps the effort it was recorded
        # with: the effort is part of the cassette key). Only batches of THIS run are listed.
        "batch_efforts": batch_efforts,
        "reasoning_effort_configured": cfg.reasoning_effort,
    }  # fmt: skip
    if gen.complete and len(gen.claims) < MIN_CLAIMS:
        meta["warning"] = f"only {len(gen.claims)} claims; the ADR's minimum is {MIN_CLAIMS}"
    sha = ""
    if gen.complete:  # only now: the labels file and the one ledger entry
        sha = write_labels(labels, gen, meta)
        record(ledger, "generated", labels_sha256=sha, **meta)
    efforts = sorted({b["reasoning_effort"] for b in batch_efforts})
    sent = (
        None if not efforts else efforts[0] if len(efforts) == 1 else "mixed: " + ", ".join(efforts)
    )
    if sent == "none":
        sent = None
    rep = {"meta": meta, "labels_sha256": sha, "token_guard": tg,
           "tokens_spent": usage.tokens_spent([run])}  # fmt: skip
    md = "\n".join([
        f"# binder-model-labels: {'COMPLETE' if gen.complete else 'INCOMPLETE'}",
        f"- labels written: {meta['labels_written']} of {gen.sentences_asked} sentences "
        f"({gen.requests} of {gen.batches_total} batches answered, {from_cassette} of them "
        f"from cassettes without a request; {counting.count} HTTP requests sent this run)",
        *([f"- stopped: {gen.stopped}", "- the one allowed generation is NOT spent: no ledger "
           "entry, no labels file; run the same command again to resume from the cassettes"]
          if not gen.complete else [f"- labels file written; sha256 {sha}"]),
        f"- usable labels: {usable} of {gen.sentences_asked} sentences asked "
        f"(at least {MIN_CLAIMS} are needed before the locked half can be scored)",
        f"- reasoning_effort per batch this run: {json.dumps(batch_efforts)}",
        *route_probe.render_token_guard(tg, sent),
    ]) + "\n"  # fmt: skip
    live.write_outputs(out_dir, rep, md, cassette, cassette_ctx, env)
    return {"meta": meta, "labels_sha256": sha, "out_dir": str(out_dir), "labels": str(labels),
            "complete": gen.complete, "markdown": md, "usable_labels": usable}  # fmt: skip
