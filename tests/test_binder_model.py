"""Model-written label binder test (ADR-0013): mocked HTTP and invented documents only. Nothing
here contacts a provider, reads the project's locked claims or measures a model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from llm_helpers import KEYS, ok_openai, settings, status
from verified_analyst.documents import binder_model as bm
from verified_analyst.documents import binder_study as bs
from verified_analyst.documents.binding import CURRENT, DEFAULT
from verified_analyst.documents.store import DocumentError

ROOT = Path(__file__).resolve().parents[1]
ADR = ROOT / "docs" / "adr" / "0013-model-written-labels.md"
TEXTS = [
    "The time limit for claiming a late or damaged parcel is 14 days. "
    "The restocking fee on returned bags is 12 percent. Standard delivery takes 5 working days.",
    "Orders above 150 pounds ship free. The warranty on zip fastenings lasts 24 months.",
    "Trade accounts get a discount of 8 percent on orders over 500 pounds.",
]


def corpus(tmp_path: Path, names_per_half: int = 3) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    made = {"dev": 0, "locked": 0}
    i = 0
    while min(made.values()) < names_per_half:
        key = f"doc{i}"
        h = bs.half_of(key)
        if made[h] < names_per_half:
            (d / f"{key}.md").write_text(TEXTS[made[h] % len(TEXTS)], encoding="utf-8")
            made[h] += 1
        i += 1
    return d


def fake_labeler(text_of=lambda sentence: "delivery time"):
    def call(req):
        user = req.messages[-1][1]
        rows = re.findall(r"^(s\d+): (.*)$", user, re.M)
        return json.dumps({"labels": [{"id": i, "label": text_of(t)} for i, t in rows]})

    return call


def test_the_adr_matches_the_code_and_names_only_current_s2_sl():
    assert bm.parse_candidates(ADR.read_text("utf-8")) == bm.PRE_REGISTERED
    assert list(bm.PRE_REGISTERED) == ["current", "S2", "SL"]
    assert bm.config_of("S2") == DEFAULT and bm.config_of("current") == CURRENT


def test_selection_is_deterministic_seeded_with_the_hostile_documents_and_capped(tmp_path):
    d = corpus(tmp_path, 4)
    first = [n.doc_key for n in bm.select_sentences(d, hostile=["doc0"], target=6)]
    assert first == [n.doc_key for n in bm.select_sentences(d, hostile=["doc0"], target=6)]
    assert first[0] == "doc0"
    sel = bm.select_sentences(d, hostile=[], target=100, max_per_doc=2)
    per_doc = {k: sum(n.doc_key == k for n in sel) for k in {n.doc_key for n in sel}}
    assert max(per_doc.values()) <= 2
    assert len({(n.doc_key, n.chunk, n.s_start) for n in sel}) == len(sel)  # one number a sentence
    assert len(bm.select_sentences()) == bm.TARGET_CLAIMS  # the real corpus: 80 sentences
    assert set(bm.hostile_docs()) <= {p.stem for p in bs.CORPUS_DIR.glob("*.md")}


def test_parse_labels_drops_bad_entries_and_never_fabricates():
    batch = bm.select_sentences(bs.CORPUS_DIR, target=3)
    good = {"labels": [{"id": "s1", "label": " standard   delivery time "},
                       {"id": "s2", "label": "the of"}, {"id": "s9", "label": "x y"},
                       {"id": "s3", "label": 5}, "junk"]}  # fmt: skip
    assert bm.parse_labels(json.dumps(good), batch) == {0: "standard delivery time"}
    assert bm.parse_labels("not json", batch) == {} and bm.parse_labels("[]", batch) == {}


def test_generation_asks_in_batches_counts_drops_and_adds_seeds(tmp_path):
    nums = bm.select_sentences(corpus(tmp_path, 6), hostile=[], target=25)
    seen = []

    def call(req):
        seen.append(req)
        return fake_labeler(lambda t: "the of" if "14" in t else "delivery time")(req)

    gen = bm.generate_labels(call, nums, seeds=[(nums[0], "courier label")])
    assert gen.requests == len(seen) == -(-len(nums) // bm.BATCH)
    assert all(r.datasets == ("docs.public",) and r.prompt_id == "label.v0" for r in seen)
    assert gen.dropped == sum("14" in bm.marked(n) for n in nums)
    assert {c["source"] for c in gen.claims} == {"model", "real_refusal"}
    assert all("<<" in r.messages[-1][1] for r in seen)


def test_cases_and_rates_have_n_and_wilson_and_swaps_move_the_number(tmp_path):
    d = corpus(tmp_path, 4)
    nums = bm.select_sentences(d, hostile=[], target=100)
    gen = bm.generate_labels(fake_labeler(lambda t: "time limit for claiming parcel"), nums)
    labels = tmp_path / "labels.yaml"
    bm.write_labels(labels, gen, {"model": "invented"})
    for half in ("dev", "locked"):
        rep = bm.half_report(half, labels, d)
        assert rep["claims"] > 0 and rep["cases"]["correct"] == rep["claims"]
        for r in rep["rows"].values():
            for k in ("false_reject", "false_accept"):
                assert set(r[k]) == {"k", "n", "wilson95"} and 0 <= r[k]["k"] <= r[k]["n"]
        assert "Wilson" in bm.render_half(rep) and "n=" in bm.render_half(rep)
    claims = bm.half_claims(bm.load_claims(labels, d), "dev")
    cases = bm.build_model_cases(claims, bs.load_numbers("dev", d))
    assert cases == bm.build_model_cases(claims, bs.load_numbers("dev", d))
    for c in cases:
        assert (c.number.id != c.source.id) == (c.kind != "correct")
        assert bs.half_of(c.number.doc_key) == "dev"  # a dev report never touches locked text


def test_a_label_that_no_longer_matches_the_corpus_is_an_error(tmp_path):
    d = corpus(tmp_path)
    labels = tmp_path / "labels.yaml"
    labels.write_text(yaml.safe_dump({"claims": [{"id": "gone#0@1", "label": "x"}]}), "utf-8")
    with pytest.raises(DocumentError, match="not a number of the corpus"):
        bm.load_claims(labels, d)


def test_seed_refusals_are_read_from_a_report_and_located_by_sentence(tmp_path):
    nums = bs.load_numbers("dev", corpus(tmp_path))
    n = nums[0]
    rep = {"injection": {"questions": [{"binder_refusals": [
        {"label": "courier label", "sentence": n.sentence.strip()[:30]},
        {"label": "", "sentence": "x"}]}]}, "router_questions": []}  # fmt: skip
    seeds = bm.seed_from_report(rep)
    assert seeds == [{"label": "courier label", "sentence": n.sentence.strip()[:30]}]
    assert bm.locate(seeds[0], nums).sentence == n.sentence
    assert bm.locate({"label": "a", "sentence": "no such sentence"}, nums) is None


def _git(committed=True, dirty=False):
    def git(args):
        if args[0] == "rev-parse":
            return 0, "true"
        if args[0] == "ls-files":
            return (0 if committed else 1), ""
        if args[0] == "status":
            return 0, " M x" if dirty else ""
        return 0, ""

    return git


def _locked_kw(tmp_path, labels, d, **over):
    kw = {"candidates_arg": "current,S2,SL", "labels": labels, "adr_path": ADR, "root": ROOT,
          "git": _git(), "ledger": tmp_path / "l.jsonl", "corpus_dir": d}  # fmt: skip
    kw.update(over)
    return kw


def test_locked_run_needs_the_adr_and_labels_committed_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MIN_CLAIMS", 5)  # the invented corpus is small
    d = corpus(tmp_path, 4)
    gen = bm.generate_labels(fake_labeler(), bm.select_sentences(d, hostile=[], target=100))
    labels = tmp_path / "labels.yaml"
    bm.write_labels(labels, gen, {})
    with pytest.raises(DocumentError, match="pre-registered"):
        bm.run_locked(**_locked_kw(tmp_path, labels, d, candidates_arg="S2"))
    for git, msg in ((_git(committed=False), "not been committed"),
                     (_git(dirty=True), "changed since")):  # fmt: skip
        with pytest.raises(DocumentError, match=msg):
            bm.run_locked(**_locked_kw(tmp_path, labels, d, git=git, root=tmp_path,
                                       adr_path=_copy(ADR, tmp_path)))  # fmt: skip
    assert not (tmp_path / "l.jsonl").exists()
    rep = bm.run_locked(**_locked_kw(tmp_path, labels, d, root=tmp_path,
                                     adr_path=_copy(ADR, tmp_path)))  # fmt: skip
    assert rep["half"] == "locked" and set(rep["rows"]) == {"current", "S2", "SL"}
    assert len(rep["adr_sha256"]) == 64 and len(rep["labels_sha256"]) == 64
    assert json.dumps(rep) and "no default" in rep["rule"]
    with pytest.raises(DocumentError, match="already scored"):
        bm.run_locked(**_locked_kw(tmp_path, labels, d, root=tmp_path,
                                   adr_path=_copy(ADR, tmp_path)))  # fmt: skip


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    dst.write_text(src.read_text("utf-8"), encoding="utf-8")
    return dst


def test_locked_run_refuses_when_the_adr_differs_from_the_code(tmp_path):
    d = corpus(tmp_path)
    labels = tmp_path / "labels.yaml"
    bm.write_labels(labels, bm.generate_labels(fake_labeler(), bm.select_sentences(d, [], 100)), {})
    adr = tmp_path / "adr.md"
    adr.write_text(ADR.read_text("utf-8").replace("frac: 1.0, window: 8", "frac: 1.0, window: 9"),
                   encoding="utf-8")  # fmt: skip
    with pytest.raises(DocumentError, match="do not match"):
        bm.run_locked(**_locked_kw(tmp_path, labels, d, root=tmp_path, adr_path=adr))


def test_the_real_locked_half_is_not_scorable_before_the_labels_are_committed(tmp_path):
    # the repository has no labels file yet: the gate refuses rather than scoring anything
    with pytest.raises(DocumentError, match="missing|not been committed|changed since"):
        bm.run_locked(candidates_arg="current,S2,SL", ledger=tmp_path / "l.jsonl")
    assert not (tmp_path / "l.jsonl").exists()


def _transport(sent, script=()):
    """Mocked HTTP: answers every batch with labels; `script` items override the first calls."""
    script = list(script)

    def transport(url, headers, body, timeout):
        data = json.loads(body)
        sent.append(data)
        if script:
            item = script.pop(0)
            if item is not None:
                return item
        req = type("R", (), {"messages": [(m["role"], m["content"]) for m in data["messages"]]})
        return ok_openai(fake_labeler()(req))

    return transport


def _gen_kw(tmp_path, d, transport):
    from verified_analyst.llm.cassette import CassetteContext

    return {"provider": "groq", "model": "m-groq", "out_dir": tmp_path / "work", "env": KEYS,
            "labels": tmp_path / "labels.yaml", "ledger": tmp_path / "ledger.jsonl",
            "corpus_dir": d, "transport": transport, "sleep": lambda s: None,
            "cassette_ctx": CassetteContext("l", "s", "w")}  # fmt: skip


def _ledger(tmp_path):
    f = tmp_path / "ledger.jsonl"
    return [json.loads(x)["event"] for x in f.read_text().splitlines()] if f.exists() else []


def test_generation_is_one_run_public_only_and_writes_one_ledger_entry(tmp_path):
    d = corpus(tmp_path, 4)
    cfg = settings(datasets={"docs.public": "public"})
    sent = []
    kw = _gen_kw(tmp_path, d, _transport(sent))
    cfg.probe.max_requests = 60
    out = bm.run_generation(cfg, **kw)
    m = out["meta"]
    assert out["complete"] and m["claims"] > 0 and m["http_requests_sent"] == len(sent)
    assert m["batches_total"] == m["batches_answered"] == len(sent) <= 60
    assert m["labels_written"] == m["claims"] and len(out["labels_sha256"]) == 64
    assert (tmp_path / "labels.yaml").exists() and _ledger(tmp_path) == ["generated"]
    n = len(sent)
    with pytest.raises(DocumentError, match="already generated"):
        bm.run_generation(cfg, **kw)
    assert len(sent) == n  # the refused second run sent nothing
    untagged = settings(datasets={"hr.salaries": "internal"})
    with pytest.raises(DocumentError, match="not tagged public"):
        bm.run_generation(untagged, **{**kw, "ledger": tmp_path / "other.jsonl"})


def test_a_rate_limit_keeps_the_batches_done_and_the_next_run_resumes_from_them(tmp_path):
    d = corpus(tmp_path, 8)  # enough sentences for 3 or more batches
    cfg = settings(datasets={"docs.public": "public"})
    cfg.probe.max_requests = 60
    sent1 = []
    hdr = {"Retry-After": "75", "x-ratelimit-limit-tokens": "8000"}
    body = b'{"error":{"message":"Rate limit on tokens per minute (TPM): Limit 8000"}}'
    first = bm.run_generation(
        cfg, **_gen_kw(tmp_path, d, _transport(sent1, [None, status(429, hdr, body)]))
    )
    m = first["meta"]
    assert not first["complete"] and m["batches_total"] >= 3 and m["batches_answered"] == 1
    assert m["labels_written"] == bm.BATCH and m["http_requests_sent"] == 2 == len(sent1)
    assert not (tmp_path / "labels.yaml").exists() and _ledger(tmp_path) == []  # not spent
    assert "INCOMPLETE" in first["markdown"] and "NOT spent" in first["markdown"]
    assert "75" in first["markdown"] and "tokens_per_minute" in first["markdown"]
    sent2 = []
    second = bm.run_generation(cfg, **_gen_kw(tmp_path, d, _transport(sent2)))
    m2 = second["meta"]
    assert second["complete"] and m2["batches_answered_from_cassettes"] == 1
    assert len(sent2) == m2["batches_total"] - 1  # the finished batch was not asked again
    assert m2["labels_written"] > m["labels_written"] and _ledger(tmp_path) == ["generated"]
    assert yaml.safe_load((tmp_path / "labels.yaml").read_text())["meta"]["complete"] is True
    with pytest.raises(DocumentError, match="already generated"):
        bm.run_generation(cfg, **_gen_kw(tmp_path, d, _transport([])))


def test_the_request_cap_also_leaves_the_generation_unspent_and_resumable(tmp_path):
    d = corpus(tmp_path, 8)
    cfg = settings(datasets={"docs.public": "public"})
    cfg.probe.max_requests = 2
    sent = []
    out = bm.run_generation(cfg, **_gen_kw(tmp_path, d, _transport(sent)))
    assert not out["complete"] and len(sent) == 2 and _ledger(tmp_path) == []
    assert out["meta"]["batches_answered"] == 2 and out["meta"]["stopped"]


def test_locked_refuses_with_the_count_when_fewer_than_40_labels_are_usable(tmp_path):
    d = corpus(tmp_path, 4)
    gen = bm.generate_labels(fake_labeler(), bm.select_sentences(d, hostile=[], target=100))
    labels = tmp_path / "labels.yaml"
    bm.write_labels(labels, gen, {})
    n = bm.usable_labels(bm.load_claims(labels, d))
    assert 0 < n < bm.MIN_CLAIMS == 40
    with pytest.raises(DocumentError, match=rf"only {n} usable labels.*at least 40"):
        bm.run_locked(**_locked_kw(tmp_path, labels, d, root=tmp_path,
                                   adr_path=_copy(ADR, tmp_path)))  # fmt: skip
    assert not (tmp_path / "l.jsonl").exists()  # nothing scored, nothing spent


def test_the_generator_reports_usable_labels_and_the_effort_per_batch(tmp_path):
    d = corpus(tmp_path, 6)
    cfg = settings(datasets={"docs.public": "public"})
    cfg.providers["groq"].reasoning_effort = "low"
    cfg.providers["groq"].models = {"default": "openai/gpt-oss-20b"}
    sent = []
    kw = _gen_kw(tmp_path, d, _transport(sent))
    out = bm.run_generation(cfg, **{**kw, "model": "openai/gpt-oss-20b"})
    m = out["meta"]
    assert {b["reasoning_effort"] for b in m["batch_efforts"]} == {"low"}
    assert {r.get("reasoning_effort") for r in sent} == {"low"}  # what the report says was sent
    assert (
        len(m["batch_efforts"]) == m["batches_total"] and m["reasoning_effort_configured"] == "low"
    )
    assert out["usable_labels"] == m["usable_labels"] == m["claims"]
    assert "reasoning_effort sent: low" in out["markdown"]
    assert "usable labels:" in out["markdown"] and "at least 40" in out["markdown"]
    assert yaml.safe_load((tmp_path / "labels.yaml").read_text())["meta"]["batch_efforts"]
    assert out["meta"]["at"].endswith("+00:00") and out["meta"]["http_requests_sent"] == len(sent)


def test_a_resumed_batch_keeps_its_recorded_effort_and_a_changed_effort_asks_again(tmp_path):
    d = corpus(tmp_path, 8)
    cfg = settings(datasets={"docs.public": "public"})
    cfg.providers["groq"].reasoning_effort = "low"
    cfg.providers["groq"].models = {"default": "openai/gpt-oss-20b"}
    cfg.probe.max_requests = 1
    kw = {**_gen_kw(tmp_path, d, _transport([])), "model": "openai/gpt-oss-20b"}
    first = bm.run_generation(cfg, **kw)
    assert not first["complete"] and first["meta"]["batch_efforts"][0]["reasoning_effort"] == "low"
    cfg.providers[
        "groq"
    ].reasoning_effort = "medium"  # a later default: the old batch is not reused
    cfg.probe.max_requests = 60
    sent = []
    second = bm.run_generation(cfg, **{**kw, "transport": _transport(sent)})
    assert second["complete"] and second["meta"]["batches_answered_from_cassettes"] == 0
    assert {b["reasoning_effort"] for b in second["meta"]["batch_efforts"]} == {"medium"}
    assert len(sent) == second["meta"]["batches_total"]
