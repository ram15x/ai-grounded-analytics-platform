from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_analyst.documents import binder_locked as bl
from verified_analyst.documents import binder_study as bs
from verified_analyst.documents.binding import DEFAULT, BinderConfig, check_binding
from verified_analyst.documents.store import DocumentError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "verified_analyst"
ADR = ROOT / "docs" / "adr" / "0011-binder-variants.md"

TEXTS = [
    "The time limit for claiming a late or damaged parcel is 14 days. "
    "The restocking fee on returned bags is 12 percent. Standard delivery takes 5 working days.",
    "Orders above 150 pounds ship free. The warranty on zip fastenings lasts 24 months.",
]


def _tmp_corpus(tmp_path: Path, half: str) -> Path:
    """Invented documents (not the project corpus), named so that they land in `half`."""
    d = tmp_path / f"corpus_{half}"
    d.mkdir()
    i = 0
    for text in TEXTS * 6:
        while bs.half_of(f"doc{i}") != half:
            i += 1
        (d / f"doc{i}.md").write_text(text, encoding="utf-8")
        i += 1
    return d


def test_default_is_the_old_r1():
    assert BinderConfig(1.0, 6, False, False, False) == DEFAULT
    text = TEXTS[0]
    s = text.index("14")
    # the shipped rule refuses a label whose words sit more than 6 words before the number ...
    far = check_binding(
        "time limit for claiming a parcel", "The time limit is 14 days.", text, s, s + 2
    )
    assert far is not None and far[0] == "label_not_in_passage_window"
    # ... and accepts one that sits next to it
    assert check_binding("damaged parcel", "The damaged parcel is 14 days.", text, s, s + 2) is None


def test_half_of_is_stable_and_splits_by_document(tmp_path):
    assert {bs.half_of(f"d{i}") for i in range(40)} == {"dev", "locked"}
    assert bs.half_of("x") == bs.half_of("x")
    dev = {n.doc_key for n in bs.load_numbers("dev", _tmp_corpus(tmp_path, "dev"))}
    assert dev and all(bs.half_of(k) == "dev" for k in dev)


def test_cases_deterministic_swaps_move_the_number_and_evaluate_counts(tmp_path):
    nums = bs.load_numbers("dev", _tmp_corpus(tmp_path, "dev"))
    a, b = bs.build_cases(nums), bs.build_cases(nums)
    assert [(c.kind, c.style, c.label) for c in a] == [(c.kind, c.style, c.label) for c in b]
    assert {c.kind for c in a} >= {"correct", "other_document"}
    for c in a:
        assert (c.number.id != c.source.id) == (c.kind != "correct")
    row = bs.evaluate(DEFAULT, a)
    assert row["false_reject"]["n"] == sum(c.kind == "correct" for c in a)
    assert row["false_accept"]["n"] == sum(c.kind != "correct" for c in a)
    out = bs.render(bs.dev_table(a), "dev", a)
    assert "false reject" in out and "n=" in out
    names = [v.name for v in bs.variants()]
    assert len(names) == len(set(names))


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


def _kw(tmp_path, **over):
    kw = {
        "candidates_arg": "current,C1,C2,S1,S2,SL",
        "ledger": tmp_path / "l.jsonl",
        "git": _git(),
        "adr_path": ADR,
        "root": ROOT,
    }
    kw.update(over)
    return kw


def test_locked_refuses_unless_adr_committed_unchanged_and_matching(tmp_path):
    with pytest.raises(DocumentError, match="not been committed"):
        bl.run_locked(**_kw(tmp_path, git=_git(committed=False)))
    with pytest.raises(DocumentError, match="changed since"):
        bl.run_locked(**_kw(tmp_path, git=_git(dirty=True)))
    adr = tmp_path / "adr.md"
    adr.write_text(ADR.read_text("utf-8").replace("0.75", "0.7"), encoding="utf-8")
    with pytest.raises(DocumentError, match="do not match"):
        bl.run_locked(**_kw(tmp_path, adr_path=adr, root=tmp_path))
    assert not (tmp_path / "l.jsonl").exists()


def test_locked_accepts_only_all_registered_variants(tmp_path):
    with pytest.raises(DocumentError, match="pre-registered"):
        bl.run_locked(**_kw(tmp_path, candidates_arg="C1"))


def test_locked_runs_once_on_invented_documents_and_decides_by_the_rule(tmp_path):
    corpus = _tmp_corpus(tmp_path, "locked")  # invented text, never the project corpus
    rep = bl.run_locked(**_kw(tmp_path, corpus_dir=corpus))
    assert set(rep["rows"]) == set(bl.PRE_REGISTERED) and rep["half"] == "locked"
    cur = rep["rows"]["current"]
    for n, d in rep["decision"].items():
        r = rep["rows"][n]
        assert d["put_forward"] == (
            r["false_accept"]["k"] <= cur["false_accept"]["k"]
            and r["false_reject"]["k"] < cur["false_reject"]["k"]
        )
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 1
    assert "locked half" in bl.render_report(rep) and json.dumps(rep)
    with pytest.raises(DocumentError, match="already scored"):
        bl.run_locked(**_kw(tmp_path, corpus_dir=corpus))


def test_only_the_locked_module_builds_the_locked_half():
    for p in SRC.rglob("*.py"):
        t = p.read_text("utf-8")
        if p.name == "binder_locked.py":
            continue
        assert 'load_numbers("locked"' not in t, p
        if p.name not in {"cli.py", "binder_study.py"}:
            assert "binder_locked" not in t, p
    assert bl.parse_candidates(ADR.read_text("utf-8")) == bl.PRE_REGISTERED


def test_sentence_scope_accepts_the_real_courier_label_and_refuses_a_swap():
    text = TEXTS[0]
    s = text.index("14")
    label = "time limit for claiming a late or damaged parcel"
    prose = f"The {label} is 14 days."
    sent = bl.config_of("S1")
    assert check_binding(label, prose, text, s, s + 2, DEFAULT) is not None
    assert check_binding(label, prose, text, s, s + 2, sent) is None
    t = text.index("12 percent")  # a number in a different sentence of the same passage
    assert check_binding(label, prose, text, t, t + 2, sent) is not None


def test_nearest_number_rule_only_applies_with_several_numbers():
    text = "Claims for a damaged parcel must arrive within 14 days of delivery, not 30 days."
    s2 = bl.config_of("S2")
    a = text.index("14")
    b = text.index("30")
    lab = "claim window for a damaged parcel"
    prose = "The claim window for a damaged parcel is 14 days."
    assert check_binding(lab, prose, text, a, a + 2, s2) is None
    assert check_binding(lab, prose, text, b, b + 2, s2) is not None
    one = "Claims for a damaged parcel must arrive within 14 days of delivery."
    c = one.index("14")
    assert check_binding(lab, prose, one, c, c + 2, s2) is None
