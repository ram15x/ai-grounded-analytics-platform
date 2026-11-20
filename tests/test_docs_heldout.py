"""`va docs heldout`: the ADR gate, the three fixed candidates, and the once-per-embedder ledger."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from verified_analyst.cli import build_parser, main
from verified_analyst.documents import heldout_run as hr
from verified_analyst.documents.settings import EmbedderSettings, RetrievalSettings
from verified_analyst.documents.store import DocumentError

ROOT = Path(__file__).parent.parent
ADR_REL = "docs/adr/0009-document-retrieval-fusion.md"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A scratch git checkout holding a committed copy of the real ADR."""
    r = tmp_path / "repo"
    (r / "docs" / "adr").mkdir(parents=True)
    shutil.copy(ROOT / ADR_REL, r / ADR_REL)
    git(r, "init", "-q")
    git(r, "config", "user.name", "t")
    git(r, "config", "user.email", "t@t")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "adr")
    return r


def adr(repo):
    return repo / ADR_REL


def test_the_real_adr_matches_the_candidates_built_into_the_code():
    assert hr.parse_candidates((ROOT / ADR_REL).read_text("utf-8")) == hr.PRE_REGISTERED
    assert set(hr.PRE_REGISTERED) == {"A", "B", "baseline"}
    assert hr.PRE_REGISTERED["B"] == {
        "method": "hybrid", "rrf_k": 10, "depth": 20, "keyword_weight": 0.25,
    }  # fmt: skip


def test_a_committed_unchanged_adr_passes_and_is_identified(repo):
    info = hr.check_adr(adr(repo), repo)
    assert len(info.sha256) == 64 and len(info.commit) == 40 and info.path == ADR_REL


def test_a_missing_adr_is_refused(repo):
    adr(repo).unlink()
    with pytest.raises(DocumentError, match="missing"):
        hr.check_adr(adr(repo), repo)


def test_an_uncommitted_adr_is_refused(repo):
    git(repo, "rm", "-q", "--cached", ADR_REL)
    with pytest.raises(DocumentError, match="not been committed"):
        hr.check_adr(adr(repo), repo)


@pytest.mark.parametrize("how", ["edit", "stage"])
def test_an_adr_changed_since_its_commit_is_refused(repo, how):
    adr(repo).write_text(adr(repo).read_text("utf-8") + "\nA late edit.\n", "utf-8")
    if how == "stage":
        git(repo, "add", "-A")
    with pytest.raises(DocumentError, match="changed since it was committed"):
        hr.check_adr(adr(repo), repo)


def test_a_committed_adr_that_disagrees_with_the_code_is_refused(repo):
    adr(repo).write_text(adr(repo).read_text("utf-8").replace("rrf_k: 10", "rrf_k: 12"), "utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "edit candidates")
    with pytest.raises(DocumentError, match="do not match"):
        hr.check_adr(adr(repo), repo)


def test_no_git_means_no_run(repo):
    with pytest.raises(DocumentError, match="no git checkout"):
        hr.check_adr(adr(repo), repo, git=lambda args: (127, "git not found"))


@pytest.mark.parametrize("arg", ["A,B", "A,B,baseline,C", "A,B,default", "B", ""])
def test_only_the_three_candidates_together_are_accepted(arg):
    with pytest.raises(DocumentError, match="only the three"):
        hr.check_candidates_arg(arg)
    hr.check_candidates_arg("baseline, A ,B")  # order and spaces do not matter


def test_the_command_has_no_way_to_pass_other_settings():
    p = build_parser()
    sub = next(
        a for a in p._subparsers._group_actions[0].choices["docs"]._subparsers._group_actions
    )
    heldout = sub.choices["heldout"]
    opts = {o for a in heldout._actions for o in a.option_strings}
    assert opts == {"-h", "--help", "--candidates", "--embedder", "--max-requests"}
    for bad in ("--rrf-k", "--depth", "--keyword-weight", "--questions", "--method"):
        with pytest.raises(SystemExit):
            p.parse_args(
                ["docs", "heldout", "--candidates", "A,B,baseline", "--embedder", "x", bad, "1"]
            )


def test_cli_refuses_other_candidates_before_touching_anything(monkeypatch, capsys):
    monkeypatch.setenv("VA_DATABASE_URL", "postgresql://x/y")
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "postgresql://x/y")
    assert main(["docs", "heldout", "--candidates", "A,B", "--embedder", "fake_256"]) == 2
    assert "only the three" in capsys.readouterr().err


FAKE = RetrievalSettings(embedders={"fake": EmbedderSettings(kind="fake", hosted=False, dims=256)})


@pytest.mark.db
def test_scores_only_the_three_candidates_once_per_embedder(vector_db, repo, tmp_path):
    ledger = tmp_path / "ledger.jsonl"

    def run(**kw):
        return hr.run_heldout(
            settings=FAKE, embedder_name="fake", candidates_arg="B,A,baseline",
            admin_url=vector_db.admin_url, readonly_url=vector_db.readonly_url, env={},
            adr_path=adr(repo), root=repo, ledger_path=ledger, **kw,
        )  # fmt: skip

    import verified_analyst.documents.heldout_run as mod

    # the default role may not exist on this server; grant to the fixture's role instead
    orig = mod._reader_role
    mod._reader_role = lambda: vector_db.role
    try:
        res = run()
        assert res.exit_code == 0 and res.report["stop"] is None
        rep = res.report
        assert (
            set(rep["results"]) == {"A", "B", "baseline"} and rep["candidates"] == hr.PRE_REGISTERED
        )
        assert rep["scope"].startswith("held-out") and rep["unreachable_anchors"] == []
        assert rep["results"]["A"]["questions_scored"] == 11  # 12 questions, one has no answer
        rows = rep["comparison"]["rows"]
        assert len(rows) == 11 and set(rows[0]) >= {"id", "kind", "A", "B", "baseline", "differs"}
        assert set(rep["comparison"]["pairs"]) == {"A vs baseline", "B vs baseline", "A vs B"}
        entries = [json.loads(x) for x in ledger.read_text().splitlines()]
        assert len(entries) == 1 and entries[0]["embedder"] == "fake"
        assert entries[0]["adr_commit"] == rep["adr"]["commit"]
        md = hr.render_markdown(rep)
        assert "| A | " in md and "not an independent benchmark" in md
        with pytest.raises(DocumentError, match="already scored"):
            run()
        assert len(ledger.read_text().splitlines()) == 1
    finally:
        mod._reader_role = orig


@pytest.mark.db
def test_a_changed_adr_stops_the_run_before_any_scoring(vector_db, repo, tmp_path):
    adr(repo).write_text(adr(repo).read_text("utf-8") + "x", "utf-8")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(DocumentError, match="changed since"):
        hr.run_heldout(
            settings=FAKE, embedder_name="fake", candidates_arg="A,B,baseline",
            admin_url=vector_db.admin_url, readonly_url=vector_db.readonly_url, env={},
            adr_path=adr(repo), root=repo, ledger_path=ledger,
        )  # fmt: skip
    assert not ledger.exists()
