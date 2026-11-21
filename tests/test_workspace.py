"""The upload flow, end to end against a real database: stage, check, load, reconcile, publish."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from test_dataset_load import scratch  # noqa: F401, I001
from verified_analyst.datasets.workspace import Workspace, WorkspaceError

pytestmark = pytest.mark.db

EX = Path(__file__).resolve().parent.parent / "examples" / "datasets"
CSV = (EX / "saas_subscriptions.csv").read_bytes()
MAN = yaml.safe_load((EX / "saas_subscriptions.manifest.yaml").read_text("utf-8"))


@pytest.fixture
def ws(scratch, tmp_path):  # noqa: F811
    admin, _ro, role = scratch
    return Workspace(tmp_path, admin, role)


def test_stage_returns_a_profile_and_a_skeleton_and_no_values_are_guessed(ws):
    st = ws.stage("My Subs 2024.csv", CSV)
    assert st["profile"]["rows"] == 1200 and st["skeleton"]["name"] == "my_subs_2024"
    assert "TODO" in st["skeleton"]["description"]  # the person must fill it in


def test_bad_uploads_are_refused(ws):
    with pytest.raises(WorkspaceError):
        ws.stage("notes.txt", b"a,b\n1,2\n")
    with pytest.raises(WorkspaceError):
        ws.stage("empty.csv", b"")
    with pytest.raises(WorkspaceError):
        ws._staged_dir("../../etc")


def test_check_then_load_publishes_and_cleans_up(ws):
    tok = ws.stage("saas.csv", CSV)["token"]
    chk = ws.check(tok, MAN)
    assert chk["ok"] and "mrr" in chk["approved"] and "draft_net_new_mrr" in chk["not_approved"]
    assert chk["without_known_figure"] == ["churn_rate"]
    res = ws.load(tok, MAN)
    assert res["rows"] == 1200 and all(r["ok"] is not False for r in res["reconciliation"])
    assert [d["name"] for d in ws.datasets()] == ["saas_subs"]
    assert ws.datasets()[0]["visibility"] == "internal"
    assert ws.layer_path("saas_subs").exists()
    with pytest.raises(WorkspaceError):  # the staged file is gone once loaded
        ws.check(tok, MAN)


def test_a_definition_that_misses_the_known_figure_is_not_published(ws):
    tok = ws.stage("saas.csv", CSV)["token"]
    bad = copy.deepcopy(MAN)
    bad["metrics"]["mrr"]["known"][0]["value"] = "999999"
    with pytest.raises(WorkspaceError, match="do not reproduce"):
        ws.load(tok, bad)
    assert ws.datasets() == []


def test_replacing_an_existing_dataset_needs_a_clear_yes(ws):
    t1 = ws.stage("saas.csv", CSV)["token"]
    ws.load(t1, MAN)
    t2 = ws.stage("saas.csv", CSV)["token"]
    with pytest.raises(WorkspaceError, match="already exists"):
        ws.load(t2, MAN)
    assert ws.load(t2, MAN, overwrite=True)["rows"] == 1200


def test_manifest_errors_are_readable_not_a_traceback(ws):
    tok = ws.stage("saas.csv", CSV)["token"]
    with pytest.raises(WorkspaceError):
        ws.check(tok, {"name": "x"})
    bad = copy.deepcopy(MAN)
    bad["columns"]["status"]["role"] = "banana"
    with pytest.raises(WorkspaceError):
        ws.check(tok, bad)
