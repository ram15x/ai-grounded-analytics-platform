import json

import pytest

from grounded_analytics import reports
from grounded_analytics.web.app import load_payload

RUNS = load_payload()["items"]


def test_only_verified_answers_enter_the_body():
    rep = reports.draft("Weekly", RUNS)
    assert len(rep["sections"]) == 7
    assert len(rep["withheld"]) == 10
    assert all(s["numbers_verified"] >= 1 for s in rep["sections"])


def test_approval_needs_the_digest_and_a_name():
    rep = reports.draft("Weekly", RUNS)
    with pytest.raises(reports.ReportError):
        reports.approve(rep, "Ram", "0" * 16)
    with pytest.raises(reports.ReportError):
        reports.approve(rep, " ", rep["digest"])
    assert reports.approve(rep, "Ram", rep["digest"])["status"] == reports.APPROVED


def test_edited_draft_cannot_be_approved():
    rep = reports.draft("Weekly", RUNS)
    seen = rep["digest"]
    rep["sections"][0]["text"] = "The United Kingdom had £9,999,999.00."
    with pytest.raises(reports.ReportError):
        reports.approve(rep, "Ram", seen)


def test_unapproved_report_is_never_written(tmp_path):
    rep = reports.draft("Weekly", RUNS)
    with pytest.raises(reports.ReportError):
        reports.publish(rep, tmp_path)
    assert list(tmp_path.iterdir()) == []
    path = reports.publish(reports.approve(rep, "Ram", rep["digest"]), tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Approved by Ram" in text and "£8,209,930.48" in text and "Withheld" in text


def test_nothing_verified_means_nothing_to_publish():
    rep = reports.draft("Empty", [r for r in RUNS if r["outcome"] != "verified"])
    with pytest.raises(reports.ReportError):
        reports.approve(rep, "Ram", rep["digest"])
    json.dumps(rep)
