"""PDF text layers become passages; scans are refused plainly; documents stay with their dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pdf_helpers import make_pdf
from test_dataset_load import scratch  # noqa: F401, I001
from verified_analyst.datasets.workspace import Workspace, WorkspaceError
from verified_analyst.documents.store import DocumentError, pdf_text, read_document_bytes, search

EX = Path(__file__).resolve().parent.parent / "examples" / "datasets"


def test_text_layer_is_extracted_page_by_page():
    pdf = make_pdf(
        ["Refunds are accepted within 30 days of delivery.", "Gift vouchers last 12 months."]
    )
    text = pdf_text(pdf)
    assert "30 days" in text and "12 months" in text and text.index("30") < text.index("12")


def test_a_page_with_no_text_is_refused_with_the_reason():
    with pytest.raises(DocumentError, match="OCR"):
        pdf_text(make_pdf([""]))


def test_garbage_is_a_plain_error_not_a_traceback():
    with pytest.raises(DocumentError, match="cannot read"):
        pdf_text(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(DocumentError):
        read_document_bytes("notes.docx", b"x")


def test_title_and_key_come_from_the_file():
    key, title, _ = read_document_bytes(
        "Returns Policy 2024.pdf", make_pdf(["Returns Policy. Thirty days."])
    )
    assert key == "returns-policy-2024" and title.startswith("Returns Policy")


@pytest.mark.db
def test_documents_belong_to_their_dataset_and_search_is_scoped(scratch, tmp_path):  # noqa: F811
    admin, ro, role = scratch
    ws = Workspace(tmp_path, admin, role)
    csv = (EX / "saas_subscriptions.csv").read_bytes()
    man = yaml.safe_load((EX / "saas_subscriptions.manifest.yaml").read_text("utf-8"))
    ws.load(ws.stage("saas.csv", csv)["token"], man)
    with pytest.raises(WorkspaceError):
        ws.add_document("nope", "a.pdf", make_pdf(["x"]))
    r = ws.add_document(
        "saas_subs", "Refund Terms.pdf", make_pdf(["Annual plans are refundable within 14 days."])
    )
    assert r["doc_key"] == "ws-saas_subs-refund-terms"[:64].replace("_", "-") or r[
        "doc_key"
    ].startswith("ws-")
    assert r["sensitivity"] == "internal" and r["chunks"] >= 1
    assert [d["title"] for d in ws.documents("saas_subs")][0].startswith("Annual plans")
    assert search(ro, "refundable days", 3, doc_prefix="ws-saas_subs-")
    assert not search(ro, "refundable days", 3, doc_prefix="ws-other-")
