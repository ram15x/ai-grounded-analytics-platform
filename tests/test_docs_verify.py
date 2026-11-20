"""`va verify` reads passage text from the document store only (needs Postgres)."""

import json

import psycopg
import pytest

from verified_analyst.cli import main, run_verify
from verified_analyst.documents.store import ingest, make_passage_id

pytestmark = pytest.mark.db
TEXT = (
    "# Scratch policy\n\nThe scratch charge is £4.95 per parcel.\n\n"
    "Parcels above 10 kg pay £1.20 per extra kilogram."
)


def _answer(tmp_path, passage_id, span, prose, value, text, passages=()):
    doc = {
        "prose": prose,
        "passages": list(passages),
        "claims": [
            {
                "id": "c1",
                "text": text,
                "value": value,
                "unit": "currency",
                "label": "courier charge",
                "provenance": {"kind": "passage", "passage_id": passage_id, "span": span},
            }
        ],
    }
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def stored(warehouse):
    key = "verify-doc"
    with psycopg.connect(warehouse.admin_url) as c:
        ingest(c, key, "Charges", "charges.md", TEXT, 400, reader_role=warehouse.role)
    text, sha, _ = _chunk_in(warehouse.admin_url, key)
    yield key, text, make_passage_id(key, 0, sha)
    with psycopg.connect(warehouse.admin_url) as c:
        c.execute("DELETE FROM docs.documents WHERE doc_key = %s", (key,))


def _chunk_in(url, key):
    with psycopg.connect(url) as c:
        return c.execute(
            "SELECT text, text_sha256, char_start FROM docs.chunks WHERE doc_key = %s", (key,)
        ).fetchone()


def _span(text, needle):
    i = text.index(needle)
    return [i, i + len(needle)]


def test_a_stored_passage_verifies_a_number_it_contains(warehouse, stored, tmp_path):
    _, text, pid = stored
    path = _answer(
        tmp_path, pid, _span(text, "£4.95"), "The courier charge is £4.95.", "4.95", "£4.95"
    )
    r = run_verify(path, warehouse.readonly_url)
    assert [v.verdict for v in r.verdicts] == ["VERIFIED_QUOTE"]


def test_a_number_outside_the_cited_span_is_unsupported(warehouse, stored, tmp_path):
    _, text, pid = stored
    path = _answer(
        tmp_path,
        pid,
        _span(text, "The scratch charge is"),
        "The courier charge is £4.95.",
        "4.95",
        "£4.95",
    )
    r = run_verify(path, warehouse.readonly_url)
    assert [(v.verdict, v.reason) for v in r.verdicts] == [
        ("UNSUPPORTED", "number_not_in_cited_span")
    ]


def test_passage_text_written_in_the_answer_file_is_ignored(warehouse, stored, tmp_path, capsys):
    """A model that writes the passage itself must not be able to verify against its own words."""
    key, text, pid = stored
    forged = {"id": pid, "text": "The courier charge is £99.00."}
    path = _answer(
        tmp_path,
        pid,
        [0, 28],
        "The courier charge is £99.00.",
        "99.00",
        "£99.00",
        passages=[forged],
    )
    r = run_verify(path, warehouse.readonly_url)
    assert [v.verdict for v in r.verdicts] == ["UNSUPPORTED"]
    assert "ignored 1 passage text" in capsys.readouterr().err


def test_a_citation_of_an_old_or_invented_id_is_unsupported(warehouse, stored, tmp_path):
    key, text, pid = stored
    for bad in (key + "#0." + "0" * 10, "no-such-doc#0." + "a" * 10, "free text"):
        path = _answer(tmp_path, bad, [0, 20], "The courier charge is £4.95.", "4.95", "£4.95")
        r = run_verify(path, warehouse.readonly_url)
        assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "unknown_passage")]


def test_the_cli_exit_code_follows_the_stored_verdict(warehouse, stored, tmp_path, monkeypatch):
    _, text, pid = stored
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", warehouse.readonly_url)
    good = _answer(
        tmp_path, pid, _span(text, "£4.95"), "The courier charge is £4.95.", "4.95", "£4.95"
    )
    assert main(["verify", str(good)]) == 0
    bad = _answer(
        tmp_path, pid, _span(text, "£4.95"), "The courier charge is £5.95.", "5.95", "£5.95"
    )
    assert main(["verify", str(bad)]) == 1


def test_without_a_docs_schema_a_passage_claim_is_unsupported_not_an_error(
    warehouse_factory, raw_fixture, tmp_path
):
    """A database that has never had documents ingested has no `docs` schema at all."""
    wh = warehouse_factory(raw_fixture)
    path = _answer(
        tmp_path, "any-doc#0." + "a" * 10, [0, 5], "The charge is £4.95.", "4.95", "£4.95"
    )
    r = run_verify(path, wh.readonly_url)
    assert [(v.verdict, v.reason) for v in r.verdicts] == [("UNSUPPORTED", "no_passage_store")]
