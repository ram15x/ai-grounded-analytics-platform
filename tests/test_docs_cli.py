"""`va docs` commands and the held-out rule for retrieval questions."""

import json
import re
from pathlib import Path

import pytest

from verified_analyst import cli
from verified_analyst.cli import build_parser, main

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src" / "verified_analyst"


def test_docs_eval_has_no_way_to_name_a_question_file():
    p = build_parser()
    args = p.parse_args(["docs", "eval"])
    assert not any(k in vars(args) for k in ("questions", "file", "path"))
    with pytest.raises(SystemExit):
        p.parse_args(["docs", "eval", "--questions", "x.yaml"])


def test_only_the_gated_command_names_the_heldout_retrieval_file():
    """`va docs heldout` is the one place: its module holds the file name, cli.py only the command
    word, and no module reached by `va docs probe`, `eval` or `search` mentions it."""
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower().replace("held-out", "")
        if path.name in ("heldout_run.py", "e2e_locked.py"):  # e2e_locked: ADR-0014, one-shot
            continue
        if path.name == "cli.py":
            assert "heldout.yaml" not in text and "doc_questions" in text
            continue
        assert "heldout" not in text, path
    importers = [
        p.name
        for p in SRC.rglob("*.py")
        if "heldout_run" in p.read_text(encoding="utf-8") and p.name != "heldout_run.py"
    ]
    assert importers == ["cli.py"]
    assert cli.DEV_DOC_QUESTIONS.name == "dev.yaml"
    assert cli.DEV_DOC_QUESTIONS.is_file()


def test_the_documents_package_reads_no_question_file_at_all():
    """Only the gated held-out module knows where a question file is; it is checked above."""
    for path in (SRC / "documents").glob("*.py"):
        if path.name == "heldout_run.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"doc_questions", text), path


@pytest.mark.db
def test_ingest_search_and_eval_through_the_cli(docs_db, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_DATABASE_URL", docs_db.admin_url)
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", docs_db.readonly_url)
    note = tmp_path / "Night Shift Rota.md"
    note.write_text("# Night shift rota\n\nThe night shift starts at 22:00.\n", encoding="utf-8")
    assert main(["docs", "ingest", str(note), "--reader-role", docs_db.role]) == 0
    assert "stored    night-shift-rota (1 chunks)" in capsys.readouterr().out
    assert main(["docs", "ingest", str(note), "--reader-role", docs_db.role]) == 0
    assert "unchanged" in capsys.readouterr().out

    assert main(["docs", "search", "when does the night shift start", "-k", "2", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["doc_key"] == "night-shift-rota" and "22:00" in hits[0]["text"]
    assert main(["docs", "search", "zzyzx"]) == 0
    assert "no passage matched" in capsys.readouterr().out

    assert main(["docs", "eval"]) == 0
    report = json.loads(capsys.readouterr().out)
    from verified_analyst.documents.metrics import load_questions

    answerable = [q for q in load_questions(cli.DEV_DOC_QUESTIONS) if q.evidence]
    assert (
        report["questions_scored"] == len(answerable) > 20 and report["unreachable_anchors"] == []
    )
    assert report["config"] == {"ts_config": "english", "mode": "any"}

    import psycopg

    with psycopg.connect(docs_db.admin_url) as c:
        c.execute("DELETE FROM docs.documents WHERE doc_key = 'night-shift-rota'")


def test_ingest_refuses_other_file_types_and_empty_folders(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_DATABASE_URL", "host=127.0.0.1 port=1 user=x dbname=y")
    assert main(["docs", "ingest", str(tmp_path)]) == 2
    assert "no .md, .txt or .pdf files" in capsys.readouterr().err
    other = tmp_path / "a.docx"
    other.write_bytes(b"PK")
    assert main(["docs", "ingest", str(other)]) == 2
    assert "only .md, .txt and .pdf" in capsys.readouterr().err
    pdf = tmp_path / "b.pdf"
    pdf.write_bytes(b"%PDF")
    assert main(["docs", "ingest", str(pdf)]) == 2
    assert "cannot read this PDF" in capsys.readouterr().err
