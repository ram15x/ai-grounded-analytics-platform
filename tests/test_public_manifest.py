"""Every real public document must carry its provenance (see docs/PUBLIC_DOCUMENTS_PLAN.md)."""

import datetime
import hashlib
from pathlib import Path

import yaml

PUBLIC = Path(__file__).parent.parent / "corpus" / "public"
ALLOWED_LICENCES = {"OGL-3.0"}
FIELDS = (
    "doc_key",
    "file",
    "title",
    "source_url",
    "licence",
    "licence_url",
    "licence_checked_on_page",
    "retrieved",
    "sha256",
    "attribution",
)


def _entries():
    return yaml.safe_load((PUBLIC / "MANIFEST.yaml").read_text(encoding="utf-8"))["documents"]


def _files():
    return {p.name for p in PUBLIC.iterdir() if p.suffix.lower() in (".md", ".txt")} - {"README.md"}


def test_the_manifest_is_a_list_and_nothing_is_added_without_an_entry():
    entries = _entries()
    assert isinstance(entries, list)
    assert _files() == {e["file"] for e in entries}, "files and manifest entries differ"


def test_every_entry_has_every_field_and_a_valid_value():
    keys = set()
    for e in _entries():
        assert all(e.get(f) not in (None, "") for f in FIELDS), e.get("doc_key")
        assert e["licence"] in ALLOWED_LICENCES, e["doc_key"]
        assert e["licence_checked_on_page"] is True, e["doc_key"]
        assert e["source_url"].startswith("https://") and e["licence_url"].startswith("https://")
        assert datetime.date.fromisoformat(str(e["retrieved"])) <= datetime.date.today()
        assert e["doc_key"] not in keys
        keys.add(e["doc_key"])
        data = (PUBLIC / e["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e["sha256"], f"{e['file']} changed"
        assert e["attribution"] in data.decode("utf-8"), f"{e['file']} lacks its attribution"


def test_public_files_are_never_in_the_synthetic_corpus_folder():
    synthetic = {p.name for p in (PUBLIC.parent / "synthetic").glob("*")}
    assert not synthetic & _files()
