"""`va verify` on the hand-written sample answers in examples/ (needs the warehouse fixture)."""

import json
from pathlib import Path

import pytest

from verified_analyst.cli import build_parser, main, run_verify

EXAMPLES = Path(__file__).parent.parent / "examples"

EXPECTED = {
    "01_verified.json": ("verified", {"VERIFIED": 4, "VERIFIED_DERIVED": 1}),
    "02_wrong_digit.json": ("partially_verified", {"UNSUPPORTED": 1}),
    "03_unsupported_number.json": ("partially_verified", {"UNSUPPORTED": 1}),
    "04_wrong_metric_label.json": ("partially_verified", {"METRIC_MISMATCH": 1}),
    "05_bad_derivation.json": ("partially_verified", {"DERIVATION_MISMATCH": 1}),
    "06_scale_error.json": ("partially_verified", {"UNSUPPORTED": 1}),
    "07_ambiguous_number.json": ("partially_verified", {"AMBIGUOUS": 1}),
}


def test_parser_has_the_documented_commands():
    p = build_parser()
    assert p.parse_args(["verify", "x.json"]).cmd == "verify"
    assert p.parse_args(["load", "--source", "f.csv"]).cmd == "load"
    assert p.parse_args(["profile", "f.csv"]).cmd == "profile"
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_every_example_has_an_expectation():
    assert {f.name for f in EXAMPLES.glob("*.json")} == set(EXPECTED)


@pytest.mark.db
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_verdicts(warehouse, name):
    status, must_include = EXPECTED[name]
    report = run_verify(EXAMPLES / name, warehouse.readonly_url)
    assert report.status == status
    counts = report.counts()
    for verdict, n in must_include.items():
        assert counts.get(verdict, 0) == n, counts


@pytest.mark.db
def test_exit_codes_and_json_output(warehouse, monkeypatch, capsys):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", warehouse.readonly_url)
    assert main(["verify", str(EXAMPLES / "01_verified.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert main(["verify", str(EXAMPLES / "02_wrong_digit.json")]) == 1


def test_invalid_answer_file_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "host=localhost")
    bad = tmp_path / "bad.json"
    bad.write_text('{"prose": "x", "claims": [{"id": 1}]}')
    assert main(["verify", str(bad)]) == 2
    assert "error" in capsys.readouterr().err
