"""`va verify` on the hand-written sample answers in examples/ (needs the warehouse fixture)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from verified_analyst.cli import build_parser, main, run_verify

EXAMPLES = Path(__file__).parent.parent / "examples"

# Verdicts of the shipped sample answers. Files unchanged. The examples say "The UK is 59.6% of the
# £17.8m total"; since the layer lists "UK" as an alias of "United Kingdom" the binder reads that
# sentence like the full-name one, which it has always rejected: the global total £17,845.25 sits
# in a sentence that names a country the total's query does not cover (scope_not_in_query). Before
# the alias was read, these examples verified only because "UK" was invisible to the binder.
EXPECTED = {
    "01_verified.json": (
        "partially_verified",
        {"VERIFIED": 3, "VERIFIED_DERIVED": 1, "UNSUPPORTED": 1},
    ),
    "02_wrong_digit.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "03_unsupported_number.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "04_wrong_metric_label.json": ("partially_verified", {"METRIC_MISMATCH": 1}),
    "05_bad_derivation.json": ("partially_verified", {"DERIVATION_MISMATCH": 1, "UNSUPPORTED": 1}),
    "06_scale_error.json": ("partially_verified", {"UNSUPPORTED": 1}),
    "07_ambiguous_number.json": ("partially_verified", {"AMBIGUOUS": 1, "UNSUPPORTED": 1}),
    "08_template_verified.json": (
        "partially_verified",
        {"VERIFIED": 2, "VERIFIED_DERIVED": 1, "UNSUPPORTED": 1},
    ),
    "09_template_number_typed_by_model.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "10_template_wrong_row.json": ("partially_verified", {"UNSUPPORTED": 2}),
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
    assert main(["verify", str(EXAMPLES / "01_verified.json"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partially_verified"
    assert main(["verify", str(EXAMPLES / "02_wrong_digit.json")]) == 1


def test_invalid_answer_file_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "host=localhost")
    bad = tmp_path / "bad.json"
    bad.write_text('{"prose": "x", "claims": [{"id": 1}]}')
    assert main(["verify", str(bad)]) == 2
    assert "error" in capsys.readouterr().err


def test_a_malformed_answer_file_never_opens_a_database_connection(tmp_path, monkeypatch, capsys):
    """Validation comes before any connection. On Windows a connect to `localhost` once hung this
    test for 130 s because the file was only checked after the catalogue query."""
    import verified_analyst.cli as cli

    def no_connection(*a, **k):
        raise AssertionError("opened a database connection for an invalid answer file")

    monkeypatch.setattr(cli, "catalogue", no_connection)
    monkeypatch.setattr(cli, "execute_guarded", no_connection)
    monkeypatch.setenv("VA_READONLY_DATABASE_URL", "host=127.0.0.1 port=1 user=x dbname=y")
    for name, text in {
        "plain.json": '{"prose": "x", "claims": [{"id": 1}]}',
        "template.json": '{"template": "x {a}", "claims": [{"id": 1}]}',
    }.items():
        bad = tmp_path / name
        bad.write_text(text)
        assert main(["verify", str(bad)]) == 2
        assert "error" in capsys.readouterr().err


@pytest.mark.db
def test_profile_db_reports_duplicates_effect(warehouse):
    from verified_analyst.cli import profile_db  # noqa: PLC0415

    out = profile_db(warehouse.readonly_url)
    assert set(out["all_rows"]) == set(out["excluding_exact_duplicates"])
    dups = sum(
        int(r["n"])
        for r in out["rows_by_line_type_and_duplicate_flag"]
        if r["is_exact_duplicate"] == "True"
    )
    assert dups == warehouse.normalized.dq["exact_duplicate_rows"]
    assert Decimal(out["all_rows"]["gross_revenue"]) >= Decimal(
        out["excluding_exact_duplicates"]["gross_revenue"]
    )


@pytest.mark.db
@pytest.mark.parametrize("name", ["01_verified.json", "08_template_verified.json"])
def test_the_uk_examples_get_the_same_verdicts_as_their_full_name_wording(
    warehouse, tmp_path, name
):
    """The alias goes through the same path as the full name; the example file is not edited, the
    reworded copy is only a witness. The total is rejected for scope_not_in_query either way."""
    raw = (EXAMPLES / name).read_text()
    assert "The UK is" in raw
    full = tmp_path / name
    full.write_text(raw.replace("The UK is", "The United Kingdom is"))

    def outcome(path):
        r = run_verify(path, warehouse.readonly_url)
        verdicts = r.report.verdicts if hasattr(r, "report") else r.verdicts
        return r.status, [(v.claim_id, v.verdict, v.reason) for v in verdicts]

    uk, full_name = outcome(EXAMPLES / name), outcome(full)
    assert uk == full_name
    assert any(reason == "scope_not_in_query" for _, _, reason in uk[1])
