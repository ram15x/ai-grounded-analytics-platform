"""`grounded verify` on the hand-written sample answers in examples/ (needs the warehouse)."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from grounded_analytics.cli import build_parser, main, run_verify

EXAMPLES = Path(__file__).parent.parent / "examples"

# Verdicts of the shipped sample answers. Samples 01 and 08 state the share without naming a country
# next to the global total, so they verify in full with the binder unchanged. Sample 11 keeps the
# old wording ("The UK is 59.6% of the £17,845.25 total") on purpose: the binder rejects the total
# (scope_not_in_query) because the sentence names a country that the total's query does not cover.
# That is a known, conservative limit (docs/VERIFIER_LIMITS.md). Samples 03, 05, 07 and 09 still
# carry that sentence, so each shows one extra UNSUPPORTED on the total besides its own defect.
EXPECTED = {
    "01_verified.json": ("verified", {"VERIFIED": 4, "VERIFIED_DERIVED": 1, "UNSUPPORTED": 0}),
    "02_wrong_digit.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "03_unsupported_number.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "04_wrong_metric_label.json": ("partially_verified", {"METRIC_MISMATCH": 1}),
    "05_bad_derivation.json": ("partially_verified", {"DERIVATION_MISMATCH": 1, "UNSUPPORTED": 1}),
    "06_scale_error.json": ("partially_verified", {"UNSUPPORTED": 1}),
    "07_ambiguous_number.json": ("partially_verified", {"AMBIGUOUS": 1, "UNSUPPORTED": 1}),
    "08_template_verified.json": ("verified", {"VERIFIED": 3, "VERIFIED_DERIVED": 1}),
    "09_template_number_typed_by_model.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "10_template_wrong_row.json": ("partially_verified", {"UNSUPPORTED": 2}),
    "11_known_limit_share_of_global_total.json": (
        "partially_verified",
        {"VERIFIED": 3, "VERIFIED_DERIVED": 1, "UNSUPPORTED": 1},
    ),
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
    monkeypatch.setenv("GA_READONLY_DATABASE_URL", warehouse.readonly_url)
    assert main(["verify", str(EXAMPLES / "01_verified.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    known_limit = str(EXAMPLES / "11_known_limit_share_of_global_total.json")
    assert main(["verify", known_limit, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "partially_verified"
    assert main(["verify", str(EXAMPLES / "02_wrong_digit.json")]) == 1


def test_invalid_answer_file_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GA_READONLY_DATABASE_URL", "host=localhost")
    bad = tmp_path / "bad.json"
    bad.write_text('{"prose": "x", "claims": [{"id": 1}]}')
    assert main(["verify", str(bad)]) == 2
    assert "error" in capsys.readouterr().err


def test_a_malformed_answer_file_never_opens_a_database_connection(tmp_path, monkeypatch, capsys):
    """Validation comes before any connection. On Windows a connect to `localhost` once hung this
    test for 130 s because the file was only checked after the catalogue query."""
    import grounded_analytics.cli as cli

    def no_connection(*a, **k):
        raise AssertionError("opened a database connection for an invalid answer file")

    monkeypatch.setattr(cli, "catalogue", no_connection)
    monkeypatch.setattr(cli, "execute_guarded", no_connection)
    monkeypatch.setenv("GA_READONLY_DATABASE_URL", "host=127.0.0.1 port=1 user=x dbname=y")
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
    from grounded_analytics.cli import profile_db  # noqa: PLC0415

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
@pytest.mark.parametrize(
    "name", ["11_known_limit_share_of_global_total.json", "09_template_number_typed_by_model.json"]
)
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


@pytest.mark.db
def test_sample_11_shows_the_known_limit_and_the_fixed_samples_do_not(warehouse):
    def reasons(name):
        r = run_verify(EXAMPLES / name, warehouse.readonly_url)
        verdicts = r.report.verdicts if hasattr(r, "report") else r.verdicts
        return {v.reason for v in verdicts}

    assert "scope_not_in_query" in reasons("11_known_limit_share_of_global_total.json")
    for name in ("01_verified.json", "08_template_verified.json"):
        assert "scope_not_in_query" not in reasons(name)
    raw = (EXAMPLES / "11_known_limit_share_of_global_total.json").read_text()
    assert "KNOWN LIMIT" in raw and "The UK is 59.6%" in raw


@pytest.mark.db
def test_the_binder_default_change_moves_no_verdict_of_the_shipped_samples(warehouse, monkeypatch):
    """ADR-0012: the samples carry no passage claim, so the binder is never consulted; verified by
    running all of them under the old rule (`current`) and under the shipped default."""
    import functools

    from grounded_analytics import verifier
    from grounded_analytics.documents.binding import CURRENT, DEFAULT, check_binding

    def run_all(cfg):
        monkeypatch.setattr(verifier, "check_binding", functools.partial(check_binding, cfg=cfg))
        out = {}
        for path in sorted(EXAMPLES.glob("*.json")):
            r = run_verify(path, warehouse.readonly_url)
            verdicts = r.report.verdicts if hasattr(r, "report") else r.verdicts
            out[path.name] = (r.status, [(v.claim_id, v.verdict, v.reason) for v in verdicts])
        return out

    old, new = run_all(CURRENT), run_all(DEFAULT)
    assert len(old) == 11 and old == new
    assert not any("passage" in p.read_text().lower() for p in EXAMPLES.glob("*.json"))
