import json

import pytest

from grounded_analytics.cli import build_parser


def parse(*a):
    return build_parser().parse_args(["eval", "e2e", *a])


def test_locked_needs_confirmation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parse("--split", "locked", "--provider", "groq", "--model", "m")
    with pytest.raises(SystemExit, match="confirm-locked"):
        args.fn(args)


def test_locked_refuses_replay_and_a_second_run_for_the_same_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = parse("--split", "locked", "--confirm-locked", "--replay")
    with pytest.raises(SystemExit, match="not run on replay"):
        args.fn(args)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "e2e_locked_ledger.jsonl").write_text(
        json.dumps({"model": "groq/m"}) + "\n"
    )
    args = parse("--split", "locked", "--confirm-locked", "--provider", "groq", "--model", "m")
    with pytest.raises(SystemExit, match="already run"):
        args.fn(args)
