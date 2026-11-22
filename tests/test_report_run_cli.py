from grounded_analytics.cli import build_parser


def test_report_run_parses_and_needs_a_title():
    a = build_parser().parse_args(["report", "run", "--title", "T", "--replay"])
    assert a.replay == "demo/recordings.jsonl" and a.questions == "demo/weekly_questions.json"
    assert a.fn.__name__ == "cmd_report_run"
