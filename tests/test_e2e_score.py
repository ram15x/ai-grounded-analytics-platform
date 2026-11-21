from verified_analyst.e2e import score, summarise, wilson


def run(outcome="verified", values=("9769872.0540",), reason=""):
    return {
        "outcome": outcome,
        "reason": reason,
        "verification": {
            "placeholders": [{"true_value": v} for v in values],
            "verification": {"verdicts": [{"verdict": "VERIFIED"} for _ in values]},
        },
    }


def test_correct_matches_at_the_question_decimals():
    assert score(run(), "9769872.054", 2) == "correct"


def test_verified_but_different_figure_is_verified_wrong():
    assert score(run(values=("9769872.99",)), "9769872.054", 2) == "verified_wrong"


def test_withheld_and_no_reply_are_not_counted_as_wrong():
    assert score(run("abstain", reason="ambiguous"), "1", 0) == "withheld"
    assert score(run("abstain", reason="provider_error"), "1", 0) == "no_reply"
    assert score(run("not_recorded", reason="no_recording"), "1", 0) == "no_reply"


def test_ratio_uses_its_own_decimals():
    assert score(run(values=("0.0835",)), "0.08354201029609074", 4) == "correct"


def test_summary_and_interval():
    rows = [{"score": s} for s in ("correct", "correct", "withheld", "no_reply")]
    s = summarise(rows)
    assert (s["n"], s["answered"], s["correct"]) == (4, 3, 2)
    lo, hi = wilson(2, 3)
    assert 0 < lo < 0.67 < hi < 1


def test_table_gold_needs_every_numeric_cell_and_ignores_labels():
    from verified_analyst.e2e import score_table

    gold = [["Germany", "100.50"], ["France", "90.00"]]
    ok = run(values=("100.50", "90.00", "7"))  # an extra figure is allowed
    assert score_table(ok, gold, 2) == "correct"
    assert score_table(run(values=("100.50",)), gold, 2) == "verified_wrong"
    assert score_table(run("abstain", reason="ambiguous"), gold, 2) == "withheld"
