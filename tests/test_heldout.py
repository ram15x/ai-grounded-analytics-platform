"""The held-out question file and its gold functions (eval/gold/t2.py): hand-checked values on a
tiny file, agreement with the other gold module, independence, and the guards that keep the
held-out set away from prompt tuning. No model, no real data."""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from decimal import Decimal as D
from inspect import signature
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
GOLD = ROOT / "eval" / "gold"
sys.path.insert(0, str(GOLD))
import t1  # noqa: E402
import t2  # noqa: E402

DOC = yaml.safe_load((ROOT / "eval" / "questions" / "heldout.yaml").read_text(encoding="utf-8"))
QS = DOC["questions"]
DEV = yaml.safe_load((ROOT / "eval" / "questions" / "dev.yaml").read_text(encoding="utf-8"))

HEADER = "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
ROWS = [
    "1001,85123A,A,2,2011-01-05 10:00:00,3.00,1.0,UK",  # sale 6
    "1001,POST,POSTAGE,1,2011-01-05 10:00:00,5.00,1.0,UK",  # sale 5, not merchandise
    "1002,22000,B,4,2011-01-20 10:00:00,2.50,,UK",  # sale 10, no customer
    "1003,85123A,A,1,2011-02-03 10:00:00,3.00,2.0,France",  # sale 3
    "1004,22001,C,10,2011-02-10 10:00:00,1.00,3.0,France",  # sale 10
    "C1005,85123A,A,-1,2011-02-11 10:00:00,3.00,2.0,France",  # cancellation 3
    "1006,D,Discount,-1,2011-02-12 10:00:00,2.00,2.0,France",  # discount 2
    "1007,22002,D2,5,2011-03-01 10:00:00,4.00,4.0,Germany",  # sale 20
    "1008,22003,E,2,2011-03-02 10:00:00,5.00,5.0,Germany",  # sale 10
    "1009,85123A,A,3,2011-03-05 10:00:00,3.00,4.0,Germany",  # sale 9
    "1010,22004,F,1,2011-03-06 10:00:00,0,6.0,Germany",  # price 0: nothing
    "1001,85123A,A,2,2011-01-05 10:00:00,3.00,1.0,UK",  # exact duplicate sale 6, kept
]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("h") / "tiny.csv"
    path.write_text(HEADER + "\n".join(ROWS), encoding="utf-8")
    return path, t2.load_raw(path)


def run(tiny, fn, **kw):
    return t2.GOLD_FUNCTIONS[fn](tiny[1], **kw)


def scalar(v):
    return {"kind": "scalar", "value": v}


def table(columns, rows):
    return {"kind": "table", "columns": columns, "rows": rows}


# hand calculation: gross 79 (UK 27, France 13, Germany 39), cancelled 3, discount 2, net 74.
# Months: Jan gross 27, Feb gross 13 net 8, Mar gross 39.
HAND = [
    ("gross_revenue_in_country", {"country": "Germany"}, scalar(D("39"))),
    ("net_revenue_in_month", {"month": "2011-02"}, scalar(D("8"))),
    (
        "top_countries_by_gross_revenue",
        {"n": 2},
        table(["country", "gross_revenue"], [["Germany", "39"], ["UK", "27"]]),
    ),
    (
        "orders_per_month",
        {"months": ["2011-03", "2011-01", "2011-02"]},
        table(["month", "order_count"], [["2011-01", "2"], ["2011-02", "2"], ["2011-03", "3"]]),
    ),
    ("active_customers_in_country", {"country": "France"}, scalar(D("2"))),
    (
        "month_with_highest_cancelled_revenue",
        {},
        table(["month", "cancelled_revenue"], [["2011-02", "3"]]),
    ),
    ("average_order_value_in_country", {"country": "UK"}, scalar(D("13.5"))),
    (
        "percent_change_gross_revenue",
        {"month_from": "2011-02", "month_to": "2011-03"},
        scalar(D("200")),
    ),  # fmt: skip
    (
        "net_revenue_in_two_months",
        {"months": ["2011-01", "2011-03"]},
        table(["month", "net_revenue"], [["2011-01", "27"], ["2011-03", "39"]]),
    ),
    (
        "net_revenue_in_country_between_months",
        {"country": "France", "first": "2011-01", "last": "2011-02"},
        scalar(D("8")),
    ),  # fmt: skip
    (
        "top_products_by_merchandise_units_in_country",
        {"country": "Germany", "n": 2},
        table(["stock_code", "units_sold"], [["22002", "5"], ["85123A", "3"]]),
    ),
    ("merchandise_units_per_order", {}, scalar(D(29) / D(7))),  # 29 merchandise units, 7 orders
    ("gross_revenue_without_customer_id", {}, scalar(D("10"))),
    ("share_of_gross_revenue_from_top_countries", {"n": 2}, scalar(D(66) / D(79) * 100)),
    ("median_order_value", {}, scalar(D("10"))),  # invoice values 3, 9, 10, 10, 10, 17, 20
]


@pytest.mark.parametrize(("fn", "kw", "expected"), HAND, ids=[h[0] for h in HAND])
def test_gold_matches_hand_calculation(tiny, fn, kw, expected):
    assert run(tiny, fn, **kw) == expected


def test_every_gold_function_has_a_hand_check_and_a_question():
    assert {h[0] for h in HAND} == set(t2.GOLD_FUNCTIONS)
    assert {q["gold_fn"] for q in QS} == set(t2.GOLD_FUNCTIONS)  # no unused, no missing


def test_a_tie_at_the_top_n_cutoff_is_refused_not_guessed(tiny):
    with pytest.raises(ValueError, match="tie at the top-1"):
        run(tiny, "top_products_by_merchandise_units_in_country", country="UK", n=1)
    # a tie that is NOT at the cut-off is fine: both UK codes fit in the top two
    got = run(tiny, "top_products_by_merchandise_units_in_country", country="UK", n=2)
    assert got["rows"] == [["22000", "4"], ["85123A", "4"]]


def test_median_matches_the_standard_library_for_odd_and_even_counts(tiny):
    df = tiny[1]
    for drop in ([], ["1003"], ["1003", "1001"]):
        d = df[~df["invoice"].isin(drop)]
        totals = [D(int(v)) / t2.TICKS for v in d[d["sale"]].groupby("invoice")["amount"].sum()]
        assert t2.median_order_value(d)["value"] == statistics.median(totals)


def test_the_two_gold_modules_agree_on_the_overall_totals(tiny):
    d1 = t1.load_raw(tiny[0])
    d2 = tiny[1]
    assert t2._money(t2._gross(d2)) == t1.gross_revenue(d1)
    assert t2._money(t2._net(d2)) == t1.net_revenue(d1)
    assert D(int(d2.loc[d2["sale"], "invoice"].nunique())) == t1.order_count(d1)
    assert D(int(d2.loc[d2["sale"], "customer"].dropna().nunique())) == t1.active_customers(d1)
    assert D(int(d2.loc[d2["sale"] & d2["is_merch"], "qty"].sum())) == t1.merchandise_units_sold(d1)


# ---- the question file ---------------------------------------------------------------------------


def test_question_file_shape():
    assert DOC["held_out"] is True and len(QS) == 15
    assert len({q["id"] for q in QS}) == 15
    assert {q["tier"] for q in QS} <= {"T2", "T3"} and {"T2", "T3"} == {q["tier"] for q in QS}
    for q in QS:
        assert q["id"].startswith("H-") and q["split"] == "held_out"
        assert q["author_written"] is True and q["answer_shape"] in ("scalar", "table")
        assert q["gold_fn"] in t2.GOLD_FUNCTIONS
        params = set(signature(t2.GOLD_FUNCTIONS[q["gold_fn"]]).parameters) - {"d"}
        assert set(q.get("gold_args", {})) == params, q["id"]
        assert "gold" not in q and "gold_value" not in q and "value" not in q
        assert ("metric_query" in q) == q["expressible_as_metric_query"]
        if not q["expressible_as_metric_query"]:
            assert q["not_expressible_reason"]
    assert len(DOC["common_rules"]) >= 8


def test_expressible_questions_compile_and_unexpressible_ones_say_why():
    from grounded_analytics.compiler import compile_query
    from grounded_analytics.restate import restate
    from grounded_analytics.semantic import load_semantic_layer

    layer = load_semantic_layer()
    expressible = [q for q in QS if q["expressible_as_metric_query"]]
    assert len(expressible) == 13 and len(QS) - len(expressible) == 2
    for q in expressible:
        compile_query(layer, q["metric_query"])
        assert restate(layer, q["metric_query"]).startswith("Compute ")


def test_the_variety_the_set_is_meant_to_cover_is_there():
    fns = " ".join(q["gold_fn"] for q in QS)
    text = " ".join(q["question"] for q in QS)
    assert "in_country" in fns  # filter by country
    assert "month" in fns  # by month
    assert re.search(r"\b(five|three) ", text)  # top-N
    assert "percent_change" in fns and "two_months" in fns  # two periods
    assert any("ratio" in q.get("needs_derived_claim", "") for q in QS)  # a ratio


# ---- held-out guards -----------------------------------------------------------------------------


def test_no_question_text_or_id_is_shared_with_the_dev_set():
    dev_text = {q["question"].strip().lower() for q in DEV["questions"]}
    assert not dev_text & {q["question"].strip().lower() for q in QS}
    assert not {q["id"] for q in DEV["questions"]} & {q["id"] for q in QS}


def test_no_product_code_reads_or_embeds_the_held_out_file():
    # `grounded docs heldout` is the one deliberate exception (heldout_run.py plus its command in
    # cli.py); it reads the DOCUMENT question file only. The text-to-SQL held-out file is still
    # named nowhere in src.
    # e2e_locked.py (ADR-0014) is the second deliberate exception: the text-to-SQL locked file is
    # read only by `grounded eval e2e --split locked --confirm-locked`, once per model (ledger).
    allowed = {"heldout_run.py", "cli.py", "e2e_locked.py"}
    hits = [
        p
        for p in (ROOT / "src").rglob("*.py")
        if "heldout" in p.read_text("utf-8").lower() and p.name not in allowed
    ]
    assert hits == []
    for p in (ROOT / "src").rglob("*.py"):
        if p.name == "e2e_locked.py":
            continue
        assert "questions/heldout" not in p.read_text("utf-8").replace("\\", "/"), p
    from grounded_analytics.llm.prompts import PROMPT_TEMPLATES

    blob = " ".join(PROMPT_TEMPLATES.values()).lower()
    assert not [q["question"] for q in QS if q["question"].lower() in blob]


def test_the_probe_reads_only_the_dev_file():
    from grounded_analytics.llm.live import default_questions_path, load_probe_questions

    assert default_questions_path().name == "dev.yaml"
    held = {q["question"] for q in QS}
    assert not held & set(load_probe_questions())


def test_held_out_values_are_git_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "heldout_values*.json" in ignore


def test_run_script_writes_values_with_the_raw_hash_and_refuses_a_missing_file(
    tiny, tmp_path, capsys
):
    spec = importlib.util.spec_from_file_location("run_heldout", GOLD / "run_heldout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--raw", str(tmp_path / "nope.xlsx"), "--out", str(tmp_path / "v.json")]) == 2
    assert "not found" in capsys.readouterr().err
    # a question file of two questions keeps the run small and still exercises both shapes
    qs = {"questions": [q for q in QS if q["id"] in ("H-T2-01", "H-T2-04")]}
    qf = tmp_path / "q.yaml"
    qf.write_text(yaml.safe_dump(qs), encoding="utf-8")
    out = tmp_path / "v.json"
    assert mod.main(["--raw", str(tiny[0]), "--out", str(out), "--questions", str(qf)]) == 0
    data = json.loads(out.read_text())
    assert data["held_out"] is True and len(data["raw_sha256"]) == 64
    assert data["values"]["H-T2-01"] == {
        "kind": "scalar",
        "value": "39",
    }  # Germany in the tiny file
    assert data["values"]["H-T2-04"]["kind"] == "table"


def test_compare_heldout_agreement_rules():
    spec = importlib.util.spec_from_file_location("ch", ROOT / "eval" / "compare_heldout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.agrees(
        {"kind": "scalar", "value": D("12.344")}, {"kind": "scalar", "value": "12.34"}, 2
    )
    assert not mod.agrees(
        {"kind": "scalar", "value": D("12.36")}, {"kind": "scalar", "value": "12.34"}, 2
    )
    gold = {"kind": "table", "columns": ["m", "v"], "rows": [["2011-01", "5"], ["2011-02", "6"]]}
    same = {"kind": "table", "columns": ["m", "v"], "rows": [["2011-01", "5.0"], ["2011-02", "6"]]}
    assert mod.agrees(same, gold, 0)
    assert not mod.agrees({**same, "rows": same["rows"][:1]}, gold, 0)  # a missing row
    assert not mod.agrees({**same, "rows": [["2011-02", "5"], ["2011-01", "6"]]}, gold, 0)  # order
    assert mod._cell(__import__("datetime").date(2011, 1, 1)) == "2011-01"
