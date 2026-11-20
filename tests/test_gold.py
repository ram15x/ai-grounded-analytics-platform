"""Gold harness: hand-checked values, independence from the product code, question-file sanity."""

import ast
import importlib.util
import json
import sys
from decimal import Decimal as D
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent
GOLD = ROOT / "eval" / "gold"
sys.path.insert(0, str(GOLD))
import t1  # noqa: E402

QUESTIONS = yaml.safe_load((ROOT / "eval" / "questions" / "dev.yaml").read_text(encoding="utf-8"))
QS = QUESTIONS["questions"]

HEADER = "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
TINY = HEADER + "\n".join(
    [
        "1001,85123A,A,2,2011-01-01 10:00:00,3.00,1.0,UK",
        "1001,POST,POSTAGE,1,2011-01-01 10:00:00,5.00,1.0,UK",
        "1002,22000,B,4,2011-01-02 10:00:00,2.50,,UK",
        "C1003,85123A,A,-1,2011-01-03 10:00:00,3.00,1.0,UK",
        "1004,D,Discount,-1,2011-01-04 10:00:00,2.00,1.0,UK",
        "1005,22001,C,1,2011-01-05 10:00:00,0,2.0,UK",
        "1006,22002,D2,-3,2011-01-06 10:00:00,1.00,3.0,UK",
        "1001,85123A,A,2,2011-01-01 10:00:00,3.00,1.0,UK",
    ]
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("g") / "tiny.csv"
    path.write_text(TINY, encoding="utf-8")
    return t1.load_raw(path)


HAND_CHECKS = [
    ("gross_revenue", D("27")),  # 6 + 5 + 10 + 6 (the exact duplicate is NOT removed)
    ("cancelled_revenue", D("3")),
    ("discount_amount", D("2")),
    ("net_revenue", D("22")),  # 27 - 3 - 2
    ("units_sold", D("9")),  # 2 + 1 (postage counts) + 4 + 2
    ("merchandise_units_sold", D("8")),  # units_sold minus the 1 postage unit
    ("order_count", D("2")),  # invoices 1001 and 1002
    ("active_customers", D("1")),  # customer 1; the blank is not counted
    ("average_order_value", D("13.5")),
    ("revenue_per_customer", D("17")),  # (6 + 5 + 6) / 1
    ("cancellation_rate", D(3) / D(27)),
    ("merchandise_gross_revenue", D("22")),  # POST excluded: 6 + 10 + 6
    ("sale_line_count", D("4")),
    ("distinct_products_sold", D("3")),
    ("cancelled_invoice_count", D("1")),
    ("identified_net_revenue", D("12")),  # 6 + 5 + 6 - 3 - 2
]


@pytest.mark.parametrize(("fn", "expected"), HAND_CHECKS)
def test_gold_matches_hand_calculation(tiny, fn, expected):
    assert t1.GOLD_FUNCTIONS[fn](tiny) == expected


def test_every_gold_function_has_a_hand_check():
    checked = {fn for fn, _ in HAND_CHECKS}
    assert checked == set(t1.GOLD_FUNCTIONS)


# ---- independence ------------------------------------------------------------------------------

ALLOWED = {"__future__", "re", "decimal", "pathlib", "pandas", "argparse", "hashlib", "json",
           "sys", "yaml", "t1"}  # fmt: skip


@pytest.mark.parametrize("path", sorted(GOLD.glob("*.py")), ids=lambda p: p.name)
def test_gold_files_import_nothing_from_the_product(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert mods <= ALLOWED, mods - ALLOWED


# ---- question file -------------------------------------------------------------------------------


def test_question_file_shape():
    assert len(QS) == 15
    assert len({q["id"] for q in QS}) == 15
    for q in QS:
        assert q["tier"] == "T1" and q["author_written"] is True
        assert q["gold_fn"] in t1.GOLD_FUNCTIONS
        assert len(q["rules"]) >= 3 and all(isinstance(r, str) for r in q["rules"])
        assert "gold" not in q and "gold_value" not in q  # values come only from the gold script
        assert ("metric_query" in q) == q["expressible_as_metric_query"]
        if not q["expressible_as_metric_query"]:
            assert q["not_expressible_reason"]
    assert "AUTHOR-WRITTEN" in QUESTIONS["description"]


def test_metric_queries_compile_and_restate():
    from verified_analyst.compiler import compile_query
    from verified_analyst.restate import restate
    from verified_analyst.semantic import load_semantic_layer

    layer = load_semantic_layer()
    for q in QS:
        if q["expressible_as_metric_query"]:
            assert len(q["metric_query"]["metrics"]) == 1 and not q["metric_query"].get(
                "dimensions"
            )
            compile_query(layer, q["metric_query"])
            assert restate(layer, q["metric_query"]).startswith("Compute ")


def test_run_script_refuses_a_missing_raw_file(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("gold_run", GOLD / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--raw", str(tmp_path / "nope.xlsx"), "--out", str(tmp_path / "v.json")]) == 2
    assert "not found" in capsys.readouterr().err


def test_run_script_writes_values_with_hash(tmp_path):
    raw = tmp_path / "tiny.csv"
    raw.write_text(TINY, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("gold_run2", GOLD / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "v.json"
    assert mod.main(["--raw", str(raw), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["author_written"] is True and len(data["raw_sha256"]) == 64
    assert data["values"]["T1-01"] == "27"


@pytest.mark.db
def test_compare_on_synthetic_fixture_finds_no_difference_after_the_t1_05_decision(
    warehouse, tmp_path
):
    from verified_analyst.semantic import load_semantic_layer

    spec = importlib.util.spec_from_file_location("compare_gold", ROOT / "eval" / "compare_gold.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = t1.load_raw(ROOT / "tests" / "fixtures" / "mini_retail.csv")
    gold = {"raw_file": "mini_retail.csv", "raw_sha256": "0" * 64, "raw_rows": len(raw),
            "values": {q["id"]: str(t1.GOLD_FUNCTIONS[q["gold_fn"]](raw)) for q in QS}}  # fmt: skip
    rows = mod.compare(QS, gold, load_semantic_layer(), warehouse.readonly_url)
    diffs = [r for r in rows if r["status"] == "DIFF"]
    assert diffs == []
    assert sum(r["status"] == "not_expressible" for r in rows) == 4
    text = mod.render(rows, gold, "pending")
    assert "SYNTHETIC FIXTURE" in text and "author-written" in text
