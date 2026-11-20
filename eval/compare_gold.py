"""Compare gold values (eval/gold/values.json) with the semantic layer, via the real compiled path.

    python eval/compare_gold.py --values eval/gold/values.json --out docs/GOLD_DIFFS.md

Needs VA_READONLY_DATABASE_URL (the warehouse loaded with `va load`). This is the ONLY eval file
that imports the product code; the gold computation itself (eval/gold/) never does.

It never changes a metric definition. A difference is classified as:
  - intended rule difference: the question file declared it up front (`declared_rule_difference`);
    the owner still has to confirm it;
  - UNCLASSIFIED: needs a human to decide between gold bug, semantic-layer bug, or intended
    rule difference. The script does not guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from verified_analyst.compiler import MetricQuery, compile_query
from verified_analyst.config import readonly_database_url
from verified_analyst.guard import execute_guarded
from verified_analyst.restate import restate
from verified_analyst.semantic import load_semantic_layer

ROOT = Path(__file__).parent
SYNTHETIC_FILES = {"mini_retail.csv"}


def layer_value(layer, url: str, mq: dict) -> tuple[Decimal, str]:
    q = MetricQuery.model_validate(mq)
    cq = compile_query(layer, q)
    res = execute_guarded(url, cq.sql)
    assert len(res.rows) == 1, f"expected one row, got {len(res.rows)}"
    return Decimal(str(res.rows[0][cq.columns.index(q.metrics[0])])), restate(layer, q)


def compare(questions: list[dict], gold: dict, layer, url: str) -> list[dict]:
    rows = []
    for q in questions:
        g = Decimal(gold["values"][q["id"]])
        row = {"id": q["id"], "question": q["question"], "gold": g, "decimals": q["decimals"]}
        if not q["expressible_as_metric_query"]:
            row.update(
                status="not_expressible", layer=None, note=q.get("not_expressible_reason", "")
            )
        else:
            val, restated = layer_value(layer, url, q["metric_query"])
            tol = Decimal(1).scaleb(-q["decimals"]) / 2
            agree = abs(val - g) <= tol
            row.update(
                layer=val, restated=restated, diff=val - g, status="agree" if agree else "DIFF"
            )
            if not agree:
                declared = q.get("declared_rule_difference")
                if declared:
                    row["classification"] = (
                        "intended rule difference "
                        "(declared by the author before comparing; owner to confirm)"
                    )
                else:
                    row["classification"] = (
                        "UNCLASSIFIED: decide gold bug, semantic-layer bug, "
                        "or intended rule difference"
                    )
                row["note"] = declared or ""
        rows.append(row)
    return rows


def fmt(v: Decimal | None, decimals: int) -> str:
    if v is None:
        return "n/a"
    d = max(decimals, 0) + (2 if decimals == 4 else 0)
    return f"{v:,.{d}f}"


def render(rows: list[dict], gold: dict, note: str | None) -> str:
    synthetic = gold["raw_file"] in SYNTHETIC_FILES
    label = (
        "SYNTHETIC FIXTURE, dry run of the pipeline. NOT real data." if synthetic else "REAL DATA"
    )
    n_expr = sum(r["status"] != "not_expressible" for r in rows)
    diffs = [r for r in rows if r["status"] == "DIFF"]
    out = ["# Gold vs semantic layer: differences", ""]
    if note:
        out += [f"> {note}", ""]
    out += [
        f"Source of gold values: **{label}**",
        f"Raw file: `{gold['raw_file']}` (sha256 `{gold['raw_sha256'][:16]}...`, "
        f"{gold['raw_rows']} rows). Generated {date.today().isoformat()} "
        "by `eval/compare_gold.py`.",
        "",
        "**Gold questions are author-written** (and the same author wrote the semantic layer), so "
        "agreement is weak evidence and every difference is a finding. No metric definition was "
        "changed to make numbers agree.",
        "",
        f"Questions: {len(rows)}. Expressible as a catalogue query (coverage): "
        f"{n_expr}/{len(rows)}. Agree: {sum(r['status'] == 'agree' for r in rows)}. "
        f"Differ: {len(diffs)}. Not expressible (no layer value to compare): "
        f"{sum(r['status'] == 'not_expressible' for r in rows)}.",
        "",
        "| ID | Question | Gold | Layer | Status |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['id']} | {r['question']} | {fmt(r['gold'], r['decimals'])} | "
            f"{fmt(r['layer'], r['decimals'])} | {r['status']} |"
        )
    out += ["", "## Differences", ""]
    if not diffs:
        out.append("None.")
    for r in diffs:
        out += [
            f"### {r['id']}: {r['question']}",
            f"- Gold: {fmt(r['gold'], r['decimals'])}; layer: {fmt(r['layer'], r['decimals'])}; "
            f"layer minus gold: {fmt(r['diff'], r['decimals'])}",
            f"- Classification: **{r['classification']}**",
            f"- Author's note: {r['note'] or '(none)'}",
            f"- Layer query, restated: {r['restated']}",
            "- Owner decision: ______",
            "",
        ]
    out += ["## Not expressible as a catalogue query", ""]
    for r in rows:
        if r["status"] == "not_expressible":
            out.append(
                f"- {r['id']}: {r['question']} (gold {fmt(r['gold'], r['decimals'])}). {r['note']}"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=Path, default=ROOT / "gold" / "values.json")
    ap.add_argument("--questions", type=Path, default=ROOT / "questions" / "dev.yaml")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "docs" / "GOLD_DIFFS.md")
    ap.add_argument("--note", help="banner line to put at the top of the report")
    a = ap.parse_args(argv)
    if not a.values.exists():
        print(
            f"error: {a.values} not found; run eval/gold/run.py on the raw file first",
            file=sys.stderr,
        )
        return 2
    gold = json.loads(a.values.read_text(encoding="utf-8"))
    qs = yaml.safe_load(a.questions.read_text(encoding="utf-8"))["questions"]
    rows = compare(qs, gold, load_semantic_layer(), readonly_database_url())
    a.out.write_text(render(rows, gold, a.note), encoding="utf-8")
    print(f"wrote {a.out}: {sum(r['status'] == 'DIFF' for r in rows)} difference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
