"""Check the HELD-OUT questions against the semantic layer: which are expressible, and for those,
does the layer's number agree with the gold? (eval/questions/heldout.yaml, gold from
eval/gold/run_heldout.py)

    python eval/compare_heldout.py --values heldout_values.json

Needs GA_READONLY_DATABASE_URL (warehouse from `grounded load`). This is a metric-definition
check, not a prompt-tuning step: it prints only counts and, for a disagreement, the two numbers. It
never writes values into the repository (use --out to write a report elsewhere). It does not call
any model.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml

from grounded_analytics.compiler import MetricQuery, compile_query
from grounded_analytics.config import readonly_database_url
from grounded_analytics.guard import execute_guarded
from grounded_analytics.semantic import load_semantic_layer

ROOT = Path(__file__).parent


def _cell(v) -> str:
    if isinstance(v, date):
        return v.strftime("%Y-%m") if v.day == 1 else v.isoformat()
    return str(v)


def layer_answer(layer, url: str, q: dict) -> dict:
    """Run the intended plan and shape the result like the gold: a scalar or a table. Derived
    claims (percent change, ratio, difference) are computed here exactly as the verifier defines
    them, from cells of the one result."""
    mq = MetricQuery.model_validate(q["metric_query"])
    cq = compile_query(layer, mq)
    res = execute_guarded(url, cq.sql)
    cols, rows = res.columns, res.rows
    derived = q.get("needs_derived_claim", "")
    if derived.startswith("percent_change"):
        a, b = (Decimal(str(r[cols.index("gross_revenue")])) for r in rows)
        return {"kind": "scalar", "value": (b - a) / abs(a) * 100}
    if derived.startswith("ratio"):
        r = rows[0]
        return {
            "kind": "scalar",
            "value": Decimal(str(r[cols.index("units_sold")]))
            / Decimal(str(r[cols.index("order_count")])),
        }
    if derived.startswith("difference"):
        r = rows[0]
        return {
            "kind": "scalar",
            "value": Decimal(str(r[cols.index("gross_revenue")]))
            - Decimal(str(r[cols.index("identified_gross_revenue")])),
        }
    if q["answer_shape"] == "scalar":
        return {"kind": "scalar", "value": Decimal(str(rows[0][cols.index(mq.metrics[0])]))}
    return {"kind": "table", "columns": cols, "rows": [[_cell(v) for v in r] for r in rows]}


def _num_eq(a: str, b: str, decimals: int) -> bool:
    try:
        return abs(Decimal(a) - Decimal(b)) <= Decimal(1).scaleb(-decimals) / 2
    except Exception:
        return a == b


def agrees(got: dict, gold: dict, decimals: int) -> bool:
    if gold["kind"] == "scalar":
        return got["kind"] == "scalar" and _num_eq(str(got["value"]), gold["value"], decimals)
    if got["kind"] != "table" or len(got["rows"]) != len(gold["rows"]):
        return False

    def same_row(g: list[str], w: list[str]) -> bool:
        return (
            len(g) == len(w)
            and g[0] == w[0]
            and all(_num_eq(x, y, decimals) for x, y in zip(g[1:], w[1:], strict=True))
        )

    return all(same_row(g, w) for g, w in zip(got["rows"], gold["rows"], strict=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=Path, required=True)
    ap.add_argument("--questions", type=Path, default=ROOT / "questions" / "heldout.yaml")
    ap.add_argument("--out", type=Path, help="optional: write the table here (not into the repo)")
    a = ap.parse_args(argv)
    gold = json.loads(a.values.read_text(encoding="utf-8"))
    qs = yaml.safe_load(a.questions.read_text(encoding="utf-8"))["questions"]
    layer, url = load_semantic_layer(), readonly_database_url()
    lines, n_agree, n_diff, n_not = [], 0, 0, 0
    for q in qs:
        if not q["expressible_as_metric_query"]:
            n_not += 1
            lines.append(f"{q['id']}: not expressible ({q['not_expressible_reason'][:90]}...)")
            continue
        got = layer_answer(layer, url, q)
        ok = agrees(got, gold["values"][q["id"]], q["decimals"])
        n_agree, n_diff = n_agree + ok, n_diff + (not ok)
        if ok:
            lines.append(f"{q['id']}: expressible, layer agrees with gold")
        else:
            lines.append(f"{q['id']}: expressible, layer DIFFERS from gold: layer {got} gold "
                         f"{gold['values'][q['id']]}")  # fmt: skip
    text = "\n".join(lines) + (
        f"\nexpressible {n_agree + n_diff}/{len(qs)}; agree {n_agree}; differ {n_diff}; "
        f"not expressible {n_not}\n"
    )
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 1 if n_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())
