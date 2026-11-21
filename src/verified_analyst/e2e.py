"""End-to-end scoring of pipeline runs against gold values (pandas over the raw file).

A run is scored from what the reader would see:
  correct        verified, and a shown figure equals the gold value at the question's decimals
  verified_wrong verified, but no shown figure equals the gold value (the case the product exists
                 to prevent; any such row is a failure to investigate, not a statistic to tune)
  withheld       the system abstained or no figure could be verified
  no_reply       the model gave no usable reply (provider error, no recording)
No accuracy claim comes out of this: n is small and the questions are author-written.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def _round(x: Decimal, decimals: int) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def score(run: dict[str, Any], gold: str, decimals: int) -> str:
    if run["outcome"] == "not_recorded" or run.get("reason") == "provider_error":
        return "no_reply"
    if run["outcome"] != "verified":
        return "withheld"
    want = _round(Decimal(gold), decimals)
    verdicts = ((run.get("verification") or {}).get("verification") or {}).get("verdicts", [])
    shown = [
        p["true_value"]
        for p in ((run.get("verification") or {}).get("placeholders") or [])
        if p.get("true_value") is not None
    ]
    for v in shown:
        if _round(Decimal(v), decimals) == want:
            return "correct"
    return "verified_wrong" if verdicts or shown else "withheld"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {k: 0 for k in ("correct", "verified_wrong", "withheld", "no_reply")}
    for r in rows:
        counts[r["score"]] += 1
    answered = len(rows) - counts["no_reply"]
    lo, hi = wilson(counts["correct"], answered)
    return {
        "n": len(rows),
        "answered": answered,
        **counts,
        "correct_ci95": [round(lo, 3), round(hi, 3)],
    }


def render_md(summary: dict[str, Any], rows: list[dict[str, Any]], title: str) -> str:
    s = summary
    lines = [
        f"# {title}",
        "",
        f"n = {s['n']} questions, {s['answered']} with a model reply.",
        f"correct {s['correct']}, verified_wrong {s['verified_wrong']}, withheld {s['withheld']}, "
        f"no_reply {s['no_reply']}.",
        f"correct among answered: {s['correct']}/{s['answered']} "
        f"(Wilson 95% interval {s['correct_ci95'][0]} to {s['correct_ci95'][1]}).",
        "",
        "| id | expressible | score | question |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['id']} | {r['expressible']} | {r['score']} | {r['question']} |" for r in rows]
    lines += ["", "Author-written questions, one model, small n: not an accuracy claim."]
    return "\n".join(lines) + "\n"
