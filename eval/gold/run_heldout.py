"""Compute gold values for the HELD-OUT questions (eval/questions/heldout.yaml) from a RAW file.

    python eval/gold/run_heldout.py --raw "C:\\path\\Online Retail.xlsx" --out heldout_values.json

The output is NOT committed (it is git-ignored): held-out answers must not sit in the repository
where a prompt-tuning step could read them. Imports only the standard library, pandas, PyYAML and
the sibling gold module t2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import t2  # noqa: E402

QUESTIONS = Path(__file__).parent.parent / "questions" / "heldout.yaml"


def compute(df, questions: list[dict]) -> dict:
    values = {}
    for q in questions:
        res = t2.GOLD_FUNCTIONS[q["gold_fn"]](df, **q.get("gold_args", {}))
        if res["kind"] == "scalar":
            values[q["id"]] = {"kind": "scalar", "value": str(res["value"])}
        else:
            values[q["id"]] = res
    return values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    a = ap.parse_args(argv)
    if not a.raw.exists():
        print(f"error: raw file not found: {a.raw}", file=sys.stderr)
        return 2
    qs = yaml.safe_load(a.questions.read_text(encoding="utf-8"))["questions"]
    df = t2.load_raw(a.raw)
    out = {
        "held_out": True,
        "raw_file": a.raw.name,
        "raw_sha256": hashlib.sha256(a.raw.read_bytes()).hexdigest(),
        "raw_rows": int(len(df)),
        "author_written": True,
        "values": compute(df, qs),
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(qs)} held-out gold values from {len(df)} raw rows); do not commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
