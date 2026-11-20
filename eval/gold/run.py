"""Compute gold values for eval/questions/dev.yaml from a RAW file.

    python eval/gold/run.py --raw "C:\\path\\Online Retail.xlsx" --out eval/gold/values.json

Imports only the standard library, pandas, PyYAML and the sibling gold module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
import t1  # noqa: E402

QUESTIONS = Path(__file__).parent.parent / "questions" / "dev.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "values.json")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    a = ap.parse_args(argv)
    if not a.raw.exists():
        print(f"error: raw file not found: {a.raw}", file=sys.stderr)
        return 2
    qs = yaml.safe_load(a.questions.read_text(encoding="utf-8"))["questions"]
    df = t1.load_raw(a.raw)
    values = {}
    for q in qs:
        fn = t1.GOLD_FUNCTIONS[q["gold_fn"]]
        values[q["id"]] = str(fn(df))
    out = {
        "raw_file": a.raw.name,
        "raw_sha256": hashlib.sha256(a.raw.read_bytes()).hexdigest(),
        "raw_rows": int(len(df)),
        "author_written": True,
        "values": values,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({len(values)} gold values from {len(df)} raw rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
