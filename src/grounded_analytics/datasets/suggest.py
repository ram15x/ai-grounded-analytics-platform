"""Model-drafted metric suggestions for an uploaded table.

The model sees column names, types, the role the person chose, the share of empty values, the
number of distinct values and at most three example values per column. Columns marked personal or
ignored are left out entirely. It never sees rows. Every suggestion is then checked against the
same closed grammar and the same profile checks as a hand-written metric; anything that does not
pass is dropped with its reason. Suggestions come back unapproved and without a known figure, so a
suggestion is published only after the person gives a figure from their own report and approves it.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from . import manifest as dm
from .profile import DatasetProfile

PROMPT_ID = "suggest.v0"
MAX_SUGGESTIONS = 8
NUMERIC = ("integer", "decimal", "float", "number", "numeric")

SYSTEM = """You help define business metrics for one table. Reply with JSON only:
{"metrics": [ ... at most 8 ... ]}
Each metric is ONE of:
  {"name": "snake_case", "description": "plain words",
   "aggregate": "count"|"sum"|"avg"|"min"|"max"|"count_distinct",
   "column": "<column name, omitted for count>",
   "filter": {"column": "<column>", "value": "<one example value>"} or null,
   "unit": "count"|"currency"|"ratio", "decimals": 0-4}
  {"name": "snake_case", "description": "plain words", "numerator": "<name of another metric here>",
   "denominator": "<name of another metric here>", "unit": "ratio", "decimals": 4}
Rules: use only the columns listed; sum and avg only on numeric columns; filter values must be one
of the example values shown; prefer the metrics a business owner of this table would ask about.
Do not invent figures. Do not write SQL."""


class SuggestError(ValueError):
    pass


def context(profile: DatasetProfile, man: dict[str, Any]) -> tuple[str, set[str]]:
    """The text the model sees, and the columns it may use."""
    cols = man.get("columns") or {}
    lines, usable = [], set()
    for c in profile.columns:
        spec = cols.get(c.name) or {}
        if spec.get("sensitivity") == "personal" or spec.get("role", "ignore") == "ignore":
            continue
        usable.add(c.name)
        samples = ", ".join(str(s) for s in (c.samples or [])[:3])
        lines.append(
            f"- {c.name}: type {c.type}, role {spec.get('role')}, {c.null_rate:.0%} empty, "
            f"{c.distinct} distinct, examples: {samples}"
        )
    head = (
        f"Table: {man.get('description') or 'not described'}. "
        f"One row is: {man.get('grain') or 'not said'}. "
        f"Currency: {man.get('currency') or 'none'}. Rows: {profile.rows}.\nColumns:\n"
    )
    return head + "\n".join(lines), usable


def _form_row(item: dict[str, Any]) -> dict[str, Any]:
    """A suggestion in the shape of one row of the upload form (never approved)."""
    f = item.get("filter") or {}
    row = {
        "name": str(item.get("name", "")).strip(),
        "description": str(item.get("description", "")).strip()[:200],
        "unit": item.get("unit", "count"),
        "decimals": item.get("decimals", 0),
        "known": "",
        "approve": False,
        "suggested": True,
    }
    if item.get("numerator") or item.get("denominator"):
        row |= {"aggregate": "ratio", "numerator": item.get("numerator", ""),
                "denominator": item.get("denominator", ""), "column": "",
                "filterCol": "", "filterVal": ""}  # fmt: skip
    else:
        row |= {
            "aggregate": item.get("aggregate", ""),
            "column": item.get("column") or "",
            "filterCol": f.get("column") or "",
            "filterVal": str(f.get("value") or ""),
        }
    return row


def _spec(row: dict[str, Any], currency: str) -> dict[str, Any]:
    unit = row["unit"]
    if unit == "currency":
        if not currency:
            raise SuggestError("money needs a currency code on the table first")
        unit = f"currency:{currency}"
    o: dict[str, Any] = {"status": "draft", "description": row["description"] or row["name"],
                         "unit": unit, "decimals": int(row["decimals"] or 0)}  # fmt: skip
    if row["aggregate"] == "ratio":
        o |= {"numerator": row["numerator"], "denominator": row["denominator"]}
    else:
        o["aggregate"] = row["aggregate"]
        if row["aggregate"] != "count":
            o["column"] = row["column"]
        if row["filterCol"]:
            o["where"] = [{"column": row["filterCol"], "op": "eq", "value": row["filterVal"]}]
    return o


def check(
    reply: str, profile: DatasetProfile, man: dict[str, Any], usable: set[str]
) -> dict[str, Any]:
    """Parse the model's reply and keep only suggestions that pass every rule."""
    try:
        items = json.loads(reply).get("metrics", [])
        if not isinstance(items, list):
            raise TypeError
    except (ValueError, TypeError, AttributeError) as exc:
        raise SuggestError("the model did not reply with the expected JSON") from exc
    taken = set((man.get("metrics") or {}).keys())
    kept: dict[str, dict[str, Any]] = {}
    dropped: list[str] = []
    # simple metrics first, so a ratio can name a suggestion that was kept
    ordered = sorted(
        (i for i in items[: MAX_SUGGESTIONS * 2] if isinstance(i, dict)),
        key=lambda i: bool(i.get("numerator") or i.get("denominator")),
    )
    for item in ordered:
        row = _form_row(item)
        name = row["name"] or "(unnamed)"
        if len(kept) >= MAX_SUGGESTIONS:
            break
        if name in taken or name in kept:
            dropped.append(f"{name}: a number with this name already exists")
            continue
        used = {row.get("column"), row.get("filterCol")} - {"", None}
        if row["aggregate"] != "ratio" and not used <= usable:
            dropped.append(f"{name}: uses a column that is personal, ignored or does not exist")
            continue
        try:
            spec = _spec(row, man.get("currency") or "")
            simple = {k: _spec(v, man.get("currency") or "") for k, v in kept.items()}
            trial = dict(man) | {
                "description": man.get("description") or "-",  # checked elsewhere, not here
                "grain": man.get("grain") or "-",
                "metrics": {**simple, name: spec},
            }
            problems = [
                p
                for p in dm.validate(dm.Manifest.model_validate(trial), profile)
                if p.startswith(f"{name}:") or name in p
            ]
        except (SuggestError, ValidationError, dm.ManifestError, ValueError, TypeError) as exc:
            msg = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            dropped.append(f"{name}: {msg}")
            continue
        if problems:
            dropped.append(problems[0])
            continue
        kept[name] = row
    return {"suggestions": list(kept.values()), "dropped": dropped}
