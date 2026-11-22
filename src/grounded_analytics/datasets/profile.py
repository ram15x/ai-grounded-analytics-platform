"""Deterministic profile of an uploaded table. No model is involved.

The profile states what is in the columns (types, nulls, ranges, samples, odd values). Anything
that guesses meaning is a *hint* and is marked as one: a person confirms the roles later.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

MAX_BYTES = 100 * 1024 * 1024  # refuse larger files; loading is a separate, bounded step
SAMPLE_VALUES = 5
SAMPLE_CHARS = 60
_BOOL = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}
_DMY = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:\s|$)")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_PHONE = re.compile(r"^\+?[\d\s().-]{8,16}$")
_ID_NAME = re.compile(r"(id|no|num|number|code|key|ref)$")
_AMOUNT_NAME = re.compile(
    r"amount|price|total|revenue|cost|value|sales|fee|charge|balance|qty|quantity|units|discount"
)


class ProfileError(ValueError):
    pass


@dataclass
class ColumnProfile:
    source_name: str
    name: str  # a safe, unique, snake_case name for use in SQL
    type: str  # integer | decimal | boolean | date | datetime | text | empty
    rows: int
    nulls: int
    null_rate: float
    distinct: int
    min: str | None = None
    max: str | None = None
    mean: float | None = None
    negatives: int | None = None
    zeros: int | None = None
    samples: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)  # guesses, never facts
    issues: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    source_file: str
    rows: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_names(raw: list[Any]) -> list[str]:
    """snake_case, ascii, starts with a letter, unique. Never empty."""
    seen: dict[str, int] = {}
    out = []
    for i, r in enumerate(raw):
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(r).strip()).strip("_").lower() or f"column_{i + 1}"
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()
        if not s[0].isalpha():
            s = "c_" + s
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.append(s if n == 0 else f"{s}_{n + 1}")
    return out


def read_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """CSV, TSV or XLSX, every cell read as given (strings stay strings)."""
    if not path.is_file():
        raise ProfileError(f"file not found: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ProfileError(f"file is larger than {MAX_BYTES // (1024 * 1024)} MB; not read")
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet, dtype=object)
    if ext in (".csv", ".tsv", ".txt"):
        last: Exception | None = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                head = path.read_bytes()[:4096].decode(enc, errors="replace")
                sep = "\t" if ext == ".tsv" else _sniff(head)
                return pd.read_csv(path, sep=sep, dtype=object, encoding=enc)
            except UnicodeDecodeError as exc:
                last = exc
        raise ProfileError(f"could not decode {path.name}: {last}")
    raise ProfileError(f"unsupported file type {ext!r}; use .csv, .tsv or .xlsx")


def _sniff(head: str) -> str:
    """The delimiter, chosen only among , ; tab and |. A single-column file gets a comma."""
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _clip(v: Any) -> str:
    s = str(v)
    return s if len(s) <= SAMPLE_CHARS else s[: SAMPLE_CHARS - 1] + "…"


def _day(ts: Any, kind: str) -> str:
    return str(ts.date()) if kind == "date" else str(ts)


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )


def _date_order(strings: pd.Series) -> tuple[bool | None, list[str]]:
    """(dayfirst, issues). dayfirst is None when the values do not settle it."""
    first_gt12 = second_gt12 = dmy = 0
    for v in strings:
        m = _DMY.match(str(v))
        if not m:
            continue
        dmy += 1
        first_gt12 += int(m.group(1)) > 12
        second_gt12 += int(m.group(2)) > 12
    if not dmy:
        return None, []
    if first_gt12 and second_gt12:
        return None, ["date_formats_conflict (some look day-first, some month-first)"]
    if first_gt12:
        return True, []
    if second_gt12:
        return False, []
    return None, ["date_order_ambiguous (day-first or month-first cannot be told from the values)"]


def profile_column(raw: pd.Series, source_name: str, name: str) -> ColumnProfile:
    rows = len(raw)
    s = raw.where(raw.notna(), None)
    s = s.map(lambda v: None if isinstance(v, str) and v.strip() == "" else v)
    present = s.dropna()
    nulls = rows - len(present)
    null_rate = round(nulls / rows, 4) if rows else 0.0
    cp = ColumnProfile(source_name, name, "empty", rows, nulls, null_rate,
                       int(present.astype(str).nunique()))  # fmt: skip
    if present.empty:
        cp.issues.append("all_values_missing")
        return cp
    cp.samples = [_clip(v) for v in present.astype(str).value_counts().head(SAMPLE_VALUES).index]
    strings = present.astype(str)
    if (strings != strings.str.strip()).any():
        cp.issues.append("leading_or_trailing_spaces")

    if all(isinstance(v, (pd.Timestamp,)) or hasattr(v, "year") for v in present.head(200)):
        dt = pd.to_datetime(present, errors="coerce")
        cp.type = "datetime" if (dt.dt.normalize() != dt).any() else "date"
        cp.min, cp.max = _day(dt.min(), cp.type), _day(dt.max(), cp.type)
    else:
        lowered = set(strings.str.lower().str.strip().unique())
        num = _numeric(present)
        ok = num.notna()
        if ok.all():
            cp.type = "integer" if (num % 1 == 0).all() else "decimal"
            cp.min, cp.max = str(num.min()), str(num.max())
            cp.mean = round(float(num.mean()), 4)
            cp.negatives, cp.zeros = int((num < 0).sum()), int((num == 0).sum())
            if lowered <= _BOOL and cp.distinct <= 2:
                cp.type = "boolean"
        elif lowered <= _BOOL and cp.distinct <= 2:
            cp.type = "boolean"
        else:
            has_sep = strings.str.contains(r"[-/.: ]", regex=True)
            dayfirst, order_issues = _date_order(strings)
            dt = pd.to_datetime(
                strings.where(has_sep), errors="coerce", format="mixed", dayfirst=bool(dayfirst)
            )
            if has_sep.mean() > 0.95 and dt.notna().sum() / len(strings) > 0.95:
                cp.type = (
                    "datetime" if (dt.dropna().dt.normalize() != dt.dropna()).any() else "date"
                )
                cp.min, cp.max = _day(dt.min(), cp.type), _day(dt.max(), cp.type)
                cp.issues += order_issues
                if dt.isna().any():
                    cp.issues.append(f"{int(dt.isna().sum())}_values_are_not_dates")
            else:
                cp.type = "text"
                share = float(ok.mean())
                if 0.5 <= share < 1.0:
                    bad = strings[~ok].head(3).map(_clip).tolist()
                    cp.issues.append(f"mixed_numbers_and_text (examples of text: {bad})")
                if (strings.str.len() > 0).any():
                    cp.min, cp.max = _clip(strings.min()), _clip(strings.max())
    cp.hints = _hints(cp, present)
    if cp.distinct == 1:
        cp.issues.append("constant_column")
    if cp.null_rate >= 0.5:
        cp.issues.append("more_than_half_missing")
    return cp


def _hints(cp: ColumnProfile, present: pd.Series) -> list[str]:
    h = []
    ratio = cp.distinct / max(len(present), 1)
    if cp.type in ("date", "datetime"):
        h.append("date")
    if cp.type in ("integer", "decimal") and _AMOUNT_NAME.search(cp.name):
        h.append("amount_or_quantity (from the name)")
    if _ID_NAME.search(cp.name) or (
        cp.type in ("integer", "text") and ratio >= 0.9 and cp.distinct > 20
    ):
        h.append("identifier")
    if (
        cp.type in ("text", "integer")
        and 2 <= cp.distinct <= min(50, max(20, int(0.05 * len(present))))
        and "identifier" not in h
    ):
        h.append("category")
    sample = present.astype(str).head(200)
    if cp.type == "text" and len(sample):
        if sample.map(lambda v: bool(_EMAIL.match(v.strip()))).mean() > 0.5:
            h.append("possible_personal_data: email addresses")
            h = [x for x in h if x != "category"]
        elif sample.map(lambda v: bool(_PHONE.match(v.strip()))).mean() > 0.5:
            h.append("possible_personal_data: phone numbers")
            h = [x for x in h if x != "category"]
    return h


def profile_frame(df: pd.DataFrame, source_file: str = "") -> DatasetProfile:
    if df.shape[1] == 0 or len(df) == 0:
        raise ProfileError("the table has no rows or no columns")
    names = safe_names(list(df.columns))
    cols = [profile_column(df[c], str(c), n) for c, n in zip(df.columns, names, strict=True)]
    dup = int(df.astype(str).duplicated().sum())
    issues = []
    if dup:
        issues.append(f"{dup} exact duplicate rows")
    renamed = [c.source_name for c in cols if c.source_name != c.name]
    if renamed:
        issues.append(f"{len(renamed)} column names were made safe for SQL")
    return DatasetProfile(source_file, len(df), cols, dup, issues)


def profile_file(path: Path, sheet: str | int = 0) -> DatasetProfile:
    return profile_frame(read_table(path, sheet), path.name)


def render_markdown(p: DatasetProfile) -> str:
    lines = [f"# Profile of {p.source_file}", "", f"{p.rows} rows, {len(p.columns)} columns.", ""]
    lines += [f"- {i}" for i in p.issues] + [""] if p.issues else []
    lines += [
        "| column | type | nulls | distinct | range | hints | issues |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in p.columns:
        rng = f"{c.min} to {c.max}" if c.min is not None else ""
        lines.append(
            f"| {c.name} | {c.type} | {c.null_rate:.1%} | {c.distinct} | {rng} | "
            f"{'; '.join(c.hints)} | {'; '.join(c.issues)} |"
        )
    lines += ["", "Hints are guesses from names and values. A person confirms the real meaning."]
    return "\n".join(lines) + "\n"
