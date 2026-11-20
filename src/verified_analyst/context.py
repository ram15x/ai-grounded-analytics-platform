"""Sentence-context binding for cell claims (extends ARCHITECTURE 4.1 step E).

A claim's `label` is written by the untrusted composer, so it cannot be the only thing that binds
a number to a row, a period or a metric. This module reads the PROSE around the number and checks
that what the sentence says the number is agrees with the cell it cites:

  1. row: if the cell is one row of a multi-row result, the sentence must name that row's
     dimension value (country, month, ...), and nearest-mention resolution must pick it;
  2. scope: a dimension value mentioned next to the number must be allowed by the query's
     filters, and a dimension the query neither groups nor filters by must not be named;
  3. metric: a metric term (name or synonym from the semantic layer) in the number's clause must
     resolve to the metric of the cited column.

Everything is deterministic string matching over the semantic layer's own vocabulary. It is a
heuristic with documented limits (docs/VERIFIER_LIMITS.md): it can be fooled by phrasing that puts
two row names around one number ("France and Germany together £X") and it cannot see synonyms the
layer does not list ("UK" for "United Kingdom"); in those cases it answers AMBIGUOUS, not VERIFIED.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

from .semantic import SemanticLayer, norm_term

AMBIGUOUS = "AMBIGUOUS"
UNSUPPORTED = "UNSUPPORTED"
METRIC_MISMATCH = "METRIC_MISMATCH"

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENT_END = re.compile(r"[.!?;](?=\s|$)|\n")
_CLAUSE_SPLIT = re.compile(r"[,;:](?=\s)|[.!?](?=\s|$)|\n")
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]  # fmt: skip
_MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ABBR["sept"] = 9
_MONTH_FULL = {m: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_RE = re.compile(
    r"\b(?P<name>" + "|".join([*MONTHS, "sept", *(m[:3] for m in MONTHS)]) + r")\b\.?"
    r"(?:\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(?P<year>(?:19|20)\d{2}))?",
    re.IGNORECASE,
)
_ISO_MONTH_RE = re.compile(r"\b(?P<y>(?:19|20)\d{2})-(?P<m>0[1-9]|1[0-2])(?:-(?P<d>[0-3]\d))?\b")


class ContextFail(Exception):
    def __init__(self, verdict: str, reason: str, detail: str = ""):
        super().__init__(reason)
        self.verdict, self.reason, self.detail = verdict, reason, detail


@dataclass(frozen=True)
class Mention:
    start: int
    end: int
    key: Any


def _stem(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def find_phrases(
    text: str, phrases: dict[tuple[str, ...], Any], offset: int = 0, stem: bool = False
) -> list[Mention]:
    """Longest-first, non-overlapping token-sequence matches of `phrases` in `text`."""
    toks = [(m.start(), m.end(), m.group().lower()) for m in _TOKEN.finditer(text)]
    if stem:
        toks = [(s, e, _stem(t)) for s, e, t in toks]
    maxlen = max((len(k) for k in phrases), default=0)
    out: list[Mention] = []
    i = 0
    while i < len(toks):
        hit = None
        for n in range(min(maxlen, len(toks) - i), 0, -1):
            key = tuple(t for _, _, t in toks[i : i + n])
            if key in phrases:
                hit = (n, phrases[key])
                break
        if hit:
            n, val = hit
            out.append(Mention(offset + toks[i][0], offset + toks[i + n - 1][1], val))
            i += n
        else:
            i += 1
    return out


def _span_bounds(prose: str, s: int, e: int, pattern: re.Pattern[str]) -> tuple[int, int]:
    start, end = 0, len(prose)
    for m in pattern.finditer(prose):
        if m.end() <= s:
            start = m.end()
        elif m.start() >= e:
            end = m.start()
            break
    return start, end


def sentence_span(prose: str, s: int, e: int) -> tuple[int, int]:
    return _span_bounds(prose, s, e, _SENT_END)


def clause_span(prose: str, s: int, e: int) -> tuple[int, int]:
    return _span_bounds(prose, s, e, _CLAUSE_SPLIT)


def _gap(prose: str, m: Mention, s: int, e: int) -> int:
    """Words between a mention and the number at [s, e) (0 if adjacent or overlapping)."""
    if m.end <= s:
        return len(_TOKEN.findall(prose[m.end : s]))
    if m.start >= e:
        return len(_TOKEN.findall(prose[e : m.start]))
    return 0


def resolve(prose: str, mentions: list[Mention], s: int, e: int, clause: tuple[int, int],
            sentence: tuple[int, int]) -> tuple[str, Any]:  # fmt: skip
    """Which key does the sentence attach to the number at [s, e)? ('one'|'none'|'many', key)."""
    inside = [m for m in mentions if clause[0] <= m.start and m.end <= clause[1]]
    if inside:
        keys = {m.key for m in inside}
        if len(keys) == 1:
            return "one", inside[0].key
        ranked = sorted(inside, key=lambda m: _gap(prose, m, s, e))
        if (
            _gap(prose, ranked[0], s, e) == _gap(prose, ranked[1], s, e)
            and ranked[0].key != ranked[1].key
        ):
            return "many", None
        return "one", ranked[0].key
    in_sent = [m for m in mentions if sentence[0] <= m.start and m.end <= sentence[1]]
    for group in ([m for m in in_sent if m.end <= s], [m for m in in_sent if m.start >= e]):
        keys = {m.key for m in group}
        if len(keys) == 1:
            return "one", group[0].key
        if len(keys) > 1:
            return "many", None
    return "none", None


# ---- dimension value mentions ------------------------------------------------------------------


def _as_date(v: Any) -> date | None:
    if isinstance(v, date):
        return v
    text = str(v).strip()[:10]
    if re.fullmatch(r"\d{4}-\d{2}", text):
        text += "-01"
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def day_mentions(text: str, offset: int) -> list[Mention]:
    """Full ISO dates only: key = datetime.date."""
    out = []
    for m in _ISO_MONTH_RE.finditer(text):
        if m["d"]:
            try:
                out.append(Mention(offset + m.start(), offset + m.end(),
                                   date(int(m["y"]), int(m["m"]), int(m["d"]))))  # fmt: skip
            except ValueError:
                continue
    return out


def month_mentions(text: str, offset: int) -> list[Mention]:
    """Mentions of a month: key = (year|None, month). ISO 2011-03 and 'March 2011' / 'March'."""
    out: list[Mention] = []
    taken: list[tuple[int, int]] = []
    for m in _ISO_MONTH_RE.finditer(text):
        if m["d"]:
            taken.append((m.start(), m.end()))  # a full date belongs to the `day` dimension
            continue
        out.append(Mention(offset + m.start(), offset + m.end(), (int(m["y"]), int(m["m"]))))
        taken.append((m.start(), m.end()))
    for m in _MONTH_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        name = m["name"].lower().rstrip(".")
        if name == "may" and not m["year"]:
            continue  # "may" is far more often a verb
        month = _MONTH_FULL.get(name) or _MONTH_ABBR.get(name[:4] if name == "sept" else name[:3])
        if month is None:
            continue
        year = int(m["year"]) if m["year"] else None
        out.append(Mention(offset + m.start(), offset + m.end(), (year, month)))
    return sorted(out, key=lambda x: x.start)


def _month_matches(mention_key: tuple[int | None, int], target: date) -> bool:
    y, mth = mention_key
    return mth == target.month and (y is None or y == target.year)


def _norm_value(v: Any) -> str:
    return norm_term(str(v))


def _dim_mentions(layer: SemanticLayer, dim: str, universe: Iterable[Any], text: str, offset: int):
    spec = layer.dimensions[dim]
    if spec.type == "date":
        return day_mentions(text, offset) if dim == "day" else month_mentions(text, offset)
    phrases = {}
    for v in universe:
        toks = tuple(_norm_value(v).split())
        if toks:
            phrases[toks] = _norm_value(v)
    return find_phrases(text, phrases, offset)


def _matches(layer: SemanticLayer, dim: str, key: Any, allowed: Iterable[Any]) -> bool:
    if layer.dimensions[dim].type == "date":
        dates = [d for a in allowed if (d := _as_date(a)) is not None]
        if dim == "day":
            return key in dates
        return any(_month_matches(key, d) for d in dates)
    return key in {_norm_value(a) for a in allowed}


def _filter_allowed(f: Any) -> list[Any] | None:
    if f.op == "eq":
        return [f.value]
    if f.op == "in":
        return list(f.value)
    return None


# ---- the check -----------------------------------------------------------------------------------


def check_cell_context(
    *,
    prose: str,
    start: int,
    end: int,
    layer: SemanticLayer,
    columns: list[str],
    rows: list[tuple[Any, ...]],
    row: tuple[Any, ...],
    row_key: dict[str, Any],
    metric_query: Any | None,
    metric: str | None,
    known_values: dict[str, set[str]] | None = None,
    number_spans: list[tuple[int, int]] | None = None,
) -> None:
    known_values = known_values or {}
    sent = sentence_span(prose, start, end)
    clause = clause_span(prose, start, end)
    filters = {f.dimension: f for f in (metric_query.filters if metric_query else [])}
    grouped = {c for c in columns if c in layer.dimensions}

    for dim in layer.dimensions:
        universe: set[Any] = set(known_values.get(dim, ()))
        if dim in columns:
            universe |= {r[columns.index(dim)] for r in rows}
        flt = filters.get(dim)
        allowed_by_filter = _filter_allowed(flt) if flt else None
        if allowed_by_filter:
            universe |= set(allowed_by_filter)
        if layer.dimensions[dim].type == "text" and not universe:
            continue
        mentions = _dim_mentions(layer, dim, universe, prose[sent[0] : sent[1]], sent[0])
        kind, key = resolve(prose, mentions, start, end, clause, sent)

        if dim in grouped:
            cited = row[columns.index(dim)]
            multi = len({r[columns.index(dim)] for r in rows}) > 1
            if dim in row_key or multi:
                if kind == "none":
                    if multi:
                        raise ContextFail(AMBIGUOUS, "row_not_named_in_sentence", dim)
                    continue
                if kind == "many":
                    raise ContextFail(AMBIGUOUS, "ambiguous_row_reference", dim)
                if not _matches(layer, dim, key, [cited]):
                    raise ContextFail(UNSUPPORTED, "row_mismatch", f"{dim}: sentence names {key}")
            continue
        if kind == "none":
            continue
        if kind == "many":
            raise ContextFail(AMBIGUOUS, "ambiguous_scope_reference", dim)
        if allowed_by_filter is None:
            raise ContextFail(UNSUPPORTED, "scope_not_in_query", f"{dim}: sentence names {key}")
        if not _matches(layer, dim, key, allowed_by_filter):
            raise ContextFail(UNSUPPORTED, "scope_mismatch", f"{dim}: sentence names {key}")

    if metric is not None:
        phrases: dict[tuple[str, ...], str] = {}
        for name, m in layer.metrics.items():
            for alias in (name, *m.synonyms):
                phrases[tuple(_stem(t) for t in norm_term(alias).split())] = name
        ms = find_phrases(prose[clause[0] : clause[1]], phrases, clause[0], stem=True)
        # a metric term belongs to the number it is closest to ("£5 from 99 orders": orders is 99's)
        others = [sp for sp in (number_spans or []) if sp != (start, end)]
        ms = [
            m for m in ms
            if all(_gap(prose, m, start, end) <= _gap(prose, m, a, b) for a, b in others)
        ]  # fmt: skip
        if ms:
            keys = {m.key for m in ms}
            pick = None
            if len(keys) == 1:
                pick = ms[0].key
            else:
                ranked = sorted(ms, key=lambda m: _gap(prose, m, start, end))
                if _gap(prose, ranked[0], start, end) == _gap(prose, ranked[1], start, end):
                    raise ContextFail(AMBIGUOUS, "ambiguous_metric_reference")
                pick = ranked[0].key
            if pick != metric:
                raise ContextFail(METRIC_MISMATCH, "sentence_metric_differs", f"{pick} vs {metric}")
