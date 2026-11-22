"""Deterministic number extraction from answer prose (ARCHITECTURE section 4.1, step B).

`extract_numbers(text)` returns every numeric expression it can find, each with a
status:
  ok         parsed; the verifier must tie it to a claim
  ambiguous  looks numeric but could not be classified safely; never silently dropped
  excluded   recognised as a date, year, ordinal, identifier, list marker, reference...
             (kept in the result so the UI and tests can see why it was not checked)

Known limits are listed in docs/EXTRACTOR.md. Everything here is rule based: no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal

Status = Literal["ok", "ambiguous", "excluded"]


@dataclass(frozen=True)
class ExtractedNumber:
    start: int
    end: int
    text: str
    status: Status
    reason: str | None = None
    value: Decimal | None = None  # mantissa * scale, sign applied (percent stays 12.5)
    mantissa: Decimal | None = None
    scale: Decimal = Decimal(1)
    decimals: int = 0  # digits shown after the decimal point in the mantissa
    unit: str = "number"  # number | currency | percent | percentage_points | basis_points
    currency: str | None = None
    qualifier: str | None = None  # approx | lower_bound | upper_bound
    direction: str | None = None  # up | down (from words like "fell by")
    kind: str = "plain"  # plain | range_low | range_high | fraction | spelled
    range_id: int | None = None

    @property
    def quantum(self) -> Decimal:
        """Smallest step of the displayed precision, in base units."""
        return (Decimal(1).scaleb(-self.decimals)) * self.scale


@dataclass(frozen=True)
class Extraction:
    text: str
    items: list[ExtractedNumber]

    @property
    def ok(self) -> list[ExtractedNumber]:
        return [i for i in self.items if i.status == "ok"]

    @property
    def ambiguous(self) -> list[ExtractedNumber]:
        return [i for i in self.items if i.status == "ambiguous"]

    @property
    def excluded(self) -> list[ExtractedNumber]:
        return [i for i in self.items if i.status == "excluded"]


# ------------------------------------------------------------------ vocabulary ------

MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
CURRENCY_SYMBOLS = {"£": "GBP", "$": "USD", "€": "EUR", "¥": "JPY", "₹": "INR"}
CURRENCY_CODES = {"GBP", "USD", "EUR", "INR", "JPY", "CNY", "AUD", "CAD", "CHF"}
CURRENCY_WORDS = {
    "pound": "GBP", "pounds": "GBP", "quid": "GBP", "sterling": "GBP",
    "dollar": "USD", "dollars": "USD", "euro": "EUR", "euros": "EUR",
    "rupee": "INR", "rupees": "INR",
}  # fmt: skip
SCALE_WORDS = {
    "thousand": Decimal(10**3), "million": Decimal(10**6), "billion": Decimal(10**9),
    "trillion": Decimal(10**12), "lakh": Decimal(10**5), "lakhs": Decimal(10**5),
    "crore": Decimal(10**7), "crores": Decimal(10**7),
}  # fmt: skip
SCALE_SUFFIX = {
    "k": Decimal(10**3), "K": Decimal(10**3), "M": Decimal(10**6), "MM": Decimal(10**6),
    "mn": Decimal(10**6), "bn": Decimal(10**9), "B": Decimal(10**9), "Bn": Decimal(10**9),
    "m": Decimal(10**6), "b": Decimal(10**9),
}  # fmt: skip
APPROX = ("about", "around", "approximately", "approx", "roughly", "circa", "nearly",
          "almost", "~", "≈", "c.")  # fmt: skip
LOWER = ("more than", "at least", "over", "above", "exceeding", "upwards of", "no fewer than",
         "greater than", "≥", ">=", ">")  # fmt: skip
UPPER = ("fewer than", "less than", "at most", "under", "below", "up to", "no more than",
         "≤", "<=", "<")  # fmt: skip
UP_WORDS = {"grew", "grow", "grown", "growth", "rose", "rise", "risen", "up", "increase",
            "increased", "increasing", "gain", "gained", "climbed", "jumped", "surged",
            "higher", "improved", "expanded"}  # fmt: skip
DOWN_WORDS = {"fell", "fall", "fallen", "dropped", "drop", "declined", "decline", "decrease",
              "decreased", "decreasing", "down", "lower", "reduction", "reduced", "shrank",
              "dipped", "plunged", "slumped", "lost", "loss", "worsened", "contracted"}  # fmt: skip

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}  # fmt: skip
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}  # fmt: skip
NUMBER_WORDS = set(UNITS) | set(TENS) | {"hundred"} | set(SCALE_WORDS) | {"and", "point", "a", "an"}
ONE_FOLLOWERS = {
    "customer", "customers", "product", "products", "order", "orders", "country", "countries",
    "month", "months", "year", "years", "unit", "units", "invoice", "invoices", "item", "items",
    "day", "days", "week", "weeks", "percent", "pound", "pounds", "market", "markets",
    "region", "regions", "category", "categories", "sku", "skus", "supplier", "suppliers",
}  # fmt: skip

NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CODES = "GBP|USD|EUR|INR|JPY|CNY|AUD|CAD|CHF"
NUM_RE = re.compile(
    rf"(?:(?<![\w.,])|(?<=(?<![A-Za-z])(?:{_CODES})))" rf"{NUM}(?![\d])|(?<![\w.,])\.\d+(?!\d)"
)

URL_RE = re.compile(r"https?://\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
DATE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"), "date"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "date"),
    (re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"), "date"),
    (re.compile(r"\b\d{4}-(?:0[1-9]|1[0-2])\b(?!-)"), "date"),
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?(?!\d)"), "time"),
    (re.compile(rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"), "date"),
    (re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTHS}\b(?:,?\s+\d{{4}})?"), "date"),
    (re.compile(rf"\b{MONTHS}\.?,?\s+\d{{4}}\b"), "date"),
    (re.compile(r"\bQ[1-4]\b(?:\s*['’]?\d{2,4})?", re.IGNORECASE), "period_label"),
    (re.compile(r"\bFY\s?['’]?\d{2,4}\b", re.IGNORECASE), "period_label"),
    (re.compile(r"\b[HQ][1-4]\b(?:\s*['’]?\d{2,4})?"), "period_label"),
]
IDENT_TOKEN_RE = re.compile(
    r"\b(?!(?:GBP|USD|EUR|INR|JPY|CNY|AUD|CAD|CHF)\d)[A-Za-z]+(?:[-_]\d|\d)[A-Za-z0-9_-]*(?:\.\d+)*\b"
)
ORDINAL_RE = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"\b(?:step|table|figure|fig|page|pages|section|chapter|row|column|query|passage|source|"
    r"appendix|slide|line|footnote|note|rank)\s*\.?\s*#?\s*\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
ID_CONTEXT_RE = re.compile(
    r"\b(?:invoice|order|customer|client|account|product|sku|stock\s*code|code|ref|reference|"
    r"po|ticket|id)\s*(?:no\.?|number|num|id)?\s*[:#]?\s*[A-Za-z]?\d[\dA-Za-z-]*",
    re.IGNORECASE,
)
HASH_RE = re.compile(r"#\s?\d+")
CITATION_RE = re.compile(r"\[\d+(?:\s*[,-]\s*\d+)*\]")
LIST_MARKER_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[.)](?=\s)")
SLASH_FRACTION_RE = re.compile(r"(?<![\d/.])(\d{1,3})/(\d{1,3})(?![\d/])")

PERCENT_AFTER = re.compile(
    r"^\s?(?:%|percent(?:age)?\b(?!\s+points?)|per\s?cent\b|pct\b)", re.IGNORECASE
)
POINTS_AFTER = re.compile(r"^\s?(?:percentage\s+points?|pp\b|ppt\b)", re.IGNORECASE)
BPS_AFTER = re.compile(r"^\s?(?:bps\b|basis\s+points?)", re.IGNORECASE)
SCALE_WORD_AFTER = re.compile(
    r"^\s?(thousand|million|billion|trillion|lakhs?|crores?)\b", re.IGNORECASE
)
SCALE_SUFFIX_AFTER = re.compile(r"^(mn|bn|MM|Bn|[kKMBmb])(?![A-Za-z])")
CURRENCY_WORD_AFTER = re.compile(
    r"^\s?(pounds?|quid|sterling|dollars?|euros?|rupees?)\b", re.IGNORECASE
)
CURRENCY_CODE_AFTER = re.compile(r"^\s?(GBP|USD|EUR|INR|JPY|CNY|AUD|CAD|CHF)\b")
CURRENCY_CODE_BEFORE = re.compile(r"(GBP|USD|EUR|INR|JPY|CNY|AUD|CAD|CHF)\s?$")
RANGE_SEP = re.compile(r"^\s*(?:-|–|—|to|and)\s*$", re.IGNORECASE)


class _Claims:
    """Tracks which character spans are already explained."""

    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []

    def free(self, start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in self.spans)

    def add(self, start: int, end: int) -> None:
        self.spans.append((start, end))


def _dec(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _qualifier(prefix: str) -> tuple[str | None, int]:
    """Qualifier phrase right before a number, and how many chars it consumed."""
    low = prefix.lower().rstrip()
    for table, name in ((LOWER, "lower_bound"), (UPPER, "upper_bound"), (APPROX, "approx")):
        for phrase in sorted(table, key=len, reverse=True):
            if low.endswith(phrase):
                before = low[: len(low) - len(phrase)]
                if phrase[0].isalpha() and before and before[-1].isalnum():
                    continue
                return name, len(prefix) - len(low) + len(phrase)
    return None, 0


_FILLERS = {"by", "of", "a", "an", "the", "about", "around", "roughly", "approximately",
            "nearly", "almost", "over", "under", "just", "only", "some", "as", "much"}  # fmt: skip


def _direction(prefix: str) -> str | None:
    """up/down only when a direction word sits directly before the number.

    "rose to 5m" (a level) and "up 12% from 1.2m (about 3 million" (a verb that belongs to a
    different number) give None.
    """
    tokens = prefix.split()
    for tok in reversed(tokens[-4:]):
        word = re.sub(r"[^A-Za-z]", "", tok).lower()
        if any(ch.isdigit() for ch in tok) or not word:
            return None
        if word == "to":
            return None
        if word in DOWN_WORDS:
            return "down"
        if word in UP_WORDS:
            return "up"
        if word not in _FILLERS:
            return None
    return None


# ------------------------------------------------------------------------- passes ----


def _excluded(items: list[ExtractedNumber], text: str, s: int, e: int, why: str) -> None:
    items.append(ExtractedNumber(s, e, text[s:e], "excluded", why))


def _exclusion_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    def run(pattern: re.Pattern[str], why: str, group: int = 0) -> None:
        for m in pattern.finditer(text):
            s, e = m.span(group)
            if claims.free(s, e):
                claims.add(s, e)
                _excluded(items, text, s, e, why)

    run(URL_RE, "url_or_email")
    for pat, why in DATE_PATTERNS:
        run(pat, why)
    run(CITATION_RE, "citation")
    run(LIST_MARKER_RE, "list_marker")
    run(REFERENCE_RE, "reference_label")
    run(ID_CONTEXT_RE, "identifier")
    run(HASH_RE, "identifier")
    run(IDENT_TOKEN_RE, "identifier")
    run(ORDINAL_RE, "ordinal")


def _fraction_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    for m in SLASH_FRACTION_RE.finditer(text):
        s, e = m.span()
        if not claims.free(s, e):
            continue
        claims.add(s, e)
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            items.append(ExtractedNumber(s, e, m.group(), "ambiguous", "zero_denominator"))
        elif a <= 31 and b <= 12:
            items.append(ExtractedNumber(s, e, m.group(), "ambiguous", "fraction_or_date"))
        else:
            val = Decimal(a) / Decimal(b)
            items.append(
                ExtractedNumber(
                    s, e, m.group(), "ok", value=val, mantissa=val, kind="fraction", decimals=4
                )  # fmt: skip
            )


WORD_FRACTION_RE = re.compile(
    r"\b(?:(?P<num>a|an|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]+)?"
    r"(?P<den>half|halves|thirds?|quarters?|fifths?|sixths?|eighths?|tenths?)\b"
    r"(?P<of>\s+of\b)?",
    re.IGNORECASE,
)
DEN = {"half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4, "fifth": 5,
       "fifths": 5, "sixth": 6, "sixths": 6, "eighth": 8, "eighths": 8, "tenth": 10,
       "tenths": 10}  # fmt: skip
NUMW = {"a": 1, "an": 1, **{k: v for k, v in UNITS.items() if 1 <= v <= 10}}


_HALF_FOLLOWERS = re.compile(
    r"^\s+(?:the|a|an|as|that|those|these|its|their|our|your|his|her|of)\b", re.IGNORECASE
)
_HALF_PRECEDERS = re.compile(
    r"\b(?:first|second|last|this|next|previous|other|fiscal|calendar|full|h)\s*$", re.IGNORECASE
)


def _bare_half_is_claim(text: str, s: int, e: int, m: re.Match[str]) -> bool:
    """'half of', 'half the total' are quantities; 'first half', 'half-year' are not."""
    if _HALF_PRECEDERS.search(text[max(0, s - 12) : s]):
        return False
    if m.group("of"):
        return True
    return bool(_HALF_FOLLOWERS.match(text[e:]))


def _word_fraction_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    for m in WORD_FRACTION_RE.finditer(text):
        s, e = m.span()
        den_word = m.group("den").lower()
        num_word = (m.group("num") or "").lower()
        if not claims.free(s, e):
            continue
        if not num_word and not (den_word == "half" and _bare_half_is_claim(text, s, e, m)):
            continue  # "the third largest", "last quarter", "first half" are not claims
        num = NUMW[num_word] if num_word else 1
        den = DEN[den_word]
        if num_word in ("a", "an") and den_word in ("quarter", "quarters"):
            before = text[max(0, s - 12) : s].lower()
            if re.search(r"(last|this|next|each|every|per|previous|current|that)\s*$", before):
                continue
        claims.add(s, e)
        frac = Fraction(num, den)
        val = Decimal(frac.numerator) / Decimal(frac.denominator)
        items.append(
            ExtractedNumber(
                s, e, m.group().strip(), "ok", value=val, mantissa=val, kind="fraction", decimals=4
            )  # fmt: skip
        )


SPELLED_RE = re.compile(
    r"\b(?:(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")(?:[\s-]+|\b))+",
    re.IGNORECASE,
)
_SCALE_ORDER = {"thousand": 3, "lakh": 5, "lakhs": 5, "million": 6, "crore": 7, "crores": 7,
                "billion": 9, "trillion": 12}  # fmt: skip


def _parse_prefix(words: list[str]) -> tuple[Decimal, int, Decimal, int] | None:
    """Parse the longest valid number at the start of `words`.

    Returns (value, decimals, scale, words_consumed) or None. Enforces English number
    grammar, so "one one" or "twenty twenty" are NOT merged into one number.
    """
    i, n = 0, len(words)
    if i < n and words[i] in ("a", "an"):
        if i + 1 < n and (words[i + 1] == "hundred" or words[i + 1] in SCALE_WORDS):
            words = ["one", *words[1:]]
        else:
            return None
    total, current = Decimal(0), Decimal(0)
    state = good_state = "start"
    last_scale = 99
    scale_used = Decimal(1)
    consumed = 0
    for idx, w in enumerate(words):
        if w == "zero":
            if idx == 0 and (len(words) == 1 or words[1] not in UNITS | TENS):
                return Decimal(0), 0, Decimal(1), 1
            break
        if w in UNITS and 1 <= UNITS[w] <= 9:
            if state not in ("start", "tens", "h", "and", "scale"):
                break
            current += UNITS[w]
            state = "u"
        elif w in UNITS:  # teens and ten
            if state not in ("start", "h", "and", "scale"):
                break
            current += UNITS[w]
            state = "teen"
        elif w in TENS:
            if state not in ("start", "h", "and", "scale"):
                break
            current += TENS[w]
            state = "tens"
        elif w == "hundred":
            if state not in ("u", "teen"):
                break
            current *= 100
            state = "h"
        elif w in SCALE_WORDS:
            if state not in ("u", "teen", "tens", "h") or _SCALE_ORDER[w] >= last_scale:
                break
            total += current * SCALE_WORDS[w]
            scale_used = SCALE_WORDS[w]
            last_scale = _SCALE_ORDER[w]
            current = Decimal(0)
            state = "scale"
        elif w == "and":
            if state not in ("h", "scale"):
                break
            state = "and"
            continue
        else:
            break
        consumed = idx + 1
        good_state = state
    if consumed == 0 or good_state in ("and", "start"):
        return None
    value = total + current
    scale = scale_used if (scale_used > 1 and current == 0) else Decimal(1)
    decimals = 0
    rest = words[consumed:]
    if rest[:1] == ["point"]:
        digits = []
        for d in rest[1:]:
            if d in UNITS and UNITS[d] <= 9:
                digits.append(str(UNITS[d]))
            else:
                break
        if digits:
            frac = Decimal("0." + "".join(digits))
            consumed += 1 + len(digits)
            decimals = len(digits)
            after = rest[1 + len(digits) :][:1]
            if after and after[0] in SCALE_WORDS and scale == 1:
                scale = SCALE_WORDS[after[0]]
                consumed += 1
                value = (value + frac) * scale
            elif scale > 1:
                value = (value / scale + frac) * scale
            else:
                value = value + frac
    return value, decimals, scale, consumed


def _spelled_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    for m in SPELLED_RE.finditer(text):
        toks = [(t.group().lower(), m.start() + t.start(), m.start() + t.end())
                for t in re.finditer(r"[A-Za-z]+", m.group())]  # fmt: skip
        i = 0
        while i < len(toks):
            words = [w for w, _, _ in toks[i:]]
            if words[0] in ("and", "point", "a", "an") and not (
                words[0] in ("a", "an")
                and len(words) > 1
                and (words[1] == "hundred" or words[1] in SCALE_WORDS)
            ):
                i += 1
                continue
            parsed = _parse_prefix(words)
            if parsed is None:
                if (
                    words[0] in UNITS
                    or words[0] in TENS
                    or words[0] in SCALE_WORDS
                    or words[0] == "hundred"
                ):
                    s0, e0 = toks[i][1], toks[i][2]
                    if claims.free(s0, e0):
                        claims.add(s0, e0)
                        items.append(ExtractedNumber(s0, e0, text[s0:e0], "ambiguous",
                                                     "unparseable_number_words"))  # fmt: skip
                i += 1
                continue
            value, decimals, scale, used = parsed
            s0, e0 = toks[i][1], toks[i + used - 1][2]
            i += used
            if not claims.free(s0, e0):
                continue
            if words[:used] == ["one"]:
                nxt = re.match(r"\s+([A-Za-z]+)", text[e0:])
                follower = nxt.group(1).lower() if nxt else ""
                if follower not in ONE_FOLLOWERS and not re.match(r"\s?[%£$€]", text[e0 : e0 + 2]):
                    continue  # "one of", "no one": pronoun or determiner use
            unit, end2, extras = _suffix_after(text, e0)
            prefix = text[:s0]
            qual, qlen = _qualifier(prefix)
            direction = _direction(prefix[: len(prefix) - qlen] if qual else prefix)
            claims.add(s0, end2)
            mantissa = value / scale if scale > 1 else value
            items.append(
                ExtractedNumber(
                    s0,
                    end2,
                    text[s0:end2],
                    "ok",
                    value=value,
                    mantissa=mantissa,
                    scale=scale,
                    decimals=decimals,
                    unit=unit,
                    currency=extras.get("currency"),
                    qualifier=qual,
                    direction=direction,
                    kind="spelled",
                )  # fmt: skip
            )


def _suffix_after(text: str, end: int) -> tuple[str, int, dict]:
    """Detect unit/scale words directly after a number. Returns (unit, new_end, extras)."""
    rest = text[end:]
    extras: dict = {}
    pos = end
    m = SCALE_WORD_AFTER.match(rest)
    if m:
        extras["scale"] = SCALE_WORDS[m.group(1).lower()]
        pos += m.end()
        rest = text[pos:]
    else:
        m = SCALE_SUFFIX_AFTER.match(rest)
        if m:
            extras["suffix"] = m.group(1)
            extras["scale"] = SCALE_SUFFIX[m.group(1)]
            pos += m.end()
            rest = text[pos:]
    unit = "number"
    for pat, name in ((POINTS_AFTER, "percentage_points"), (BPS_AFTER, "basis_points"),
                      (PERCENT_AFTER, "percent")):  # fmt: skip
        m = pat.match(rest)
        if m:
            unit, pos = name, pos + m.end()
            return unit, pos, extras
    m = CURRENCY_WORD_AFTER.match(rest) or CURRENCY_CODE_AFTER.match(rest)
    if m:
        word = m.group(1)
        extras["currency"] = CURRENCY_WORDS.get(word.lower(), word.upper())
        unit, pos = "currency", pos + m.end()
    return unit, pos, extras


def _is_year_context(before: str) -> bool:
    return bool(
        re.search(
            r"(?:\b(?:in|since|during|from|until|till|by|of|through|thru|to|and|between|year|"
            r"fy|fiscal|for|before|after|early|late|mid|end of|the)\s*|[,(]\s*)$",
            before.lower(),
        )
    )


@dataclass
class _Cand:
    s: int
    e: int
    m_start: int
    m_end: int
    item: ExtractedNumber | None


def _digit_candidate(text: str, m: re.Match[str], claims: _Claims) -> _Cand | None:  # noqa: PLR0911, PLR0912
    num_s, num_e = m.span()
    raw = m.group()
    letters = re.match(r"[A-Za-z]+", text[num_e:])
    # --- attached letters ---------------------------------------------------------
    if letters and not SCALE_SUFFIX_AFTER.match(text[num_e:]):
        suffix = letters.group()
        if suffix.lower() in ("st", "nd", "rd", "th"):
            return None  # ordinals are handled by the exclusion pass
        if suffix.lower() in ("pct", "pp", "ppt", "bps"):
            pass
        else:
            end = num_e + len(suffix)
            claims.add(num_s, end)
            identifier_like = len(raw) >= 4 and suffix.isupper() and len(suffix) <= 3
            status = "excluded" if identifier_like else "ambiguous"
            reason = "identifier" if identifier_like else "unknown_suffix"
            item = ExtractedNumber(num_s, end, text[num_s:end], status, reason)
            return _Cand(num_s, end, num_s, num_e, item)
    # --- prefix: sign, currency, qualifier -----------------------------------------
    start = num_s
    sign = 1
    cur: str | None = None
    prefix_end = num_s
    # currency symbol (optionally after a sign: -£5 or £-5 is rare, support -£5)
    j = prefix_end
    if j > 0 and text[j - 1] == " " and j > 1 and text[j - 2] in CURRENCY_SYMBOLS:
        j -= 1
    if j > 0 and text[j - 1] in CURRENCY_SYMBOLS:
        cur = CURRENCY_SYMBOLS[text[j - 1]]
        start = j - 1
    else:
        mcode = CURRENCY_CODE_BEFORE.search(text[:prefix_end])
        if mcode:
            cur = mcode.group(1)
            start = mcode.start()
    k = start
    if k > 0 and text[k - 1] in "-−" and (k == 1 or text[k - 2] in " \t\n(=:,;"):
        sign = -1
        start = k - 1
    elif k > 0 and text[k - 1] == "+" and (k == 1 or text[k - 2] in " \t\n(=:,;"):
        start = k - 1
    unit, end, extras = _suffix_after(text, num_e)
    parens = text[start - 1 : start] == "(" and text[end : end + 1] == ")"
    paren_span = None
    if parens and sign == 1 and cur is None and unit == "number" and not extras:
        # "(1,234)" is an accounting negative in finance but a plain aside in prose.
        paren_span = (start - 1, end + 1)
    scale = extras.get("scale", Decimal(1))
    currency = cur or extras.get("currency")
    if currency and unit == "number":
        unit = "currency"
    mant = _dec(raw if not raw.startswith(".") else "0" + raw)
    decimals = _decimals(raw)
    ambiguity = None
    if extras.get("suffix") in ("m", "b") and not currency:
        ambiguity = "lowercase_m_or_b_suffix"
    qual, qlen = _qualifier(text[:start])
    pre = text[: start - qlen] if qual else text[:start]
    direction = _direction(pre)  # also before a negative number: the verifier refuses that
    s0 = start - qlen if qual else start
    value = mant * scale * sign
    item = ExtractedNumber(
        s0, end, text[s0:end], "ok", value=value, mantissa=mant * sign, scale=scale,
        decimals=decimals, unit=unit, currency=currency, qualifier=qual, direction=direction,
    )  # fmt: skip
    if ambiguity:
        item = _replace(item, status="ambiguous", reason=ambiguity)
    if paren_span:
        item = _replace(
            item,
            start=paren_span[0],
            end=paren_span[1],
            text=text[paren_span[0] : paren_span[1]],
            status="ambiguous",
            reason="parenthesised_number",
        )
        s0, end = paren_span
    if (
        not currency
        and unit == "number"
        and scale == 1
        and decimals == 0
        and "," not in raw
        and 1900 <= int(mant) <= 2100
        and sign == 1
        and not qual
    ):
        if _is_year_context(text[:num_s]):
            item = ExtractedNumber(num_s, num_e, raw, "excluded", "year")
        else:
            item = ExtractedNumber(num_s, num_e, raw, "ambiguous", "possible_year")
    return _Cand(s0, end, num_s, num_e, item)


def _replace(item: ExtractedNumber, **changes) -> ExtractedNumber:
    from dataclasses import replace

    return replace(item, **changes)


EU_FORMAT_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:\.\d{3})+,\d{1,2}(?![\d])")
SPACE_GROUP_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\d])")
COMMA_DECIMAL_RE = re.compile(r"(?<![\w.,])\d+,\d{1,2}(?![\d,])")


def _format_ambiguity_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    patterns = (
        (EU_FORMAT_RE, "european_number_format"),
        (SPACE_GROUP_RE, "space_grouped_number"),
        (COMMA_DECIMAL_RE, "comma_decimal_or_list"),
    )
    for pat, why in patterns:
        for m in pat.finditer(text):
            s, e = m.span()
            if claims.free(s, e):
                claims.add(s, e)
                items.append(ExtractedNumber(s, e, m.group(), "ambiguous", why))


def _digit_pass(text: str, claims: _Claims, items: list[ExtractedNumber]) -> None:
    cands: list[_Cand] = []
    for m in NUM_RE.finditer(text):
        if not claims.free(*m.span()):
            continue
        c = _digit_candidate(text, m, claims)
        if c is not None:
            cands.append(c)
    # pair ranges: "10-20", "10 to 20", "between 10 and 20", "£1.2-1.5m"
    out: list[ExtractedNumber] = []
    i, rid = 0, 0
    while i < len(cands):
        a = cands[i]
        b = cands[i + 1] if i + 1 < len(cands) else None
        if (
            b is not None
            and a.item is not None and b.item is not None
            and _pairable(a.item) and _pairable(b.item)
            and RANGE_SEP.match(text[a.e : b.s])
            and _range_ok(text, a, b)
            and _compatible(a.item, b.item)
        ):  # fmt: skip
            ai, bi = a.item, b.item
            if _both_years(ai, bi):
                for it in (ai, bi):
                    out.append(ExtractedNumber(it.start, it.end, it.text, "excluded", "year_range"))
                claims.add(a.s, b.e)
            else:
                rid += 1
                lo, hi = _inherit(ai, bi)
                out.append(_replace(lo, kind="range_low", range_id=rid))
                out.append(_replace(hi, kind="range_high", range_id=rid))
            i += 2
            continue
        if a.item is not None:
            out.append(a.item)
        i += 1
    for it in out:
        claims.add(it.start, it.end)
    items.extend(out)


def _pairable(item: ExtractedNumber) -> bool:
    return item.status in ("ok", "ambiguous") or item.reason == "year"


def _compatible(a: ExtractedNumber, b: ExtractedNumber) -> bool:
    """Range ends may omit a unit, but may not state different ones ("5.2% to £1.1m")."""
    if "number" not in (a.unit, b.unit) and a.unit != b.unit:
        return False
    if a.currency and b.currency and a.currency != b.currency:
        return False
    return not (a.unit == "percent" and b.currency) and not (b.unit == "percent" and a.currency)


def _range_ok(text: str, a: _Cand, b: _Cand) -> bool:
    sep = text[a.e : b.s].strip().lower()
    if sep == "and":
        return bool(re.search(r"\bbetween\s+\S*\s*$", text[max(0, a.s - 25) : a.s].lower() + " "))
    return True


def _both_years(a: ExtractedNumber, b: ExtractedNumber) -> bool:
    def yr(x: ExtractedNumber) -> bool:
        return x.reason in ("year", "possible_year") or (
            x.mantissa is not None
            and x.unit == "number"
            and x.decimals == 0
            and 1900 <= abs(int(x.mantissa)) <= 2100
        )

    return yr(a) and yr(b)


def _inherit(lo: ExtractedNumber, hi: ExtractedNumber) -> tuple[ExtractedNumber, ExtractedNumber]:
    """Ends of "£1.2-1.5m" or "10-20%" share unit, currency and scale when only one states it."""
    unit = lo.unit if lo.unit != "number" else hi.unit
    if hi.unit == "number":
        unit = lo.unit
    currency = lo.currency or hi.currency
    scale = lo.scale if lo.scale != 1 else hi.scale
    qualifier = lo.qualifier or hi.qualifier
    out = []
    for it in (lo, hi):
        if it.status == "excluded" or it.mantissa is None:
            out.append(it)
            continue
        status, reason = it.status, it.reason
        if reason == "lowercase_m_or_b_suffix" and currency:
            status, reason = "ok", None
        new_scale = scale
        out.append(
            _replace(
                it,
                unit=unit,
                currency=currency,
                scale=new_scale,
                value=it.mantissa * new_scale,
                qualifier=it.qualifier or qualifier,
                status=status,
                reason=reason,
            )
        )
    return out[0], out[1]


def extract_numbers(text: str) -> Extraction:
    """Find every numeric expression in `text`. Items are ordered by position."""
    claims = _Claims()
    items: list[ExtractedNumber] = []
    _exclusion_pass(text, claims, items)
    _format_ambiguity_pass(text, claims, items)
    _fraction_pass(text, claims, items)
    _digit_pass(text, claims, items)
    _word_fraction_pass(text, claims, items)
    _spelled_pass(text, claims, items)
    items.sort(key=lambda i: (i.start, i.end))
    return Extraction(text, items)
