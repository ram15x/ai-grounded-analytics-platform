"""Placeholder composer contract (ADR-0008): the model writes `{c1}`, the system writes the number.

Renderer: each placeholder is replaced by a value formatted from the RECOMPUTED cell or derived
value (never from anything the composer typed), in a unit-aware format, with the displayed
precision recorded. Validator: unknown placeholders, unused claims, stray braces, placeholders that
merge with neighbouring characters, and any digit or spelled-out number outside a placeholder.
The rendered text then goes through the ordinary verifier (extractor, number check, label binding,
sentence-context binder) as a second, independent check.

No LLM, no network, no database.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from .claims import Claim, TemplateAnswer, TemplateClaim
from .extract import ExtractedNumber, extract_numbers
from .verifier import (
    UNSUPPORTED,
    ExecutedQuery,
    Report,
    ResolveFailure,
    ShadowFn,
    Verifier,
)

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
CURRENCY_SYMBOL = {"GBP": "£", "USD": "$", "EUR": "€", "INR": "₹", "JPY": "¥"}
SCALE = {
    "thousand": (Decimal(1_000), "k", " thousand"),
    "million": (Decimal(1_000_000), "m", " million"),
    "billion": (Decimal(1_000_000_000), "bn", " billion"),
}
DEFAULT_DECIMALS = {"currency": 2, "count": 0, "percent": 1, "percentage_points": 1,
                    "ratio": 4, "number": 2}  # fmt: skip
# extractor exclusions that may legitimately appear outside placeholders
ALLOWED_EXCLUSIONS = {"date", "time", "period_label", "year", "year_range", "ordinal",
                      "url_or_email", "list_marker"}  # fmt: skip
ID_EXCLUSIONS = {"identifier", "reference_label", "citation"}


@dataclass
class Issue:
    code: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass
class RenderedClaim:
    claim_id: str
    text: str
    start: int
    end: int
    true_value: Decimal
    shown_value: Decimal
    decimals: int
    scale: str
    unit: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id, "text": self.text, "span": [self.start, self.end],
            "true_value": str(self.true_value), "shown_value": str(self.shown_value),
            "decimals": self.decimals, "scale": self.scale, "unit": self.unit,
        }  # fmt: skip


@dataclass
class TemplateReport:
    rendered: str
    placeholders: list[RenderedClaim] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    report: Report = field(default_factory=Report)

    @property
    def status(self) -> str:
        return (
            "verified"
            if not self.issues and self.report.status == "verified"
            else "partially_verified"
        )

    def counts(self) -> dict[str, int]:
        out = dict(self.report.counts())
        for i in self.issues:
            out[f"ISSUE:{i.code}"] = out.get(f"ISSUE:{i.code}", 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rendered": self.rendered,
            "placeholders": [p.as_dict() for p in self.placeholders],
            "issues": [i.as_dict() for i in self.issues],
            "verification": self.report.as_dict(),
        }


# ---- formatting ----


def _quantize(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def format_value(value: Decimal, unit: str, currency: str | None, decimals: int,
                 scale: str) -> tuple[str, Decimal]:  # fmt: skip
    """Format `value` for display. Returns (text, shown_value in full units)."""
    if scale != "none" and unit not in ("currency", "count"):
        raise ValueError(f"scale {scale!r} is only allowed for currency and count")
    factor, suffix, word = SCALE.get(scale, (Decimal(1), "", ""))
    shown_scaled = _quantize(value / factor, decimals)
    shown = shown_scaled * factor
    neg = shown_scaled < 0
    body = f"{abs(shown_scaled):,.{decimals}f}"
    if unit == "currency":
        sym = CURRENCY_SYMBOL.get(currency or "")
        if sym is None:
            if not currency:
                raise ValueError("cannot infer a currency for this claim")
            text = f"{currency} {body}{suffix}"
        else:
            text = f"{sym}{body}{suffix}"
    elif unit in ("percent", "percentage_points"):
        text = f"{body}%" if unit == "percent" else f"{body} percentage points"
    else:
        text = f"{body}{word}"
    return ("-" if neg else "") + text, shown


# ---- helpers ----


def _currency_of(layer: Any, metrics: list[str]) -> str | None:
    for m in metrics:
        unit = layer.metrics[m].unit
        if unit.startswith("currency:"):
            return unit.split(":", 1)[1]
    return None


def _default_decimals(layer: Any, claim: TemplateClaim, metrics: list[str]) -> int:
    if claim.decimals is not None:
        return claim.decimals
    if claim.provenance.kind == "cell" and metrics:
        m = layer.metrics[metrics[0]]
        if m.unit_kind == "ratio" and claim.unit == "percent":
            return max(m.decimals - 2, 0)
        if claim.unit in ("currency", "count", "ratio", "number"):
            return m.decimals
    return DEFAULT_DECIMALS[claim.unit]


def _known_ids(queries: dict[str, ExecutedQuery]) -> set[str]:
    ids: set[str] = set()
    for q in queries.values():
        for row in q.rows:
            for v in row:
                if isinstance(v, str | int):
                    ids.add(str(v).upper())
    return ids


# ---- render + validate + verify ----


def verify_template(
    layer: Any,
    answer: TemplateAnswer,
    executed: dict[str, ExecutedQuery],
    shadow: ShadowFn | None = None,
    known_values: dict[str, set[str]] | None = None,
) -> TemplateReport:
    verifier = Verifier(layer, executed, None, shadow, known_values)
    issues: list[Issue] = []
    template = answer.template

    by_id: dict[str, TemplateClaim] = {}
    for c in answer.claims:
        if c.id in by_id:
            issues.append(Issue("duplicate_claim_id", c.id))
        by_id.setdefault(c.id, c)

    for ch in template:
        if not ch.isascii() and unicodedata.category(ch) in ("Nd", "No", "Nl"):
            issues.append(Issue("non_ascii_digit", repr(ch)))
            break
    leftover = PLACEHOLDER.sub("", template)
    if "{" in leftover or "}" in leftover:
        issues.append(Issue("stray_brace", "unbalanced or nested braces"))

    for m in PLACEHOLDER.finditer(template):
        before = template[m.start() - 1] if m.start() > 0 else " "
        after = template[m.end()] if m.end() < len(template) else " "
        nxt = template[m.end() + 1] if m.end() + 1 < len(template) else " "
        if (
            before.isalnum()
            or before in "}£$€₹¥+-−~"
            or after.isalnum()
            or after in "{%"
            or (after in ".," and nxt.isdigit())
        ):
            issues.append(
                Issue("placeholder_adjacent_text", f"{m.group(0)} touches {before!r}/{after!r}")
            )

    out: list[str] = []
    placed: list[RenderedClaim] = []
    synth: list[Claim] = []
    used: set[str] = set()
    pos = 0
    seen: dict[str, int] = {}
    for m in PLACEHOLDER.finditer(template):
        out.append(template[pos : m.start()])
        pos = m.end()
        cid = m.group(1).strip()
        claim = by_id.get(cid)
        start = sum(len(x) for x in out)
        if claim is None:
            issues.append(Issue("unknown_placeholder", cid))
            out.append(f"[?{cid}]")
            continue
        used.add(cid)
        try:
            res = verifier.resolve(claim.provenance)
            currency = _currency_of(layer, res.metrics)
            true_value = res.value
            if (
                claim.provenance.kind == "cell"
                and res.metrics
                and layer.metrics[res.metrics[0]].unit_kind == "ratio"
                and claim.unit == "percent"
            ):
                true_value = res.value * 100
            decimals = _default_decimals(layer, claim, res.metrics)
            text, shown = format_value(true_value, claim.unit, currency, decimals, claim.scale)
        except ResolveFailure as f:
            issues.append(
                Issue("placeholder_unresolvable", f"{cid}: {f.reason} {f.detail}".strip())
            )
            out.append(f"[{cid} unavailable]")
            continue
        except ValueError as exc:
            issues.append(Issue("placeholder_unformattable", f"{cid}: {exc}"))
            out.append(f"[{cid} unavailable]")
            continue
        rt = extract_numbers(text).ok
        if len(rt) != 1 or rt[0].value != shown or (rt[0].start, rt[0].end) != (0, len(text)):
            issues.append(Issue("render_round_trip_failed", f"{cid}: {text!r}"))
        out.append(text)
        seen[cid] = seen.get(cid, 0) + 1
        placed.append(RenderedClaim(cid, text, start, start + len(text), true_value, shown,
                                    decimals, claim.scale, claim.unit))  # fmt: skip
        synth.append(Claim.model_validate({
            "id": cid if seen[cid] == 1 else f"{cid}#{seen[cid]}",
            "text": text, "value": shown, "unit": claim.unit, "label": claim.label,
            "provenance": claim.provenance.model_dump(),
        }))  # fmt: skip
    out.append(template[pos:])
    rendered = "".join(out)

    for cid in by_id:
        if cid not in used:
            issues.append(Issue("unused_claim", cid))

    extraction = extract_numbers(rendered)
    spans = {(p.start, p.end) for p in placed}
    for p in placed:
        hit = [i for i in extraction.items if i.start < p.end and p.start < i.end]
        if not any((i.start, i.end) == (p.start, p.end) and i.status == "ok" for i in hit):
            seen_txt = ", ".join(repr(i.text) for i in hit) or "nothing"
            issues.append(
                Issue("placeholder_merged_with_text", f"{p.claim_id}: extractor saw {seen_txt}")
            )

    known = _known_ids(executed)
    covered = [(i.start, i.end) for i in extraction.items if i.status != "excluded"]
    covered += [(p.start, p.end) for p in placed]
    for item in extraction.excluded:
        if (item.start, item.end) in spans:
            continue
        reason = item.reason or ""
        if reason in ALLOWED_EXCLUSIONS:
            continue
        if reason in ID_EXCLUSIONS:
            toks = re.findall(r"[A-Za-z0-9]*\d[A-Za-z0-9]*", item.text)
            if toks and all(t.upper() in known for t in toks):
                continue
            issues.append(Issue("identifier_not_in_results", item.text))
            continue
        issues.append(Issue("digits_in_unrecognised_form", item.text))
    skip = [(i.start, i.end) for i in extraction.excluded] + covered
    for d in re.finditer(r"[0-9]+", rendered):
        if not any(a <= d.start() < b for a, b in skip):
            issues.append(
                Issue("digit_outside_placeholder", rendered[max(0, d.start() - 10) : d.end() + 10])
            )
            break

    inner = verifier.verify(rendered, synth)
    for v in inner.verdicts:
        if v.verdict == UNSUPPORTED and v.reason == "no_claim":
            v.reason = "number_outside_placeholder"
    return TemplateReport(rendered, placed, issues, inner)


def is_template_doc(data: dict) -> bool:
    return "template" in data and "prose" not in data


__all__ = [
    "TemplateReport",
    "format_value",
    "is_template_doc",
    "verify_template",
    "ExtractedNumber",
]
