"""Alias adversarial answers and controls. "UK" is listed in semantic/metrics.yaml as an alias of
"United Kingdom"; the binder must treat the two spellings alike everywhere. Every LIE below reuses a
REAL cell, so the number check passes and only the sentence binding can stop it. Each prose has a
`{who}` slot filled with "UK" and with "United Kingdom"; both must give the same result.
All data is synthetic (see adversarial_cases.py)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D

from adversarial_cases import KNOWN, _claim, _q, cell, queries

KNOWN_K = {**KNOWN, "product_kind": {"merchandise", "non_merchandise"}}
WHO = ("UK", "United Kingdom")


def alias_queries():
    qs = dict(queries())
    qk = _q(
        "qk", ["net_revenue"], [], [(D("4444.44"),)],
        filters=[{"dimension": "country", "op": "eq", "value": "United Kingdom"}],
    )  # fmt: skip
    qku = _q(
        "qku", ["units_sold"], [], [(D("404"),)],
        filters=[{"dimension": "country", "op": "eq", "value": "United Kingdom"}],
    )  # fmt: skip
    qs["qk"], qs["qku"] = qk, qku
    return qs


@dataclass
class AliasCase:
    name: str
    prose: str  # contains {who}; numbers appear as text
    claims: list  # (id, text, value, unit, label, provenance)
    expect: str  # "reject" | "verify"


def cur(cid, text, value, label, prov):
    return (cid, text, value, "currency", label, prov)


def cnt(cid, text, value, label, prov):
    return (cid, text, value, "count", label, prov)


def case(name, prose, claims, expect):
    return AliasCase(name, prose, claims, expect)


UK, FR, DE = "United Kingdom", "France", "Germany"
NET = "net revenue"

LIES = [
    case(
        "alias-01 UK net revenue on a global total",
        "{who} net revenue was £9,000.09.",
        [cur("c1", "£9,000.09", "9000.09", NET, cell("qt", "net_revenue"))],
        "reject",
    ),
    case(
        "alias-02 UK net revenue on a France-filtered query",
        "{who} net revenue was £1,111.11.",
        [cur("c1", "£1,111.11", "1111.11", NET, cell("qf", "net_revenue"))],
        "reject",
    ),
    case(
        "alias-03 UK net revenue on a France+EIRE-filtered query",
        "{who} net revenue was £3,333.33.",
        [cur("c1", "£3,333.33", "3333.33", NET, cell("qi", "net_revenue"))],
        "reject",
    ),
    case(
        "alias-04 UK prose, France row",
        "{who} net revenue was £1,111.11.",
        [cur("c1", "£1,111.11", "1111.11", NET, cell("qc", "net_revenue", country=FR))],
        "reject",
    ),
    case(
        "alias-05 UK orders, Germany row",
        "{who} placed 22 orders.",
        [cnt("c1", "22", "22", "orders", cell("qc", "order_count", country=DE))],
        "reject",
    ),
    case(
        "alias-06 UK net revenue on a January-filtered query",
        "{who} net revenue was £5,101.01.",
        [cur("c1", "£5,101.01", "5101.01", NET, cell("qj", "net_revenue"))],
        "reject",
    ),
    case(
        "alias-07 UK merchandise units on a global total",
        "{who} merchandise units were 777.",
        [cnt("c1", "777", "777", "units", cell("qt", "units_sold"))],
        "reject",
    ),
    case(
        "alias-08 UK units, Germany row",
        "{who} units were 202.",
        [cnt("c1", "202", "202", "units", cell("qc", "units_sold", country=DE))],
        "reject",
    ),
    case(
        "alias-09 UK gross revenue, net revenue cell of the UK row",
        "{who} gross revenue was £4,444.44.",
        [cur("c1", "£4,444.44", "4444.44", "gross revenue", cell("qc", "net_revenue", country=UK))],
        "reject",
    ),
    case(
        "alias-10 swapped pair with the alias",
        "France made £4,444.44, and {who} made £1,111.11.",
        [
            cur("a", "£4,444.44", "4444.44", NET, cell("qc", "net_revenue", country=UK)),
            cur("b", "£1,111.11", "1111.11", NET, cell("qc", "net_revenue", country=FR)),
        ],
        "reject",
    ),
]

CONTROLS = [
    case(
        "alias-ctl-01 UK net revenue, UK row",
        "{who} net revenue was £4,444.44.",
        [cur("c1", "£4,444.44", "4444.44", NET, cell("qc", "net_revenue", country=UK))],
        "verify",
    ),
    case(
        "alias-ctl-02 UK net revenue, UK-filtered query",
        "{who} net revenue was £4,444.44.",
        [cur("c1", "£4,444.44", "4444.44", NET, cell("qk", "net_revenue"))],
        "verify",
    ),
    case(
        "alias-ctl-03 two rows, comma between",
        "France made £1,111.11, and {who} made £4,444.44, of net revenue.",
        [
            cur("a", "£1,111.11", "1111.11", NET, cell("qc", "net_revenue", country=FR)),
            cur("b", "£4,444.44", "4444.44", NET, cell("qc", "net_revenue", country=UK)),
        ],
        "verify",
    ),
    case(
        "alias-ctl-04 UK merchandise units, UK row",
        "{who} merchandise units were 404.",
        [cnt("c1", "404", "404", "units", cell("qc", "units_sold", country=UK))],
        "verify",
    ),
    case(
        "alias-ctl-05 UK merchandise units, UK-filtered query",
        "{who} merchandise units were 404.",
        [cnt("c1", "404", "404", "units", cell("qku", "units_sold"))],
        "verify",
    ),
]
ALL = [*LIES, *CONTROLS]


def prose_claims(case: AliasCase, who: str):
    prose = case.prose.format(who=who)
    return prose, [_claim(*c) for c in case.claims]


def template_claims(case: AliasCase, who: str):
    prose = case.prose.format(who=who)
    claims = []
    for cid, text, _value, unit, label, prov in case.claims:
        assert text in prose
        prose = prose.replace(text, "{" + cid + "}", 1)
        claims.append({"id": cid, "label": label, "unit": unit, "provenance": prov})
    return prose, claims
