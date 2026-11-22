"""30 hand-built adversarial answers. Every one is a LIE that reuses a REAL number from the
executed results, so the number check alone passes: the lie is in what the prose says the number
is (wrong row, wrong period, wrong dimension value, wrong metric of the same unit).

A case counts as a FALSE ACCEPT when the verifier reports status 'verified' for it.
All data below is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal as D

from grounded_analytics.claims import Claim
from grounded_analytics.compiler import MetricQuery, compile_query
from grounded_analytics.semantic import load_semantic_layer
from grounded_analytics.verifier import ExecutedQuery

LAYER = load_semantic_layer()
COUNTRIES = ["France", "Germany", "EIRE", "United Kingdom", "Spain"]


def _q(qid, metrics, dims, rows, **kw):
    mq = MetricQuery(metrics=metrics, dimensions=dims, **kw)
    cq = compile_query(LAYER, mq)
    return ExecutedQuery(qid, cq.sql, cq.columns, rows, metric_query=mq)


def queries() -> dict[str, ExecutedQuery]:
    by_country = _q(
        "qc", ["net_revenue", "gross_revenue", "order_count", "units_sold"], ["country"],
        [("France", D("1111.11"), D("1211.11"), D("11"), D("101")),
         ("Germany", D("2222.22"), D("2322.22"), D("22"), D("202")),
         ("EIRE", D("3333.33"), D("3433.33"), D("33"), D("303")),
         ("United Kingdom", D("4444.44"), D("4544.44"), D("44"), D("404"))],
    )  # fmt: skip
    by_month = _q(
        "qm", ["net_revenue"], ["month"],
        [(date(2011, 1, 1), D("5101.01")), (date(2011, 2, 1), D("5202.02")),
         (date(2011, 3, 1), D("5303.03"))],
    )  # fmt: skip
    total = _q(
        "qt", ["net_revenue", "gross_revenue", "cancelled_revenue", "discount_amount", "aov",
               "revenue_per_customer", "order_count", "active_customers", "units_sold"], [],
        [(D("9000.09"), D("9500.95"), D("400.04"), D("100.01"), D("90.09"), D("180.18"),
          D("100"), D("50"), D("777"))],
    )  # fmt: skip
    filt = _q(
        "qf",
        ["net_revenue"],
        [],
        [(D("1111.11"),)],
        filters=[{"dimension": "country", "op": "eq", "value": "France"}],
    )
    filt_in = _q(
        "qi",
        ["net_revenue"],
        [],
        [(D("3333.33"),)],
        filters=[{"dimension": "country", "op": "in", "value": ["France", "EIRE"]}],
    )
    filt_month = _q(
        "qj",
        ["net_revenue"],
        [],
        [(D("5101.01"),)],
        filters=[{"dimension": "month", "op": "eq", "value": "2011-01"}],
    )
    return {q.id: q for q in (by_country, by_month, total, filt, filt_in, filt_month)}


KNOWN = {"country": set(COUNTRIES)}


def cell(qid, col, **row):
    d = {"kind": "cell", "query_id": qid, "column": col}
    if row:
        d["row_key"] = row
    return d


@dataclass
class Case:
    name: str
    category: str
    prose: str
    claims: list[Claim]


def _claim(cid, text, value, unit, label, prov):
    return Claim.model_validate({"id": cid, "text": text, "value": value, "unit": unit,
                                 "label": label, "provenance": prov})  # fmt: skip


def gbp(t, v, label, prov, cid="c1"):
    return _claim(cid, t, v, "currency", label, prov)


def cases() -> list[Case]:
    out: list[Case] = []

    def add(name, cat, prose, *claims):
        out.append(Case(name, cat, prose, list(claims)))

    # ---- wrong row (8): the number is real but belongs to a different row ----
    add(
        "row-01 Germany prose, France cell",
        "wrong_row",
        "Germany net revenue was £1,111.11.",
        gbp("£1,111.11", "1111.11", "net revenue", cell("qc", "net_revenue", country="France")),
    )
    add(
        "row-02 UK prose, EIRE cell",
        "wrong_row",
        "United Kingdom net revenue was £3,333.33.",
        gbp("£3,333.33", "3333.33", "net revenue", cell("qc", "net_revenue", country="EIRE")),
    )
    add(
        "row-03 label names the cited row, prose names another",
        "wrong_row",
        "Germany net revenue was £1,111.11.",
        gbp(
            "£1,111.11",
            "1111.11",
            "France net revenue",
            cell("qc", "net_revenue", country="France"),
        ),
    )
    add(
        "row-04 swapped pair",
        "wrong_row",
        "France net revenue was £2,222.22 and Germany net revenue was £1,111.11.",
        gbp(
            "£2,222.22", "2222.22", "net revenue", cell("qc", "net_revenue", country="Germany"), "a"
        ),
        gbp(
            "£1,111.11", "1111.11", "net revenue", cell("qc", "net_revenue", country="France"), "b"
        ),
    )
    add(
        "row-05 order count of another country",
        "wrong_row",
        "EIRE placed 11 orders.",
        _claim(
            "c", "11", "11", "count", "order count", cell("qc", "order_count", country="France")
        ),
    )
    add(
        "row-06 units of another country",
        "wrong_row",
        "Germany sold 404 units.",
        _claim(
            "c",
            "404",
            "404",
            "count",
            "units sold",
            cell("qc", "units_sold", country="United Kingdom"),
        ),
    )
    add(
        "row-07 best country is actually someone else",
        "wrong_row",
        "The top country, France, made £4,444.44 of net revenue.",
        gbp(
            "£4,444.44",
            "4444.44",
            "net revenue",
            cell("qc", "net_revenue", country="United Kingdom"),
        ),
    )
    add(
        "row-08 row never named in the sentence",
        "wrong_row",
        "Revenue for that market was £2,222.22.",
        gbp("£2,222.22", "2222.22", "net revenue", cell("qc", "net_revenue", country="Germany")),
    )

    # ---- wrong period (6) ----
    add(
        "per-01 February prose, January cell",
        "wrong_period",
        "In February net revenue was £5,101.01.",
        gbp("£5,101.01", "5101.01", "net revenue", cell("qm", "net_revenue", month="2011-01-01")),
    )
    add(
        "per-02 March 2011 prose, February cell",
        "wrong_period",
        "March 2011 net revenue was £5,202.02.",
        gbp("£5,202.02", "5202.02", "net revenue", cell("qm", "net_revenue", month="2011-02-01")),
    )
    add(
        "per-03 ISO month wrong",
        "wrong_period",
        "Net revenue for 2011-01 was £5,303.03.",
        gbp("£5,303.03", "5303.03", "net revenue", cell("qm", "net_revenue", month="2011-03-01")),
    )
    add(
        "per-04 swapped months",
        "wrong_period",
        "January net revenue was £5,202.02, February net revenue was £5,101.01.",
        gbp(
            "£5,202.02",
            "5202.02",
            "net revenue",
            cell("qm", "net_revenue", month="2011-02-01"),
            "a",
        ),
        gbp(
            "£5,101.01",
            "5101.01",
            "net revenue",
            cell("qm", "net_revenue", month="2011-01-01"),
            "b",
        ),
    )
    add(
        "per-05 December 2010 not in data, cites January",
        "wrong_period",
        "December 2010 net revenue was £5,101.01.",
        gbp("£5,101.01", "5101.01", "net revenue", cell("qm", "net_revenue", month="2011-01-01")),
    )
    add(
        "per-06 month never named",
        "wrong_period",
        "Net revenue that month was £5,303.03.",
        gbp("£5,303.03", "5303.03", "net revenue", cell("qm", "net_revenue", month="2011-03-01")),
    )

    # ---- wrong dimension value (5): prose scope differs from the query's filter ----
    add(
        "dim-01 query filtered to France, prose says Germany",
        "wrong_dimension_value",
        "Germany net revenue was £1,111.11.",
        gbp("£1,111.11", "1111.11", "net revenue", cell("qf", "net_revenue")),
    )
    add(
        "dim-02 filtered to France, prose says United Kingdom",
        "wrong_dimension_value",
        "The United Kingdom made £1,111.11 of net revenue.",
        gbp("£1,111.11", "1111.11", "net revenue", cell("qf", "net_revenue")),
    )
    add(
        "dim-03 filtered to France+EIRE, prose says Spain",
        "wrong_dimension_value",
        "Spain net revenue was £3,333.33.",
        gbp("£3,333.33", "3333.33", "net revenue", cell("qi", "net_revenue")),
    )
    add(
        "dim-04 filtered to January, prose says March",
        "wrong_dimension_value",
        "March net revenue was £5,101.01.",
        gbp("£5,101.01", "5101.01", "net revenue", cell("qj", "net_revenue")),
    )
    add(
        "dim-05 filtered to France, prose says Spain",
        "wrong_dimension_value",
        "Spain net revenue was £1,111.11.",
        gbp("£1,111.11", "1111.11", "net revenue", cell("qf", "net_revenue")),
    )

    # ---- right number, different metric, same unit (8) ----
    add(
        "met-01 gross cell called net revenue",
        "wrong_metric_same_unit",
        "Net revenue was £9,500.95.",
        gbp("£9,500.95", "9500.95", "gross revenue", cell("qt", "gross_revenue")),
    )
    add(
        "met-02 net cell called gross revenue (prose)",
        "wrong_metric_same_unit",
        "Gross revenue was £9,000.09.",
        gbp("£9,000.09", "9000.09", "net revenue", cell("qt", "net_revenue")),
    )
    add(
        "met-03 cancelled called discount",
        "wrong_metric_same_unit",
        "Discounts totalled £400.04.",
        gbp("£400.04", "400.04", "cancelled revenue", cell("qt", "cancelled_revenue")),
    )
    add(
        "met-04 aov called revenue per customer",
        "wrong_metric_same_unit",
        "Revenue per customer was £90.09.",
        gbp("£90.09", "90.09", "average order value", cell("qt", "aov")),
    )
    add(
        "met-05 active customers called order count",
        "wrong_metric_same_unit",
        "There were 50 orders.",
        _claim("c", "50", "50", "count", "active customers", cell("qt", "active_customers")),
    )
    add(
        "met-06 orders called units sold",
        "wrong_metric_same_unit",
        "We sold 100 units.",
        _claim("c", "100", "100", "count", "order count", cell("qt", "order_count")),
    )
    add(
        "met-07 label and column agree, prose says sales (=net)",
        "wrong_metric_same_unit",
        "Sales were £9,500.95.",
        gbp("£9,500.95", "9500.95", "gross revenue", cell("qt", "gross_revenue")),
    )
    add(
        "met-08 discount called cancelled revenue",
        "wrong_metric_same_unit",
        "Cancelled revenue was £100.01.",
        gbp("£100.01", "100.01", "discount amount", cell("qt", "discount_amount")),
    )

    # ---- mixed / edge (3) ----
    add(
        "mix-01 two rows named, number between them",
        "wrong_row",
        "France and Germany together: £2,222.22.",
        gbp("£2,222.22", "2222.22", "net revenue", cell("qc", "net_revenue", country="Germany")),
    )
    add(
        "mix-02 wrong row AND wrong metric",
        "wrong_row",
        "Germany gross revenue was £1,111.11.",
        gbp("£1,111.11", "1111.11", "net revenue", cell("qc", "net_revenue", country="France")),
    )
    add(
        "mix-03 prose names a month for a total",
        "wrong_period",
        "February net revenue was £9,000.09.",
        gbp("£9,000.09", "9000.09", "net revenue", cell("qt", "net_revenue")),
    )
    assert len(out) == 30, len(out)
    return out
