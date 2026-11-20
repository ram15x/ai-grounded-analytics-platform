"""Placeholder contract (ADR-0008): renderer, validator, and adversarial templates."""

from decimal import Decimal as D

import pytest
from pydantic import ValidationError

import verified_analyst.verifier as verifier_mod
from adversarial_cases import KNOWN, LAYER, cases, cell, queries
from verified_analyst.claims import TemplateAnswer
from verified_analyst.template import format_value, verify_template

QS = queries()


def claim(cid, prov, label="net revenue", unit="currency", **kw):
    return {"id": cid, "label": label, "unit": unit, "provenance": prov, **kw}


def run(template, claims, qs=None):
    ans = TemplateAnswer.model_validate({"template": template, "claims": claims})
    return verify_template(LAYER, ans, qs or QS, None, KNOWN)


FR = cell("qc", "net_revenue", country="France")
GE = cell("qc", "net_revenue", country="Germany")


def codes(r):
    return {i.code for i in r.issues}


def reasons(r):
    return {v.reason for v in r.report.verdicts if not v.ok}


# ---- formatting -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "cur", "dec", "scale", "text", "shown"),
    [
        (D("1234.5678"), "currency", "GBP", 2, "none", "£1,234.57", D("1234.57")),
        (D("-5"), "currency", "GBP", 2, "none", "-£5.00", D("-5.00")),
        (D("1234567.89"), "currency", "GBP", 1, "million", "£1.2m", D("1200000.0")),
        (D("1234567.89"), "currency", "USD", 0, "thousand", "$1,235k", D("1235000")),
        (D("12"), "count", None, 0, "none", "12", D("12")),
        (D("1234567"), "count", None, 1, "million", "1.2 million", D("1200000.0")),
        (D("12.345"), "percent", None, 1, "none", "12.3%", D("12.3")),
        (D("2.5"), "percentage_points", None, 1, "none", "2.5 percentage points", D("2.5")),
        (D("0.5"), "ratio", None, 2, "none", "0.50", D("0.50")),
        (D("1234.5"), "currency", "CHF", 2, "none", "CHF 1,234.50", D("1234.50")),
        (D("0.125"), "number", None, 2, "none", "0.13", D("0.13")),
    ],
)
def test_format_value(value, unit, cur, dec, scale, text, shown):
    assert format_value(value, unit, cur, dec, scale) == (text, shown)


def test_scale_only_for_currency_and_count():
    with pytest.raises(ValueError):
        format_value(D("5"), "percent", None, 1, "million")


# ---- rendering ---------------------------------------------------------------------------------------


def test_render_and_verify_happy_path():
    r = run("France net revenue was {a}, Germany net revenue was {b}.",
            [claim("a", FR), claim("b", GE)])  # fmt: skip
    assert r.status == "verified" and r.issues == []
    assert r.rendered == "France net revenue was £1,111.11, Germany net revenue was £2,222.22."
    p = r.placeholders[0]
    assert (p.true_value, p.shown_value, p.decimals) == (D("1111.11"), D("1111.11"), 2)
    assert r.rendered[p.start : p.end] == p.text


def test_value_comes_from_the_cell_not_the_composer():
    # there is nowhere to type a number: extra fields are rejected by the schema
    with pytest.raises(ValidationError):
        TemplateAnswer.model_validate(
            {"template": "{a}", "claims": [{**claim("a", FR), "value": "9"}]}
        )
    with pytest.raises(ValidationError):
        TemplateAnswer.model_validate(
            {"template": "{a}", "claims": [{**claim("a", FR), "text": "£9"}]}
        )


def test_derived_percent_and_ratio_metric_render():
    prov = {"kind": "derived", "op": "percent_of", "args": [FR, cell("qt", "net_revenue")]}
    r = run("France is {a} of the total.", [claim("a", prov, "net revenue share", "percent")])
    assert r.rendered == "France is 12.3% of the total." and r.status == "verified"


def test_scale_and_decimals_hints_are_recorded():
    r = run("France net revenue was {a}.", [claim("a", FR, scale="thousand", decimals=1)])
    assert r.rendered == "France net revenue was £1.1k." and r.status == "verified"
    p = r.placeholders[0]
    assert (p.scale, p.decimals, p.true_value, p.shown_value) == (
        "thousand",
        1,
        D("1111.11"),
        D("1100.0"),
    )


def test_count_and_row_count_render():
    r = run(
        "France placed {a} orders; the query returned {b} rows.",
        [claim("a", cell("qc", "order_count", country="France"), "order count", "count"),
         claim("b", {"kind": "row_count", "query_id": "qc"}, "rows", "count")],
    )  # fmt: skip
    assert (
        r.rendered == "France placed 11 orders; the query returned 4 rows."
        and r.status == "verified"
    )


def test_months_and_codes_outside_placeholders_are_allowed():
    r = run(
        "In March 2011 net revenue was {a}.",
        [claim("a", cell("qm", "net_revenue", month="2011-03-01"))],
    )
    assert r.status == "verified", (r.issues, r.report.verdicts)


# ---- the 30 old adversarial answers in placeholder form -------------------------------------------------


def to_template(case):
    prose, claims = case.prose, []
    for c in case.claims:
        assert c.text in prose
        prose = prose.replace(c.text, "{" + c.id + "}", 1)
        claims.append(claim(c.id, c.provenance.model_dump(), c.label, c.unit))
    return prose, claims


OLD = cases()


@pytest.mark.parametrize("case", OLD, ids=[c.name for c in OLD])
def test_old_adversarial_answers_in_placeholder_form_are_not_verified(case):
    template, claims = to_template(case)
    assert run(template, claims).status != "verified"


def test_old_adversarial_false_accept_count_in_placeholder_form():
    """Unit test on 30 author-written lies; not an accuracy claim."""
    accepted = [c.name for c in OLD if run(*to_template(c)).status == "verified"]
    assert accepted == []


def test_placeholders_alone_do_not_stop_description_lies(monkeypatch):
    """With the sentence binder switched off, placeholders catch none of the 30: they remove
    transcription errors, not mis-description. The binder is therefore kept as the second check."""
    monkeypatch.setattr(verifier_mod, "check_cell_context", lambda **kw: None)
    accepted = [c.name for c in OLD if run(*to_template(c)).status == "verified"]
    assert len(accepted) >= 20, accepted  # most lies survive without the binder


# ---- structure: 25+ adversarial templates ------------------------------------------------------------------

A = claim("a", FR)
B = claim("b", GE)

ADVERSARIAL = [
    # (name, template, claims, expected issue code or verdict reason)
    ("unknown placeholder", "France net revenue was {zz}.", [A], "unknown_placeholder"),
    ("unused claim", "France net revenue was {a}.", [A, B], "unused_claim"),
    ("duplicate claim ids", "France net revenue was {a}.", [A, A], "duplicate_claim_id"),
    ("stray open brace", "France net revenue was {a} {", [A], "stray_brace"),
    ("stray close brace", "France net revenue was {a} }", [A], "stray_brace"),
    ("nested braces", "France net revenue was {{a}}.", [A], "stray_brace"),
    ("digit appended", "France net revenue was {a}0.", [A], "placeholder_adjacent_text"),
    ("decimal appended", "France net revenue was {a}.5 or so.", [A], "placeholder_adjacent_text"),
    ("digit prepended", "France net revenue was 1{a}.", [A], "placeholder_adjacent_text"),
    ("suffix appended", "France net revenue was {a}k.", [A], "placeholder_adjacent_text"),
    ("percent appended", "France net revenue was {a}%.", [A], "placeholder_adjacent_text"),
    ("second currency symbol", "France net revenue was £{a}.", [A], "placeholder_adjacent_text"),
    ("adjacent placeholders", "France and Germany {a}{b} net revenue.", [A, B], "placeholder_adjacent_text"),
    ("digits outside", "France net revenue was {a} from 99 orders.", [A], "number_outside_placeholder"),
    ("spelled-out outside", "France net revenue was {a}, about two thousand more.", [A], "number_outside_placeholder"),
    ("half outside", "France net revenue was {a}, half of the total.", [A], "number_outside_placeholder"),
    ("percent outside", "France net revenue was {a}, up 12% on last month.", [A], "number_outside_placeholder"),
    ("hundred and five", "France net revenue was {a} across one hundred and five invoices.", [A], "number_outside_placeholder"),
    ("range outside", "France net revenue was {a} (between 10 and 20 percent of sales).", [A], "number_outside_placeholder"),
    ("approx outside", "France net revenue was {a}, approximately 100 more.", [A], "number_outside_placeholder"),
    ("bare digit sentence", "France net revenue was {a}. 7.", [A], "number_outside_placeholder"),
    ("fullwidth digits", "France net revenue was {a}, plus １２３.", [A], "non_ascii_digit"),
    ("identifier smuggling", "France net revenue was {a} on invoice 1234.", [A], "identifier_not_in_results"),
    ("reference label smuggling", "France net revenue was {a} (see page 7).", [A], "identifier_not_in_results"),
    ("row not in data", "Spain net revenue was {a}.", [claim("a", cell("qc", "net_revenue", country="Spain"))], "placeholder_unresolvable"),
    ("divide by zero", "Ratio {a}.", [claim("a", {"kind": "derived", "op": "ratio", "args": [FR, cell("qc", "order_count", country="Nowhere")]}, "ratio", "number")], "placeholder_unresolvable"),
    ("currency unit on a count cell", "France placed {a} orders.", [claim("a", cell("qc", "order_count", country="France"), "order count", "currency")], "placeholder_unformattable"),
    ("percent unit on a currency cell", "France net revenue was {a}.", [claim("a", FR, "net revenue", "percent")], "unit_incompatible"),
    ("wrong cell in another country's sentence", "Germany net revenue was {a}.", [A], "row_mismatch"),
    ("same placeholder, contradictory sentences", "France net revenue was {a}; Germany net revenue was {a}.", [A], "row_mismatch"),
    ("label contradicts column", "France net revenue was {a}.", [claim("a", cell("qc", "gross_revenue", country="France"), "net revenue")], "label_column_mismatch"),
    ("sentence metric contradicts column", "France gross revenue was {a}.", [A], "sentence_metric_differs"),
    ("literal argument", "Total {a}.", [claim("a", {"kind": "derived", "op": "sum", "args": [FR, {"kind": "literal", "value": "5"}]}, "total")], "schema"),
    ("unscaled scale on percent", "Share {a}.", [claim("a", FR, "share", "percent", scale="million")], "placeholder_unformattable"),
]  # fmt: skip


@pytest.mark.parametrize(
    "name,template,claims,expected", ADVERSARIAL, ids=[a[0] for a in ADVERSARIAL]
)
def test_adversarial_template(name, template, claims, expected):
    if expected == "schema":
        with pytest.raises(ValidationError):
            TemplateAnswer.model_validate({"template": template, "claims": claims})
        return
    r = run(template, claims)
    assert r.status != "verified"
    assert expected in codes(r) | reasons(r), (r.issues, [v.as_dict() for v in r.report.verdicts])


def test_at_least_25_adversarial_templates():
    assert len(ADVERSARIAL) >= 25


# ---- controls: things that must still verify -------------------------------------------------------------------


def test_ordinals_dates_and_known_codes_are_allowed():
    qs = dict(QS)
    r = run("On the 3rd, in Q4, France net revenue was {a}.", [A], qs)
    assert r.status == "verified", (r.issues, [v.as_dict() for v in r.report.verdicts])


def test_identifier_present_in_results_is_allowed():
    from verified_analyst.compiler import MetricQuery
    from verified_analyst.verifier import ExecutedQuery

    q = ExecutedQuery("qp", "SELECT 1", ["stock_code", "net_revenue"], [("85123A", D("10"))],
                      metric_query=None, declared_metric_query=MetricQuery(metrics=["net_revenue"]))  # fmt: skip
    ans = TemplateAnswer.model_validate({"template": "Product 85123A is in the results; revenue {a}.",
        "claims": [claim("a", {"kind": "row_count", "query_id": "qp"}, "rows", "count")]})  # fmt: skip
    r = verify_template(LAYER, ans, {"qp": q}, None, KNOWN)
    assert not any(i.code == "identifier_not_in_results" for i in r.issues)


def test_same_placeholder_reused_consistently_is_fine():
    r = run("France net revenue was {a}; again, France net revenue was {a}.", [A])
    assert r.status == "verified" and len(r.placeholders) == 2


def test_report_serialises():
    r = run("France net revenue was {a}.", [A])
    d = r.as_dict()
    assert d["status"] == "verified" and d["placeholders"][0]["shown_value"] == "1111.11"
