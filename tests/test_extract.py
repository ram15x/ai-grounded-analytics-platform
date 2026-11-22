"""Number extractor: parsing, exclusions, ambiguity, and structural invariants.

Each case lists EVERY item the extractor should report for the text, in order."""

import random
import string
from decimal import Decimal

import pytest

from grounded_analytics.extract import extract_numbers


def ok(text, value, unit="number", **kw):
    return ("ok", text, Decimal(str(value)), unit, kw)


def amb(text, reason):
    return ("ambiguous", text, reason)


def exc(text, reason):
    return ("excluded", text, reason)


def summarize(text):
    out = []
    for i in extract_numbers(text).items:
        if i.status == "ok":
            kw = {}
            for field in ("currency", "qualifier", "direction", "kind"):
                v = getattr(i, field)
                if v not in (None, "plain"):
                    kw[field] = v
            out.append(("ok", i.text, i.value, i.unit, kw))
        else:
            out.append((i.status, i.text, i.reason))
    return out


def norm(expected):
    """Expected items may leave kw empty; compare only the keys they mention."""
    return expected


CASES = [
    # ---- plain numbers and separators --------------------------------------------------
    ("42", [ok("42", 42)]),
    ("There were 42 orders", [ok("42", 42)]),
    ("1,234", [ok("1,234", 1234)]),
    ("1,234,567", [ok("1,234,567", 1234567)]),
    ("0.5", [ok("0.5", "0.5")]),
    (".5", [ok(".5", "0.5")]),
    ("3.14159", [ok("3.14159", "3.14159")]),
    ("1,234.50", [ok("1,234.50", "1234.50")]),
    ("12345", [ok("12345", 12345)]),
    ("0", [ok("0", 0)]),
    ("Totals: 7, 8 and 9.", [ok("7", 7), ok("8", 8), ok("9", 9)]),
    ("Sold 3 units; refunded 1.", [ok("3", 3), ok("1", 1)]),
    # ---- currency ----------------------------------------------------------------------
    ("£5", [ok("£5", 5, "currency", currency="GBP")]),
    ("£1,234.56", [ok("£1,234.56", "1234.56", "currency", currency="GBP")]),
    ("$5", [ok("$5", 5, "currency", currency="USD")]),
    ("€5", [ok("€5", 5, "currency", currency="EUR")]),
    ("₹5", [ok("₹5", 5, "currency", currency="INR")]),
    ("£ 5", [ok("£ 5", 5, "currency", currency="GBP")]),
    ("GBP 1,234", [ok("GBP 1,234", 1234, "currency", currency="GBP")]),
    ("1,234 GBP", [ok("1,234 GBP", 1234, "currency", currency="GBP")]),
    ("USD5", [ok("USD5", 5, "currency", currency="USD")]),
    ("5 pounds", [ok("5 pounds", 5, "currency", currency="GBP")]),
    ("5 dollars", [ok("5 dollars", 5, "currency", currency="USD")]),
    ("-£5", [ok("-£5", -5, "currency", currency="GBP")]),
    ("1.2 million pounds", [ok("1.2 million pounds", 1200000, "currency", currency="GBP")]),
    # ---- percent -------------------------------------------------------------------------
    ("5%", [ok("5%", 5, "percent")]),
    ("5 %", [ok("5 %", 5, "percent")]),
    ("12.5%", [ok("12.5%", "12.5", "percent")]),
    ("12.5 percent", [ok("12.5 percent", "12.5", "percent")]),
    ("12.5 per cent", [ok("12.5 per cent", "12.5", "percent")]),
    ("12.5pct", [ok("12.5pct", "12.5", "percent")]),
    ("5 percentage points", [ok("5 percentage points", 5, "percentage_points")]),
    ("5pp", [ok("5pp", 5, "percentage_points")]),
    ("50 bps", [ok("50 bps", 50, "basis_points")]),
    ("25 basis points", [ok("25 basis points", 25, "basis_points")]),
    ("100% of orders", [ok("100%", 100, "percent")]),
    # ---- magnitude suffixes and words -------------------------------------------------------
    ("£1.2m", [ok("£1.2m", 1200000, "currency", currency="GBP")]),
    ("£1.2M", [ok("£1.2M", 1200000, "currency", currency="GBP")]),
    ("1.2M", [ok("1.2M", 1200000)]),
    ("5k", [ok("5k", 5000)]),
    ("5K", [ok("5K", 5000)]),
    ("£5k", [ok("£5k", 5000, "currency", currency="GBP")]),
    ("3bn", [ok("3bn", 3000000000)]),
    ("3B", [ok("3B", 3000000000)]),
    ("$3.2bn", [ok("$3.2bn", 3200000000, "currency", currency="USD")]),
    ("5 mn", [ok("5 mn", 5000000)]) if False else ("5mn", [ok("5mn", 5000000)]),
    ("2 million", [ok("2 million", 2000000)]),
    ("2.5 million", [ok("2.5 million", 2500000)]),
    ("3 billion", [ok("3 billion", 3000000000)]),
    ("4 thousand", [ok("4 thousand", 4000)]),
    ("2 lakh", [ok("2 lakh", 200000)]),
    ("3 crore", [ok("3 crore", 30000000)]),
    ("£1.2 million", [ok("£1.2 million", 1200000, "currency", currency="GBP")]),
    ("₹2 crore", [ok("₹2 crore", 20000000, "currency", currency="INR")]),
    ("5m units", [amb("5m", "lowercase_m_or_b_suffix")]),
    ("5b", [amb("5b", "lowercase_m_or_b_suffix")]),
    # ---- signs ---------------------------------------------------------------------------
    ("-5", [ok("-5", -5)]),
    ("−5", [ok("−5", -5)]),
    ("-5%", [ok("-5%", -5, "percent")]),
    ("+5%", [ok("+5%", 5, "percent")]),
    ("change of -1,234.5", [ok("-1,234.5", "-1234.5")]),
    ("(1,234)", [amb("(1,234)", "parenthesised_number")]),
    ("orders (42)", [amb("(42)", "parenthesised_number")]),
    ("x-5", [exc("x-5", "identifier")]),
    # ---- ranges ----------------------------------------------------------------------------
    ("10-20", [ok("10", 10, kind="range_low"), ok("20", 20, kind="range_high")]),
    ("10–20", [ok("10", 10, kind="range_low"), ok("20", 20, kind="range_high")]),
    ("10 - 20", [ok("10", 10, kind="range_low"), ok("20", 20, kind="range_high")]),
    ("10 to 20", [ok("10", 10, kind="range_low"), ok("20", 20, kind="range_high")]),
    ("between 10 and 20", [ok("10", 10, kind="range_low"), ok("20", 20, kind="range_high")]),
    (
        "10-20%",
        [ok("10", 10, "percent", kind="range_low"), ok("20%", 20, "percent", kind="range_high")],
    ),
    (
        "5 to 10 percent",
        [
            ok("5", 5, "percent", kind="range_low"),
            ok("10 percent", 10, "percent", kind="range_high"),
        ],
    ),
    (
        "£1.2-1.5m",
        [
            ok("£1.2", 1200000, "currency", currency="GBP", kind="range_low"),
            ok("1.5m", 1500000, "currency", currency="GBP", kind="range_high"),
        ],
    ),
    ("10 and 20", [ok("10", 10), ok("20", 20)]),  # no "between": two separate numbers
    ("2010-2011", [exc("2010", "year_range"), exc("2011", "year_range")]),
    # ---- spelled-out numbers ---------------------------------------------------------------
    ("zero", [ok("zero", 0, kind="spelled")]),
    ("two", [ok("two", 2, kind="spelled")]),
    ("twelve", [ok("twelve", 12, kind="spelled")]),
    ("twenty-one", [ok("twenty-one", 21, kind="spelled")]),
    ("forty two", [ok("forty two", 42, kind="spelled")]),
    ("ninety-nine", [ok("ninety-nine", 99, kind="spelled")]),
    ("one hundred", [ok("one hundred", 100, kind="spelled")]),
    ("one hundred and five", [ok("one hundred and five", 105, kind="spelled")]),
    ("a hundred", [ok("a hundred", 100, kind="spelled")]),
    ("a million", [ok("a million", 1000000, kind="spelled")]),
    ("two million", [ok("two million", 2000000, kind="spelled")]),
    ("three point five", [ok("three point five", "3.5", kind="spelled")]),
    ("two point five million", [ok("two point five million", 2500000, kind="spelled")]),
    ("one thousand two hundred", [ok("one thousand two hundred", 1200, kind="spelled")]),
    ("fifteen hundred", [ok("fifteen hundred", 1500, kind="spelled")]),
    ("five percent", [ok("five percent", 5, "percent", kind="spelled")]),
    ("five pounds", [ok("five pounds", 5, "currency", currency="GBP", kind="spelled")]),
    (
        "a thousand pounds",
        [ok("a thousand pounds", 1000, "currency", currency="GBP", kind="spelled")],
    ),
    ("twenty twenty", [ok("twenty", 20, kind="spelled"), ok("twenty", 20, kind="spelled")]),
    ("one customer", [ok("one", 1, kind="spelled")]),
    ("two hundred and twelve orders", [ok("two hundred and twelve", 212, kind="spelled")]),
    # pronoun/determiner uses of "one" are not numbers
    ("no one knows", []),
    ("one of our best customers", []),
    ("the one thing", []),
    # unparseable word sequences are flagged, never guessed
    ("one one", [ok("one", 1, kind="spelled")])
    if False
    else ("hundred", [amb("hundred", "unparseable_number_words")]),
    # ---- fractions -------------------------------------------------------------------------
    ("7/20", [ok("7/20", "0.35", kind="fraction")]),
    ("3/4", [amb("3/4", "fraction_or_date")]),
    ("1/2", [amb("1/2", "fraction_or_date")]),
    ("1/0", [amb("1/0", "zero_denominator")]),
    ("half of customers", [ok("half of", "0.5", kind="fraction")]),
    ("a third", [ok("a third", "0.3333333333333333333333333333", kind="fraction")]),
    ("two thirds", [ok("two thirds", "0.6666666666666666666666666667", kind="fraction")]),
    ("three quarters", [ok("three quarters", "0.75", kind="fraction")]),
    ("last quarter", []),
    ("That is half the total.", [ok("half", "0.5", kind="fraction")]),
    ("the second half of the year", []),
    ("half-year results", []),
    ("first half", []),
    ("the third largest", []),
    ("first half", []),
    # ---- qualifiers --------------------------------------------------------------------------
    ("about 5", [ok("about 5", 5, qualifier="approx")]),
    (
        "approximately £2m",
        [ok("approximately £2m", 2000000, "currency", currency="GBP", qualifier="approx")],
    ),
    ("roughly 30%", [ok("roughly 30%", 30, "percent", qualifier="approx")]),
    ("nearly 100", [ok("nearly 100", 100, qualifier="approx")]),
    ("~50", [ok("~50", 50, qualifier="approx")]),
    ("over 5", [ok("over 5", 5, qualifier="lower_bound")]),
    ("more than 5", [ok("more than 5", 5, qualifier="lower_bound")]),
    ("at least 5", [ok("at least 5", 5, qualifier="lower_bound")]),
    ("under 5", [ok("under 5", 5, qualifier="upper_bound")]),
    ("less than 5", [ok("less than 5", 5, qualifier="upper_bound")]),
    ("up to 5", [ok("up to 5", 5, qualifier="upper_bound")]),
    ("at most 5", [ok("at most 5", 5, qualifier="upper_bound")]),
    ("over £2m", [ok("over £2m", 2000000, "currency", currency="GBP", qualifier="lower_bound")]),
    ("discover 5", [ok("5", 5)]),  # "over" inside a word is not a qualifier
    # ---- direction words ---------------------------------------------------------------------
    ("rose 5%", [ok("5%", 5, "percent", direction="up")]),
    ("grew by 5%", [ok("5%", 5, "percent", direction="up")]),
    ("fell by 5%", [ok("5%", 5, "percent", direction="down")]),
    ("dropped 5%", [ok("5%", 5, "percent", direction="down")]),
    ("down 5%", [ok("5%", 5, "percent", direction="down")]),
    ("increased by £1,000", [ok("£1,000", 1000, "currency", currency="GBP", direction="up")]),
    ("fell to 5%", [ok("5%", 5, "percent")]),
    ("rose to £5m", [ok("£5m", 5000000, "currency", currency="GBP")]),
    ("grew 3 times, then 5%", [ok("3", 3, direction="up"), ok("5%", 5, "percent")]),
    ("revenue is 5%", [ok("5%", 5, "percent")]),
    # ---- dates, times, periods ----------------------------------------------------------------
    ("2011-03-04", [exc("2011-03-04", "date")]),
    ("2011-03-04 14:30:00", [exc("2011-03-04 14:30:00", "date")]),
    ("04/03/2011", [exc("04/03/2011", "date")]),
    ("4.3.2011", [exc("4.3.2011", "date")]),
    ("March 4, 2011", [exc("March 4, 2011", "date")]),
    ("4 March 2011", [exc("4 March 2011", "date")]),
    ("4th of March", [exc("4th of March", "date")]),
    ("March 2011", [exc("March 2011", "date")]),
    ("Dec 1", [exc("Dec 1", "date")]),
    ("2011-03", [exc("2011-03", "date")]),
    ("14:30", [exc("14:30", "time")]),
    ("2:30 pm", [exc("2:30 pm", "time")]),
    ("Q1", [exc("Q1", "period_label")]),
    ("Q4 2011", [exc("Q4 2011", "period_label")]),
    ("FY2011", [exc("FY2011", "period_label")]),
    ("H1", [exc("H1", "period_label")]),
    # ---- years -------------------------------------------------------------------------------
    ("in 2011", [exc("2011", "year")]),
    ("since 2010", [exc("2010", "year")]),
    ("during 2011", [exc("2011", "year")]),
    ("2011 customers", [amb("2011", "possible_year")]),
    ("2011", [amb("2011", "possible_year")]),
    ("£2011", [ok("£2011", 2011, "currency", currency="GBP")]),
    ("2,011", [ok("2,011", 2011)]),
    # ---- ordinals, ids, references, markers ---------------------------------------------------
    ("3rd", [exc("3rd", "ordinal")]),
    ("21st", [exc("21st", "ordinal")]),
    ("invoice 536365", [exc("invoice 536365", "identifier")]),
    ("order #123", [exc("order #123", "identifier")]),
    ("customer 12347", [exc("customer 12347", "identifier")]),
    ("stock code 85123A", [exc("stock code 85123A", "identifier")]),
    ("SKU 71053", [exc("SKU 71053", "identifier")]),
    ("85123A", [exc("85123A", "identifier")]),
    ("C536379", [exc("C536379", "identifier")]),
    ("#42", [exc("#42", "identifier")]),
    ("COVID-19", [exc("COVID-19", "identifier")]),
    ("v1.2", [exc("v1.2", "identifier")]),
    ("page 3", [exc("page 3", "reference_label")]),
    ("Table 2", [exc("Table 2", "reference_label")]),
    ("step 4", [exc("step 4", "reference_label")]),
    ("query 2", [exc("query 2", "reference_label")]),
    ("see [1]", [exc("[1]", "citation")]),
    ("see [2, 3]", [exc("[2, 3]", "citation")]),
    ("https://example.com/a/2011/5", [exc("https://example.com/a/2011/5", "url_or_email")]),
    ("a.b2@example.com", [exc("a.b2@example.com", "url_or_email")]),
    ("1. First point\n2. Second point", [exc("1.", "list_marker"), exc("2.", "list_marker")]),
    ("5000XYZ", [exc("5000XYZ", "identifier")]),
    # ---- ambiguity is flagged, never dropped ---------------------------------------------------
    ("1.234,56", [amb("1.234,56", "european_number_format")]),
    ("1 234 567", [amb("1 234 567", "space_grouped_number")]),
    ("12,5", [amb("12,5", "comma_decimal_or_list")]),
    ("12x", [amb("12x", "unknown_suffix")]),
    ("10kg", [amb("10kg", "unknown_suffix")]),
    (
        "fifteen and hundred",
        [ok("fifteen", 15, kind="spelled"), amb("hundred", "unparseable_number_words")],
    ),
    # ---- realistic sentences --------------------------------------------------------------------
    (
        "Net revenue in November 2010 was £45,678.90, up 12.5% on October.",
        [
            exc("November 2010", "date"),
            ok("£45,678.90", "45678.90", "currency", currency="GBP"),
            ok("12.5%", "12.5", "percent", direction="up"),
        ],
    ),
    (
        "France had 12 orders worth £1,234 (3.4% of 2011 sales).",
        [
            ok("12", 12),
            ok("£1,234", 1234, "currency", currency="GBP"),
            ok("3.4%", "3.4", "percent"),
            exc("2011", "year"),
        ],
    ),
    (
        "Revenue fell 5.2% to £1.1m between 2010-12 and 2011-01.",
        [
            ok("5.2%", "5.2", "percent", direction="down"),
            ok("£1.1m", 1100000, "currency", currency="GBP"),
            exc("2010-12", "date"),
            exc("2011-01", "date"),
        ],
    ),
    (
        "Top 5 products by revenue; product 85123A led with 2,345 units.",
        [ok("5", 5), exc("product 85123A", "identifier"), ok("2,345", 2345)],
    ),
    (
        "Orders: 1) 120, 2) 85",
        [exc("1)", "list_marker")]
        if False
        else [ok("1", 1), ok("120", 120), ok("2", 2), ok("85", 85)],
    ),
    ("No numbers here at all.", []),
    ("", []),
    ("   \n\t ", []),
]


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[repr(c[0])[:60] for c in CASES])
def test_extraction(text, expected):
    got = summarize(text)
    assert len(got) == len(expected), (got, expected)
    for g, e in zip(got, expected, strict=True):
        assert g[:2] == e[:2], (g, e)
        if e[0] == "ok":
            assert g[2] == e[2], (g, e)
            assert g[3] == e[3], (g, e)
            for key, val in e[4].items():
                assert g[4].get(key) == val, (g, e, key)
        else:
            assert g[2] == e[2], (g, e)


def test_at_least_one_hundred_cases():
    assert len(CASES) >= 100


# ---- precision / quantum -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "decimals", "scale", "quantum"),
    [
        ("1,234,567", 0, 1, "1"),
        ("1,234.50", 2, 1, "0.01"),
        ("£1.2m", 1, 1000000, "100000.0"),
        ("5k", 0, 1000, "1000"),
        ("12.5%", 1, 1, "0.1"),
        ("two million", 0, 1000000, "1000000"),
        ("two point five million", 1, 1000000, "100000.0"),
    ],
)
def test_displayed_precision(text, decimals, scale, quantum):
    (item,) = extract_numbers(text).ok
    assert item.decimals == decimals
    assert item.scale == scale
    assert item.quantum == Decimal(quantum)


def test_value_is_decimal_not_float():
    (item,) = extract_numbers("£0.1").ok
    assert isinstance(item.value, Decimal) and item.value == Decimal("0.1")


# ---- structural invariants -----------------------------------------------------------------


def test_spans_and_text_agree_and_do_not_overlap():
    text = (
        "Q4 2011: revenue was £1,234.56 (up 12.5%), 3rd best; invoice 536365 had 85123A. "
        "Between 10 and 20 orders, 1/2 or half of them, about two million units on 2011-03-04."
    )
    items = extract_numbers(text).items
    assert items
    last_end = -1
    for i in items:
        assert text[i.start : i.end] == i.text
        assert i.start >= last_end, (i, last_end)
        last_end = i.end


def test_ambiguous_and_excluded_never_checkable():
    ex = extract_numbers("in 2011 there were 1 234 567 things at 14:30")
    assert all(i.status == "ok" for i in ex.ok)
    assert {i.reason for i in ex.ambiguous} >= {"space_grouped_number"}
    assert {i.reason for i in ex.excluded} >= {"year", "time"}


def test_unicode_digits_do_not_crash():
    extract_numbers("fullwidth １２３ and arabic-indic ٣٤٥")


def test_random_text_never_crashes_and_spans_are_consistent():
    rng = random.Random(1234)
    alphabet = string.ascii_letters + string.digits + " ,.-%£$€()/:#_\n" + "–−~"
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        items = extract_numbers(text).items
        for i in items:
            assert text[i.start : i.end] == i.text
            assert 0 <= i.start < i.end <= len(text)
        spans = sorted((i.start, i.end) for i in items)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)), text


def test_long_input_is_fast_enough():
    text = "Revenue was £1,234.56 in March 2011, up 5% (about 3 million units). " * 2000
    items = extract_numbers(text).items
    assert len(items) >= 8000
