# Number extractor: scope and known limits

`verified_analyst.extract.extract_numbers(text)` finds every numeric claim in an answer
(ARCHITECTURE section 4.1, step B). Each item has one of three statuses:

- **ok**: parsed into an exact `Decimal` value, unit, currency, scale, displayed precision
  (`quantum`), and optional qualifier (about / lower bound / upper bound) and direction.
- **ambiguous**: something number-like was found but its meaning is unclear. The verifier
  reports `AMBIGUOUS`; it never guesses.
- **excluded**: dates, times, years, ordinals, identifiers, list markers, citations, URLs.
  These are not claims, and the exclusion reason is kept so it can be audited.

Items never overlap, and `text[start:end]` always equals the item text.

## What it parses

Digits with thousands separators and decimals; currency symbols and codes (GBP, USD, EUR, INR,
JPY...); percentages, percentage points and basis points; magnitude suffixes (k, m, bn, mn) and
words (thousand, million, billion, lakh, crore); signs; ranges ("10-20", "10 to 20",
"between 10 and 20", "£1.2-1.5m"); spelled-out numbers up to the trillions; simple word
fractions ("half", "two thirds") and numeric fractions ("7/20"); qualifiers ("about", "over",
"at most"); direction words next to a number ("rose 5%").

## What it cannot parse (reported AMBIGUOUS, excluded, or missed)

| Form | Behaviour |
|---|---|
| European grouping `1.234,56`, space grouping `1 234 567`, `12,5` | AMBIGUOUS |
| Lowercase `5m` / `5b` without a currency | AMBIGUOUS (million vs metre, billion vs byte) |
| Unknown unit suffixes (`12x`, `10kg`) | AMBIGUOUS |
| Parenthesised numbers `(42)`: negative or footnote? | AMBIGUOUS |
| `1/2`, `3/4`: fraction or date? | AMBIGUOUS (`7/20` is read as a fraction) |
| Bare four-digit numbers (`2011 customers`) | AMBIGUOUS (`possible_year`) |
| Unparseable number words (`hundred` alone) | AMBIGUOUS |
| Multiples (`3.5x`), "double", "triple", "dozen", "half-year" | not parsed as quantities |
| Spelled-out ranges ("two to three million") | parsed as separate numbers, no range link |
| Compound spelled numbers beyond simple grammar | AMBIGUOUS or partial |
| Non-Latin digits, other scripts, other languages | not supported; must not crash |
| Phone numbers, postcodes, version strings | excluded when identifier-shaped, otherwise may be extracted |
| Numbers inside images, tables rendered as images | out of scope |
| Durations ("3 months", "last 30 days") | extracted as plain numbers; the claim must be a `parameter` claim |
| Adjectival numbers ("24-hour") | extracted, may not be a claim |

## Known imprecision

- Spelled-out numbers have coarse precision: "two million" has a quantum of 1,000,000, so it
  verifies against anything that rounds to 2m.
- Direction words are a heuristic (a fixed word list immediately before the number). "fell to 5%"
  deliberately carries no direction because it states a level, not a change.
- Fractions given in words are approximations of the true ratio.
- Exclusion rules can wrongly hide a real claim (for example "order 5" as a reference when it
  means five orders). The reason is recorded so it can be reviewed, but the number is not verified.
- Range pairing refuses ends with different units ("5.2% to £1.1m"), and treats "and" as a range
  separator only after "between".

## Tests

`tests/test_extract.py` has over 200 cases (parametrised) plus invariants: spans are consistent,
items never overlap, random text never crashes, and long input stays fast.
