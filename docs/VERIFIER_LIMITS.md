# Verifier: sentence-context binding and its limits

The number check (does the cell hold this value?) is necessary, not sufficient. A composer can
cite a REAL cell and describe it wrongly: another country, another month, another metric of the
same unit. The claim's `label` is composer-written, so it cannot be trusted to bind the number.

`src/verified_analyst/context.py` therefore reads the PROSE around each `cell` claim:

1. **Row.** If the cell is one row of a multi-row result, the sentence must name that row's
   dimension value, and the value nearest the number (in words) must be the cited row.
2. **Scope.** A dimension value near the number must be allowed by the query's `eq`/`in` filters;
   naming a dimension the query neither groups nor filters by is rejected.
3. **Metric.** A metric term (name or synonym from `semantic/metrics.yaml`) belonging to the
   number must resolve to the cited column's metric.

Verdicts: wrong row/scope is `UNSUPPORTED`; unnamed or equidistant references are `AMBIGUOUS`;
a different metric is `METRIC_MISMATCH`.

## Measured, on 30 hand-built adversarial answers (`tests/test_adversarial.py`)

Before the check: **30 of 30 falsely accepted** (the verifier trusted the label). After: **0 of 30**.
This is a unit test of 30 specific lies written by the same author as the check, with synthetic data.
It is not an accuracy claim, and 0/30 does not mean 0% on real model output.

## Implicit scopes (a metric's own definition)

Some metrics already apply a scope: `units_sold` counts merchandise sale lines only. A sentence
such as "The total merchandise units were X" names the dimension value `merchandise` without the
query filtering on it, which the scope rule above rejects (`scope_not_in_query`). `semantic/
metrics.yaml` can declare `implicit_scopes` for a metric (today: `units_sold`, `product_kind:
[merchandise]`). The binder then accepts that scope word for that metric only, when the query has
no filter on that dimension. Guards, all tested:

- The layer refuses a declaration unless the metric's `expr` filters on that column and value, the
  dimension is an enumerable text dimension the metric cannot also group by, and the metric is
  not a ratio. Ratios do not inherit scopes from their parts.
- Other values (`non_merchandise`), other metrics ("merchandise gross revenue"), and other scopes
  in the same sentence (a country, even by the alias "UK") are still `UNSUPPORTED`.
- Measured on the same 30 adversarial answers, in prose and placeholder form, with and without
  `product_kind` values loaded: per-case results identical before and after, 0 false accepts.
  A unit test by the author of the check; not an accuracy claim.

## Composer conventions this implies

One number per clause (comma or semicolon between numbers); name the row or period in the same
sentence as the number; use names exactly as in the data or in the layer's metric vocabulary.

## Known limits (pinned by tests)

- Names are matched literally. "UK" for "United Kingdom" is `AMBIGUOUS` when the UK row is the
  cited row of a country-grouped result. On a global total, or a query filtered to another
  country, "UK net revenue was £X" is NOT seen and verifies (a false accept, pinned in
  `tests/test_implicit_scopes.py`; the 30 adversarial answers do not cover it). The layer lists
  "UK" as an alias of "United Kingdom", but the binder uses it only as a veto on an implicit
  scope (below): recognising aliases everywhere would also reject "The UK is 60% of the total",
  which the shipped examples contain and which verifies today. Decision left to the owner.
- "France and Germany together made £X" attaches X to the nearer name (a false accept).
- Two numbers in one clause ("France made £A and Germany made £B") can come back `AMBIGUOUS`.
- Only `month` and `day` periods are recognised; "last quarter", "Q4" and relative dates are not.
- "May" alone is not read as a month (verb); "May 2011" is.
- The check applies to `cell` claims. Cells used inside `derived` claims are only checked to exist.
- Dimension catalogue (values of `country`, `product_kind`) comes from `enumerable: true` in the
  layer; other dimensions only know values present in the executed results.
- Free-form wording that names a scope without the dimension value (for example "that market")
  is `AMBIGUOUS`; the plain-English restatement of the query (see PHASE_1C.md) is the reader's check.

## Template mode (ADR-0008)

With placeholders the model types no numbers, so transcription, scale and rounding errors cannot
occur. The limits above still apply to the rendered text. Additional limits:

- Placeholders do not stop mis-description: with the binder off, at least 20 of the 30 old
  adversarial answers (placeholder form) still pass. The binder is kept for that reason.
- Measured on those 30 plus 33 adversarial templates: 0 false accepts of the 30 with the binder on.
  Unit tests by the author of the checks, synthetic data; not an accuracy claim.
- Digits outside placeholders are `UNSUPPORTED`; date, year, ordinal, list-marker and URL forms are
  excluded. Identifier-like tokens (order or stock codes) pass only if they appear in the executed
  result cells; a coincidental match would pass.
- Not supported yet: passage and parameter claims, literal arguments in derived claims.
- Whether small models follow the placeholder format is not measured.

## Pipeline rule: an answer with no figure is not "verified" (Phase 1c-1)

`verify_template` passes a text that contains no placeholder and no number, because there is nothing
to check. The scripted pipeline found this: a prose-only answer came back `verified`. The pipeline
(`llm/pipeline.py`) now marks such an attempt `no_figures`, feeds it back for repair, and abstains
with reason `unverified` if no attempt verified a number. The verifier itself is unchanged, so
calling `verify_template` directly on a figure-free text still returns `verified`.
