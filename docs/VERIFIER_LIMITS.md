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

## Value aliases, measured (`tests/test_aliases.py`)

Ten alias lies (a real cell described with "UK" or "United Kingdom" for a scope the query does not
cover, a wrong row, a wrong metric, a swapped pair), each in prose and placeholder form, with both
spellings. Before the alias was read: 4 of 10 falsely accepted with "UK", 0 of 10 with the full
name. After: 0 of 10 with either spelling, and every case has the same verdicts and reasons under
both spellings. The 30 adversarial answers: 0 false accepts before and after, per-case results
identical. The 12 legitimate controls: 12 of 12 verify before and after. Five true alias sentences
(UK row, UK-filtered query, two rows, merchandise units): verify 5 of 5 under both spellings
(2 of 5 under "UK" before). Unit tests by the author of the check, synthetic data; not an accuracy
claim.

## Composer conventions this implies

One number per clause (comma or semicolon between numbers); name the row or period in the same
sentence as the number; use names exactly as in the data or in the layer's metric vocabulary.

## Known limits (pinned by tests)

- Names are matched literally, except aliases the layer lists (`value_aliases` in
  `semantic/metrics.yaml`; today "UK" for "United Kingdom"). A listed alias is the same mention as
  the value it stands for, everywhere in the binder: same row, scope and filter rules, same
  verdicts and reasons (pinned by `tests/test_aliases.py`). Before this, "UK" was invisible, so
  "UK net revenue was £X" on a global total or a France-filtered query verified (4 of 10 alias
  lies were falsely accepted; now 0 of 10). A spelling the layer does NOT list ("Britain",
  "Great Britain", "U.K.") is still not read: `AMBIGUOUS` when it names the cited row of a
  country-grouped result, and not seen at all on a total or a differently filtered query, where
  "Britain net revenue was £X" verifies (a false accept of the same kind; not in the 30).
- A true sentence that names a country next to a number from a query that does not cover that
  country is `UNSUPPORTED` (`scope_not_in_query`), alias or full name alike. Example: "The United
  Kingdom is 59.6% of the £17,845.25 total." The total comes from a global query, and the sentence
  gives the country as its scope. This is a conservative false reject and is not loosened.
  `examples/11_known_limit_share_of_global_total.json` shows it on purpose ("The UK is 59.6% of the
  £17,845.25 total": `partially_verified`). Samples 01 and 08 state the same share without naming
  a country beside the total ("Across all countries net revenue was £17,845.25, and the leader's
  share of that total is 59.6%") and verify in full with the binder unchanged. Samples 03, 05, 07
  and 09 still use the old wording, so each shows one extra `UNSUPPORTED` on the total.
  Backlog, not built: a derived-share rule for sentences that name the country of a share and
  its global total; to be designed with its own adversarial tests.
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

## Passage claims and flagged text

A passage claim is `VERIFIED_QUOTE` only if the cited number sits in the cited span of the stored
chunk. If that span overlaps text the scanner flags (an instruction to an AI, a role marker, a
hidden comment, invisible characters or a spoofed citation or verification mark), the claim is
`UNSUPPORTED` with reason `passage_span_flagged`. This is a heuristic filter and a quote that
really is in a document still verifies as a quote; see docs/INJECTION.md.
