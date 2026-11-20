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

## Composer conventions this implies

One number per clause (comma or semicolon between numbers); name the row or period in the same
sentence as the number; use names exactly as in the data or in the layer's metric vocabulary.

## Known limits (pinned by tests)

- Names are matched literally. "UK" for "United Kingdom" is `AMBIGUOUS`, not verified.
- "France and Germany together made £X" attaches X to the nearer name (a false accept).
- Two numbers in one clause ("France made £A and Germany made £B") can come back `AMBIGUOUS`.
- Only `month` and `day` periods are recognised; "last quarter", "Q4" and relative dates are not.
- "May" alone is not read as a month (verb); "May 2011" is.
- The check applies to `cell` claims. Cells used inside `derived` claims are only checked to exist.
- Dimension catalogue (values of `country`, `product_kind`) comes from `enumerable: true` in the
  layer; other dimensions only know values present in the executed results.
- Free-form wording that names a scope without the dimension value (for example "that market")
  is `AMBIGUOUS`; the plain-English restatement of the query (see PHASE_1C.md) is the reader's check.
