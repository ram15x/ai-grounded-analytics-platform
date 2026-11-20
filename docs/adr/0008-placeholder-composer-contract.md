# ADR-0008: Placeholder composer contract

Status: ACCEPTED for the deterministic part (renderer, validator, schema). The model-facing part
is PROPOSED and unmeasured; no LLM was called to produce anything in this ADR.

## Problem

The adversarial test of the sentence binder (docs/VERIFIER_LIMITS.md) showed that when a model
types the numbers itself, the verifier has to catch transcription, scale and rounding slips after
the fact, and every extractor or binder gap is a way for a wrong number to pass. The cheapest
fix is to stop the model typing numbers at all.

## Decision

1. The composer outputs a `TemplateAnswer`: `question`, `template` (prose), `queries`, `claims`.
   The prose contains placeholders such as `{uk}`. Each placeholder names one claim with an `id`,
   a `label`, a `unit` and a `provenance` (a `cell`, `derived` or `row_count` reference). There is
   no `text` or `value` field, so the schema gives the model nowhere to type a number.
2. The system renders. For each placeholder it takes the value from the executed cell or from the
   recomputed derived value, and formats it using the unit from the semantic layer (currency
   symbol, thousands separator, percent for ratio metrics, k/m/bn suffix when `scale` is set).
   The displayed precision (decimals, scale, true value, shown value) is recorded per claim.
3. The validator rejects: duplicate claim ids, unknown placeholders, claims never used in the
   template, stray braces, non-ASCII digits, placeholders glued to text or symbols (`£{a}`,
   `{a}0`, `{a}%`), values that cannot be formatted in the stated unit, and literal arguments in
   derived claims.
4. Any digit or spelled-out number outside a placeholder is passed through the existing extractor
   and comes back `UNSUPPORTED` (`number_outside_placeholder`). Dates, years, ordinals, list
   markers and URLs are excluded as before. Identifier-like tokens are allowed only if they
   appear in the executed result cells.
5. The rendered text is then verified as an ordinary answer: every rendered number must round-trip
   through the extractor to the same value, and the sentence-context binder (`context.py`) runs on
   it unchanged. It is a second check, not replaced.

Passage and parameter claims are not supported in template mode yet.

## What this removes

- Transcription errors (a digit changed, dropped or invented).
- Scale errors (thousands vs millions, percent vs ratio).
- Rounding and formatting inconsistencies (formatting is code, not model output).
- Model arithmetic: derived values are recomputed by the verifier, never accepted.

## What this does not remove

- A wrong claim chosen for a sentence ("EIRE led with {uk}"). The number is real but described as
  something else. Measured: with the binder switched off, at least 20 of the 30 old adversarial
  answers, converted to placeholder form, still pass. The binder stays and is the only defence here.
- A wrong question or wrong query for the question asked. The plain-English restatement
  (PHASE_1C.md) is the reader's check for that.
- Binder limits (docs/VERIFIER_LIMITS.md): literal name matching, "France and Germany together".
- Identifier tokens that happen to match a result cell.

## Measured (unit tests, not accuracy claims)

The 30 old adversarial answers rewritten in placeholder form: 0 of 30 falsely accepted with the
binder on. 33 further adversarial templates (wrong cell, placeholder reused in a contradictory
sentence, digits smuggled next to a placeholder, spelled-out numbers) behave as expected. These
cases were written by the author of the checks with synthetic data; 0 does not mean 0% on real
model output.

## Open risk

Small models may not follow the placeholder format: they may type numbers anyway, invent ids,
leave claims unused, or choose the wrong claim for a sentence. Failures are safe (flagged, not
accepted), but a model that fails often yields mostly flagged answers and wasted repair attempts.

Measure before building the composer on top of this, per model:

- placeholder adherence: share of answers with no digits outside placeholders;
- schema-valid rate (valid JSON, known ids, no unused claims);
- how often the binder returns AMBIGUOUS or a row/scope mismatch on real phrasing;
- repair attempts needed and token cost versus the free-typing composer.

Until those are measured, no claim is made that placeholders improve end-to-end results.
