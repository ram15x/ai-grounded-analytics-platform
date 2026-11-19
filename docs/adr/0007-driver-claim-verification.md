# ADR-0007: Verifying "driver" claims

Status: PROPOSED (design only, not implemented)

## Problem

The verifier (ADR-0001) checks that a number equals a cell, or a recomputation from cells.
Analysts also say things like "revenue fell mainly because of France" or "growth came from
higher prices, not volume". These are claims about **cause or attribution**. A number can be
right while the attribution is wrong, and a free-text sentence cannot be checked directly.

## Decision

Do not let the model assert drivers in prose. A driver claim must be expressed as a
**decomposition the verifier can recompute**, and the sentence is only allowed if the
decomposition supports it. The word "driver" is reserved for what arithmetic shows;
anything else is phrased as a hypothesis and flagged as unverified.

### 1. Contribution decomposition (which segment moved the total)

For a metric M that is additive over a dimension D (net revenue by country), between periods 0 and 1:

    delta_total = sum_s (M1_s - M0_s)
    contribution_s = (M1_s - M0_s) / delta_total        (share of the change)

A claim "France drove the decline" is checked as: `contribution_France` is the largest in
magnitude with the same sign as `delta_total`, and exceeds a stated threshold (default 50% for
"mainly", 30% for "a major factor"). Segments absent in one period count as 0 for that period.
If `delta_total` is 0 or the segments' changes largely offset (sum of absolute changes much
larger than |delta_total|), contributions are unstable; the verifier returns `AMBIGUOUS`.

Only additive metrics qualify. Ratios (AOV, rates) need mix decomposition (below); the
semantic layer marks each metric `additive: true|false` (a new field).

### 2. Price-volume-mix (PVM) decomposition (why revenue changed)

For revenue R = sum_i p_i x q_i over products i, between periods 0 and 1:

- Volume effect = (Q1 - Q0) x P0_avg, where Q is total units and P0_avg the period-0 average price
- Mix effect = sum_i (share1_i - share0_i) x Q1 x p0_i, minus the average-price part (price differences across the mix)
- Price effect = sum_i (p1_i - p0_i) x q1_i

The three effects must sum exactly to delta R. This is verified with `Decimal` arithmetic and
is a hard check: a decomposition that does not reconcile is rejected as `DERIVATION_MISMATCH`.
New and discontinued products need an explicit rule (their whole revenue goes to mix; stated in the output).
The ordering of effects (price at new volume vs old volume) changes the split, so the convention
is fixed and printed with every result.

### 3. Claim form

A new provenance kind `driver`:

    {kind: "driver", method: "contribution" | "pvm", metric, dimension, period_a, period_b,
     subject, threshold}

The verifier runs the decomposition itself from the warehouse using compiled metric queries
(never model-written SQL), then checks the sentence's word against the result: "mainly",
"entirely" and "offset" map to thresholds; "because of" is allowed only for `contribution` >= threshold
or the largest PVM effect.

### 4. Verdicts

Reuses existing ones: `VERIFIED_DERIVED` when the decomposition supports the wording;
`DERIVATION_MISMATCH` when it contradicts it or does not reconcile; `AMBIGUOUS` when effects are
unstable or the wording is stronger than the data can support. A new `NOT_CAUSAL` note is attached
to every driver verdict.

## What this does not do

- It does not establish causation. It shows **arithmetic attribution** within the data: France
  accounts for X% of the change, not that France's customers behaved differently for a reason.
- It cannot verify external causes (a promotion, a stockout, weather) unless that fact is itself
  in a cited passage or table, in which case only the quote is verified.
- Results depend on the decomposition convention; two defensible conventions give different splits.

## Consequences

- Needs `additive` metadata in `semantic/metrics.yaml`, and a period-over-period query shape in the compiler.
- Needs tests against an independent pandas PVM derivation, as for the metrics, plus mutation tests (swap price and volume effects, drop a segment).
- Implementation is deferred until the SQL agent exists and the driver claims it actually produces are known.
