# ADR 0011: Pre-registered passage label binder variants

Status: proposed, pre-registration. Written before the locked half of the binder study was scored.

## Context

The passage label binder (`documents/binding.py`, rule R1: every label word the passage uses must
sit within 6 words of the number) refused correct labels on real model wording. The binder study
(`documents/binder_study.py`, `va docs binder-study`) measures variants on generated cases from the
80 synthetic documents: correct cases (a label from the number's own sentence, four styles) and
swapped cases (the same label on a different number, in the same passage or another document).
Documents are split by a hash of the document key; the dev half is used to choose candidates, the
locked half is built only by `documents/binder_locked.py`.

On the dev half the shipped rule has false rejects 229 of 461 correct cases and false accepts 73
of 873 swapped cases. No variant measured on dev reaches 0 false accepts, so the absolute target
of 0 of N is stated here as not expected to be met.

## Decision rule (fixed before the locked half is scored)

A candidate is put forward for the owner's review only if, on the locked half, its false accepts on
the swapped cases are no higher than `current`'s AND its false rejects on the correct cases are
lower than `current`'s. Otherwise the shipped rule stays. No default changes in this ADR or in the
run; the owner decides after reading the table. The locked half is scored once (ledger).

## Candidates

All use window 6, no ignore-repeated rule, no head-noun rule. `skip_units` means unit and quantity
words ("days", "units", "percent") do not count as label evidence.

<!-- candidates:begin -->
```yaml
candidates:
  current: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: false}
  C1: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: true}
  C2: {frac: 0.75, window: 6, ignore_repeated: false, head_noun: false, skip_units: true}
  C3: {frac: 0.6, window: 6, ignore_repeated: false, head_noun: false, skip_units: true}
```
<!-- candidates:end -->

## Limits

Synthetic corpus written by the project author; labels are generated, not written by a model; some
swapped pairs may still be fair descriptions. This is a screen for gross failures, not a benchmark,
and says nothing about accuracy on real answers.
