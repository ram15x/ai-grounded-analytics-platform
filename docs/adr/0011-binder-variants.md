# ADR 0011: Pre-registered passage label binder variants

Status: proposed, pre-registration. Written before the locked half of the binder study was scored.

## Context

The passage label binder (`documents/binding.py`, rule R1: every label word the passage uses must
sit within 6 words of the number) refused correct labels on real model wording. The binder study
(`documents/binder_study.py`, `grounded docs binder-study`) measures variants on generated cases from the
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

## Candidates (amendment 1, replaces the original list of four)

Amendment 1 was made before the locked half was scored. It amends the ADR as committed in
`19195ffff4e34d957a05757bf086b0fa9aaf1502`, whose candidates were `current`, `C1`, `C2` and `C3`.
The locked half has not been read or scored: the ledger (`data/binder_locked_ledger.jsonl`) does
not exist. Reasons for the change, all from the dev half only: `C3` raised false accepts above
current's (80 of 873 against 73) and is dropped; every window variant refused "distant" labels 65 of
65, so a family that checks the label against the whole sentence of the cited number was added. The
decision rule is unchanged.

- `current`: the shipped rule (label words within 6 words of the number).
- `C1`: `current` with unit and quantity words ("days", "units", "percent") not counted as evidence.
- `C2`: `C1` with 75% of the label words the passage uses required in the window.
- `S1`: label words must occur in the same sentence of the passage as the cited number (all of the
  label words the passage uses; units skipped). Sentence from `context.sentence_span`, the splitter
  the verifier uses.
- `S2`: `S1`, and when that sentence holds more than one number, the cited number must be the number
  closest to a matched label word or within 8 tokens of one.
- `SL`: the sentence rule only, units not skipped.

Dev half, for reference (false rejects of 461 correct, false accepts of 873 swapped): current 229 and
73; C1 228 and 58; C2 153 and 66; S1 6 and 37; S2 37 and 33; SL 6 and 52. Every false accept of the
sentence family is a swap onto another document; none is a swap inside the same passage. Correct
cases take their label from the number's own sentence, so the sentence family passes them almost by
construction; its false-reject figure is a floor, not evidence that it accepts real composer
wording. The one real refusal seen so far (a label for the time limit for claiming a late or damaged
parcel, beside the number) is covered by a unit test; one case is not a measurement.

<!-- candidates:begin -->
```yaml
candidates:
  current: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: false, scope: window, nearest: false}
  C1: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: true, scope: window, nearest: false}
  C2: {frac: 0.75, window: 6, ignore_repeated: false, head_noun: false, skip_units: true, scope: window, nearest: false}
  S1: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: true, scope: sentence, nearest: false}
  S2: {frac: 1.0, window: 8, ignore_repeated: false, head_noun: false, skip_units: true, scope: sentence, nearest: true}
  SL: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: false, scope: sentence, nearest: false}
```
<!-- candidates:end -->

## Limits

Synthetic corpus written by the project author; labels are generated, not written by a model; some
swapped pairs may still be fair descriptions. This is a screen for gross failures, not a benchmark,
and says nothing about accuracy on real answers.
