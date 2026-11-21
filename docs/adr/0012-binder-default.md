# ADR 0012: The passage label binder default becomes S2

Status: accepted by the owner, 2026-11-21.

Links: [ADR-0011](0011-binder-variants.md) (pre-registration, committed in
`19195ffff4e34d957a05757bf086b0fa9aaf1502`) and its amendment 1 (`5b8d83931b8301e5d8c1dcbe4e6a07b7b6e9247e`,
made before the locked half was scored). The decision rule of ADR-0011 is applied unchanged below.

## Result of the one locked run

`va docs binder-locked --candidates current,C1,C2,S1,S2,SL`, report
`data\live_runs\20261121_115112_binder_locked`, run by the owner once. Locked half: 664 correct
cases and 1289 swapped cases, generated from the synthetic documents whose key hashes to "locked".
False rejects are of the 664 correct cases, false accepts of the 1289 swapped cases.

| Variant | False rejects | False accepts | Rule of ADR-0011 |
|---|---|---|---|
| current (label words within 6 words of the number) | 193 of 664 | 251 of 1289 | the baseline |
| C1 (units skipped) | 193 of 664 | 232 of 1289 | not put forward: false rejects not lower |
| C2 (units skipped, 75% of label words) | 145 of 664 | 244 of 1289 | put forward |
| S1 (same sentence, units skipped) | 14 of 664 | 16 of 1289 | put forward |
| S2 (S1 plus the nearest-number rule, window 8) | 40 of 664 | 15 of 1289 | put forward |
| SL (same sentence only, units not skipped) | 14 of 664 | 25 of 1289 | put forward |

The absolute target of 0 false accepts was not met by any candidate; S2's 15 of 1289 is 1.2%.

## Decision

S2 becomes the shipped default of `documents/binding.py`: the label words that the passage uses must
occur in the sentence of the passage that holds the cited number (unit words such as "days" not
counted); when that sentence holds more than one number, the cited number must be the one nearest a
matched label word, or within 8 tokens of one. R2 and R3 are unchanged. The earlier rule is kept as
the named variant `current` (`documents.binding.CURRENT`), used by the study and by tests.

Reason: a false accept is the unsafe direction, because it lets a real passage number carry a label
that describes something else. S2 had the fewest false accepts (15 of 1289) and is the only
candidate with a rule for two numbers in one sentence. Its higher false-reject count against S1
(40 of 664 against 14) costs abstentions, not wrong answers.

## What this does not show

- S1 and S2 differ by one false accept (16 and 15 of 1289). That difference is within noise and is
  not evidence that S2 is safer. The choice of S2 rests on the design argument (the nearest-number
  rule), not on the count.
- False rejects on labels written by a model are unmeasured. In the study, every correct case takes
  its label from the number's own sentence, so a sentence-scoped rule passes it almost by
  construction (S1 and SL: 14 of 664). Real composer wording may be refused more often, or accepted
  where it should not be. The one real refusal seen (the time limit for claiming a late or damaged
  parcel, whose label words sat more than six words before the number) is accepted by S2 and
  refused by `current`; that is one case, not a measurement.
- The test set is synthetic, written by the project author, with generated labels and some swapped
  pairs that may still be fair. It is a screen for gross failures, not a benchmark, and no accuracy
  claim follows.
- Other-document swaps are still accepted. On the dev half all of S2's false accepts (33 of 873) are
  swaps onto another document, none onto another sentence of the same passage. The cause is generic
  label words ("charge", "policy", "order") that occur in the cited sentence of the other document.
  The locked report as relayed gives totals only, so the same split on the locked half is not
  stated here.

## Effect on verdicts, measured on dev data only

The locked half was not read or re-scored for this ADR. On the dev half (36 documents), case by
case from `current` to S2: of 461 correct cases, 192 refused by `current` are accepted and none
changes the other way; of 412 labels swapped inside the same passage, 19 accepted by `current` are
now refused and none changes the other way; of 461 labels swapped to another document, 30 accepted by
`current` are now refused, and 9 refused by `current` are now accepted. The 11 shipped sample answers
carry no passage claim, so the binder is not consulted and all 11 keep their verdicts (a test runs
them under both rules and compares). The FakeProvider end-to-end tests, including the five hostile
documents, pass with the new default.

## Consequences

The locked half is spent (`binder_locked.SPENT_BY` names this ADR); the command refuses to run again,
and a new rule needs a new test set and a new ADR. The refusal code for a label that is not in the
number's sentence is `label_not_in_passage_sentence`; `label_not_in_passage_window` is still used by
`current`. Assumption rows 46 and 47 record what stays unverified.
