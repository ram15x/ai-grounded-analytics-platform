# ADR 0013: Pre-registered test of model-written passage labels

Status: proposed, pre-registration. Written before any label was generated and before anything was
scored.

## Context

ADR-0012 made S2 (sentence scope) the passage label binder default. Its numbers came from labels
that a program takes from the number's own sentence, which a sentence rule passes almost by
construction (ADR-0012, "What this does not show"). The real refusals seen under the old rule
(`20261121T064538Z`, 4 labels over 2 documents) were not rescued by S2 in an offline replay: their
subject words sit in another sentence or line than the number (assumptions row 46). This ADR
registers a test in which a real model writes the labels.

## Procedure (fixed here, before any label exists)

1. Sentences. `binder_model.select_sentences`: every number-carrying sentence of the five hostile
   documents named in `eval/router_questions/dev.yaml` (the seed), at most 3 per document, then
   sentences of the other synthetic documents round robin in a fixed hash order, at most 3 per
   document, up to 80 sentences (target; 40 is the minimum for a valid run). One number per
   sentence. The corpus is the synthetic, public corpus only.
2. Labels. One run of `grounded docs binder-model-labels` against one Groq model: 8 requests of 10
   sentences (prompt `label.v0`, JSON mode, `reasoning_effort` as configured), recorded in
   cassettes. The model sees the sentence with the number marked and writes a 2 to 6 word label.
   A label with no content word is dropped and counted. A refused or failed request is not retried
   by hand and the run is not repeated: the ledger allows one generation. Real refusals from a
   report (`--seed-run`) may be added as seed claims with the label the real model wrote.
3. Cases. For every claim: one correct case (the model's label on its own number) and up to two
   matched swapped cases from the same passage: the same label on another number of the same chunk
   (different sentence) and on a number of another document, dropped when at least half of the
   label's words occur in the other number's own sentence (`binder_study._fair`).
4. Split. Documents split dev or locked by `binder_study.half_of` (hash of the document key). The
   dev half may be scored any time. The locked half is scored only by `grounded docs binder-model-locked`,
   which refuses unless this ADR is committed and unchanged, the labels file is committed and
   unchanged, and the ledger has no earlier locked run. Nobody reads or tunes on the locked claims.
5. Candidates and report: `current`, `S2` and `SL` (below), nothing else. False rejects (correct
   cases refused) and false accepts (swapped cases accepted) as k of n with a 95% Wilson interval,
   per half, by swap kind, with the list of refused labels (dev only).

## Decision rule

None that changes a default. The report is read by the owner; any change of default needs its own
ADR. Not expected: 0 false accepts, or enough n to separate current, S2 and SL with these
intervals; the report states what the intervals allow.

<!-- candidates:begin -->
```yaml
candidates:
  current: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: false, scope: window, nearest: false}
  S2: {frac: 1.0, window: 8, ignore_repeated: false, head_noun: false, skip_units: true, scope: sentence, nearest: true}
  SL: {frac: 1.0, window: 6, ignore_repeated: false, head_noun: false, skip_units: false, scope: sentence, nearest: false}
```
<!-- candidates:end -->

## Limits

Synthetic documents written by the project author; one model, one run, one prompt (a first draft);
the label model sees only one sentence, while a composer sees passages and a question; the swapped
cases are built by the generator, not by a model. With about 40 claims per half the intervals are
wide. This is a screen, not a measurement of accuracy on real answers, and not a basis for any
accuracy claim.

## Amendment 1 (before any label was generated)

Written after the first generation attempt failed on its first request with a rate limit
(`all providers failed (groq:rate_limit)`); no label existed and nothing was scored, so this
changes no candidate and no data. Procedure step 2 reads as follows from now on:

- The run is resumable. Its cassettes live in one fixed folder (`data/binder_model_labels_work`).
  A batch already recorded there is answered from the cassette without a request; only missing
  batches go to the provider.
- A rate limit (the router giving up on a 429) or the request cap ends the run at that batch.
  Batches before it are kept, nothing after it is tried, nothing is retried by hand.
- The ledger entry and the labels file are written only when every batch has been answered. A
  sentence whose answer has no usable label is still counted as dropped, as before; it is not
  asked again, because the same request would replay the same answer. The report says how many
  labels were written, how many batches were answered and how many came from cassettes.
- An incomplete run does not spend the one allowed generation. After a complete run the ledger
  refuses another, as before.

