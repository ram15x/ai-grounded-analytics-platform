# ADR-0010: Fusion held-out outcome (ADR-0009 rule applied)

Status: Accepted. The shipped default is **unchanged**. The choice of fusion setting is marked
**undecided**, not settled.

## What was run

`va docs heldout --candidates A,B,baseline` once per embedder, from ADR-0009 as committed, on the
12 held-out document questions (11 answerable, 1 with no answer). Reports are in
`data\live_runs\20261121_001646_docs_heldout_gemini_embedding_2_768` and
`data\live_runs\20261121_001708_docs_heldout_local_bge_small_en_v15`. The figures below are copied
from the owner's message of 2026-11-21; I did not see the report files.

| embedder | candidate | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| gemini-embedding-2 | A vector-only | 0.64 | 0.82 | 0.75 |
| gemini-embedding-2 | B (k=10, depth 20, kw 0.25) | 0.64 | 0.91 | 0.76 |
| gemini-embedding-2 | baseline (default hybrid) | 0.73 | 1.00 | 0.84 |
| bge-small-en-v1.5 | A vector-only | 0.64 | 0.91 | 0.77 |
| bge-small-en-v1.5 | B | 0.64 | 0.91 | 0.78 |
| bge-small-en-v1.5 | baseline | 0.73 | 1.00 | 0.86 |

B ranks lower than the baseline on H-D-04, H-D-05 and H-D-06 (Gemini) and on H-D-09 and H-D-11
(bge-small). Vector-only A did not retrieve H-D-11 in the top 10 with bge-small.

## Outcome by the pre-registered rule

The rule written before the run: B had to reach MRR at least equal to the baseline on both
embedders, with at most one question ranked worse. B fails on both embedders: its MRR is below the
baseline's, and it is worse on 3 and 2 questions (limit 1). **No change is made.** The default
stays RRF constant 60, 50 candidates per method, equal weights.

## What this does and does not show

- The dev finding (default hybrid significantly worse than vector-only) **did not replicate on the
  held-out questions; the direction reversed on both embedders.** On held-out the default scores
  highest of the three.
- This is evidence that the setting read off the dev grid was fitted to the dev set. Had B been
  adopted from the dev table, a worse setting would have shipped. It is not evidence that the
  default is good.
- Limits: 11 answerable questions, a difference of two or three questions, one author, one
  synthetic corpus, two embedders.
- Hypothesis, **unverified**: the held-out set has more lookup and exact-term questions than the
  dev set, where keyword matching helps, while the dev set holds 12 low-overlap paraphrases that
  favour vectors. One observed case fits it (H-D-11 with bge-small: vector search missed the
  passage, the keyword leg of the hybrid found it). One case is an observation, not a test of the
  hypothesis.
- Keeping the default also keeps exact keyword matching for identifiers and codes. That is a
  reason to keep it, not a measurement of it.

## Reranker (dev probe, local cross-encoder)

Reported by the owner from the dev probe with the local MiniLM cross-encoder
(`local_minilm_l6_cross_encoder`): MRR +0.105 on the default hybrid; no measurable effect on
vector-only or on B; recall@5 on the default hybrid fell from 1.00 to 0.96; 28 to 63 seconds per
probe run. Dev only, not tested on held-out, and the same caveats about one author apply.

**Decision:** no reranker in the default pipeline. The interface (local and hosted) stays in the
code. Recheck on harder public data.

## The 12 held-out document questions are SPENT

They have now informed a decision, so they cannot be used to choose between settings again. A new,
larger held-out set, ideally on real public documents and written without sight of the old one, is
needed for any further document-retrieval decision. `va docs heldout` now refuses to run (the
constant `SPENT_BY` in `documents/heldout_run.py` names this ADR). The file itself is not edited.

## Next

Revisit fusion with real public documents (docs/PUBLIC_DOCUMENTS_PLAN.md; the owner is asked
before anything is fetched) and a new held-out set. Until then the open question stays open.
