# ADR-0009: Pre-registered fusion candidates for document search

Status: Accepted as a pre-registration. The shipped default is **not** changed by this ADR.

## Context

`va docs probe` runs keyword, vector, hybrid and reranked search on the 65 dev document questions.
With two real embedders (local `BAAI/bge-small-en-v1.5` and Gemini `gemini-embedding-2`, both on
the dev questions) the owner found that:

- the default hybrid (RRF constant 60, 50 candidates per method, equal weights) is significantly
  worse than vector-only on both (the paired bootstrap interval excludes zero);
- hybrid with keyword weight 0.25, RRF constant 10 and 20 candidates improves on the default on
  both embedders but does not beat vector-only.

The second finding came from a 27-setting grid scored on the same dev questions, so it is a result
of searching, not of a test.

## Pre-registered candidates

Fixed here, before any held-out scoring. `va docs heldout` scores exactly these three and reads
them back from the block below; the code also holds them and refuses to run if they differ.

<!-- candidates:begin -->
```yaml
candidates:
  A:
    method: vector
  B:
    method: hybrid
    rrf_k: 10
    depth: 20
    keyword_weight: 0.25
  baseline:
    method: hybrid
    rrf_k: 60
    depth: 50
    keyword_weight: 1.0
```
<!-- candidates:end -->

- **A** vector-only.
- **B** hybrid, RRF constant 10, 20 candidates per method, keyword weight 0.25 (vector weight 1).
- **baseline** the shipped default hybrid.

## Evidence (dev questions only)

Values below are copied from the probe reports named in the source column.

| embedder | method | recall@1 | MRR | source |
|---|---|---|---|---|
| gemini-embedding-2 | vector-only | 0.885 | 0.927 | `data\live_runs\20261120_234517_docs_gemini_embedding_2_768\report.md`, fusion table |
| gemini-embedding-2 | default hybrid (baseline) | 0.712 | 0.820 | same |
| gemini-embedding-2 | B (k=10 depth=20 kw 0.25) | 0.865 | 0.911 | same |
| bge-small-en-v1.5 | vector-only | 0.827 | 0.878 | `data\live_runs\20261120_234404_docs_local_bge_small_en_v15\report.md`, fusion table |
| bge-small-en-v1.5 | default hybrid (baseline) | 0.673 | 0.780 | same |
| bge-small-en-v1.5 | B (k=10 depth=20 kw 0.25) | 0.808 | 0.868 | same |

All six cells are on the 52 scored dev questions and were read from the two report files by the
project's CTO review, not typed from memory. The first Gemini table (0.88 / 0.93 vector, 0.71 / 0.82
hybrid) agrees with these to two decimals.

## Caveats

- One author wrote the documents and the questions.
- 52 of the 65 dev questions are scored (13 have no answer in the corpus).
- B was read off a 27-setting grid scored on those same 52 questions, so its dev advantage over the
  baseline is optimistic.
- 12 dev questions are paraphrases with low word overlap with their passage. Those favour vector
  search by construction.
- Real company data has identifiers, SKUs and codes that a vector model may blur and that keyword
  search matches exactly. Vector-only could do worse there than it does here.
- The held-out set has 12 questions, one with no answer. It cannot separate small differences.
- Two embedders is a small sample of embedders.

## Procedure

1. Commit this file (with the pasted numbers). `va docs heldout` refuses to run if the file is
   missing, not committed, or changed since it was committed.
2. Run `va docs heldout --candidates A,B,baseline --embedder <name>` once per embedder. It scores
   only these three candidates on the 12 held-out document questions and prints per-question
   differences. It records each embedder in `data/heldout_ledger.jsonl` and refuses a second run
   for the same embedder. (Deleting that file by hand defeats this; it is a guard against
   accidents, not against the owner.)
3. Decision rule, fixed here BEFORE any held-out scoring (no `data/heldout_ledger.jsonl` exists at
   the time of writing). The held-out set has 12 questions, so it can only veto, never prove:
   - B replaces the baseline as the shipped default only if, on BOTH embedders, B's held-out MRR is
     at least the baseline's and B ranks the first relevant passage worse than the baseline on at
     most one question.
   - A (vector-only) is preferred over B only if A's held-out MRR beats B's by more than 0.03 on
     BOTH embedders. Otherwise B is kept, so the keyword side stays available for identifiers and
     codes, which this synthetic corpus under-represents.
   - If B fails its test against the baseline on either embedder, the default stays as it is and
     the failure is recorded as a finding. The rule is not changed after seeing the output.
   - Whatever happens, the result is reported as "did not contradict the dev finding on 12 held-out
     questions", never as an accuracy figure.

## Consequences

The held-out questions stay out of every tuning step. They are used once per embedder, on settings
fixed in advance. They are still not an independent benchmark: the same author wrote them.
