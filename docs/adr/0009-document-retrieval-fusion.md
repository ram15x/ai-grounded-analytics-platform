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

Numbers the owner has reported in chat are quoted as given. Everything else is to be pasted from the
probe's `report.md` before the held-out run; a blank cell is not a result.

| embedder | method | recall@1 | MRR | source |
|---|---|---|---|---|
| gemini-embedding-2 | vector-only | 0.88 | 0.93 | owner, chat |
| gemini-embedding-2 | default hybrid (baseline) | 0.71 | 0.82 | owner, chat |
| gemini-embedding-2 | B | OWNER TO PASTE | OWNER TO PASTE | report.md |
| bge-small-en-v1.5 | vector-only | OWNER TO PASTE | OWNER TO PASTE | report.md |
| bge-small-en-v1.5 | default hybrid (baseline) | OWNER TO PASTE | OWNER TO PASTE | report.md |
| bge-small-en-v1.5 | B | OWNER TO PASTE | OWNER TO PASTE | report.md |

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
3. This ADR sets no decision rule. Changing the shipped default is a separate decision, made by
   the owner after reading the per-question output.

## Consequences

The held-out questions stay out of every tuning step. They are used once per embedder, on settings
fixed in advance. They are still not an independent benchmark: the same author wrote them.
