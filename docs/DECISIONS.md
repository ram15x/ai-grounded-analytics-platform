# Decisions and open questions

| ADR | Title | Status |
|---|---|---|
| [0001](adr/0001-deterministic-verification.md) | Deterministic number verification | Accepted (design) |
| [0002](adr/0002-postgres-and-duckdb.md) | Postgres + DuckDB | Accepted (design) |
| [0003](adr/0003-model-router.md) | Model router and data-sensitivity rule | Accepted (design) |
| [0004](adr/0004-postgres-job-queue.md) | Postgres-backed job queue | Accepted (design) |
| [0005](adr/0005-hybrid-retrieval.md) | Hybrid retrieval | Accepted; reranker undecided |
| [0006](adr/0006-dataset-choice.md) | Dataset choice | **Decided: UCI Online Retail only** (owner, Phase 1b) |
| [0007](adr/0007-driver-claim-verification.md) | Verifying "driver" claims | Proposed (design only) |
| [0009](adr/0009-document-retrieval-fusion.md) | Pre-registered fusion candidates for document search | Accepted as a pre-registration; default unchanged |

"Accepted (design)" means agreed as a design direction in Phase 1a, not yet proven by code.

## Decided by the owner (Phase 1b)

- Dataset: UCI Online Retail only (CC BY 4.0, credited in the README). Olist is not used.
- Repository licence: MIT, copyright holder Ram Narayanan.
- Hosting: deferred.

## Open questions for the owner

1. Hosted model providers: which free accounts do you already have (Google AI Studio, Groq, OpenRouter)? Needed from Phase 2.
2. The repo sits inside a OneDrive folder. Git inside OneDrive can corrupt `.git` during sync; consider excluding it or moving the repo.
3. Exact duplicate rows are flagged but still counted in metrics (docs/DATA_PROFILE.md, decision 7). Confirm, or choose to exclude them, after the real-data profile.
