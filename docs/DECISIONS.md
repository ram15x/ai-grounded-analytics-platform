# Decisions and open questions

| ADR | Title | Status |
|---|---|---|
| [0001](adr/0001-deterministic-verification.md) | Deterministic number verification | Accepted (design) |
| [0002](adr/0002-postgres-and-duckdb.md) | Postgres + DuckDB | Accepted (design) |
| [0003](adr/0003-model-router.md) | Model router and data-sensitivity rule | Accepted (design) |
| [0004](adr/0004-postgres-job-queue.md) | Postgres-backed job queue | Accepted (design) |
| [0005](adr/0005-hybrid-retrieval.md) | Hybrid retrieval | Accepted; reranker undecided |
| [0006](adr/0006-dataset-choice.md) | Dataset choice | **Proposed, awaiting owner** |

"Accepted (design)" means agreed as a design direction in Phase 1a, not yet proven by code.

## Open questions for the owner

1. Dataset: is CC BY 4.0 UCI Online Retail as the primary dataset acceptable, and do you want Olist (non-commercial, share-alike) as well?
2. Repository licence: MIT, Apache-2.0 or none for now? No LICENSE file has been added.
3. Hosting: pick the API host for the demo (Render vs Cloud Run vs other), knowing the cold-start and billing-account trade-offs in docs/RESEARCH.md.
4. Hosted model providers: which free accounts do you already have (Google AI Studio, Groq, OpenRouter)?
5. The repo sits inside a OneDrive folder. Git inside OneDrive can corrupt `.git` during sync; consider excluding it or moving the repo.
