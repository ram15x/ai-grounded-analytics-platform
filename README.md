# Verified Analyst

An AI analyst for business teams. Ask a question in plain English about company data and documents. The planned answer shows the SQL used, the source passages, and a verdict for every number. Numbers are checked by deterministic code before display; unsupported claims are flagged, never guessed. Scheduled reports go out only after an approval step.

**Current status: Phase 1c-2 tooling (`va live models` / `va live probe`, built but not yet run; no real provider has been called). The deterministic core exists (warehouse loader, semantic layer, metric compiler, SQL guard, number extractor, verifier, CLI). There is no LLM, no agent, no UI, and no accuracy figure is claimed.**

## Status

| Area | State |
|---|---|
| Research, architecture, ADRs | Done, see [docs/RESEARCH.md](docs/RESEARCH.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DECISIONS.md](docs/DECISIONS.md) |
| Warehouse (Postgres 16 via docker-compose; dims, facts, read-only views, read-only role) | Loaded and reconciled on the real UCI file (541,909 rows); tests use a synthetic 368-row fixture. `docker-compose.yml` has not been run. |
| Data profile and data-quality decisions | Decisions implemented with measured sizes on the real file, see [docs/DATA_PROFILE.md](docs/DATA_PROFILE.md). |
| Semantic layer (`semantic/metrics.yaml`, 11 metrics) | Implemented. Each metric is tested against an independent pandas derivation (shares the business rules, not the code). |
| Metric compiler (metric + dimensions + filters to SQL) | Implemented and tested |
| SQL guard (sqlglot, one SELECT, deny-lists, allow-listed relations, LIMIT, timeout, read-only role) | Implemented; 100+ adversarial and accepted cases. Not a proof of safety. |
| Number extractor | Implemented; 200+ cases. Limits listed in [docs/EXTRACTOR.md](docs/EXTRACTOR.md) |
| Verifier (claims schema, Decimal recomputation, label binding, verdicts) | Implemented, deterministic, no LLM |
| Mutation tests | Implemented as unit tests (see note below) |
| CLI: `va load`, `va profile`, `va verify answer.json` | Implemented; sample answers in `examples/` |
| Driver-claim verification | Design only, [ADR-0007](docs/adr/0007-driver-claim-verification.md). Not implemented. |
| Pre-commit config and GitHub Actions CI (lint, format, tests with a Postgres service, Python 3.11 and 3.12) | Files added. CI has not run on GitHub yet; nothing is pushed. |
| Model router, provider adapters (Groq, Gemini, OpenRouter), cassettes, scripted plan/compose/verify pipeline | Implemented and tested with **mocked HTTP and a fake provider only. No real provider has been called**; every provider assumption is listed in [docs/PHASE_1C1_ASSUMPTIONS.md](docs/PHASE_1C1_ASSUMPTIONS.md). No prompts for real models yet. |
| Documents (Phase 2a): store with character spans, keyword search, passage citations checked against stored text, retrieval metrics | Implemented and tested; synthetic corpus in `corpus/synthetic`. No vectors, reranker or router yet. See [docs/PHASE_2.md](docs/PHASE_2.md) |
| SQL agent with a real model, vector search and reranking, orchestration, approvals, scheduled reports | Does not exist |
| API, auth, web UI, deployment, observability | Does not exist |
| Evals and benchmark scores | Does not exist. **No accuracy figure is claimed.** |

### About the mutation tests

`tests/test_mutations.py` takes a fully verified answer, changes one number (a digit, the scale,
the rounding, the sign, the column, or adds an unsupported number) and checks that the verifier
refuses it. These are **unit tests of specific, synthetic corruptions. They are not an accuracy
claim**: they do not measure how many real model errors would be caught, and a passing suite says
nothing about errors the verifier was not written to see (for example a correct number that
answers the wrong question). Likewise the pandas oracle and the fixture are synthetic.

## Build phases

1. Warehouse, SQL agent, verification
2. Document RAG, router, citations
3. Multi-agent orchestration, approvals, n8n scheduled reports
4. Evals, CI gates, observability
5. Deployment, UI, demo video, write-up

Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 7.

## Development

Requires Python 3.11+.

```
python -m pip install -e ".[dev]"
pre-commit install   # once .pre-commit-config.yaml is added
ruff check . && ruff format --check .
pytest
```

Database tests run only when `VA_TEST_DATABASE_URL` points at a Postgres server where the user can
create databases and roles; otherwise they are skipped.

### Try the verifier

```
docker compose up -d --wait                # needs POSTGRES_PASSWORD; set VA_PG_PORT if 5432 is taken
export VA_DATABASE_URL=... VA_READONLY_PASSWORD=... VA_READONLY_DATABASE_URL=...
# optional: VA_CONNECT_TIMEOUT_S (seconds to wait for a connection, default 10, max 300)
va load --download                         # or: va load --source Online_Retail.xlsx
va profile-db                              # metric totals, with and without exact duplicates
va docs ingest corpus/synthetic            # then: va docs search "how long do refunds take"
va verify examples/01_verified.json        # exit 0 when fully verified, 1 otherwise
va verify examples/11_known_limit_share_of_global_total.json   # exit 1 on purpose: a known limit
```

The sample answers are written against the synthetic fixture (`va load --source tests/fixtures/mini_retail.csv`);
against the real data their numbers will differ.


API keys, when they are needed, are read from environment variables only. Nothing in this repo stores a key.

## Data and known limits

Measured on the real file (`Online Retail.xlsx`, 541,909 rows, 2010-12-01 to 2011-12-09; details in
[docs/DATA_PROFILE.md](docs/DATA_PROFILE.md)):

- Net revenue is 9,769,872.05 (gross 10,666,684.54, cancelled 891,116.27, discounts 5,696.22); 19,960 orders; 4,338 identified customers.
- **Exact duplicates stay counted.** 5,268 rows (0.97%). Without them net revenue would be 9,748,131.07 (-0.22%) and units -0.28%; orders and customers do not change.
- **24.93% of rows (135,080) have no customer.** They are kept in revenue; 15.04% of net revenue (1,469,806.24) cannot be tied to a customer, and customer metrics ignore them.
- **Units sold means merchandise units** (postage, fees and manual lines excluded). Counting every sale row would give 11,380 more.
- 2,517 rows with price <= 0 are `adjustment` and excluded from all metrics; 2,995 rows are non-merchandise codes.
- Gold check: 15 author-written questions, 11 expressible as catalogue queries, 11 agree, 4 not expressible ([docs/GOLD_DIFFS.md](docs/GOLD_DIFFS.md)). The gold author also wrote the semantic layer, so agreement is weak evidence. No accuracy figure is claimed.
- The verifier's sentence binder is a heuristic with documented limits: [docs/VERIFIER_LIMITS.md](docs/VERIFIER_LIMITS.md).

## Data and licence

Code: MIT, see [LICENSE](LICENSE).

Dataset: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33. Licensed CC BY 4.0. The dataset is not stored in this repository.
