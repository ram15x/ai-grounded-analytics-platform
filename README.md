# Verified Analyst

An AI analyst for business teams. Ask a question in plain English about company data and documents. The planned answer shows the SQL used, the source passages, and a verdict for every number. Numbers are checked by deterministic code before display; unsupported claims are flagged, never guessed. Scheduled reports go out only after an approval step.

**Current status: Phase 1a. Research, design and repo skeleton only. No product features exist yet.**

## Status

| Area | State |
|---|---|
| Research (models, hosting, datasets, competitors) | Done, see [docs/RESEARCH.md](docs/RESEARCH.md) |
| Architecture and verification design | Written, not implemented, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Decisions (ADRs) | Written; dataset choice awaiting owner, see [docs/DECISIONS.md](docs/DECISIONS.md) |
| Repo skeleton: pyproject, package, one import test | Exists |
| Pre-commit config and GitHub Actions CI (lint + tests) | Drafted and tested outside the repo, **not yet in this repo**: my tools may not write those two protected paths. Add `.pre-commit-config.yaml` and `.github/workflows/ci.yml` by hand. |
| Data warehouse and datasets | Does not exist |
| SQL agent | Does not exist |
| Number verifier and semantic layer | Does not exist |
| Document RAG | Does not exist |
| Router, multi-agent orchestration, approvals, n8n reports | Does not exist |
| API, auth, web UI | Does not exist |
| Evals and benchmark scores | Does not exist. No accuracy figure is claimed. |
| Deployment, observability | Does not exist |

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

API keys, when they are needed, are read from environment variables only. Nothing in this repo stores a key.
