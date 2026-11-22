# AI Grounded Analytics Platform

[![CI](https://github.com/ram15x/ai-grounded-analytics-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ram15x/ai-grounded-analytics-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-1E3A6D)
![License](https://img.shields.io/badge/license-MIT-1E3A6D)

Ask business questions in plain English over your data and documents. Every number in the answer
is checked against the database or a stored passage before it is shown. If a number cannot be
traced to its source, the answer is withheld.

**[Live demo](https://ram15x.github.io/ai-grounded-analytics-platform/)** (recorded answers, no sign-up)

![An answer with every figure tied to its source](docs/assets/answer.png)

## Why

Language models write confident numbers that may not exist in your data. This platform never lets
the model type a number. The model plans a query and writes the sentence with placeholders.
Deterministic code then fills each placeholder from the query result or a quoted passage and checks
it. Anything it cannot check is withheld.

## How it works

```mermaid
flowchart LR
    Q[Question] --> R{Route}
    R -->|data| P[Plan a metric query]
    R -->|documents| S[Retrieve passages]
    R -->|out of scope| A[Withhold, with the reason]
    P --> G[SQL guard, read-only role] --> W[(Warehouse)]
    W --> C[Compose with placeholders]
    S --> C
    C --> V{Verify every figure}
    V -->|all tied| OK[Answer with tie-out notes]
    V -->|any not tied| A
```

1. **Route.** The model decides whether the question needs data, documents, both, or neither.
2. **Plan.** Data questions become a query over *approved* metrics only (a semantic layer), not free SQL.
3. **Guard.** The SQL is parsed (sqlglot), allow-listed and run under a read-only database role.
4. **Compose.** The model writes the answer with placeholders such as `{c1}`, never digits.
5. **Verify.** Code fills each placeholder from a result cell or a stored passage and checks the label
   next to it. The answer is shown only if every figure ties out.

## Features

- **Verified answers.** Each figure shows its verdict (tied, derived, or quoted), the SQL, the result
  row, or the source passage.
- **Bring your own data.** Upload CSV, TSV or Excel, review column roles, and define metrics from a
  closed set (count, sum, average, min, max, distinct count, one filter). A metric is published only
  if it reproduces a figure you already know. Personal-looking columns are never loaded.
- **Documents.** Attach PDF (text layer), Markdown or text files. Quotes are checked against the stored
  text, and text that looks like instructions to a model is withheld from prompts.
- **Your key, never stored.** Groq, OpenRouter or Gemini keys are pasted into the page, used for one
  request and never written anywhere. The page shows the key's remaining tokens when the provider
  reports them.
- **Sensitivity rule.** Data is internal by default. Internal data is never sent to a hosted model,
  only to a local one (Ollama).
- **Approved reports.** A report is drafted, then written only after a named approval.

## Quick start

Requires Python 3.11+ and Docker (for Postgres with pgvector).

```bash
git clone https://github.com/ram15x/ai-grounded-analytics-platform
cd ai-grounded-analytics-platform
python -m pip install -e ".[dev]"

# view the recorded answers, no database or key needed
grounded serve                       # http://127.0.0.1:8000
```

To ask your own questions on your own data:

```bash
export POSTGRES_PASSWORD=<admin password>
docker compose up -d --wait

export GA_DATABASE_URL="postgresql://ga_admin:$POSTGRES_PASSWORD@127.0.0.1:5432/grounded_analytics"
export GA_READONLY_PASSWORD=<read-only password>
export GA_READONLY_DATABASE_URL="postgresql://ga_readonly:$GA_READONLY_PASSWORD@127.0.0.1:5432/grounded_analytics"

grounded serve --workspace           # upload data, then ask; this computer only
```

On Windows PowerShell, set variables with `$env:NAME = "value"`.

## Modes

| Command | What it serves | Needs |
|---|---|---|
| `grounded serve` | Recorded answers with live checks | Nothing |
| `grounded serve --workspace` | Upload your data and documents, then ask anything | Postgres; a model key or local Ollama |
| `grounded serve --byok` | Public tryout: 12 questions (3 easy, 3 medium, 3 hard, 3 your own), one per group per visitor | Postgres with the sample warehouse |
| `grounded ask "<question>"` | One answer in the terminal, with a JSON trace | Postgres; `--replay` or a key |
| `grounded report draft` / `approve` | A report written only after named approval | Recorded runs |
| `grounded verify answer.json` | Check a hand-written answer against the warehouse | Postgres |

The sample warehouse is the UCI *Online Retail* dataset: `grounded load --download`, then
`grounded docs ingest corpus/synthetic --sensitivity public` for the sample documents (public, so a hosted model may read them).

## Configuration

| Variable | Purpose |
|---|---|
| `GA_DATABASE_URL` | Admin connection, used for loading data |
| `GA_READONLY_DATABASE_URL` | Read-only connection, used for every query an answer runs |
| `GA_READONLY_PASSWORD` | Creates or refreshes the read-only role during a load |
| `GA_PG_PORT` | Host port for the Docker Postgres (default 5432) |
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY` | Only for terminal commands; the web page takes the key in the page instead |

Models and providers are in `config/router.yaml`, retrieval in `config/retrieval.yaml`, the tryout
questions in `config/tryout.yaml`, and the metric catalogue in `semantic/metrics.yaml`.

## Project structure

```
src/grounded_analytics/
  verifier.py, extract.py, claims.py   figure extraction and verification (no model)
  compiler.py, guard.py, semantic.py   metric queries, SQL guard, semantic layer
  llm/                                 router, provider adapters, prompts, recordings
  documents/                           document store, hybrid search, passage citations
  datasets/                            upload, profiling, reconciliation
  web/                                 the web app (single page, no build step)
config/  semantic/  corpus/  eval/  tests/  docs/
```

## Testing

```bash
ruff check . && ruff format --check .
pytest                                   # database tests run when GA_TEST_DATABASE_URL is set
```

CI runs lint and the full suite on Python 3.11 and 3.12 against Postgres with pgvector.

## Results and limitations

What has been measured, each with its sample size, is in [docs/RESULTS.md](docs/RESULTS.md).
Highlights: 30 of 30 adversarial wrong answers were refused, and a real upload-to-answer run on
Groq `openai/gpt-oss-20b` returned a verified answer. **No accuracy figure is claimed.**

Limitations:

- A tied figure matches its source. That does not prove the metric was the right one to ask for.
- Document answers prove a number appears in a passage, not that the passage was read correctly.
- No sign-in and no multi-user separation. Uploads run only on your own computer.
- Scanned PDFs are refused (no OCR). Real-world PDF layouts are untested.
- The locked held-out evaluation has not been run.

## Documentation

- [Write-up](docs/WRITEUP.md): the problem, the design, what was measured
- [Architecture](docs/ARCHITECTURE.md) and [decision records](docs/DECISIONS.md)
- [Verifier limits](docs/VERIFIER_LIMITS.md): what the checker cannot catch
- [Results](docs/RESULTS.md): every measurement with its sample size
- [Hosting](docs/HOSTING.md): what is deployed and why the rest is local

## License

Code: MIT, see [LICENSE](LICENSE).
Dataset: Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository,
https://doi.org/10.24432/C5BW33, CC BY 4.0. Not stored in this repository.
