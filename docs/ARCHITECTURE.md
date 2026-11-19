# Verified Analyst: Architecture

Status: design only. Nothing in this document is implemented yet. See README for the status table.

## 1. Goal and non-goals

A business user asks a plain-English question about company data and documents. The system returns:

1. the answer text,
2. the SQL it ran,
3. the source passages it used,
4. a verdict for every number in the answer.

Every number is checked by deterministic code before display. A number that cannot be tied to an executed query, a recomputation or a source passage is flagged. It is never guessed and never silently corrected.

Non-goals: write access to company data, free-form dashboards, fine-tuning models, handling confidential production data on free hosted LLM tiers.

## 2. Components

```
            +------------------+        +-----------------------+
 n8n  ----> | ingestion jobs   | -----> | Postgres (warehouse)  |
 (cron,     | (CSV/docs ->     |        | + pgvector, FTS       |
 webhooks)  |  tables, chunks) | -----> | DuckDB (analytics     |
            +------------------+        |  over parquet)        |
                                        +-----------+-----------+
 Web UI  <---->  FastAPI  <---->  job queue  <---->  LangGraph workers
 (answer, SQL,   (auth, API,      (Postgres-       router -> SQL agent
  citations,      approvals)       backed)          \-> doc agent
  trace view)                                        -> verifier -> answer
                                         |
                                  model router (hosted free tier / local Ollama)
                                         |
                                  observability (traces, cost, latency)
```

| Component | Choice | Notes |
|---|---|---|
| Ingestion and scheduling | n8n | Triggers ingestion, scheduled reports and approval notifications. n8n is orchestration only; it holds no business logic and calls the FastAPI backend. |
| Warehouse | Postgres (Neon free in the demo, Docker locally) | Source of truth for tabular data, plus pgvector and full-text search for RAG. |
| Analytical engine | DuckDB | Fast aggregate queries over Parquet exports, also used by the eval harness (BIRD/Spider databases are SQLite, loaded into DuckDB). See ADR-0002. |
| Retrieval | Hybrid: keyword (Postgres FTS/BM25-style) + vector (pgvector) + reranker | Reciprocal-rank fusion, then a cross-encoder reranker. Reranker choice deferred to Phase 2 (ADR-0005). |
| Agents | LangGraph | Router, SQL agent, doc agent, verifier. The verifier is plain Python, not an LLM. |
| Model router | Python module | Per-task model choice, rate-limit aware, data-sensitivity aware. ADR-0003. |
| API | FastAPI | REST plus server-sent events for streaming trace steps. |
| Job queue | Postgres-backed (e.g. a `jobs` table with `SKIP LOCKED`) | Avoids another free-tier service. Revisit if throughput demands. ADR-0004. |
| Auth | Session or JWT in FastAPI, roles: viewer, analyst, approver, admin | Approvers gate scheduled reports. |
| Web UI | Small SPA or server-rendered pages | Must show: answer with per-number verdict badges, SQL, source passages with highlighted spans, trace view of agent steps. |
| Container / CI | Docker Compose locally, GitHub Actions | CI in this skeleton runs lint and tests only. |
| Observability | OpenTelemetry-style traces, plus a table of per-call tokens, cost estimate, latency, provider, model | Self-hosted viewer or plain tables first. Vendor choice deferred. |

Secrets: only environment variables (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `DATABASE_URL`, etc.). No `.env` file is committed and no key appears in any tracked file. Names above are examples; final names are set in Phase 1.

## 3. Request flow

1. UI posts a question to FastAPI; a job is queued.
2. Router classifies: SQL-only, document-only, or mixed.
3. SQL agent: reads the semantic layer and schema, proposes a structured query plan, generates SQL, validates it (section 5), executes read-only, stores the result with a `query_id`.
4. Doc agent: hybrid retrieval, returns passages with `passage_id` and character spans. It has no SQL tool and no write tools.
5. Answer composer (LLM) writes the answer as structured claims, each pointing at result cells, a derivation, or passage spans.
6. Verifier (deterministic, section 4) checks every number in the final text.
7. UI shows the answer with verdicts. Unverified numbers are visibly flagged. The user can inspect SQL, rows and passages.

## 4. Verification design

This is the core of the product. The old synthetic checker is not reused; only these ideas carry over: check numbers separately from prose, keep the checker free of the generating model, and report per-number verdicts.

### 4.1 Pipeline

**Step A: structured claims from the composer.** The composer must output JSON: prose with claim markers, and for each claim a `value`, `unit`, `label`, and exactly one provenance reference:

- `cell`: `{query_id, row_key, column}`
- `derived`: `{op, args}` where each arg is a `cell` or a literal, from a whitelisted set of operations (sum, difference, ratio, percent_change, mean, min, max, count, rank)
- `passage`: `{passage_id, span}`

**Step B: independent number extraction from the final text.** Do not trust the composer's claim list. A deterministic extractor scans the rendered prose and finds every numeric expression: digits with thousands separators and decimals, currency symbols and codes, percentages, magnitude suffixes (k, M, B, "million"), spelled-out numbers ("twelve"), ranges, and fractions. It normalises each to `Decimal` with unit, scale and displayed precision, and records the character span. It excludes dates, years, ordinals, list numbering and identifiers using explicit rules, and when a number is ambiguous (for example "2011" could be a year or a count) it marks it `AMBIGUOUS` rather than skipping it silently.

**Step C: reconciliation.** Every extracted number must map to exactly one claim from step A. Extracted numbers with no claim become `UNSUPPORTED`. Claims whose text does not appear in the prose are dropped and logged.

**Step D: provenance check.**
- `cell` claims: look up the stored result table (immutable, hashed). Compare after applying the displayed precision: `round(cell_value, displayed_decimals) == displayed_value`, with unit and scale normalisation (for example 1,234,567 shown as "1.2M" is accepted; shown as "1.3M" is not). Match is exact within rounding, not fuzzy.
- `derived` claims: the verifier recomputes with `Decimal` from the cited cells. The LLM never does arithmetic that is trusted. A mismatch beyond rounding gives `DERIVATION_MISMATCH`.
- `passage` claims: the number must appear in the cited passage span after the same normalisation. This proves the document contains the number, not that the document is right.

**Step E: label binding.** To stop a number matching an unrelated cell by coincidence, a cell claim must also match its label to the column name or alias, and the unit must be compatible (currency vs count vs percent). Binding uses the semantic layer's synonyms (4.2), not an LLM.

**Step F: semantic-layer check** (4.2).

**Step G: verdicts.** One per number:

| Verdict | Meaning |
|---|---|
| `VERIFIED` | Matches a result cell (within displayed precision) and label binds. |
| `VERIFIED_DERIVED` | Recomputed by the verifier from cited cells. |
| `VERIFIED_QUOTE` | Present in the cited passage. Says nothing about the passage being true. |
| `DERIVATION_MISMATCH` | A `derived` claim does not recompute from its cited cells within displayed precision (also: wrong arity, division by zero, a direction word that contradicts the sign). |
| `METRIC_MISMATCH` | Claim names a defined metric but the executed query disagrees with the metric definition or its canonical recomputation. |
| `UNSUPPORTED` | No provenance, or provenance does not contain the number. |
| `AMBIGUOUS` | Could not classify the expression, or multiple candidate cells fit. |

Policy: if any number is not in a `VERIFIED*` state, the answer's header status becomes "partially verified" and the bad numbers are highlighted. The system may make at most N regeneration attempts (default 2) with the failure reasons fed back, then shows the best attempt with flags. It never edits a number itself.

**Phase 1b notes (as built).** Claim kinds are `cell`, `derived`, `passage`, `parameter` (a chosen number that appears as a SQL literal, such as `LIMIT 5`) and `row_count`. Derived ops implemented: sum, difference, ratio, percent_change, percent_of, mean, min, max, count (`rank` is not implemented). An extracted number the extractor marks ambiguous is always `AMBIGUOUS`; a claim cannot resolve it. A literal argument to a derived claim must itself appear in the prose. "About" claims are held to the same rounding check as exact ones. Label binding and the definition check apply to `cell` claims; the cells inside `derived` claims are only checked for existence.

### 4.2 YAML semantic layer

A versioned `semantic/metrics.yaml` defines business meaning once:

```yaml
metrics:
  net_revenue:
    description: Sum of line revenue excluding cancelled invoices and returns
    sql: SUM(quantity * unit_price)
    from: fact_invoice_lines
    filters: ["NOT is_cancelled"]
    unit: currency:GBP
    allowed_dimensions: [month, country, product_category]
    synonyms: [revenue, sales, turnover]
```

The verifier uses it two ways:

1. **Compiled path (preferred):** the SQL agent is steered to request a metric by name plus dimensions and filters, and the system compiles the SQL from the YAML. A number tied to a compiled metric query is correct by construction relative to the definition.
2. **Shadow recomputation:** when the agent wrote free-form SQL but the claim label maps to a metric, the verifier compiles the canonical metric query for the same dimensions and filters, executes it, and compares. A difference gives `METRIC_MISMATCH` with both queries shown.

The layer also supplies units, synonyms for label binding, and allowed dimensions. Metric definitions are reviewed by a human and covered by unit tests against fixture data.

### 4.3 What this catches

- Numbers in the prose that appear in no query result (hallucinated or mis-copied).
- Transcription errors: wrong digit, wrong scale (thousands vs millions), rounding beyond displayed precision, sign flips.
- Arithmetic errors in derived values (percent change, totals, ratios).
- Numbers that bind to the wrong column or unit.
- Violations of a defined metric (for example revenue that includes cancelled orders) when the metric is in the YAML.
- Document numbers not actually present in the cited passage.
- Unsupported numbers slipped into the prose that the composer left out of its claim list.

### 4.4 What this cannot catch

State this plainly in the UI help and README.

- **Right number to the wrong question.** If the SQL runs and the prose reports its result faithfully, but the SQL answers a different question than the user meant (wrong join, wrong date window, wrong filter not covered by a metric), the verdict is `VERIFIED`. Mitigations: show the SQL, show a plain-English restatement of the query, prefer compiled metrics, and measure this failure rate in evals. It stays the largest residual risk.
- Non-numeric claims ("sales grew steadily", "the main driver was X") and causal or interpretive statements.
- Numbers expressed in forms the extractor does not parse, such as unusual phrasing or numbers inside images.
- Errors in the data itself, or a source document that is wrong, stale or malicious. `VERIFIED_QUOTE` only means the text was present.
- Coincidental matches the label binding cannot separate (two columns with the same unit and similar synonyms).
- Metrics not in the YAML. They get provenance checks but no definition check.
- Qualitative misreadings of a retrieved passage.

A sixth guard, optional and costly: generate a second independent SQL for the same question and compare results, flagging disagreement. Deferred; evaluate after Phase 1.

## 5. Security

| Threat | Control |
|---|---|
| LLM writes destructive or exfiltrating SQL | Dedicated **read-only database role** with `SELECT` only on an exposed schema of views. `default_transaction_read_only = on`. No access to system catalogs beyond what is needed. |
| Multiple or non-SELECT statements | Parse with a real SQL parser (sqlglot). Accept exactly **one statement** and only `SELECT` or `WITH ... SELECT`. Reject on any parse error. Deny-list dangerous functions and features (`pg_read_file`, `COPY`, `dblink`, `set_config`, file and network table functions, DuckDB `read_csv` on arbitrary paths, `ATTACH`, extension loading). |
| Runaway queries | `statement_timeout` (default 10 s), row limit enforced by wrapping the query in an outer `LIMIT`, max result bytes, and per-user rate limits. |
| DuckDB file or network access | Open read-only, `enable_external_access=false`, autoload of extensions off. |
| Prompt injection via documents | Retrieved text is treated as untrusted data: wrapped in clearly delimited blocks, with a system rule that it is never instructions. The doc agent has no SQL tool and no side-effect tools. The verifier is deterministic and cannot be influenced by document text. Instruction-like patterns in documents are flagged at ingest. Injection documents are part of the eval set (section 6). |
| Injection via data values | Column values are returned to the model as data fields in structured form, and the same untrusted-data rule applies. |
| Leaking private data to free hosted tiers | Model router refuses hosted free providers for data marked non-public (ADR-0003). Gemini's unpaid terms allow training use and human review. |
| Scheduled reports sent without review | Approval step: a report is generated, stored, and held until an approver releases it. n8n cannot send without the approval flag. |
| Secrets | Environment variables only. CI uses repository secrets. Pre-commit has only a private-key detector today; a full secret scanner (for example gitleaks in CI) is planned for Phase 4. |
| AuthN/Z | Role checks on every endpoint. Row-level scoping per user is a later option, not Phase 1. |

## 6. Evaluation plan

Principle: the eval must be able to **fail** the system and must not share data with prompts or few-shot examples.

1. **In-house gold set** (about 100 questions on the chosen sales dataset): each has a gold SQL, gold result, and gold answer values. Split by difficulty (single table, join, window function/CTE, metric-defined, mixed SQL+document). A held-out portion is never used for prompt tuning.
2. **Abstention set** (about 30): unanswerable (data not present), ambiguous, out of scope, questions whose documents conflict with data, and documents containing injection text. The correct behaviour is refusal, a clarification request, or flagged numbers. Metrics: abstention precision and recall.
3. **Verifier mutation tests:** take correct answers and inject faults (change a digit, scale, rounding, sign, swap a column, add an unsupported number) and measure the verifier's false-negative rate. This tests the verifier itself, independent of the LLM. Target: 100% on mechanical faults; the "wrong question" class is reported separately as out of scope for the verifier.
4. **Independent public benchmark:** BIRD dev set (1,534 examples, CC BY-SA 4.0), execution accuracy, run with a pinned config and reported as dev-set EX with model and date. Optional test submission later. Spider subset as a cheap smoke test. See docs/RESEARCH.md.
5. **CI regression gate:** deterministic parts run on every PR: verifier unit and mutation tests, SQL guard tests, semantic-layer tests, and a small agent eval replayed from recorded model responses (cassettes) so CI needs no API keys and no network. Gate: no metric drops more than a set tolerance versus the stored baseline. Full live evals run manually or nightly because free-tier daily limits make them impractical per PR.
6. **Failure analysis format:** every failing case gets a record in `evals/failures/<run_id>.yaml`: question, expected, actual, verdict shown, `failure_class` from a fixed taxonomy (schema-linking, join, filter, date-window, metric-definition, arithmetic, extraction-miss, retrieval-miss, injection-followed, wrong-abstention, provider-error), root cause note, and fix status. Reports aggregate counts per class so work is driven by the biggest class.
7. **Cost and latency** are reported alongside accuracy for every run.

No accuracy number will appear in the README until a reproducible run produces it.

## 7. Build phases

Each phase ends in something that works and can be demoed alone.

| Phase | Scope | Done when |
|---|---|---|
| 1 | Warehouse load for the chosen dataset, semantic-layer YAML, SQL guard, SQL agent, model router (minimal), verifier (sections 4-5), CLI or minimal API | A question returns answer + SQL + per-number verdicts; verifier mutation tests pass; read-only role proven by tests. |
| 2 | Document ingestion, hybrid RAG with reranker, router agent, citations with spans, injection handling | Mixed questions answered with passage citations; retrieval metrics reported; injection tests pass. |
| 3 | Multi-agent orchestration in LangGraph, job queue, auth, approvals, n8n scheduled reports | A scheduled report is generated, held for approval, released by an approver, and delivered. |
| 4 | Full eval suite, CI gates with cassettes, observability (traces, cost, latency), failure-analysis reports | CI blocks regressions; first published eval table with reproducible command. |
| 5 | Deployment to free hosting, web UI with citations and trace view, demo video, architecture write-up | Public demo URL with documented cold-start behaviour. |

## 8. Recommended runtime setup (provisional)

- Development: local Postgres in Docker, DuckDB local, Groq and Gemini free keys for hosted calls, Ollama for the offline path.
- Demo hosting: Neon free Postgres plus one free web host for the API. Which host is an open question (docs/RESEARCH.md, section 1b: Render spins down after 15 minutes; Hugging Face Docker Spaces look unreliable; Cloud Run needs a billing account).
- Expect cold starts: web host wake about a minute (Render) plus Neon resume after 5 idle minutes. The UI must show a "warming up" state.
- Demo data is public/synthetic only, since free hosted LLM tiers may use prompts for training.
