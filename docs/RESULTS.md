# Results and measurements

Everything measured on this project, with its n. These are small, single-author screens, not
benchmarks: one author wrote the questions, the semantic layer and the synthetic documents.
**No accuracy figure is claimed.** Run folders are under `data/live_runs/` (git-ignored).

## End-to-end evaluation (2026-11-22, groq `openai/gpt-oss-20b`)

Scored from what a reader would see (ADR-0014): **correct** (verified, gold figure shown),
**verified_wrong** (shown as verified, gold figure missing: the failure this product exists to
prevent), **withheld** (abstained), **no_reply** (no usable model reply). Questions and gold values
are author-written; one model; small n. **Not an accuracy claim.**

### Dev set (15 questions; may be run repeatedly)

| Run | Plan prompt | Correct | Verified wrong | Withheld | No reply |
|---|---|---|---|---|---|
| `20261121T144216Z` | plan.v1 | 10 | 0 | 2 | 3 |
| `20261122T141403Z` | plan.v2 | 9 | 0 | 4 | 2 |

With `plan.v2`, the 4 withheld questions are the 4 marked not expressible with the approved metrics
(withholding is the right answer), and both no-replies were HTTP 429 (8,000 tokens per minute).
Every expressible question that got a reply was correct (9 of 9).

### Locked held-out set (15 questions, run once, `20261122T143125Z`)

Settings fixed before the run in ADR-0014: `plan.v2`, `--pause-s 45`.

| Correct | Verified wrong | Withheld | No reply | Correct among answered (Wilson 95%) |
|---|---|---|---|---|
| 8 | **2** | 5 | 0 | 8/15 (30.1% to 75.2%) |

Both verified-wrong answers were investigated (ADR-0014 requires it):

- **H-T3-01**, percent change in gross revenue from October to November 2011. The data shows an
  increase of 30.7% (1,154,979.30 to 1,509,496.33). The answer said "increased by -23.5%". The
  model passed the two months to the percent-change check in reverse order, and the verifier
  trusted that order, so the backwards figure matched itself. The word "increased" before a
  negative number was not checked either.
- **H-T3-02**, December 2010 against December 2011 net revenue. The plan filtered `month = Dec 2010
  AND month = Dec 2011`, which no row can satisfy; the query returned nothing, and the answer
  reported "net revenue ... 0 rows", which the verifier accepted as a row count.

Fixed after the run, each with a regression test (`tests/test_template.py`): a percent change
between periods must run from the earlier to the later, and between groups the base must be named
("from", "than", "on", "compared with"); a direction word before a negative number is refused; a row
count cannot be shown as a measured value; the compiler rejects one column filtered to two
different values. **The locked score above was produced by the code before these fixes and is not
re-scored**: the held-out questions are spent for this model, and re-running them after a fix would
no longer be a held-out measurement.

### Model-written labels for document numbers (ADR-0013)

80 of 80 labels written by `openai/gpt-oss-20b` (labels file sha256 `71583e31`). False reject =
a correct label refused; false accept = a label swapped onto another number accepted.

| Variant | Dev: false reject | Dev: false accept | Locked: false reject | Locked: false accept |
|---|---|---|---|---|
| current | 10 of 38 | 6 of 68 | 15 of 42 (23.0% to 50.8%) | 5 of 76 (2.8% to 14.5%) |
| S2 (default) | 5 of 38 | 4 of 68 | 16 of 42 (25.0% to 53.2%) | 1 of 76 (0.2% to 7.1%) |
| SL | 5 of 38 | 4 of 68 | 15 of 42 (23.0% to 50.8%) | 1 of 76 (0.2% to 7.1%) |

Reading: on the locked half S2 and SL accept far fewer swapped labels than `current`, at about the
same false-reject rate. That false-reject rate is high: roughly one correct label in three is
refused, which shows up as withheld document answers. Per ADR-0013 no default changes on this
result alone. Locked report: `data/live_runs/20261122_194905_binder_model_locked`.

## Provider and pipeline runs

Provider probes, 10 single-metric questions on `analytics.sales_lines` (`grounded live probe`; "verified" means every number passed the verifier, not that the answer is correct). The prompts were first drafts and changed between runs, so the rows are not repeats of one experiment.

| Run folder | Model | max_tokens | Questions | Verified | Typed a digit outside a placeholder | Other |
|---|---|---|---|---|---|---|
| `20261120T125956Z_groq_openai-gpt-oss-20b` | groq `openai/gpt-oss-20b`, untuned v0 prompts | 1024 | 5 (request cap) | 0 of 5 | n/a | compose/repair valid 0 of 10; 3 HTTP 429 |
| `20261120T132347Z_groq_openai-gpt-oss-20b` | same, compose.v1 | 1024 | 10 | 8 of 10 | 0 of 8 | 1 HTTP 400 `json_validate_failed`; finish `stop` 18 of 18 |
| `20261120T143412Z_groq_openai-gpt-oss-20b` | same, plan.v1 | 1024 | 9 (request cap) | 8 of 9 | 0 of 10 | 2 HTTP 400 `json_validate_failed` |
| `20261120T171922Z_groq_openai-gpt-oss-20b` | same | 2048 | 10 | 10 of 10 | 0 of 11 | no HTTP 400; finish `stop` 21 of 21 |
| `20261120T172754Z_openrouter_nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-free` | openrouter `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 2048 | 10 | 10 of 10 | 0 of 10 | finish `stop` 20 of 20; p50 latency 4652 ms |
| `20261120T144945Z_openrouter_google-gemma-4-31b-it-free` | openrouter `google/gemma-4-31b-it:free` | 2048 | 0 | n/a | n/a | stopped after 5 failed requests (4 HTTP 429 upstream rate limit, 1 HTTP 400): nothing measured |
| `20261120T140739Z_gemini_gemini-3.5-flash` | gemini `gemini-3.5-flash` | 2048 | 10 | 2 of 10 | 0 of 2 | 5 of 23 requests got HTTP 200; 13 timeouts, 3 HTTP 429, 1 HTTP 503; p50 latency 26170 ms |
| `20261120T135432Z_gemini_gemini-3.8-flash` | gemini `gemini-3.8-flash` | 2048 | 6 (request cap) | 0 of 6 | n/a | 0 HTTP 200: 10 HTTP 503, 11 timeouts, 3 HTTP 429 |
| `20261120T135315Z_gemini_gemini-2.5-flash` | gemini `gemini-2.5-flash` | 1024 | 10 | 0 of 10 | n/a | 13 HTTP 404: model no longer available to new users |

So only Groq `gpt-oss-20b` (one run) and the OpenRouter nemotron model (one run) verified 10 of 10; Gemini was unavailable or slow in all three runs. The free tiers rate-limit: Groq HTTP 429 in the first run, OpenRouter gemma 429 upstream, Gemini 429 "exceeded your current quota".

Router and injection probes (`grounded route probe`, Groq `openai/gpt-oss-20b`, 12 dev router questions, one author):

- **Route prompt `route.v1`:** route matched the author's label 9 of 12 (`20261120T192006Z_groq_openai-gpt-oss-20b`, route only; 9 of 17 route replies valid) and 7 of 12 (`20261120T192555Z_groq_openai-gpt-oss-20b`, with the full answer; 11 of 23 replies valid).
- **Route prompt `route.v2`,** written after reading those failures on the same 12 questions (so a tuned dev figure, not held-out): matched 12 of 12 with 12 of 12 replies valid (`20261121T043707Z_groq_openai-gpt-oss-20b`, route only); matched 12 of 12 with 14 of 14 valid (`20261121T044057Z_groq_openai-gpt-oss-20b`, full answer). In the full run 9 questions had a valid data, documents or both route: 7 reached the composer, 2 stopped at `unsupported_metric`; compose/repair replies valid 10 of 14; outcomes verified 5 of 12, abstain 7 of 12; the run stopped on its token cap after 2 of the 5 injection questions. The 12 locked router questions are unused.
- **Injection** (5 hostile-document questions, one model): in `20261121T064538Z_groq_openai-gpt-oss-20b` all 5 reached the composer and 0 of 5 composer replies contained an injected marker string; outcomes were I-D-01 partially verified, I-D-02 and I-D-03 verified, I-D-04 and I-D-05 abstained (`provider_error`: JSON refusals, and one HTTP 429). In `20261121T055258Z_groq_openai-gpt-oss-20b` 4 of 5 reached the composer, 0 of 4 contained a marker. n=5, one model, two runs, not a safety claim.
- **Label binder on real wording** (rule before ADR-0012, window of 6 words): 6 refusals in `20261121T044057Z_groq_openai-gpt-oss-20b` and 4 in `20261121T064538Z_groq_openai-gpt-oss-20b` (3 labels for I-D-01, 1 for I-D-05; the report lists each with its label, sentence and missing label words). Replayed offline against the stored hostile documents (no model call), the S2 default also refuses all 4 of those labels: the words they use for the subject are not in the number's own sentence (n=4). The S2 default has not been run on live model output.
- **Groq empty replies:** in `20261121T064538Z`, 2 of 2 replies to the JSON-mode-off retry were HTTP 200 with empty content, 1024 completion tokens of which 1022 were reasoning, finish `length` (I-D-01 repair, I-D-04 compose). In the five Groq reports that show the HTTP 400 `json_validate_failed` bodies (`20261120T132347Z`, `20261120T143412Z`, `20261121T044057Z`, `20261121T055258Z`, `20261121T064538Z`), 16 of 18 had an empty `failed_generation`; the other 2 (`20261121T044057Z`, R-D-02) carried a cut-off template. That reasoning exhaustion also causes the 400s is a hypothesis, unverified ([assumptions rows 42 and 45](PHASE_1C1_ASSUMPTIONS.md)). With `max_tokens` 2048 the Groq probe had no 400 (`20261120T171922Z`, n=10 questions); at 1024 it had 1 and 2.
- **Document retrieval** (12 held-out questions, 11 answerable, `grounded docs heldout`): default hybrid recall@5 1.00 and MRR 0.84 with `gemini-embedding-2` (768 dims) (`20261121_001646_docs_heldout_gemini_embedding_2_768`), and recall@5 1.00 and MRR 0.86 with `BAAI/bge-small-en-v1.5` (`20261121_001708_docs_heldout_local_bge_small_en_v15`). The pre-registered alternative B had MRR 0.76 and 0.78 and failed its rule, so the default stays and the fusion choice is marked undecided ([ADR-0010](adr/0010-fusion-heldout-outcome.md)). The one question without an answer still got passages back (1 of 1 on both embedders).
- **Label binder S2 chosen from a locked synthetic study** (`20261121_115112_binder_locked`, 39 documents): S2 accepted 15 of 1289 swapped labels and refused 40 of 664 correct labels; the earlier rule accepted 251 and refused 193. The labels are generated from each number's own sentence, so refusals of labels a model writes are unmeasured, and swaps onto another document are still accepted ([ADR-0011](adr/0011-binder-variants.md), [ADR-0012](adr/0012-binder-default.md)).
- **CI:** GitHub Actions run #1 on commit `c7aa432` passed in 1m 46s (status only). The workflow sets `GA_REQUIRE_DB=1`, under which the suite fails instead of skipping when the database is unreachable (`tests/test_require_db.py`), so a green run implies the database tests ran; that is read from the workflow and test code, not from the run logs, which were not read.

## Status

**Project closed on 2026-11-22.** Last checked on a real machine with a real Groq key: upload,
reconcile, a verified answer, a correct refusal, and key headroom all worked (details in
docs/WRITEUP.md). Open items are listed under "What I'd do next"; none is in progress.

| Area | State |
|---|---|
| Research, architecture, ADRs | Done, see [docs/RESEARCH.md](RESEARCH.md), [docs/ARCHITECTURE.md](ARCHITECTURE.md), [docs/DECISIONS.md](DECISIONS.md) |
| Warehouse (Postgres 16 via docker-compose; dims, facts, read-only views, read-only role) | Loaded and reconciled on the real UCI file (541,909 rows); tests use a synthetic 368-row fixture. `docker-compose.yml` uses `pgvector/pgvector:pg16`, which CI also uses. |
| Data profile and data-quality decisions | Decisions implemented with measured sizes on the real file, see [docs/DATA_PROFILE.md](DATA_PROFILE.md). |
| Semantic layer (`semantic/metrics.yaml`, 11 metrics) | Implemented. Each metric is tested against an independent pandas derivation (shares the business rules, not the code). |
| Metric compiler (metric + dimensions + filters to SQL) | Implemented and tested |
| SQL guard (sqlglot, one SELECT, deny-lists, allow-listed relations, LIMIT, timeout, read-only role) | Implemented; 100+ adversarial and accepted cases. Not a proof of safety. |
| Number extractor | Implemented; 200+ cases. Limits listed in [docs/EXTRACTOR.md](EXTRACTOR.md) |
| Verifier (claims schema, Decimal recomputation, label binding, verdicts) | Implemented, deterministic, no LLM |
| Mutation tests | Implemented as unit tests (see note below) |
| CLI: `grounded load`, `grounded profile`, `grounded verify answer.json` | Implemented; sample answers in `examples/` |
| Driver-claim verification | Design only, [ADR-0007](adr/0007-driver-claim-verification.md). Not implemented. |
| Pre-commit config and GitHub Actions CI (lint, format, tests with a Postgres service, Python 3.11 and 3.12) | Run #1 passed on GitHub (see above). |
| Model router, provider adapters (Groq, Gemini, OpenRouter), cassettes, scripted plan/compose/verify pipeline | Implemented; tests use mocked HTTP and a fake provider. Real providers were probed (table above) with first-draft prompts. Every provider assumption, with its status, is in [docs/PHASE_1C1_ASSUMPTIONS.md](PHASE_1C1_ASSUMPTIONS.md); most are still unverified. After a JSON-mode refusal the router makes at most 2 requests and drops a provider after 2 refusals in a row for one question. |
| Documents (Phase 2a-2c): store with character spans, keyword, vector and hybrid search, local and hosted embedder and reranker clients, passage citations checked against stored text, retrieval metrics | Implemented and tested on a synthetic corpus. Two embedders were probed on dev and held-out; the shipped hybrid default stays and the fusion choice is **undecided** (ADR-0010; the 12 held-out document questions are spent). No reranker in the default pipeline. Slice 2d adds injection handling (scanner, delimited prompt block, flagged text cannot be cited); one model, 5 questions, see above ([docs/INJECTION.md](INJECTION.md)). Slice 2e adds a router (data, documents, both, abstain), mixed answers with system-written passage figures and a heuristic label binder (default S2, ADR-0012). Not an accuracy claim: one author, one corpus. See [docs/PHASE_2.md](PHASE_2.md) |
| `grounded ask`, approved reports, local web view, stored-results gate | Implemented. `grounded ask "<question>" --replay` runs the real pipeline on recorded model replies (needs the local database, no key); `--provider/--model` runs it live on public data. Each run writes a JSON trace to `data/traces/`. `grounded report draft` then `grounded report approve` writes a Markdown file only after a named approval by digest; nothing is sent. `grounded serve` shows 17 recorded dev runs with SQL, rows, passages and verdicts. A test fails if a verified figure has no trace to its stored cell or passage. Scheduled runs, auth and a live model in a hosted product do not exist. |
| Auth, deployment | No sign-in. The recorded view is deployed to GitHub Pages (static, no model calls); uploads (`--workspace`, loopback only) and the key-based tryout (`--byok`) run locally. |
| Evals and benchmark scores | Does not exist. The measurements above are small single-author screens with their n, not benchmarks. **No accuracy figure is claimed.** |

## Page citations on a real PDF (2026-11-22)

One real, public PDF, not written for this project: "Attention Is All You Need" (arXiv 1706.03762,
15 pages, two-column paper with tables and figures). It was ingested with the same code path as
`docs ingest`, into a scratch Postgres.

- Text was found on all 15 pages; the page offsets matched the stored text exactly, so pages are shown.
- 56 chunks; **56 of 56 carry a page range.** No model and no key were used.
- Seven phrases whose page was read from the PDF itself were each cited to that page: abstract
  figure (p. 1), "Scaled Dot-Product Attention" (p. 4), "Multi-Head Attention" (pp. 2 and 4),
  "Positional Encoding" (p. 6), "Adam optimizer" (p. 7), "Attention Visualizations" (p. 13), the
  conclusion (p. 10). **7 of 7 matched.**
- This checks the page labels, not answer quality: no question was asked of this document. Table
  contents come out of the PDF in whatever order it stores them, so a figure in a table can sit in
  an odd place in the text. Scanned PDFs are refused (no OCR). Only this one document was tried; a
  second document (an IRS publication) could not be fetched.

## Not measured

- **OpenRouter and Gemini rate-limit headroom.** The headroom probe was measured on Groq only
  (about 8,000 tokens a minute, 200k a day). No OpenRouter or Gemini key was supplied, so their
  limits are not measured here; for those the page says it cannot tell instead of guessing.

## About the mutation tests

`tests/test_mutations.py` takes a fully verified answer, changes one number (a digit, the scale,
the rounding, the sign, the column, or adds an unsupported number) and checks that the verifier
refuses it. These are **unit tests of specific, synthetic corruptions. They are not an accuracy
claim**: they do not measure how many real model errors would be caught, and a passing suite says
nothing about errors the verifier was not written to see (for example a correct number that
answers the wrong question). Likewise the pandas oracle and the fixture are synthetic.

## Data and known limits

Measured on the real file (`Online Retail.xlsx`, 541,909 rows, 2010-12-01 to 2011-12-09; details in
[docs/DATA_PROFILE.md](DATA_PROFILE.md)):

- Net revenue is 9,769,872.05 (gross 10,666,684.54, cancelled 891,116.27, discounts 5,696.22); 19,960 orders; 4,338 identified customers.
- **Exact duplicates stay counted.** 5,268 rows (0.97%). Without them net revenue would be 9,748,131.07 (-0.22%) and units -0.28%; orders and customers do not change.
- **24.93% of rows (135,080) have no customer.** They are kept in revenue; 15.04% of net revenue (1,469,806.24) cannot be tied to a customer, and customer metrics ignore them.
- **Units sold means merchandise units** (postage, fees and manual lines excluded). Counting every sale row would give 11,380 more.
- 2,517 rows with price <= 0 are `adjustment` and excluded from all metrics; 2,995 rows are non-merchandise codes.
- Gold check: 15 author-written questions, 11 expressible as catalogue queries, 11 agree, 4 not expressible ([docs/GOLD_DIFFS.md](GOLD_DIFFS.md)). The gold author also wrote the semantic layer, so agreement is weak evidence. No accuracy figure is claimed.
- The verifier's sentence binder is a heuristic with documented limits: [docs/VERIFIER_LIMITS.md](VERIFIER_LIMITS.md).
