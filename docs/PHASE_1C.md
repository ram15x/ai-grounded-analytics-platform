# Phase 1c: SQL agent (proposal)

Status: PROPOSED. Design only; nothing in this file is implemented except where it says so.
Items marked [UNVERIFIED] are assumptions I could not confirm. Phase 1c proper starts only
after the real-data profile (docs/DATA_PROFILE.md) is filled in and the owner has decided on
duplicate rows. Phase 1c-0 (deterministic prerequisites, no LLM) is described in section g.

## a. Agent graph and answer contract

Nodes (LLM nodes marked *; everything else is deterministic code):

    route* -> plan* -> compile -> guard -> execute -> compose* -> verify -> render
                ^                                          |          |
                +---------------- repair* <----------------+----------+   (max 2)
                                  abstain (terminal)

- route*: classify intent (data question / document question / out of scope) and sensitivity.
- plan*: choose a `MetricQuery` from the semantic-layer catalogue (metric names, dimensions,
  filters). It never writes SQL. Output is validated by the existing `MetricQuery` model.
  Free-form SQL is a fallback only and must carry `declared_metric_query` (already in the schema).
- compile / guard / execute: existing code (`compile_query`, `validate_sql`, `execute_guarded`).
  Failures return to plan* as structured errors, not free text.
- **restate (deterministic, no LLM)**: every compiled `MetricQuery` is restated in plain English by
  a template over the query object and the semantic layer (`restate()`, implemented in 1c-0). The
  restatement is shown with every answer, so a reader sees what was asked of the warehouse (which
  metrics, grouping, filters, ordering, limit, definitions, units) in words that cannot drift from
  the SQL. Free-form SQL has no restatement and is labelled as such.
- compose*: receives the executed result tables and outputs a `TemplateAnswer` (ADR-0008): prose
  with placeholders like `{c1}`, plus `claims` that each name a cell, a derived value or a row
  count. The model never types a number. The system renders each placeholder from the verified cell
  or the recomputed derived value, formatted by the semantic layer's unit, and records the displayed
  precision. Digits outside placeholders are `UNSUPPORTED`. The old `AnswerDoc` (model-typed
  numbers) remains for existing examples and as a comparison baseline.
  Style rules the verifier still relies on (docs/VERIFIER_LIMITS.md): one number per clause, and the
  row or period named in the same sentence as the placeholder.
- verify: `verify_template` renders, then runs `verify_answer` (steps A, C-G plus sentence-context
  binding) on the rendered text. `verified` ends the run.
- repair*: on any non-VERIFIED verdict the composer is re-called with the same result tables, its
  previous `TemplateAnswer`, and the failing verdicts as structured data (text, verdict, reason,
  claim_id). Giving the expected value back is allowed because the final text is still checked
  against the cells. A plan-level failure (METRIC_MISMATCH, unknown column, row_not_found) goes
  back to plan*. The attempt budget is shared: at most 2 repairs.
- After the last attempt: show the attempt with most verified numbers as "partially verified", with
  flags. The system never edits a number. With no verified number, or no mapping to metrics: abstain.
- Abstain is a first-class outcome (unsupported metric, forecast, causal "why", out-of-period,
  personal data, ambiguity that needs clarification). Abstention and "partially verified" are
  scored separately.

## b. Model router

Rules live in config; model IDs are never in code (ADR-0003).

| Task | Provider order | Notes |
|---|---|---|
| route | local 3-4B, then Groq small model | classification only |
| plan | Groq (JSON mode), then Gemini free, then OpenRouter `:free` | gpt-oss models have no parallel tool calls (RESEARCH.md): use JSON mode |
| compose / repair | same chain as plan | strict schema validation; invalid JSON counts as an attempt |
| local 4 GB model | route, intent/sensitivity checks, offline demo | not used for scored results unless labelled "degraded"; its tool-calling reliability is measured by the eval, not assumed |

- Data sensitivity: each dataset/table is tagged `public` or `internal`. Hosted free tiers (Gemini
  free: prompts may be used and read by humans; OpenRouter `:free` data policies [UNVERIFIED])
  receive only `public` data. Internal data goes to local models or the request is refused. The
  check is deterministic code before the call, with a test that an internal payload cannot reach a
  hosted adapter.
- Rate limits: honour `retry-after` on HTTP 429 (documented for Groq and OpenRouter); otherwise
  exponential backoff with jitter, max 3 retries per call, a per-provider token bucket at about 50%
  of the listed limit, a circuit breaker after 3 consecutive 429s, then the next provider. Whether
  failed OpenRouter requests count against its daily limit is [UNVERIFIED].
- Budget: Groq lists 200K tokens/day, 30 RPM, 1K RPD for the models in RESEARCH.md; which plan
  (free or developer) that table applies to is [UNVERIFIED], so I plan for 200K/day.
  Assumption [UNVERIFIED until measured]: about 3-4K tokens per LLM call, about 8K per question at
  one attempt, up to about 20K with two repairs; at 8K average one model covers about 25 questions
  a day. Hence a hard per-run cap (default 30K tokens, then abstain with reason "budget"), and a
  100-question eval cannot run live in one day on one model, which is why cassettes (d) exist.
  Measure real token use on 5 questions before fixing any number.

### Implementation status (Phase 1c-1)

Built and tested with mocked HTTP and a FakeProvider only; **no real provider has been called**.
Adapters for Groq, Gemini and OpenRouter (`src/grounded_analytics/llm/providers.py`), the router
(per-provider token buckets at `bucket.fraction` of the documented limit, retry with backoff and
jitter, `retry-after`, circuit breaker after 3 consecutive 429s, fall-through, per-run token cap that
ends in abstain/`budget`), and the sensitivity rule (hosted providers never see non-`public`
datasets; unknown datasets are internal; checked in the router before every call and again in each
hosted adapter). Limits and model IDs are in `config/router.yaml`, all marked UNVERIFIED. Every
assumption about provider behaviour is listed in docs/PHASE_1C1_ASSUMPTIONS.md. Not built: the
`route` node and the local-model tier, routing of METRIC_MISMATCH back to the planner, the free-form
SQL fallback, `grounded eval`.

### Phase 1c-2 tooling (built, not yet run)

`grounded live models` and `grounded live probe` (`llm/live.py`, runbook in docs/LIVE_RUNBOOK.md) are the only
code meant to reach a real provider, and only when the owner runs them with keys in the
environment. The probe is pinned to one provider and one model, refuses datasets not tagged public,
sends at most 25 HTTP requests (retries included), stops on the first 401/403, and records
cassettes only from its own live calls, to `data/live_runs/` (git-ignored) for review before any
commit. Its report gives raw rates, each with n and a Wilson interval, and no ranking or accuracy
claim. Nothing has been run yet; docs/PHASE_1C1_ASSUMPTIONS.md has a status column that stays
STILL UNKNOWN until real output is pasted back.

## c. Gold set

- **Labelling: the gold questions and their rules are author-written** (by the project author, not
  an independent party or a public benchmark). Every report states this next to every rate.
- Independence rule: gold answers are produced by `eval/gold/*.py`, plain pandas over the RAW file,
  with each question's rules written out in plain English in the question file. Gold code imports
  nothing from `semantic.py`, `compiler.py`, `warehouse/normalize.py` or the pandas oracle. Rules
  are restated per question, not shared.
- Disagreement is a finding: `eval/compare_gold.py` compares gold with `grounded profile-db` and the
  semantic layer. Each difference is classified as gold bug, semantic-layer bug, or intended rule
  difference, and written to docs/GOLD_DIFFS.md for the owner. A metric definition is never changed
  just to make numbers agree.
- A second reader checks the rules text and a 20% sample of gold values against the spreadsheet.
  [Assumption: the owner has time for this.]
- Size and tiers: 100 questions. 80 answerable: T1 single metric, no dimension (15); T2 one
  dimension or filter (20); T3 multiple metrics or derived values such as share or percent change
  (20); T4 top-N, ranking, month-over-month (15); T5 trap questions where a naive reading gives a
  different number (cancellations, missing customers, duplicates) (10). 20 abstention questions.
- Abstention set: written by hand in five groups of four: not in the data (no cost or margin
  column); forecasts; causal "why" without a decomposition; out of period; ambiguous or
  personal-data requests. Each has the expected reason code.
- Dev vs held-out: 60 dev (stratified by tier) and 40 held-out. Held-out files are committed only
  as a hash until a release point; prompts and routing are tuned on dev only. Each held-out run is
  logged; a second look at held-out after changes is flagged as contamination.
- **Coverage metric**: the share of gold questions whose gold query can be expressed as a catalogue
  `MetricQuery` (so the compiled path can answer them). It is computed from the question file
  (each question carries `expressible_as_metric_query: true|false` plus the query when true),
  reported by tier, and reported before any model is involved. Questions that are not expressible
  go through the free-form SQL fallback, and **the fallback is scored separately** from the
  compiled path in every report. External text-to-SQL benchmarks (BIRD) are deferred to Phase 4 and
  apply to the free-form fallback only; they say nothing about the catalogue path.

## d. Cassettes

- Recording: the router wraps every provider call. In record mode it writes one JSONL line per call
  to `tests/cassettes/<suite>/`: key, provider, model ID, request (messages, schema, temperature 0),
  response, token counts, latency, timestamp, and the hashes below. Headers and keys are never
  stored. Only `public`-tagged payloads are recorded.
- Key = SHA-256 of (provider, model ID, normalised request, prompt-template hash, semantic-layer
  hash, claim-schema hash, warehouse content hash).
- Replay mode (default in CI): a key miss raises an error; there is no network fallback, so CI
  needs no keys. Record mode needs keys from the environment and is run by hand.
- Stale detection: any change to a prompt, the layer, the schema or the data changes the key, so a
  stale cassette fails as a miss. `grounded cassettes check` lists missing and unused entries and cassette
  age; a periodic manual re-record against live providers reveals drift and retired model IDs.
  Replay proves the pipeline is deterministic given the outputs; it says nothing about current
  model quality. Temperature 0 does not guarantee identical hosted output [UNVERIFIED per
  provider], so a cassette freezes one sample.

### Implementation status (Phase 1c-1)

Record/replay, key hashing, the secret scan and `grounded cassettes check` are implemented
(`llm/cassette.py`). No cassette is committed, because the only calls made so far were to a
FakeProvider. `check` reports stale entries (recorded hashes differ from the current prompt, layer,
schema or warehouse), cassette age, and, given a `usage.json` written by a replay run, unused and
missing entries. There is no record command yet; recording is a manual step with keys in the
environment.

## e. Eval report

`grounded eval --set dev|heldout --replay|--live` writes `reports/<run-id>.json` and `.md`. Header: git
commit, dataset hash, semantic-layer hash, cassette-set hash, model IDs, date, n, and the line
"Gold questions are author-written."

Implemented so far (1c-1): per-call and per-run token and latency accounting in this report's
format (`llm/accounting.py`; tokens, p50/p95 with n, by provider, cost 0). There is no `grounded eval`
and no accuracy figure.

Per question: outcome, attempts, verdicts, gold match (tolerance = displayed precision), the
restatement, tokens, latency. Aggregates by tier, by path (compiled vs free-form) and by provider.

- **Every rate is printed with its n and a Wilson 95% interval, with no exceptions.** The held-out
  set has 40 questions, so its intervals are wide (a 90% observed rate on n=40 is roughly 77% to
  96%); the report says so and does not rank variants whose intervals overlap. Per-tier cells have
  n of 10 to 20 and are wider still.
- Failure classes: (1) verified-but-wrong (verifier false accept, the worst class, always listed in
  full); (2) wrong plan; (3) right answer rejected; (4) abstained when answerable; (5) answered when
  it should abstain; (6) schema or JSON failure; (7) guard block, execution error or timeout;
  (8) budget exhausted; (9) provider error or rate limit.
- Also: share verified / partially verified / abstained, repair success rate, coverage metric,
  cost (0 on free tiers; tokens and a would-be paid estimate), latency p50/p95.
- Rule: no accuracy figure goes into the README or any other doc until a reproducible command
  (`grounded eval ... --live`, or `--replay` against committed cassettes) has produced it at a stated
  commit, with n, tier and interval beside it. Held-out figures come from a separate report with
  the held-out run count.

## f. Five biggest risks and what to test first

1. Wrong plan: the model maps the question to the wrong metric or filter and everything after it
   "verifies". Test: plan-only eval (no composer) comparing the chosen `MetricQuery` with the gold
   query on the T1-T3 questions.
2. Right number, wrong question: the verifier accepts a real cell for the wrong claim. Done in
   1c-0 for 30 hand-built lies (0 false accepts after the sentence-context check, 30 before; see
   docs/VERIFIER_LIMITS.md). Next: adversarial composer outputs generated by a model, on dev only.
3. Free-tier limits and model churn make runs costly or unreproducible. Test: measure tokens per
   question on 5 questions on the real warehouse, then freeze budgets and cassettes.
4. Small or free models fail structured output. Test: schema-valid rate per model on 20 questions,
   with and without constrained JSON mode, to decide the chain.
5. Gold contamination or rule drift: gold shares its author's assumptions with the layer. Test:
   gold-vs-`grounded profile-db` comparison on all T1 questions before any agent work (1c-0), with every
   difference triaged. Gold is author-written; an independent reviewer is needed for the sample.

## g. Phase 1c-0 (deterministic only, no LLM, no provider code, no keys)

1. CI-hard database tests: `GA_REQUIRE_DB=1` makes an unreachable test database a failure. Done.
2. Gold set, first 15 T1 questions (`eval/questions/dev.yaml`, `eval/gold/`), pandas over the raw
   `Online Retail.xlsx`. Harness built; values need the raw file.
3. `eval/compare_gold.py` and docs/GOLD_DIFFS.md.
4. 30 adversarial answers for the verifier. Done, see docs/VERIFIER_LIMITS.md.
5. `restate()` for `MetricQuery`. Done.
