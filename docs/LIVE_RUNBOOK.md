# Live runbook (Phase 1c-2): first real provider calls

These commands are run by the owner on their own machine, in PowerShell. Nothing here is run by the
build chat. No key is ever written to disk: it lives only in the environment of the PowerShell
window you set it in. Cassettes and reports go to `data\live_runs\` (git-ignored). **Nothing is
committed by these commands, and no cassette is committed until you have reviewed the files.**

## 0. Once: get the code and check it

```powershell
cd C:\Users\ramna\OneDrive\Desktop\Projects\verified-analyst
Get-ChildItem ..\p4 -Filter *.patch | Sort-Object Name | ForEach-Object { git am $_.FullName }
git log --oneline | Measure-Object -Line
python -m pytest -q
```

## 1. Put the keys in this window only (they are not echoed and not saved)

```powershell
$k = Read-Host "Groq key" -AsSecureString
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password
$k = Read-Host "Gemini key" -AsSecureString
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password
$k = Read-Host "OpenRouter key" -AsSecureString
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new('', $k).Password
Remove-Variable k
```

The probe also needs the database holding the real data, with the same variables you used for
`va profile-db`: `VA_READONLY_DATABASE_URL` (and `VA_DATABASE_URL` if you reload). It sends only the
10 T1 dev questions, the metric catalogue and single-row aggregate results from the public UCI
dataset; nothing at customer level.

## 2. List the models each key can use (1 to 3 requests per provider)

```powershell
va live models
```

Paste back the full output. It prints model IDs, the rate-limit headers seen, and whether the IDs
configured in `config\router.yaml` (for example `openai/gpt-oss-20b` on Groq) are PRESENT or ABSENT.
For OpenRouter it lists zero-price models only; add `--all` for every ID. Pick one model ID per
provider from this output before probing.

## 3. Probe one model at a time (25 requests by default, at most 60)

```powershell
va live probe --provider groq --model openai/gpt-oss-20b --quota-check
```

`--quota-check` spends 3 of the request cap on one good request, one deliberately invalid one and one
good one, to see whether a failed request changes the remaining-requests header. Leave it off to use
the whole cap on the battery. The cap is 25 requests unless you pass `--max-requests N` (1 to 60;
anything else is refused). The run stops at once on the first 401/403, at the request cap, or at the
token cap in `config\router.yaml` (`probe.token_cap`, 60000; `--token-cap` can only lower it).

The probe paces itself with a client-side token guard (`tpm: 8000` for Groq in
`config\router.yaml`, used at `bucket.fraction` 0.9). Before each request it estimates prompt tokens
as characters / 4 plus `max_tokens`; if the bucket cannot cover that, it waits (at most 60 s per
wait, each wait is reported). After an HTTP 429 it reads `Retry-After`, waits up to 60 s and retries
ONCE; it never retries blindly. A wait counts against `--max-requests` only if a request is then
sent. The report's 'Token guard and rate limits' section gives the waits, the 429s and the retries.
A full run can take 10 to 20 minutes.

`reasoning_effort` (config, Groq `low`) is sent only to Groq gpt-oss models; override it for one run
with `--reasoning-effort medium`. That `low` reduces reasoning tokens is UNVERIFIED. To check it, run
once (two runs of the 5 injection questions, `low` then `medium`, each with its own request cap):

```powershell
va route probe --provider groq --model <GPT-OSS ID> --injection-only --compare-reasoning low,medium --record
```

It writes `compare.md` and `compare.json` plus one sub-folder per setting. Compose and repair ask
for at least 2048 completion tokens on models that report reasoning tokens; an empty reply with
finish `length` is retried once with the allowance doubled (cap 4096).
`--max-tokens N` sets the allowance for every call, `--token-cap N` lowers the cap (`va live
probe` also has `--no-record`; `va route probe` records only with `--record`).

Repeat with IDs taken from step 2 (replace the angle-bracket parts):

```powershell
va live probe --provider gemini --model <GEMINI ID FROM STEP 2> --quota-check
va live probe --provider openrouter --model <OPENROUTER ID FROM STEP 2> --quota-check
```

Each run prints its report and the path of its output folder, then the cassette key scan and check
result. Do not run the same model twice back to back unless asked: it spends quota.

## 4. Check the cassettes

```powershell
Get-ChildItem data\live_runs | Sort-Object Name
va cassettes check data\live_runs\<FOLDER NAME FROM THE LIST>\cassettes
```

Expected last line: `OK`. Open `data\live_runs\<folder>\cassettes\calls.jsonl` and read it yourself:
it holds the prompts sent and the raw model replies. It must contain no key and no data other than
public UCI aggregates.

## 5. Paste back

For each command above: the full console output (it is redacted of keys by construction, but glance
through it before pasting). Specifically:
1. `va live models` (all of it).
2. Each `va live probe` report (the whole `# Live probe` block, including the Rates, Tokens,
   Rate-limit headers and "Does a failed request use quota?" sections, and the last two lines).
3. The output of `va cassettes check` for each run folder.
4. Anything that looks odd in `calls.jsonl`, described in words (do not paste the file).

Then close the session's keys:

```powershell
Remove-Item Env:GROQ_API_KEY, Env:GEMINI_API_KEY, Env:OPENROUTER_API_KEY
```

## JSON-mode rejections

If Groq answers HTTP 400 `json_validate_failed`, the router sends the request once more, with JSON mode off (it counts
as a request against the cap). If that retry returns text that does not parse, the question goes through the normal repair step.
The identical-request retry recovered 0 of 4 on the first full route probe, so it was replaced
(assumptions row 44, UNVERIFIED). The route probe report has a 'JSON-mode refusals' section.

## Early stops

The probe stops on its own in three cases, and says so in a bold line at the top of the report:

- First HTTP 401 or 403: exit code 3.
- Five requests in a row fail (a timeout, a network error, or any non-2xx other than 401/403;
  429s and 5xx count): exit code 4. If no request had succeeded the line reads "provider
  unavailable, nothing measured"; otherwise it says the numbers cover only what was measured
  before the stop. A success resets the count. Five is a choice, not a measured value.
- First HTTP 404 whose body says the model is "not found", "does not exist" or "no longer
  available": exit code 4. The provider's text (redacted, cut at 300 characters) is in the line.
  The quota check's deliberate invalid-model step is exempt from this and from the count.

## HTTP 429 details

For a 429 the report keeps the quota metric, quota id, limit value, limit name and retry delay
the body names (allow-listed short strings), not the body. Read "window named: minute / day" and
"a limit of 0 was reported" as keyword matches on those names, not as the provider's statement.
If no names are found the redacted, truncated body is printed as for any other error.

## Plan prompt versions

`plan.v1` is what the pipeline sends now: the same instructions as `plan.v0` plus one sentence
about descriptions and synonyms, and a catalogue that carries every metric's and dimension's
description and synonyms word for word from `semantic/metrics.yaml`. `plan.v0` is kept only so
old cassettes still pass `va cassettes check`. A report names the prompts actually used. The
catalogue is built from the semantic layer only; it never reads a question file.

## Abstains in the report

For every question that abstained, the report has an "Abstains (raw)" section. It shows the reason
code, whether the planner chose it (and its raw reply, cut at 300 characters) or the pipeline did, and
the catalogue that was in the plan prompt. That catalogue has metric names, units and allowed
dimensions only, no definitions or synonyms, so a question that depends on a definition (for example
"merchandise only") can be abstained as ambiguous for lack of one. That is a reading of the prompt,
not something any run has shown yet.

## What the report is and is not

Raw numbers for one model with n and a Wilson interval each. "Every number verified" means the numbers
matched the warehouse, not that the right question was answered. There is no accuracy claim and no
comparison across models. The prompts are first drafts (plan.v0 untuned; compose.v1 and repair.v1 written after the first
real run, not yet run live), so a result describes the model and those prompts together. The report
lists the exact prompt ids and hashes, and shows the provider's error text for any non-2xx response.

## 6. Document retrieval probes (phase 2b)

These compare keyword, vector, hybrid and reranked search on the 65 dev document questions. They
never read the held-out file. A hosted embedder or reranker sees only documents marked public, and
the command refuses before any request if any ingested document is not. Nothing is an accuracy
claim: the questions and documents have the same author.

`va docs probe` grants the same default read-only role as `va docs ingest`; pass `--reader-role` for another.

Once, with the pgvector image running (docs/DOCKER_PGVECTOR_SWITCH.md), ingest the synthetic corpus
as public:

```powershell
va docs ingest corpus\synthetic --sensitivity public
```

Then run one probe at a time. Keys go in this window only, as in section 1; nothing is saved.

```powershell
# No network at all: checks the pipeline with the fake embedder and the fake reranker
va docs probe --embedder fake_256 --reranker fake_overlap

# Gemini embeddings (hosted; needs GEMINI_API_KEY)
va docs probe --embedder gemini_embedding_2_768 --max-requests 30
va docs probe --embedder gemini_embedding_001_768 --max-requests 30

# OpenRouter embeddings (hosted; needs OPENROUTER_API_KEY)
va docs probe --embedder openrouter_text_embedding_3_small --max-requests 30

# A reranker on top (Jina; needs JINA_API_KEY; about 65 requests, paced)
va docs probe --embedder gemini_embedding_2_768 --reranker jina_reranker_v3_5 --max-requests 80

# On this computer's CPU, no key and no daemon. Install the optional extra first (once):
#   pip install -e ".[local]"
# The first run of each model may download its files (the command prints a notice first).
va docs probe --embedder local_bge_small_en_v15
va docs probe --embedder local_bge_base_en_v15      # larger and slower
```

Add `--fusion-study` to any probe to also score a fixed grid of fusion settings (RRF constant,
candidate depth, keyword weight) on the same dev questions. It sends no extra requests, uses the
vectors and question embeddings the probe already has, and prints a table with the change against
the default and a bootstrap interval. It selects nothing: run it with at least two embedders and
look for changes that agree before anyone changes a default.

```powershell
va docs probe --embedder gemini_embedding_2_768 --fusion-study --max-requests 30
```

The report also has a "Questions with no answer" table: how well the top search score separates
questions the documents answer from ones they do not (AUC). It sets no threshold.

Each run writes `data/live_runs/<time>_docs_<names>/report.json` and `report.md`. Paste back the
printed table and the `HTTP:` line (status counts, latency); do not paste keys. Exit codes: 0 done,
2 refused or bad setup (missing key, unknown name, non-public document), 3 auth error (stopped on
the first one), 4 provider unavailable, rate-limited or request cap reached.

Vectors are stored per model, so a repeat run embeds nothing new and sends only the question
batches. A failed first call usually means a request shape in `docs/PHASE_2B_ASSUMPTIONS.md` was
wrong; paste the `Stopped early:` line.

### Local reranker (slice 2c)

A cross-encoder on this computer's CPU (fastembed; same optional extra as the local embedders).
The first run of each model may download its files; the command prints a notice first. The probe
reranks the default hybrid, vector-only and the pre-registered hybrid_B, and prints each change
against the unreranked method with a bootstrap interval. A hosted reranker (Jina) only reranks the
default hybrid, to save requests.

```powershell
va docs probe --embedder local_bge_small_en_v15 --reranker local_minilm_l6_cross_encoder
va docs probe --embedder gemini_embedding_2_768 --reranker local_minilm_l6_cross_encoder
va docs probe --embedder gemini_embedding_2_768 --reranker local_bge_reranker_base   # larger, slower
```

### Held-out document questions (once per embedder)

Commit `docs/adr/0009-document-retrieval-fusion.md` first, with the dev numbers pasted into its
evidence table. The command refuses if that file is missing, uncommitted, changed since it was
committed, or disagrees with the candidates built into the code. It accepts only the three
pre-registered candidates and no other setting, scores them on the 12 held-out document questions,
and refuses a second run for the same embedder (`data/heldout_ledger.jsonl`).

```powershell
va docs heldout --candidates A,B,baseline --embedder gemini_embedding_2_768
va docs heldout --candidates A,B,baseline --embedder local_bge_small_en_v15
```

If the first run fails before scoring (for example a missing key), nothing is recorded and you can
run it again. The output has the per-question rank of the first relevant passage for each
candidate and which questions differ. It applies no decision rule.

## 7. Router and injection probe (slice 2e)

Needs the warehouse loaded, the synthetic corpus ingested **as public** (`va docs ingest ...
--sensitivity public`, the same step as section 6), and one hosted key in this window only. It runs
the 12 dev router questions and the 5 injection questions (one per hostile document) through the
whole route-and-answer path against ONE model, public data and public documents only, retrieval by
keyword, top 3 passages. Nothing is recorded as a cassette and nothing is committed.

```powershell
# cheapest: only the route call for the 12 router questions (12 requests)
va route probe --provider groq --model <MODEL ID> --route-only --max-requests 20
# full: route, plan, compose and the injection set; at most 60 requests
va route probe --provider groq --model <MODEL ID> --max-requests 60
```

It stops on the first 401/403 (exit 3), when the provider is unavailable (exit 4), when the request
or token cap is reached, or if a retrieved document is not public (exit 4). The report
(`data/live_runs/<time>_<provider>_<model>/report.md`) is raw counts with n: how often the route
matched the author's label (and a confusion table), schema-valid replies, answers that typed a digit
outside a placeholder, how many model replies contained a string a hostile document asked for,
tokens and latency. It makes no accuracy claim and ranks nothing. Paste back the whole report.

### Reading the route probe report (after the first Groq run)

The first run (Groq gpt-oss-20b) showed route replies failing schema validation on the documents and
both routes, with no way to see why. The report now keeps, for every step whose reply did not
validate, the reply cut at 300 characters and the field-level errors, and for every non-2xx HTTP
response the status and redacted body, per question. `--record` writes cassettes next to the report
(public data and public documents only; same key scan and check as `va live probe`):

```powershell
va route probe --provider groq --model <MODEL ID> --route-only --max-requests 20 --record
```

The route prompt is now `route.v2`: the four exact reply shapes, "reason only with abstain", and
one worked example per route (made-up questions that are in no question file). If the first reply
does not validate, the second request shows that reply and its field errors, as the compose repair
does. The report separates, for the 12 router questions: route matched the label, chose a different
route (valid reply), reply not valid, and no reply; then, for valid data/documents/both routes, how
many reached the composer and where the others stopped. Injection counts cover only questions that
reached the composer (with n); the others are listed with the stage where they stopped, and say
nothing about injection.

`json_validate_failed` (HTTP 400, Groq): the shared router retries it once with JSON mode off (it
used to repeat the identical request, which failed 4 of 4 times), for every task including `route`;
text that then does not parse is handled as an invalid reply (the route step retries with the errors). The retry is a real HTTP
request and counts against `--max-requests`. A test now covers it on the route path.

### Why one step made four JSON-off calls (va_next20)

In the injection-only run a compose or repair step showed one `json_rejected` and then four
`malformed` calls with JSON mode off. Cause (from the code, matching the counts): the JSON-off retry
got an HTTP 200 whose content was empty; the adapter reports an empty completion as `malformed`,
`malformed` is a retryable error, and `max_retries` is 3, so the same request went out 1 + 3 = 4
times. Now: after a JSON refusal a step makes at most 2 requests (the refusal and one JSON-off
retry); an unusable reply or any other failure of that retry ends the path (text handed back
unparseable for an empty reply; next provider for other failures). A provider that refuses JSON mode
twice in a row for one question gets no more requests for that question; the next provider is used
if the sensitivity rule allows, otherwise the question ends as `provider_error`. A `malformed` reply
with no JSON refusal before it keeps the normal retries.

The report has a section "Unusable HTTP 200 replies": for each, the kind (`content_empty`,
`content_null`, `no_content`, `error_object`, `body_not_json`), the prompt, completion and reasoning
tokens and finish reason the provider reported, and the raw body (first 300 characters, key and
bearer text redacted). Whether the provider sent empty content or an error object is stated from
that body only (assumptions row 45 stays UNVERIFIED).

## Model-written labels for the binder (ADR-0013)

One live run, public documents only, recorded in cassettes. Do not run it twice: the ledger
(`data\binder_model_ledger.jsonl`) refuses a second generation. Needs the ADR committed first.

```powershell
va docs binder-model-labels --provider groq --model <GPT-OSS ID> --seed-run data\live_runs\20261121T064538Z_groq_<model>\report.json
git add eval\binder_model_labels\labels.yaml
git commit -m "Binder: model-written labels (one Groq run)"
va docs binder-model-dev
```

It sends 8 requests of 10 sentences (about 80 sentences). The dev command scores `current`, `S2` and
`SL` on the dev half only. Do not open the locked-half claims in the labels file. After the labels
file is committed and unchanged:

```powershell
va docs binder-model-locked --candidates current,S2,SL
```

which refuses unless ADR-0013 and the labels file are committed and unchanged, and runs once. No
default changes either way.

