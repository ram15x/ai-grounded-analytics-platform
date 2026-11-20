# Phase 1c-1: assumptions about provider behaviour (all UNVERIFIED)

Nothing in the model router, the adapters or the cassettes has touched a real provider. Every item
below is an assumption the code makes. Each must be checked against real calls (by hand, with keys
in the environment) before any number from a live run is trusted. Tests use mocked HTTP only, so a
passing test proves the code does what these assumptions say, not that the assumptions are true.

**Status column.** Every row is STILL UNKNOWN until real output from `va live models` or `va live probe`
has been pasted back and read. A row is changed to CONFIRMED or REFUTED only from that output, with the
date and the line that shows it, never from memory. "Settled by" names the output that can decide it.
Real output so far: one `va live models --provider groq` run (2026-11-20). Rows it settles are marked below; everything else is STILL UNKNOWN.

## Request and response shapes

| # | Assumption | Status | Settled by | Evidence (date, source line) |
|---|---|---|---|---|
| 1 | Groq and OpenRouter accept OpenAI-style chat completions at `https://api.groq.com/openai/v1/chat/completions` and `https://openrouter.ai/api/v1/chat/completions`, with `Authorization: Bearer <key>`. | STILL UNKNOWN | `va live models` (HTTP 200 on the listing) and `va live probe` (HTTP 200 on completions) |  |
| 2 | Both accept `max_tokens` (Groq may prefer `max_completion_tokens`) and `temperature: 0`. | STILL UNKNOWN | probe: no HTTP 400 complaining about `max_tokens` or `temperature` |  |
| 3 | `response_format: {"type": "json_object"}` is honoured by the chosen Groq model, and by whichever `:free` OpenRouter model is chosen (support varies by model; some may ignore it and return free text). | STILL UNKNOWN | probe: 'plan/compose responses that parse as JSON' rate, per provider |  |
| 4 | JSON mode may require the word "JSON" in the prompt; the minimal prompts contain it, nothing else was checked. | STILL UNKNOWN | probe: JSON-parse rate; a 400 about JSON mode would show in `by status` |  |
| 5 | Gemini accepts `POST .../v1beta/models/{model}:generateContent` with the key in `x-goog-api-key`, a `systemInstruction` field, and `generationConfig.responseMimeType: "application/json"`. | STILL UNKNOWN | probe on Gemini: HTTP 200 and a JSON-parse rate |  |
| 6 | Responses carry `usage.prompt_tokens` / `completion_tokens` (Groq, OpenRouter) and `usageMetadata.promptTokenCount` / `candidatesTokenCount` (Gemini). If absent, the router estimates characters / 4 and flags the call. | STILL UNKNOWN | probe: 'estimated-usage calls' is 0 and token totals are non-zero |  |
| 7 | Gemini "thinking" tokens and Groq/gpt-oss reasoning tokens may be reported separately or inside completion tokens; if separate, the budget under-counts. | STILL UNKNOWN | probe: 'reasoning reported' vs completion tokens vs finish reasons |  |
| 8 | A reasoning model can spend `max_tokens` on reasoning and return empty or truncated JSON (`finish_reason: length`). Empty content is treated as malformed and retried; truncated JSON becomes a schema failure. `finish_reason` is recorded but not acted on. | STILL UNKNOWN | probe: finish reasons (`length`), empty/malformed responses, JSON-parse rate |  |
| 9 | OpenRouter can return HTTP 200 with an `error` object and no `choices` (upstream failure). The code reads that as malformed and retries; whether that is the right class is unchecked. | STILL UNKNOWN | OpenRouter probe: any 200 with an error body shows as malformed in errors by provider |  |
| 10 | Models return bare JSON. A markdown code fence is stripped; `<think>` blocks or prose around the JSON are not handled. | STILL UNKNOWN | probe: JSON-parse rate and the raw failing text in the reviewed cassette files |  |

## Errors and rate limits

| # | Assumption | Status | Settled by | Evidence (date, source line) |
|---|---|---|---|---|
| 11 | HTTP 429 means rate limited on all three; 5xx and 408 are transient; 401/403 are key problems; other 4xx are our fault and not retried. | STILL UNKNOWN | probe: by-status counts (any 4xx other than 429/401/403, any 408/5xx) |  |
| 12 | `retry-after` is an integer or decimal number of seconds. Documented for Groq (RESEARCH.md); not found for OpenRouter (it documents `X-RateLimit-*` headers) or Gemini (its 429 body may carry a retry delay instead of a header). The HTTP-date form is ignored. Absent, the router backs off exponentially with jitter. | STILL UNKNOWN | probe: 'retry-after headers' section; needs a 429 to occur, otherwise stays unknown |  |
| 13 | A retry-after longer than `max_retry_after_s` (30 s) usually means a daily limit; the router falls through rather than waiting. | STILL UNKNOWN | a 429 with a long retry-after; unlikely in one probe, likely STILL UNKNOWN |  |
| 14 | Retrying a 5xx or timeout is safe and does not bill or consume quota twice. Whether failed or 429 requests count against daily limits (OpenRouter: 50 requests/day without credits) is unknown. | STILL UNKNOWN | `--quota-check` output (only if a remaining-requests header is reported) |  |
| 15 | Groq limits of 30 RPM, 1K RPD, 8K TPM, 200K TPD, and which plan column applies to a free account, are unverified. Gemini and OpenRouter limits are not in the config at all, so those providers have no bucket. | STILL UNKNOWN | `va live models` rate-limit headers; the provider's own limits page |  |
| 16 | Limits are per organisation or project, not per key. The local token bucket cannot see other consumers of the same quota, and two router processes would each assume a full bucket. | STILL UNKNOWN | cannot be settled by one probe; STILL UNKNOWN unless limits are seen shared |  |
| 17 | A smooth-refill token bucket at 50% of the documented number keeps a sliding or fixed-window server limit from firing. This is a guess about the server's windowing. | STILL UNKNOWN | probe: any 429 during a run that stayed under the bucket rate |  |
| 18 | Three consecutive 429s mean the provider is saturated; a 60 s cooldown then one probe is a reasonable recovery. Neither number is measured. | STILL UNKNOWN | needs a real 429 sequence |  |
| 19 | A 30 s timeout is enough for every task; reasoning models may need longer. | STILL UNKNOWN | probe: p50/p95 latency and any timeout in errors by provider |  |

## Model IDs and data policy

| # | Assumption | Status | Settled by | Evidence (date, source line) |
|---|---|---|---|---|
| 20 | `openai/gpt-oss-20b` is the string Groq expects. The Gemini and OpenRouter IDs are deliberately unset (`REPLACE-...`) and the adapters refuse to call until they are filled in. RESEARCH.md notes Gemini names churn and that 2.5 models were reported to retire on 2026-10-16. | CONFIRMED (Groq ID); STILL UNKNOWN (Gemini, OpenRouter) | `va live models`: configured-model line PRESENT/ABSENT; Gemini/OpenRouter ID lists | 2026-11-20, owner's `va live models --provider groq`: `configured model (default): openai/gpt-oss-20b -> PRESENT` in a 200 listing of 11 models. Exists for this key; not yet used in a completion. Gemini and OpenRouter IDs are still unset. |
| 21 | Sending the UCI Online Retail data (CC BY 4.0, public) to free hosted tiers is acceptable even though Gemini's unpaid terms allow human review and training use, and OpenRouter `:free` data policies are unresearched. The `public` tag on `analytics.sales_lines` is the owner's decision to confirm. Gemini's terms also say EEA/UK/CH use needs paid services. | STILL UNKNOWN | the owner's decision and the providers' current terms, not a program output |  |

## Behaviour that matters for the eval

| # | Assumption | Status | Settled by | Evidence (date, source line) |
|---|---|---|---|---|
| 22 | Temperature 0 does not give identical output on hosted models; a cassette freezes one sample, so replay proves the plumbing, not stability. | STILL UNKNOWN | re-running the probe twice with the same prompts and comparing cassette text |  |
| 23 | Small models may not follow the placeholder format (ADR-0008 open risk), may not emit valid JSON, or may choose the wrong claim for a sentence. Measure placeholder adherence, schema-valid rate, AMBIGUOUS rate and tokens per question first. | STILL UNKNOWN | probe: placeholder-adherence, schema-valid, and abstain/repair rates |  |
| 24 | The characters / 4 token estimate used for the per-run cap and the buckets is rough, and deliberately includes `max_tokens` so the cap errs on the side of abstaining early. The default 30K cap and the 3-4K tokens per call figure are assumptions from PHASE_1C.md. | STILL UNKNOWN | probe: tokens per call (prompt/completion) against the 30K and 60K caps |  |
| 25 | Prompt text in `llm/prompts.py` is a minimum for the fake test and has never been shown to a real model. | STILL UNKNOWN | probe: the same rates; prompts are v0 and untuned |  |

## Added in Phase 1c-2 (live tooling)

| # | Assumption | Status | Settled by | Evidence (date, source line) |
|---|---|---|---|---|
| 26 | Listing endpoints work as coded: Groq `GET .../openai/v1/models`, OpenRouter `GET .../api/v1/models`, Gemini `GET .../v1beta/models?pageSize=1000` with `nextPageToken`; response fields `data[].id`, `pricing.prompt`, `models[].name`, `supportedGenerationMethods`. | CONFIRMED (Groq only); STILL UNKNOWN (Gemini, OpenRouter) | `va live models` output: HTTP status and model counts per provider | 2026-11-20, owner's `va live models --provider groq`: `HTTP 200: 11 models listed`, IDs parsed from `data[].id`. Gemini and OpenRouter listings not yet run. |
| 27 | The headers matched by the pattern `ratelimit|rate-limit|retry-after|quota|remaining` are the ones each provider uses for limits. Gemini may send none. | STILL UNKNOWN | `va live models` and probe 'rate-limit headers seen' | 2026-11-20, owner's `va live models --provider groq`: `rate-limit headers seen: none` on the Groq listing response. That says nothing about completion responses, where Groq may still send them. |
| 28 | Reasoning tokens are reported at `usage.completion_tokens_details.reasoning_tokens` (Groq, OpenRouter) and `usageMetadata.thoughtsTokenCount` (Gemini). | STILL UNKNOWN | probe tokens line: reasoning reported > 0 for a reasoning model |  |
| 29 | A request with an unknown model ID fails with a 4xx without being served, so it is a clean probe for whether failed requests use quota. | STILL UNKNOWN | `--quota-check`: invalid_model step shows HTTP 4xx |  |
| 30 | A 25-request probe stays within each free tier's per-minute limits and so produces no 429s. | STILL UNKNOWN | probe: by-status counts and the retry-after section |  |
| 31 | Sending the 10 T1 questions, the catalogue and single-row result tables (public UCI aggregates) is all the data a probe sends; no customer-level rows leave the machine. | STILL UNKNOWN | review `calls.jsonl` in the recorded cassette before anything is committed |  |
| 32 | Groq's edge rejects Python's default User-Agent (HTTP 403) and accepts an explicit one. | CONFIRMED (Groq, listing, n=1 each) | `va live models --provider groq` before and after the User-Agent change | Before: owner saw HTTP 403 through urllib and 200 through curl.exe with the same key and URL. After the explicit `verified-analyst/<version>` header (commit 293d5a8): 2026-11-20, owner's `va live models --provider groq`: `HTTP 200`. One run each, not repeated; the User-Agent was the only intended change, so the cause is inferred, not isolated. |
