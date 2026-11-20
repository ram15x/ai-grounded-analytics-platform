# Phase 1c-1: assumptions about provider behaviour (all UNVERIFIED)

Nothing in the model router, the adapters or the cassettes has touched a real provider. Every item
below is an assumption the code makes. Each must be checked against real calls (by hand, with keys
in the environment) before any number from a live run is trusted. Tests use mocked HTTP only, so a
passing test proves the code does what these assumptions say, not that the assumptions are true.

## Request and response shapes
1. Groq and OpenRouter accept OpenAI-style chat completions at `https://api.groq.com/openai/v1/chat/completions` and `https://openrouter.ai/api/v1/chat/completions`, with `Authorization: Bearer <key>`.
2. Both accept `max_tokens` (Groq may prefer `max_completion_tokens`) and `temperature: 0`.
3. `response_format: {"type": "json_object"}` is honoured by the chosen Groq model, and by whichever `:free` OpenRouter model is chosen (support varies by model; some may ignore it and return free text).
4. JSON mode may require the word "JSON" in the prompt; the minimal prompts contain it, nothing else was checked.
5. Gemini accepts `POST .../v1beta/models/{model}:generateContent` with the key in `x-goog-api-key`, a `systemInstruction` field, and `generationConfig.responseMimeType: "application/json"`.
6. Responses carry `usage.prompt_tokens` / `completion_tokens` (Groq, OpenRouter) and `usageMetadata.promptTokenCount` / `candidatesTokenCount` (Gemini). If absent, the router estimates characters / 4 and flags the call.
7. Gemini "thinking" tokens and Groq/gpt-oss reasoning tokens may be reported separately or inside completion tokens; if separate, the budget under-counts.
8. A reasoning model can spend `max_tokens` on reasoning and return empty or truncated JSON (`finish_reason: length`). Empty content is treated as malformed and retried; truncated JSON becomes a schema failure. `finish_reason` is recorded but not acted on.
9. OpenRouter can return HTTP 200 with an `error` object and no `choices` (upstream failure). The code reads that as malformed and retries; whether that is the right class is unchecked.
10. Models return bare JSON. A markdown code fence is stripped; `<think>` blocks or prose around the JSON are not handled.

## Errors and rate limits
11. HTTP 429 means rate limited on all three; 5xx and 408 are transient; 401/403 are key problems; other 4xx are our fault and not retried.
12. `retry-after` is an integer or decimal number of seconds. Documented for Groq (RESEARCH.md); not found for OpenRouter (it documents `X-RateLimit-*` headers) or Gemini (its 429 body may carry a retry delay instead of a header). The HTTP-date form is ignored. Absent, the router backs off exponentially with jitter.
13. A retry-after longer than `max_retry_after_s` (30 s) usually means a daily limit; the router falls through rather than waiting.
14. Retrying a 5xx or timeout is safe and does not bill or consume quota twice. Whether failed or 429 requests count against daily limits (OpenRouter: 50 requests/day without credits) is unknown.
15. Groq limits of 30 RPM, 1K RPD, 8K TPM, 200K TPD, and which plan column applies to a free account, are unverified. Gemini and OpenRouter limits are not in the config at all, so those providers have no bucket.
16. Limits are per organisation or project, not per key. The local token bucket cannot see other consumers of the same quota, and two router processes would each assume a full bucket.
17. A smooth-refill token bucket at 50% of the documented number keeps a sliding or fixed-window server limit from firing. This is a guess about the server's windowing.
18. Three consecutive 429s mean the provider is saturated; a 60 s cooldown then one probe is a reasonable recovery. Neither number is measured.
19. A 30 s timeout is enough for every task; reasoning models may need longer.

## Model IDs and data policy
20. `openai/gpt-oss-20b` is the string Groq expects. The Gemini and OpenRouter IDs are deliberately unset (`REPLACE-...`) and the adapters refuse to call until they are filled in. RESEARCH.md notes Gemini names churn and that 2.5 models were reported to retire on 2026-10-16.
21. Sending the UCI Online Retail data (CC BY 4.0, public) to free hosted tiers is acceptable even though Gemini's unpaid terms allow human review and training use, and OpenRouter `:free` data policies are unresearched. The `public` tag on `analytics.sales_lines` is the owner's decision to confirm. Gemini's terms also say EEA/UK/CH use needs paid services.

## Behaviour that matters for the eval
22. Temperature 0 does not give identical output on hosted models; a cassette freezes one sample, so replay proves the plumbing, not stability.
23. Small models may not follow the placeholder format (ADR-0008 open risk), may not emit valid JSON, or may choose the wrong claim for a sentence. Measure placeholder adherence, schema-valid rate, AMBIGUOUS rate and tokens per question first.
24. The characters / 4 token estimate used for the per-run cap and the buckets is rough, and deliberately includes `max_tokens` so the cap errs on the side of abstaining early. The default 30K cap and the 3-4K tokens per call figure are assumptions from PHASE_1C.md.
25. Prompt text in `llm/prompts.py` is a minimum for the fake test and has never been shown to a real model.
