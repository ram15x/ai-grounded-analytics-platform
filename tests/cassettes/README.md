# Cassettes

Recorded model calls live here, one directory per suite (`tests/cassettes/<suite>/calls.jsonl`).

**None are committed yet.** No real provider has been called; the tests in
`tests/test_cassette.py` and `tests/test_llm_pipeline.py` write cassettes into temporary
directories from a scripted FakeProvider. A cassette recorded from a fake would say nothing about
any model, so it is not committed.

Rules (PHASE_1C.md section d):
- Only public-tagged payloads are recorded. Headers and keys are never stored; a scan refuses any
  line that looks like a credential.
- Replay is the default in CI. A key miss raises an error; there is no network fallback.
- The key covers provider, model ID, the request, the prompt-template hash, the semantic-layer
  hash, the claim-schema hash and a warehouse content hash, so any change makes an entry stale.
- `va cassettes check <dir>` lists stale entries and their age; with `--usage usage.json` (written
  by a replay run) it also lists unused and missing entries.
- Recording is done by hand with keys in the environment (GROQ_API_KEY, GEMINI_API_KEY,
  OPENROUTER_API_KEY). There is no command for that yet.
