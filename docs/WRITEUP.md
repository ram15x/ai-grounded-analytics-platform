# AI Grounded Analytics Platform: write-up

## The problem
A language model asked a business question will state numbers fluently, and some will be wrong.
Business users cannot tell which. AI Grounded Analytics Platform puts a deterministic check between the model and
the reader: no figure is shown unless it matches the warehouse or a stored passage.

## How a figure gets to the screen
1. A router decides whether the question needs data, documents, both, or neither.
2. For data, the model plans a query over named metrics (`semantic/metrics.yaml`). A compiler turns it
   into SQL, a sqlglot guard allows one read-only SELECT, and it runs under a read-only role.
3. The model writes an answer with placeholders such as `{c1}`. It never types a digit. A digit it
   types itself is rejected.
4. The verifier binds each placeholder to a result cell (or a stored passage figure), recomputes it
   with `Decimal`, and returns a verdict per figure. If any figure fails, the answer is repaired
   or withheld.

## What was measured, and what was not
- 30 adversarial answers (wrong row, period, dimension value, metric with the same unit): all 30
  were accepted by the first version of the verifier; after the sentence-context check, 0 are.
  They were written before the fix. They test this verifier, not models in general.
- The recorded demo: 17 development questions through the real pipeline on recorded replies from
  one model (openai/gpt-oss-20b on Groq, free plan): 7 verified, 5 withheld (ambiguous, forecast,
  unsupported metric), 5 with no stored reply. **No accuracy figure is claimed.** n is 17 and one
  author wrote the questions.
- Injection: 5 questions, one model, 0 obeyed. Not a safety claim.
- Not yet run: the model-written label test for the document-number binder (built, pre-registered in
  ADR-0013, blocked on free-plan quota).

## Bring your own data (built, lightly tested)
The system is no longer tied to one dataset. `grounded serve --workspace` (local only) takes a CSV, TSV or
Excel file, profiles it, and asks the person what it is and what each number means, using a closed
set of metric definitions (count, sum, average, min, max, distinct count, one filter). A definition
is published only if it reproduces a figure the person already knows, recomputed by a separate path.
Personal-looking columns are never loaded. Data is `internal` by default, so a hosted provider is
refused; a local Ollama model (no key) or an explicit `public` mark is needed. PDF, Markdown and text
documents can be attached; only a PDF's existing text layer is read, scans are refused (no OCR).
A public tryout (`--byok`) gives each visitor one question per difficulty group on the demo data,
with their own API key used once and never stored.
Proved on a second, differently shaped synthetic dataset (subscriptions): 6 of 6 known figures
reconciled and a scripted run was verified. **Not measured:** how well any real model plans questions
on uploaded data, and whether any local model is usable on a 4 GB GPU.

**Measured on 2026-11-22, on the author's Windows machine, real Groq key, `openai/gpt-oss-20b`:**
- Uploaded the subscriptions CSV through the browser; 1,200 rows loaded, the email column left out,
  the known figure 1,200 reconciled ("matches").
- "How many subscriptions are there?" came back verified: "There are 1,200 subscriptions", with the
  SQL shown, 4 model calls, 4,196 tokens. One question, not a benchmark.
- "What is the total MRR?" was withheld (out of scope), correctly: no MRR number had been approved.
  The page first printed `[object Object]` for the reason; fixed.
- Key headroom: Groq's model-list call carries no rate-limit headers (observed), so the check now
  makes one capped chat call (93 tokens measured) and reads them: 8,000 tokens per minute, remaining
  count shown. OpenRouter and Gemini headroom are untested. A daily cap is not in those headers.

## Decisions worth reading
Pre-registered evals with locked held-out sets (ADR-0009 to 0013), a sensitivity rule so only public
data reaches hosted models, cassettes so tests never call a network, and an offline gate that fails
the build if a verified figure has no trace to its stored cell or passage.

## Limits
Verification shows a figure matches its source, not that the source answers the question. The free
Groq plan caps tokens per day (200,000, measured once), which limits live runs. Only the recorded
view is deployed (GitHub Pages, static, no model calls); uploads and the key-based tryout run
locally. Single user, no sign-in. Not run: the locked held-out eval, real-world PDFs, the Docker
image build.
