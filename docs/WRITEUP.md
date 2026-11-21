# Verified Analyst: write-up

## The problem
A language model asked a business question will state numbers fluently, and some will be wrong.
Business users cannot tell which. Verified Analyst puts a deterministic check between the model and
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

## Decisions worth reading
Pre-registered evals with locked held-out sets (ADR-0009 to 0013), a sensitivity rule so only public
data reaches hosted models, cassettes so tests never call a network, and an offline gate that fails
the build if a verified figure has no trace to its stored cell or passage.

## Limits
Verification shows a figure matches its source, not that the source answers the question. The free
Groq plan caps tokens per day (200,000, measured once), which limits live runs. Nothing is deployed;
the web view is local and replays recordings.
