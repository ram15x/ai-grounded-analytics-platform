# ADR-0014: End-to-end evaluation, dev and locked

Status: accepted (written before any end-to-end run on either split; no live end-to-end run exists
at the time of writing).

## What is measured
Whole-pipeline behaviour (route, plan, query, compose, verify) on author-written questions with gold
values computed by pandas over the raw UCI file (eval/gold). Each run is scored from what the reader
would see:

- `correct`: verified, and the gold figure(s) appear among the shown figures at the question's decimals.
  For a table gold, every numeric cell must appear; extra shown figures are allowed.
- `verified_wrong`: verified, but a gold figure is missing. This is the failure the product exists to
  prevent. Any such row is investigated; the command exits 1.
- `withheld`: the system abstained or no figure could be verified. Not counted as wrong.
- `no_reply`: no usable model reply.

## Splits
- dev: eval/questions/dev.yaml (15). May be run repeatedly while developing.
- locked: eval/questions/heldout.yaml (15). One run per model id, ever. `--confirm-locked` is
  required; the one-shot ledger `data/e2e_locked_ledger.jsonl` is written only after the whole run
  completes, and a second run for the same model is refused. The locked gold values are computed at
  run time and are not written to the report. Not run on replay.
- No prompt, config or code change may be made after reading locked results and then re-scored on the
  same locked set; a change means a new, clearly labelled non-locked claim.

## Reporting
Counts per outcome, `correct` over answered with a Wilson 95% interval, and the full per-question
table. No claim beyond "n questions, one author, one model". Withheld questions are reported, not hidden.

## Known limits
Author-written questions; one corpus; a figure-level match does not prove the answer addressed the
question asked; free-plan quota may leave `no_reply` rows, which are reported as such.

## Guard tests relaxed on purpose
`tests/test_heldout.py` and `tests/test_docs_cli.py` forbid any product module from naming the locked
question file. They now allow exactly one more module, `src/grounded_analytics/e2e_locked.py`, which
holds only the path. The locked file is read by one command, `grounded eval e2e --split locked
--confirm-locked`, which refuses a second run per model. This mirrors `grounded docs heldout`.

## Settings for the first locked run (written before it, after the dev runs only)

- Model: groq `openai/gpt-oss-20b`. Plan prompt: `plan.v2`.
- Why `plan.v2`: on the 15 dev questions (run `20261122T141403Z`), it answered 13, with 9 correct,
  0 verified_wrong, 4 withheld (all four are the questions marked not expressible) and 2 no_reply
  (HTTP 429, tokens per minute). Every expressible question that got a reply was correct (9 of 9).
  With `plan.v1`, T1-11 was withheld twice and answered correctly once with `plan.v2` (n=1).
- `--pause-s 45` between questions, because each question builds a fresh router and the 8000
  tokens-per-minute limit otherwise turns questions into no_reply. This changes no prompt, model or
  scoring.
- Nothing above may be changed and re-scored on the locked set after its results are read.
