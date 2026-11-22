# Scheduled reports

`grounded report run` answers a fixed list of questions (default `demo/weekly_questions.json`) and writes a
**pending** draft. It never approves and never sends anything. A person runs
`grounded report approve --who <name> --digest <digest>` after reading the draft.

```
grounded report run --title "Weekly check" --replay          # recorded replies, needs the local database
grounded report run --title "Weekly check" --provider groq --model <id>   # live, public data only
```

Run it from the operating system's scheduler. The scheduler is outside this project.

- Windows Task Scheduler: a daily or weekly task running `.venv\\Scripts\\grounded.exe report run --title "Weekly check" --replay`, start in the project folder.
- Linux or macOS cron: `0 8 * * 1  cd /path/to/ai-grounded-analytics-platform && .venv/bin/va report run --title "Weekly check" --provider groq --model <id>`

A live run needs the provider key in the scheduler's environment and counts against the free
plan's daily token cap (see the usage line `grounded` prints). If a question gets no answer, it is listed
under "Withheld" in the draft with its reason.

# End-to-end dev run

`grounded eval e2e --provider groq --model <id>` runs the 15 dev questions (eval/questions/dev.yaml)
through the pipeline and scores each against the gold value computed from the raw file
(`--raw "data/Online Retail.xlsx"`). Scores: `correct`, `verified_wrong` (verified but no shown figure
equals the gold value; the command exits 1), `withheld`, `no_reply`. It writes `report.md` and
`results.json` under `data/e2e_runs/`. Live runs spend free-plan tokens: read the usage line first.
`--replay` only exercises the plumbing (the recordings do not cover these questions, so every row
is `no_reply`). The held-out questions are not wired in; they need a locked, one-shot procedure.

`grounded eval e2e --plan-prompt plan.v2` sends the planner prompt that adds the filter shape (and the
allowed operators) to plan.v1. It exists to test one hypothesis for the T1-11 plan failure
(UNVERIFIED). Compare the two prompts on the same dev questions before changing the default.
