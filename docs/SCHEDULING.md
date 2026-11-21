# Scheduled reports

`va report run` answers a fixed list of questions (default `demo/weekly_questions.json`) and writes a
**pending** draft. It never approves and never sends anything. A person runs
`va report approve --who <name> --digest <digest>` after reading the draft.

```
va report run --title "Weekly check" --replay          # recorded replies, needs the local database
va report run --title "Weekly check" --provider groq --model <id>   # live, public data only
```

Run it from the operating system's scheduler. The scheduler is outside this project.

- Windows Task Scheduler: a daily or weekly task running `.venv\\Scripts\\va.exe report run --title "Weekly check" --replay`, start in the project folder.
- Linux or macOS cron: `0 8 * * 1  cd /path/to/verified-analyst && .venv/bin/va report run --title "Weekly check" --provider groq --model <id>`

A live run needs the provider key in the scheduler's environment and counts against the free
plan's daily token cap (see the usage line `va` prints). If a question gets no answer, it is listed
under "Withheld" in the draft with its reason.
