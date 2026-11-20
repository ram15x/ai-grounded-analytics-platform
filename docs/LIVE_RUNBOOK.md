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

The probe now paces itself with a client-side tokens-per-minute bucket (`tpm: 8000` for Groq in
`config\router.yaml`, used at 50%) and waits up to 120 s for it, so a run takes longer than the
first one did (a full 10-question battery can take 10 to 20 minutes) and should hit fewer 429s. The
report header prints the bucket settings and whether `reasoning_effort` was sent. It is not sent
unless you add `reasoning_effort: low` under the Groq provider in the config; whether Groq accepts it
for this model is UNKNOWN, so try it as its own run and paste back the status counts and the error
text, which the report now shows for any non-2xx response. Optional: `--max-tokens 2048` if the report shows
`length` among the finish reasons, `--token-cap N` to lower the cap, `--no-record` to skip cassettes.

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

If Groq answers HTTP 400 `json_validate_failed`, the router sends the same request once more (it counts
as a request against the cap). If that is also rejected, the question goes through the normal repair
step, and the report's accounting lists `json_rejected` under errors by provider. Whether the retry
helps is not yet measured; look at `by status` and `json_rejected` counts in the next run.

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
