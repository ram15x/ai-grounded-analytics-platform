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

## 3. Probe one model at a time (at most 25 requests each)

```powershell
va live probe --provider groq --model openai/gpt-oss-20b --quota-check
```

`--quota-check` spends 3 of the 25 requests on one good request, one deliberately invalid one and one
good one, to see whether a failed request changes the remaining-requests header. Leave it off to use
all 25 on the battery. The run stops at once on the first 401/403, at 25 requests, or at the token cap
in `config\router.yaml` (`probe.token_cap`). Optional: `--max-tokens 2048` if the report shows
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

## What the report is and is not

Raw numbers for one model with n and a Wilson interval each. "Every number verified" means the numbers
matched the warehouse, not that the right question was answered. There is no accuracy claim and no
comparison across models. The prompts are the untuned v0 minimum, so a result describes the model and
those prompts together.
