# Hosting

Status: written, not deployed. Nothing here has run on a host yet. I can't create accounts
for you, so the last step is yours.

## What I measured and what I did not

| Item | Result |
|---|---|
| `pip install -e .` in a clean venv, then `va serve` (recorded view) | Works. Serves `/` with 200, about 85 MB resident memory. |
| Non-editable `pip install .` | **Fails**: the app looks for `demo/` next to the installed package. The Dockerfile uses `-e`, so it avoids this. Don't change it to a plain install. |
| `docker build` of the Dockerfile | **Not run.** The sandbox has the docker CLI but no daemon. The steps are the same ones I ran by hand, but the image itself is unbuilt. |
| Live questions (`--byok`) on a host | Not run. Needs Postgres, the seeded warehouse, and memory I have not measured. |

## Option A (recommended for today): recorded view on Render's free tier

What visitors get: the recorded runs, the verification view, the trust page. No key, no database,
no cost, nothing to leak. Live questions are off and the page says so.

Free-tier terms, from Render's docs when I read them today (they change; check before relying):
- A free web service spins down after 15 minutes without traffic and takes up to about a minute
  to wake. The first visitor after a quiet period waits.
- Free instance hours: 750 per month per workspace.
- The filesystem is ephemeral. That is fine here: this mode writes nothing.

Card check: when I tried "New Blueprint Instance" on a fresh Render account, it asked for a
credit card to "verify your identity" (the dialog said a temporary $1 authorization, not a
charge). I did not enter one. I have not checked whether creating a plain free Web Service
instead of a Blueprint skips this. If you don't want to add a card, that is the thing to try,
or use another host.

Steps (yours; card and password entry stay with you):
1. `git pull`, then check `render.yaml` at the repo root.
2. In Render: New, Blueprint, pick the `verified-analyst` GitHub repo, apply.
3. Wait for the build, open the URL, confirm the recorded run opens.

If the build fails, send me the log. I could not test the image build here.

## Option B: live tryout (visitors bring their own key)

Needs a Postgres holding the sales warehouse, a read-only role, and `va serve --byok`.
- Render's free Postgres expires after 30 days (14-day grace), one per account. You would
  re-seed it each month.
- The free web instance is small. I have not measured how much memory the document search plus
  a question needs, so I can't promise it fits.
- Visitor keys: sent once per request in a header, used for that call, scrubbed from errors,
  never written to disk or logs. The per-visitor allowance is in memory, so a restart resets it.
  It is a courtesy limit, not security.
- Cost risk to you: none from model calls (visitors use their own key). Abuse risk: someone can
  hit the free tier's limits. There is no login.

I would not do B until A is live and you've seen how it behaves.

## Not recommended

- **Hugging Face Docker Spaces:** the current docs I read say creating a compute Space (Docker or
  Gradio) needs a paid plan, with free hardware sleeping after 48 hours. A forum thread suggests
  this is unsettled. Verify on their page before choosing it.
- **Uploading your own data to any public host.** The upload workspace (`--workspace`) refuses
  non-loopback hosts on purpose: no auth, one user. Keep it local.
- **Internal-sensitivity data on a hosted model.** Hosted adapters refuse it by design.

## Before you share the link

- The tryout key and cache rules are in README ("Public tryout").
- Read the trust page yourself once, as a stranger would.
- The limits are in `docs/VERIFIER_LIMITS.md`. Don't claim more than they say.
