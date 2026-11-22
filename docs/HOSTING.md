# Hosting

## What is deployed

The recorded view is live at https://ram15x.github.io/ai-grounded-analytics-platform/
(GitHub Pages, free).

- `.github/workflows/pages.yml` rebuilds it on every push to `main`: `grounded serve --build-static _site`
  writes one HTML page with the recorded runs embedded, plus the two fonts and their licences.
- It shows recorded model replies; the checks re-run in the browser. It cannot call a model, holds
  no key and stores nothing.
- Build and deploy are separate jobs: deploying in the same job as the upload raced the artifact
  listing and failed with "Found 0 artifact(s)" (observed twice).

The tryout (`grounded serve --byok`) is live at https://ai-grounded-analytics-platform.vercel.app/
(Vercel Hobby plus Neon free Postgres, no card on either).

- `api/index.py` is the Vercel entry. `vercel.json` rewrites every path to `/api/index/<path>`;
  the handler strips that prefix. Vercel's Python runtime only detects a top-level `class handler`,
  so it is a class, not a variable assigned from a call.
- The warehouse and the public documents live in Neon. The site connects as the read-only role
  `ga_readonly`, set as `GA_READONLY_DATABASE_URL` in the Vercel project settings. The admin login
  is used only from the machine that loads the data.
- The visitor's key travels with each question and is not stored. The reply cache is `/tmp`, which
  is empty on a cold start, so a menu question needs the visitor's key unless it is still warm.
- Checked live on 2026-11-22: "How many orders were placed?" returned 19,960, tied to the query result.

## What is not deployed, and why

| Mode | Why it runs locally only |
|---|---|
| `grounded serve --workspace` (uploads) | No sign-in or per-user separation, so it refuses any host but 127.0.0.1. Hosting it would expose uploaded data. |

## Checked, not used

- **Render:** creating a Blueprint asked for a credit card to verify identity; the project is kept
  free of card-backed services.
- **Hugging Face Docker Spaces:** current docs suggested a paid plan for compute Spaces; sources
  disagreed, so it was not pursued.

## Docker

`Dockerfile` installs the package in editable mode (`pip install -e .`): a plain install cannot
find `demo/` next to the installed package. The image has not been built where this was written
(no Docker daemon).
