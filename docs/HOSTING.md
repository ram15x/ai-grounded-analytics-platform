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

## What is not deployed, and why

| Mode | Why it runs locally only |
|---|---|
| `grounded serve --workspace` (uploads) | No sign-in or per-user separation, so it refuses any host but 127.0.0.1. Hosting it would expose uploaded data. |
| `grounded serve --byok` (visitor's own key) | Needs Postgres with the warehouse loaded and a long-running server. The free hosts checked either asked for a card (Render's Blueprint flow) or expire free Postgres after 30 days. |

## Checked, not used

- **Render:** creating a Blueprint asked for a credit card to verify identity; the project is kept
  free of card-backed services.
- **Hugging Face Docker Spaces:** current docs suggested a paid plan for compute Spaces; sources
  disagreed, so it was not pursued.

## Docker

`Dockerfile` installs the package in editable mode (`pip install -e .`): a plain install cannot
find `demo/` next to the installed package. The image has not been built where this was written
(no Docker daemon).
