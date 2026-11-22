"""Vercel entry point for the public tryout (visitors bring their own model key).

Vercel's Python runtime loads `handler`, a BaseHTTPRequestHandler class. Everything here mirrors
`grounded serve --byok`; only the reply cache moves to /tmp, the one writable place on Vercel.
Needs GA_READONLY_DATABASE_URL (the hosted warehouse, read-only role) in the project settings.
The per-visitor allowance is held in memory, so with several warm instances it is best-effort:
it is a courtesy limit for the visitor's own key, never a security control.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounded_analytics.cli import _byok_ask, _byok_models  # noqa: E402
from grounded_analytics.tryout import load_config  # noqa: E402
from grounded_analytics.web.app import configure  # noqa: E402

CFG = load_config()
CACHE = os.environ.get("GA_TRYOUT_CACHE", "/tmp/tryout_cache")  # noqa: S108

handler = configure(
    byok_fn=lambda q, p, m, k, tier: _byok_ask(q, p, m, k, tier, CACHE),
    models_fn=_byok_models,
    tryout_cfg=CFG,
)
