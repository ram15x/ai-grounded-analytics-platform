"""The Vercel entry point configures the same handler as `grounded serve --byok`."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vercel_handler_serves_the_tryout_page_and_menu():
    spec = importlib.util.spec_from_file_location("vercel_index", ROOT / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.handler.mode == "byok" and str(mod.CACHE).startswith("/tmp")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "AI Grounded Analytics Platform" in page
        menu = json.loads(urllib.request.urlopen(base + "/api/tryout").read())
        assert sum(len(v) for v in menu["questions"].values()) == 9 and menu["own_slots"] == 3
    finally:
        srv.shutdown()


def test_vercel_config_routes_everything_to_the_handler():
    cfg = json.loads((ROOT / "vercel.json").read_text())
    assert cfg["functions"]["api/index.py"]["maxDuration"] <= 300  # the free plan's ceiling
    assert cfg["rewrites"] == [{"source": "/(.*)", "destination": "/api/index"}]
    req = (ROOT / "requirements.txt").read_text()
    for dep in ("pandas", "psycopg", "sqlglot", "pydantic", "pyyaml", "pypdf", "openpyxl"):
        assert dep in req
