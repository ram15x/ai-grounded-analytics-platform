"""Read-only web view of recorded question runs. No model call and no database is needed:
each record was produced by the real pipeline (see demo/README.md) and is shown as stored."""

from __future__ import annotations

import contextlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "demo" / "results.json"
PAGE = Path(__file__).with_name("index.html")

META = ROOT / "demo" / "meta.json"

NOTE = (
    "Model replies are recorded from real calls to {model} and "
    "replayed, not re-run. The SQL, the result table, the passage text and every verdict were "
    "recomputed against the real UCI Online Retail warehouse (541,909 rows) and the synthetic "
    "document corpus. 'No recording' means the demo has no stored reply for that question; it "
    "is not a model abstention. No accuracy figure is claimed: 17 development questions only."
)


def load_payload(path: Path = RESULTS) -> dict:
    model = json.loads(META.read_text(encoding="utf-8"))["model"]
    return {"items": json.loads(path.read_text(encoding="utf-8")), "note": NOTE.format(model=model)}


def static_page(payload: dict) -> str:
    html = PAGE.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("/*DATA*/null", data)


class _Handler(BaseHTTPRequestHandler):
    payload: dict = {}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/results":
            body = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args) -> None:
        pass


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    _Handler.payload = load_payload()
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"http://{host}:{port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
