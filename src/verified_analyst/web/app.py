"""Read-only web view of recorded question runs. No model call and no database is needed:
each record was produced by the real pipeline (see demo/README.md) and is shown as stored."""

from __future__ import annotations

import contextlib
import json
import threading
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


MAX_QUESTION_CHARS = 300


class _Handler(BaseHTTPRequestHandler):
    payload: dict = {}
    ask_fn = None  # callable(question) -> dict, or None when the server cannot answer new questions
    mode = "static"
    lock = threading.Lock()  # one live question at a time: the free provider plans are small

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            self._send(
                200,
                json.dumps({"mode": self.mode, "can_ask": self.ask_fn is not None}).encode(),
                "application/json",
            )
        elif self.path == "/api/results":
            body = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask" or self.ask_fn is None:
            return self._send(404, b'{"error":"not available"}', "application/json")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            q = str(json.loads(self.rfile.read(min(n, 4096)) or b"{}").get("question", "")).strip()
        except (ValueError, TypeError):
            return self._send(400, b'{"error":"bad request"}', "application/json")
        if not q or len(q) > MAX_QUESTION_CHARS:
            msg = {"error": f"question must be 1 to {MAX_QUESTION_CHARS} characters"}
            return self._send(400, json.dumps(msg).encode(), "application/json")
        with self.lock:
            try:
                out = self.ask_fn(q)
            except Exception as exc:  # noqa: BLE001 - a failed question must not stop the server
                msg = {"error": f"could not answer: {type(exc).__name__}"}
                return self._send(500, json.dumps(msg).encode(), "application/json")
        self._send(
            200, json.dumps(out, ensure_ascii=False, default=str).encode(), "application/json"
        )

    def log_message(self, *args) -> None:
        pass


def serve(port: int = 8000, host: str = "127.0.0.1", ask_fn=None, mode: str = "static") -> None:
    _Handler.payload = load_payload()
    _Handler.ask_fn, _Handler.mode = (staticmethod(ask_fn) if ask_fn else None), mode
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"http://{host}:{port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
