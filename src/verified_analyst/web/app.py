"""Read-only web view of recorded question runs. No model call and no database is needed:
each record was produced by the real pipeline (see demo/README.md) and is shown as stored."""

from __future__ import annotations

import contextlib
import json
import secrets
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
    tryout: dict | None = None  # config view; set in "byok" mode
    allowance = None  # tryout.Allowance
    byok_fn = None  # callable(question, provider, model, key) -> dict
    models_fn = None  # callable(provider, key) -> dict

    def _visitor(self) -> tuple[str, bool]:
        """The visitor id from the cookie, or a new one. A random id with no meaning of its own."""
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "va_visitor" and 16 <= len(v) <= 64 and v.isalnum():
                return v, False
        return secrets.token_hex(16), True

    def _send(self, code: int, body: bytes, ctype: str, cookie: str | None = None) -> None:
        self.send_response(code)
        if cookie:
            self.send_header("Set-Cookie", cookie)
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
                json.dumps(
                    {
                        "mode": self.mode,
                        "can_ask": self.ask_fn is not None or self.tryout is not None,
                    }
                ).encode(),
                "application/json",
            )
        elif self.path == "/api/tryout" and self.tryout is not None:
            vid, new = self._visitor()
            body = {**self.tryout, "remaining": self.allowance.remaining(vid)}
            cookie = f"va_visitor={vid}; Path=/; HttpOnly; SameSite=Strict" if new else None
            self._send(200, json.dumps(body).encode(), "application/json", cookie)
        elif self.path == "/api/results":
            body = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _key(self) -> str:
        """The visitor's own API key, from a header so it is never in a URL or the body we keep.
        It is held in this call's local variables only: not stored, not logged, not traced."""
        return (self.headers.get("X-Provider-Key") or "").strip()[:300]

    def _byok(self) -> None:
        cfg = self.tryout
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
            q = str(body.get("question", "")).strip()
            tier = str(body.get("tier", ""))
            provider = str(body.get("provider", ""))
            model = str(body.get("model", "") or "").strip()[:120]
        except (ValueError, TypeError, AttributeError):
            return self._send(400, b'{"error":"bad request"}', "application/json")
        key = self._key()
        if provider not in cfg["providers"]:
            return self._send(400, b'{"error":"choose a provider"}', "application/json")
        if not key and tier == "own":
            msg = b'{"error":"paste your own API key; it is used for this question only"}'
            return self._send(400, msg, "application/json")
        why = self.allowance.validate(tier, q)
        if why:
            return self._send(400, json.dumps({"error": why}).encode(), "application/json")
        vid, new = self._visitor()
        cookie = f"va_visitor={vid}; Path=/; HttpOnly; SameSite=Strict" if new else None
        if not self.allowance.reserve(vid, tier):
            msg = {
                "error": f"you have used your {tier} question",
                "remaining": self.allowance.remaining(vid),
            }
            return self._send(429, json.dumps(msg).encode(), "application/json", cookie)
        answered = False
        try:
            with self.lock:
                out = self.byok_fn(
                    q, provider, model or cfg["providers"][provider]["model"], key, tier
                )
            answered = out.get("reason") not in ("provider_error", "budget", "sensitivity")
        except Exception as exc:  # noqa: BLE001
            msg = {"error": f"could not answer: {type(exc).__name__}"}
            return self._send(500, json.dumps(msg).encode(), "application/json", cookie)
        finally:
            if not answered:
                self.allowance.refund(vid, tier)
        out = {**out, "remaining": self.allowance.remaining(vid)}
        self._send(
            200,
            json.dumps(out, ensure_ascii=False, default=str).encode(),
            "application/json",
            cookie,
        )

    def _models(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            provider = str(json.loads(self.rfile.read(min(n, 1024)) or b"{}").get("provider", ""))
        except (ValueError, TypeError, AttributeError):
            return self._send(400, b'{"error":"bad request"}', "application/json")
        key = self._key()
        if provider not in self.tryout["providers"] or not key:
            return self._send(400, b'{"error":"provider and key needed"}', "application/json")
        try:
            out = self.models_fn(provider, key)
        except Exception as exc:  # noqa: BLE001
            return self._send(
                500, json.dumps({"error": type(exc).__name__}).encode(), "application/json"
            )
        self._send(200, json.dumps(out).encode(), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if self.tryout is not None and self.path == "/api/ask":
            return self._byok()
        if self.tryout is not None and self.path == "/api/models":
            return self._models()
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


def serve(
    port: int = 8000,
    host: str = "127.0.0.1",
    ask_fn=None,
    mode: str = "static",
    *,
    byok_fn=None,
    models_fn=None,
    tryout_cfg: dict | None = None,
) -> None:
    from ..tryout import Allowance, public_view  # noqa: PLC0415

    _Handler.payload = load_payload()
    _Handler.ask_fn, _Handler.mode = (staticmethod(ask_fn) if ask_fn else None), mode
    if tryout_cfg is not None and byok_fn is not None:
        _Handler.tryout, _Handler.allowance = public_view(tryout_cfg), Allowance(tryout_cfg)
        _Handler.byok_fn = staticmethod(byok_fn)
        _Handler.models_fn = staticmethod(models_fn) if models_fn else None
        _Handler.mode = "byok"
    srv = ThreadingHTTPServer((host, port), _Handler)
    print(f"http://{host}:{port}  (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
