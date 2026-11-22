"""The "Your data" endpoints over real HTTP, with a real database and a scripted answerer."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from grounded_analytics.datasets.workspace import Workspace
from grounded_analytics.web import app
from test_dataset_load import scratch  # noqa: F401, I001

pytestmark = pytest.mark.db

EX = Path(__file__).resolve().parent.parent / "examples" / "datasets"
CSV = (EX / "saas_subscriptions.csv").read_bytes()
MAN = yaml.safe_load((EX / "saas_subscriptions.manifest.yaml").read_text("utf-8"))


@pytest.fixture
def srv(scratch, tmp_path):  # noqa: F811
    admin, _ro, role = scratch
    asked = []
    saved = {k: getattr(app._Handler, k) for k in ("workspace", "ws_ask_fn", "tryout", "models_fn")}
    app._Handler.workspace = Workspace(tmp_path, admin, role)
    app._Handler.ws_ask_fn = staticmethod(lambda *a: asked.append(a) or {"outcome": "verified"})
    app._Handler.tryout = None
    s = ThreadingHTTPServer(("127.0.0.1", 0), app._Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    s.asked = asked
    yield s
    s.shutdown()
    for k, v in saved.items():
        setattr(app._Handler, k, v)


def call(s, path, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(
        f"http://127.0.0.1:{s.server_port}{path}", data, headers or {},
        method="POST" if data is not None else "GET",
    )  # fmt: skip
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_upload_check_load_list_and_ask(srv):
    code, st = call(srv, "/api/ws/upload", raw=CSV, headers={"X-Filename": "saas.csv"})
    assert code == 200 and st["profile"]["rows"] == 1200
    code, chk = call(srv, "/api/ws/check", {"token": st["token"], "manifest": MAN})
    assert code == 200 and chk["ok"]
    code, res = call(srv, "/api/ws/load", {"token": st["token"], "manifest": MAN})
    assert code == 200 and res["rows"] == 1200
    code, lst = call(srv, "/api/ws/datasets")
    assert [d["name"] for d in lst] == ["saas_subs"]
    body = {"dataset": "saas_subs", "question": "What is MRR?", "provider": "groq"}
    assert call(srv, "/api/ws/ask", body, headers={"X-Provider-Key": "k"})[0] == 200
    assert srv.asked[0][-1] == "k"  # the key reaches the answerer and nothing else holds it


def test_bad_requests_get_a_plain_error(srv):
    assert call(srv, "/api/ws/upload", raw=b"x", headers={"X-Filename": "a.txt"})[0] == 400
    assert call(srv, "/api/ws/check", {"token": "nope", "manifest": {}})[0] == 400
    assert call(srv, "/api/ws/ask", {"dataset": "../x", "question": "q"})[0] == 400
    assert call(srv, "/api/ws/ask", {"dataset": "saas_subs", "question": ""})[0] == 400
    assert call(srv, "/api/ws/other", {})[0] == 404


def test_workspace_refuses_a_public_host():
    from grounded_analytics import cli

    assert cli.main(["serve", "--workspace", "--host", "0.0.0.0"]) == 2


def test_document_upload_and_listing_over_http(srv):
    from pdf_helpers import make_pdf

    _, st = call(srv, "/api/ws/upload", raw=CSV, headers={"X-Filename": "saas.csv"})
    call(srv, "/api/ws/load", {"token": st["token"], "manifest": MAN})
    h = {"X-Filename": "Terms.pdf", "X-Dataset": "saas_subs", "X-Sensitivity": "public"}
    code, out = call(
        srv, "/api/ws/doc", raw=make_pdf(["Seats may be added at any time."]), headers=h
    )
    assert code == 200 and out["sensitivity"] == "public" and out["chunks"] >= 1
    code, docs = call(srv, "/api/ws/docs/saas_subs")
    assert code == 200 and [d["sensitivity"] for d in docs] == ["public"]
    bad = {**h, "X-Filename": "scan.pdf"}
    code, err = call(srv, "/api/ws/doc", raw=make_pdf([""]), headers=bad)
    assert code == 400 and "OCR" in err["error"]
    assert call(srv, "/api/ws/doc", raw=b"x", headers={**h, "X-Dataset": "nope"})[0] == 400
    assert call(srv, "/api/ws/docs/nope")[0] == 404
