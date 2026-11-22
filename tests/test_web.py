import urllib.request

from grounded_analytics.web import app
from grounded_analytics.web.app import load_payload, static_page


def test_payload_shape_and_static_page():
    p = load_payload()
    assert len(p["items"]) == 17
    for x in p["items"]:
        for v in (x.get("verification") or {}).get("verification", {}).get("verdicts", []):
            s, e = v["span"]
            assert x["rendered"][s:e] == v["text"]  # highlight spans line up with the text
    html = static_page(p)
    assert "/*DATA*/" not in html and "AI Grounded Analytics Platform" in html


def test_no_recording_is_not_shown_as_abstain():
    p = load_payload()
    nr = [x for x in p["items"] if x["outcome"] == "not_recorded"]
    assert nr and all(x["reason"] == "no_recording" for x in nr)


def _server(ask_fn=None):
    import threading
    from http.server import ThreadingHTTPServer

    from grounded_analytics.web import app

    handler = app.configure(ask_fn, "replay" if ask_fn else "static")  # a clean handler each time
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(srv, body):
    import json
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"http://127.0.0.1:{srv.server_port}/api/ask", data=json.dumps(body).encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_ask_endpoint_answers_validates_and_survives_a_failing_question():
    def ask(q):
        if "boom" in q:
            raise RuntimeError("secret detail")
        return {"id": "X", "question": q, "outcome": "abstain", "reason": "ambiguous"}

    srv = _server(ask)
    try:
        assert _post(srv, {"question": "hi"})[0] == 200
        assert _post(srv, {"question": ""})[0] == 400
        assert _post(srv, {"question": "x" * 301})[0] == 400
        code, body = _post(srv, {"question": "boom"})
        assert code == 500 and "secret detail" not in str(body)  # no internals leaked
        assert _post(srv, {"question": "still alive"})[0] == 200
    finally:
        srv.shutdown()


def test_static_server_refuses_new_questions():
    srv = _server(None)
    try:
        assert _post(srv, {"question": "hi"})[0] == 404
    finally:
        srv.shutdown()


def test_fonts_are_served_locally_and_the_page_loads_nothing_from_another_site():
    page = app.PAGE.read_text("utf-8")
    assert (
        "googleapis" not in page
        and "gstatic" not in page
        and "https://" not in page.split("</style>")[0]
    )
    s = _server()
    try:
        for path in app.FONTS:
            with urllib.request.urlopen(f"http://127.0.0.1:{s.server_port}{path}") as r:
                assert r.status == 200 and r.headers["Content-Type"] == "font/woff2"
                assert r.read(4) == b"wOF2"
    finally:
        s.shutdown()


def test_page_has_no_canned_sample_questions_for_uploaded_data():
    page = app.PAGE.read_text("utf-8")
    assert 'id="sugg"' not in page and "examples[0]" not in page


def test_build_static_writes_a_page_that_works_under_a_subpath(tmp_path):
    import argparse

    from grounded_analytics import cli

    out = tmp_path / "site"
    rc = cli.cmd_serve(argparse.Namespace(build_static=str(out)))
    page = (out / "index.html").read_text("utf-8")
    assert rc == 0 and "url(fonts/" in page and "url(/fonts/" not in page
    assert (out / "fonts" / "public-sans.woff2").read_bytes()[:4] == b"wOF2"
    assert (out / ".nojekyll").exists()
