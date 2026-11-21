from verified_analyst.web.app import load_payload, static_page


def test_payload_shape_and_static_page():
    p = load_payload()
    assert len(p["items"]) == 17
    for x in p["items"]:
        for v in (x.get("verification") or {}).get("verification", {}).get("verdicts", []):
            s, e = v["span"]
            assert x["rendered"][s:e] == v["text"]  # highlight spans line up with the text
    html = static_page(p)
    assert "/*DATA*/" not in html and "Verified Analyst" in html


def test_no_recording_is_not_shown_as_abstain():
    p = load_payload()
    nr = [x for x in p["items"] if x["outcome"] == "not_recorded"]
    assert nr and all(x["reason"] == "no_recording" for x in nr)
