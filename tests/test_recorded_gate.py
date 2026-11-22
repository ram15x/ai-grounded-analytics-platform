"""Offline gate over demo/results.json (no network, no database, no model).

Each verified answer must trace to its own stored evidence: a cell-bound figure to a cell of the
stored result rows, a quoted figure to the stored passage text. A verified answer with a figure
that has no such trace fails the build. This guards the stored records, not the model's accuracy.
"""

import re
from decimal import Decimal

import pytest

from grounded_analytics.web.app import load_payload

GOOD = {"VERIFIED", "VERIFIED_DERIVED", "VERIFIED_QUOTE"}
ITEMS = load_payload()["items"]
VERIFIED = [x for x in ITEMS if x["outcome"] == "verified"]


def verdicts(x):
    return x["verification"]["verification"]["verdicts"]


def digits(s):
    return re.sub(r"[^\d.]", "", s)


@pytest.mark.parametrize("x", VERIFIED, ids=[x["id"] for x in VERIFIED])
def test_every_figure_in_a_verified_answer_has_a_good_verdict_and_a_trace(x):
    vs = verdicts(x)
    assert vs, "verified answer states no verified figure"
    ev = x["evidence"]
    cells = [Decimal(c) for r in ev.get("rows", []) for c in r if re.fullmatch(r"-?\d+(\.\d+)?", c)]
    passages = " ".join(p["text"] for p in (ev.get("passages") or {}).values())
    for v in vs:
        assert v["verdict"] in GOOD, v
        assert x["rendered"][v["span"][0] : v["span"][1]] == v["text"]
        if v["verdict"] == "VERIFIED":
            shown = Decimal(digits(v["text"]))
            assert any(
                abs(shown - c) <= Decimal("0.005") or abs(shown - c) / (abs(c) or 1) < 1e-4
                for c in cells
            ), f"{v['text']} is in no stored result cell"
        elif v["verdict"] == "VERIFIED_QUOTE":
            assert re.search(rf"(?<![\d.,]){re.escape(digits(v['text']))}(?![\d])", passages), v


def test_nothing_shown_as_verified_contains_a_figure_outside_a_verdict():
    for x in VERIFIED:
        text = x["rendered"]
        for v in sorted(verdicts(x), key=lambda v: -v["span"][0]):
            text = text[: v["span"][0]] + text[v["span"][1] :]
        # figures left over after removing verified spans (words like "12" in prose would show here)
        assert not re.search(r"\d", text), (x["id"], text)
