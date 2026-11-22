"""Weighted fusion, the dev-only fusion study and the abstention profile (no database)."""

import pytest

from grounded_analytics.documents import abstention
from grounded_analytics.documents.fusion import fuse_hits, hybrid, rrf
from grounded_analytics.documents.fusion_study import (
    DEFAULT,
    Setting,
    fused,
    grid,
    paired_bootstrap,
    render_markdown,
    run_study,
)
from grounded_analytics.documents.metrics import Evidence, Question, Retrieved
from grounded_analytics.documents.store import Hit


def hit(doc, ordinal=0, score=1.0, text=None):
    return Hit(f"{doc}#{ordinal}.abc", doc, "t", ordinal, 0, 5, text or f"text of {doc}", score)


def test_unit_weights_equal_plain_rrf_and_weights_shift_the_order():
    lists = [["a", "b"], ["b", "a"]]
    assert rrf(lists, 60) == rrf(lists, 60, [1.0, 1.0])
    assert rrf(lists, 60, [3.0, 1.0])[0][0] == "a"  # list 0 prefers a
    assert rrf(lists, 60, [1.0, 3.0])[0][0] == "b"
    with pytest.raises(ValueError):
        rrf(lists, 60, [1.0])
    with pytest.raises(ValueError):
        rrf(lists, 60, [1.0, -1.0])


def test_hybrid_defaults_are_unchanged_and_accept_weights():
    kw = lambda q, k: [hit("a"), hit("b")][:k]  # noqa: E731
    vec = lambda q, k: [hit("b"), hit("a")][:k]  # noqa: E731
    assert [h.doc_key for h in hybrid(kw, vec)("q", 2)] == ["a", "b"]  # tie: earliest list first
    assert [h.doc_key for h in hybrid(kw, vec, weights=(1, 3))("q", 2)] == ["b", "a"]
    assert fuse_hits([[hit("a")], [hit("a")]], 1, 10)[0].score == pytest.approx(2 / 11)


def test_grid_is_fixed_and_contains_the_default():
    g = grid()
    assert len(g) == len(set(g)) == 27
    assert Setting(DEFAULT[1], DEFAULT[2], DEFAULT[3]) in g
    assert all(s.depth <= 50 for s in g)


def test_bootstrap_is_deterministic_and_detects_a_clear_change():
    before = [0.0] * 30 + [1.0] * 10
    after = [1.0] * 30 + [1.0] * 10
    a, b = paired_bootstrap(before, after), paired_bootstrap(before, after)
    assert a == b and a["mean_change"] == pytest.approx(0.75) and a["excludes_zero"]
    assert a["questions_better"] == 30 and a["questions_worse"] == 0
    same = paired_bootstrap(before, before)
    assert same["mean_change"] == 0 and not same["excludes_zero"]
    noisy = paired_bootstrap([1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1])
    assert not noisy["excludes_zero"]
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


QUESTIONS = [
    Question("Q1", "t", "alpha question", (Evidence("a", "text of a"),)),
    Question("Q2", "t", "beta question", (Evidence("b", "text of b"),)),
    Question("Q3", "t", "nothing here", ()),
]
CORPUS = [Retrieved("a", "text of a"), Retrieved("b", "text of b"), Retrieved("c", "text of c")]


def kw(q, k):
    return {
        "alpha question": [hit("a", score=2.0), hit("c")],
        "beta question": [hit("c"), hit("b")],
    }.get(q, [])[:k]


def vec(q, k):
    return {"alpha question": [hit("c"), hit("a")], "beta question": [hit("b"), hit("c")]}.get(
        q, [hit("c", score=0.1)]
    )[:k]


def test_study_reports_every_grid_row_in_grid_order_and_selects_nothing():
    st = run_study(kw, vec, QUESTIONS, CORPUS)
    assert [r["label"] for r in st["grid"]] == [s.label for s in grid()]
    assert st["questions_scored"] == 2 and st["scope"] == "dev questions only"
    assert "Nothing is selected" in st["note"] and "selected" not in {k for k in st}
    default_row = next(r for r in st["grid"] if r["label"] == st["default"]["label"])
    assert default_row["vs_default"]["mrr"]["mean_change"] == 0
    assert st["vector_only"]["overall"]["mrr"] == pytest.approx(0.75)
    md = render_markdown(st)
    assert "vector only" in md and "keyword only" in md and "**default**" in md


def test_study_searches_each_question_once_per_method():
    calls = []

    def counted(name, fn):
        def f(q, k):
            calls.append((name, q, k))
            return fn(q, k)

        return f

    run_study(counted("kw", kw), counted("vec", vec), QUESTIONS, CORPUS)
    assert len(calls) == 2 * len(QUESTIONS) and {c[2] for c in calls} == {50}


def test_fused_applies_the_keyword_weight():
    s_hi, s_lo = Setting(60, 50, 1.0), Setting(60, 50, 0.1)
    k1 = lambda q, k: [hit("a"), hit("b")]  # noqa: E731
    v1 = lambda q, k: [hit("b"), hit("a")]  # noqa: E731
    assert fused(k1, v1, s_hi)("q", 1)[0].doc_key == "a"
    assert fused(k1, v1, s_lo)("q", 1)[0].doc_key == "b"


def test_abstention_auc_and_profile():
    assert abstention.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert abstention.auc([0.5], [0.5]) == 0.5
    assert abstention.auc([], [0.5]) is None

    def search(q, k):
        return (
            [hit("a", score=0.9)] if "alpha" in q else [hit("c", score=0.2)] if "beta" in q else []
        )

    p = abstention.profile(search, QUESTIONS)
    assert p["answerable"]["n"] == 2 and p["unanswerable"]["n"] == 1
    assert p["returned_nothing"] == {"answerable": 0, "unanswerable": 1}
    assert p["auc"] == 1.0 and p["unanswerable"]["top_score"] is None
    assert "No threshold is set" in abstention.NOTE
