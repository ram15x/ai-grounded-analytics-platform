"""A second dataset of a different shape (subscriptions, no sales lines, one date column, a
personal column) goes through the SAME plan -> compile -> guard -> compose -> verify pipeline.
The model is scripted; this tests the plumbing and the verifier on new data, not a model."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_helpers import settings
from test_dataset_load import scratch  # noqa: F401
from verified_analyst.datasets import load as L
from verified_analyst.datasets import manifest as M
from verified_analyst.datasets.profile import profile_file
from verified_analyst.guard import allowed_for
from verified_analyst.llm.fake import FakeProvider
from verified_analyst.llm.pipeline import run_question
from verified_analyst.llm.router import Router

pytestmark = pytest.mark.db

EX = Path(__file__).resolve().parent.parent / "examples" / "datasets"


@pytest.fixture(scope="module")
def saas(scratch):  # noqa: F811
    admin, ro, role = scratch
    prof = profile_file(EX / "saas_subscriptions.csv")
    m = M.load_manifest(EX / "saas_subscriptions.manifest.yaml")
    assert not M.validate(m, prof)
    return admin, ro, role, prof, m


def _run(saas, script):
    admin, ro, role, prof, m = saas
    from verified_analyst.datasets.profile import read_table

    df = read_table(EX / "saas_subscriptions.csv")
    L.load_dataset(df, prof, m, admin, role)
    layer = M.build_layer(m, prof)
    assert allowed_for(layer)
    cfg = settings(datasets={"ds_saas_subs.data": "public"})
    cfg.tasks = {t: ["local"] for t in ("plan", "compose", "repair")}
    fake = FakeProvider(script, name="local")
    router = Router(cfg, {"local": fake})
    out = run_question(
        "What is our monthly recurring revenue in EMEA?",
        layer=layer, readonly_url=ro, router=router, run=router.new_run("e2e", None),
        known_values={},
    )  # fmt: skip
    return out


EMEA = {"dimension": "region", "op": "eq", "value": "EMEA"}
PLAN = {"metric_query": {"metrics": ["mrr"], "dimensions": ["region"], "filters": [EMEA]}}
CLAIM = {"id": "m", "label": "monthly recurring revenue", "unit": "currency",
         "provenance": {"kind": "cell", "query_id": "q1", "column": "mrr",
                        "row_key": {"region": "EMEA"}}}  # fmt: skip


def test_new_dataset_answer_is_verified_against_the_warehouse(saas):
    out = _run(saas, {"plan": [PLAN], "compose": [
        {"template": "EMEA active MRR is {m}.", "claims": [CLAIM]}]})  # fmt: skip
    assert out.outcome == "verified", (out.reason, out.rendered)
    assert "48,434" in out.rendered  # matches the independent pandas figure


def test_a_typed_wrong_number_is_not_accepted(saas):
    bad = {"template": "EMEA active MRR is 50,000 USD.", "claims": []}
    out = _run(saas, {"plan": [PLAN], "compose": [bad], "repair": [bad] * 5})
    assert out.outcome == "abstain" and "50,000" not in out.rendered
