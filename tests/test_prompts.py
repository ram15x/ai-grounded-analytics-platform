"""The compose and repair prompts and the field-level problems fed back to the model. No model and
no database. These check what the text says and that its worked example is schema-valid; they say
nothing about how any real model responds to it."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

import yaml
from pydantic import ValidationError

from verified_analyst.claims import Op, TemplateClaim
from verified_analyst.llm.pipeline import schema_problems
from verified_analyst.llm.prompts import (
    COMPOSE_SYSTEM,
    PROMPT_TEMPLATES,
    REPAIR_SYSTEM,
    compose_request,
    prompt_hashes,
)
from verified_analyst.llm.schemas import ComposeOutput
from verified_analyst.semantic import load_semantic_layer

ROOT = Path(__file__).resolve().parents[1]


def _example() -> dict:
    text = COMPOSE_SYSTEM.split("Answer: ", 1)[1]
    return json.loads(text)


def test_the_worked_example_is_schema_valid_and_its_placeholders_match_its_claim_ids():
    out = ComposeOutput.model_validate(_example())
    ids = {c.id for c in out.claims}
    assert ids == set(re.findall(r"\{(\w+)\}", out.template)) == {"c1", "c2", "c3"}
    assert not re.search(r"\d", re.sub(r"\{\w+\}", "", out.template))  # types no number


def test_the_worked_example_uses_no_metric_dimension_or_figure_from_the_dev_questions():
    layer = load_semantic_layer()
    dev = (ROOT / "eval" / "questions" / "dev.yaml").read_text()
    example_part = COMPOSE_SYSTEM.split("Example, on a made-up dataset", 1)[1]
    for word in ("tickets_closed", "region"):
        assert word in example_part
        assert word not in layer.metrics and word not in layer.dimensions
        assert word not in dev
    assert not set(layer.metrics) & set(re.findall(r"[a-z]+(?:_[a-z]+)+", example_part))
    questions = [q["question"] for q in yaml.safe_load(dev)["questions"]]
    assert not any(q in example_part for q in questions)


def test_the_prompt_names_every_allowed_unit_op_scale_and_provenance_kind():
    units = get_args(TemplateClaim.model_fields["unit"].annotation)
    scales = get_args(TemplateClaim.model_fields["scale"].annotation)
    assert units and scales and get_args(Op)
    for word in (*units, *scales, *get_args(Op)):
        assert f'"{word}"' in COMPOSE_SYSTEM or word in COMPOSE_SYSTEM.replace('"', "")
    for kind in ("cell", "row_count", "derived"):
        assert f'"kind": "{kind}"' in COMPOSE_SYSTEM
    assert '"row_0"' in COMPOSE_SYSTEM and "row_key" in COMPOSE_SYSTEM  # the observed mistake
    assert "GBP is not a unit" in COMPOSE_SYSTEM
    assert "comma or a full stop" in COMPOSE_SYSTEM  # "X ... and Y ..." fails the row binder


def test_the_prompt_describes_placeholders_without_the_literal_id_trap():
    assert "{c1}" in COMPOSE_SYSTEM
    assert "{id}" not in COMPOSE_SYSTEM  # v0 invited the model to write the word id in braces


def test_repair_prompt_extends_compose_and_points_at_field_paths():
    assert REPAIR_SYSTEM.startswith(COMPOSE_SYSTEM)
    assert "field path" in REPAIR_SYSTEM and "claims.0.unit" in REPAIR_SYSTEM


def test_prompt_ids_and_hashes_are_versioned_and_distinct():
    assert set(PROMPT_TEMPLATES) == {"plan.v0", "plan.v1", "compose.v1", "repair.v1"}
    assert len(set(prompt_hashes().values())) == 4


def test_compose_and_repair_requests_carry_their_own_prompt_id():
    layer_free = dict(
        question="q", restatement="r", executed={}, datasets=("analytics.sales_lines",)
    )
    c = compose_request(**layer_free)
    r = compose_request(
        **layer_free, previous="{}", problems=["schema_failure at claims.0.unit: x"]
    )
    assert (c.prompt_id, r.prompt_id) == ("compose.v1", "repair.v1")
    assert "claims.0.unit" in r.messages[1][1] and "Problems:" in r.messages[1][1]


# ---- field-level problems, modelled on the kinds of failure the first real run showed


def _problems_for(obj) -> list[str]:
    try:
        ComposeOutput.model_validate(obj)
    except ValidationError as exc:
        return schema_problems(exc)
    raise AssertionError("expected a schema failure")


def test_free_text_unit_and_untagged_provenance_are_named_by_path():
    obj = {
        "template": "Gross revenue was {c1}.",
        "claims": [{"id": "c1", "label": "gross", "unit": "GBP",
                    "provenance": {"query_id": "q1", "column": "gross_revenue"}}],
    }  # fmt: skip
    probs = _problems_for(obj)
    assert any(p.startswith("schema_failure at claims.0.unit:") for p in probs)
    assert any(p.startswith("schema_failure at claims.0.provenance:") for p in probs)
    assert "GBP" not in " ".join(probs)  # the offending input is never echoed back


def test_a_literal_id_and_an_extra_field_are_named_by_path():
    obj = {
        "template": "x {id}",
        "claims": [{"id": "{id}", "label": "g", "unit": "currency", "value": "5696.22",
                    "provenance": {"kind": "row_count", "query_id": "q1"}}],
    }  # fmt: skip
    probs = _problems_for(obj)
    assert any("claims.0.id" in p for p in probs) and any("claims.0.value" in p for p in probs)
    assert not any("5696" in p for p in probs)


def test_problems_are_capped_and_each_message_is_cut():
    prov = {"kind": "row_count", "query_id": "q1"}
    claims = [
        {"id": f"c{i}", "label": "l", "unit": "bad", "extra": 1, "provenance": prov}
        for i in range(20)
    ]
    probs = _problems_for({"template": "t", "claims": claims})
    assert len(probs) == 9 and probs[-1].startswith("schema_failure: and ")
    assert all(len(p) < 200 for p in probs)


def test_non_pydantic_errors_give_one_short_line():
    assert schema_problems(ValueError("not JSON\nsecond line")) == ["schema_failure: not JSON"]


# ---- plan.v1 ----


def test_plan_v0_is_frozen_and_kept_for_old_cassettes():
    from verified_analyst.llm.cassette import sha256_text

    assert sha256_text(PROMPT_TEMPLATES["plan.v0"]) == (
        "f627489111a83c410774d1987f11c94fe1bac6575e714e80da070544d1707f28"
    )


def test_the_pipeline_sends_plan_v1_and_the_report_names_only_active_prompts():
    from verified_analyst.llm.prompts import active_prompt_hashes, plan_request
    from verified_analyst.semantic import load_semantic_layer

    r = plan_request(load_semantic_layer(), "q")
    assert r.prompt_id == "plan.v1" and r.prompt_hash == prompt_hashes()["plan.v1"]
    assert r.messages[0][1] == PROMPT_TEMPLATES["plan.v1"]
    assert PROMPT_TEMPLATES["plan.v1"].startswith(PROMPT_TEMPLATES["plan.v0"])
    assert set(active_prompt_hashes()) == {"plan.v1", "compose.v1", "repair.v1"}


def test_the_v1_catalogue_is_the_semantic_layers_own_words_nothing_added():
    import json

    from verified_analyst.llm.prompts import catalogue_text
    from verified_analyst.semantic import load_semantic_layer

    layer = load_semantic_layer()
    cat = json.loads(catalogue_text(layer))
    assert set(cat) == {"metrics", "dimensions"} and set(cat["metrics"]) == set(layer.metrics)
    for name, m in layer.metrics.items():
        assert cat["metrics"][name] == {
            "unit": m.unit,
            "description": m.description,
            "synonyms": m.synonyms,
            "dimensions": m.allowed_dimensions,
        }
    for name, d in layer.dimensions.items():
        assert cat["dimensions"][name] == {"description": d.description, "synonyms": d.synonyms}
    text = catalogue_text(layer)
    assert "COALESCE" not in text and "l.line_type" not in text  # no SQL, no column names


def test_the_v1_catalogue_carries_what_the_merchandise_units_question_needs():
    import json

    from verified_analyst.llm.prompts import catalogue_text
    from verified_analyst.semantic import load_semantic_layer

    cat = json.loads(catalogue_text(load_semantic_layer()))
    units = cat["metrics"]["units_sold"]
    assert "merchandise" in units["description"] and "units" in units["synonyms"]


def test_v0_catalogue_is_still_available_unchanged_in_shape():
    import json

    from verified_analyst.llm.prompts import catalogue_text_v0
    from verified_analyst.semantic import load_semantic_layer

    cat = json.loads(catalogue_text_v0(load_semantic_layer()))
    assert set(cat) == {"metrics", "dimensions"}
    assert all(set(m) == {"unit", "dimensions"} for m in cat["metrics"].values())


def test_prompt_building_reads_no_question_file():
    """The catalogue is built from the semantic layer only; the prompt module never opens a file
    of questions (dev or held-out)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "src/verified_analyst/llm/prompts.py").read_text()
    for needle in ("open(", "read_text", "yaml", "eval/", "questions", "held"):
        assert needle not in src, needle
