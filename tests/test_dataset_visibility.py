"""Uploaded data is internal unless its manifest says public; hosted models never see internal."""

from __future__ import annotations

import json

from grounded_analytics.llm.settings import apply_dataset_visibility, load_router_settings

REL = "ds_saas_subs.data"


def _layer(tmp_path, body):
    (tmp_path / "layer.yaml").write_text("x", "utf-8")
    if body is not None:
        (tmp_path / "visibility.json").write_text(body, "utf-8")
    return tmp_path / "layer.yaml"


def test_default_is_internal_even_without_a_visibility_file(tmp_path):
    cfg = load_router_settings()
    apply_dataset_visibility(cfg, _layer(tmp_path, None))
    assert cfg.sensitivity_of((REL,)) == "internal"


def test_malformed_file_fails_closed(tmp_path):
    cfg = load_router_settings()
    apply_dataset_visibility(cfg, _layer(tmp_path, "{not json"))
    apply_dataset_visibility(
        cfg, _layer(tmp_path, json.dumps({"relation": REL, "visibility": "x"}))
    )
    assert cfg.sensitivity_of((REL,)) == "internal"


def test_public_manifest_registers_the_relation(tmp_path):
    cfg = load_router_settings()
    body = json.dumps({"relation": REL, "visibility": "public"})
    apply_dataset_visibility(cfg, _layer(tmp_path, body))
    assert cfg.sensitivity_of((REL,)) == "public"
    # one internal dataset in the mix makes the whole payload internal
    assert cfg.sensitivity_of((REL, "ds_other.data")) == "internal"


def test_env_path_is_honoured(tmp_path, monkeypatch):
    p = _layer(tmp_path, json.dumps({"relation": REL, "visibility": "public"}))
    monkeypatch.setenv("GA_SEMANTIC_PATH", str(p))
    assert load_router_settings().sensitivity_of((REL,)) == "public"
