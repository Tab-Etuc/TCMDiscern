"""CPU regression for content-addressed Case-RAG checkpoints and inputs."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tcmdiscern.case_rag_identity import (
    expected_run_identity,
    prompt_fingerprints,
    validate_saved_identity,
    validate_score_only_inputs,
)


def _checkpoint(folder):
    for name in (
        "manifest.json", "probe_logits.json", "head_config.json",
        "adapter/adapter_config.json",
    ):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version":"pinned"}\n', encoding="utf-8")
    for name in ("head.safetensors", "adapter/adapter_model.safetensors"):
        path = folder / name
        path.write_bytes(b"weight-v1")


def _rows():
    return {
        "baseline": [
            {"id": "dev-000001", "state": {"description": "胃痛"}, "options": ["a", "b"]},
            {"id": "dev-000002", "state": {"description": "腹脹"}, "options": ["a", "b"]},
        ],
        "paired2": [
            {"id": "dev-000001", "state": {"description": "胃痛", "retrieved_cases": [
                {"candidate_syndrome": "a", "description": "患者無口渴"},
                {"candidate_syndrome": "b", "description": "患者有口渴"},
            ]}, "options": ["a", "b"]},
            {"id": "dev-000002", "state": {"description": "腹脹"}, "options": ["a", "b"]},
        ],
    }


def test_fingerprint_survives_json_key_order_but_catches_evidence_change():
    cases = _rows()
    baseline = prompt_fingerprints(cases)
    shuffled = {arm: [dict(reversed(list(row.items()))) for row in rows]
                for arm, rows in cases.items()}
    assert prompt_fingerprints(shuffled) == baseline
    modified = deepcopy(cases)
    modified["paired2"][0]["state"]["retrieved_cases"][0]["description"] = "患者有口渴"
    new_digest = prompt_fingerprints(modified)
    assert new_digest["baseline"] == baseline["baseline"]
    assert new_digest["paired2"] != baseline["paired2"]
    modified = deepcopy(cases)
    modified["paired2"].reverse()
    assert prompt_fingerprints(modified)["paired2"] != baseline["paired2"]


def test_same_path_checkpoint_overwrite_and_prompt_edits_reject_resume(tmp_path):
    directory = tmp_path / "s2"
    _checkpoint(directory)
    data = _rows()
    baseline = expected_run_identity(data, tmp_path / "model", directory, source="frozen")
    manifest = {**baseline, "checkpoint_method": "s2", "checkpoint_run_id": "run"}
    validate_saved_identity(manifest, baseline)
    validate_saved_identity(manifest, manifest, strict=True)
    validate_score_only_inputs(manifest, data, source="frozen")
    edits = deepcopy(data)
    edits["baseline"][1]["state"]["description"] = "更改病歷"
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_saved_identity(
            manifest,
            expected_run_identity(edits, tmp_path / "model", directory, source="frozen"),
        )
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_score_only_inputs(manifest, edits, source="frozen")
    (directory / "manifest.json").write_text('{"version":"replaced"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_provenance"):
        validate_saved_identity(
            manifest,
            expected_run_identity(data, tmp_path / "model", directory, source="frozen"),
        )
    (directory / "manifest.json").write_text('{"version":"pinned"}\n', encoding="utf-8")
    (directory / "adapter/adapter_model.safetensors").write_bytes(b"more-new-weights")
    with pytest.raises(ValueError, match="checkpoint_provenance"):
        validate_saved_identity(
            manifest,
            expected_run_identity(data, tmp_path / "model", directory, source="frozen"),
        )


def test_legacy_run_without_prompt_fingerprint_is_not_silently_adopted(tmp_path):
    cp = tmp_path / "checkpoint"
    _checkpoint(cp)
    expected = expected_run_identity(_rows(), tmp_path / "model", cp, source="frozen")
    old = {key: value for key, value in expected.items() if key != "prompt_sha256"}
    with pytest.raises(ValueError, match="prompt_sha256"):
        validate_saved_identity(old, expected)
    with pytest.raises(ValueError, match="schema"):
        validate_saved_identity({**expected, "unexpected": True}, expected, strict=True)


def test_checkpoint_provenance_requires_train_metadata(tmp_path):
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    with pytest.raises(FileNotFoundError, match="provenance"):
        expected_run_identity(_rows(), tmp_path / "model", cp, source="frozen")
