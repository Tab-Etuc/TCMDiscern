"""Clean-H200 tokenizer fallback and deterministic checkpoint discovery."""
from __future__ import annotations

import json

import pytest

from scripts.run_case_rag_h200 import check, checkpoints, choose_checkpoint
from tcmdiscern.case_rag_tokenizer import locate_jeff_tokenizer


def _complete_checkpoint(folder, *, method="s2", update=1350, epoch=2):
    folder.mkdir(parents=True)
    for name in (
        "manifest.json", "head_config.json", "head.safetensors",
        "probe_logits.json", "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    ):
        file = folder / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}", encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({
        "loss_method": method,
        "run_id": f"{method}-synthetic",
        "global_update": update,
        "cursor_epoch": epoch,
    }), encoding="utf-8")


def test_model_only_h200_tokenizer_without_local_duplicate(tmp_path, monkeypatch):
    monkeypatch.delenv("H200_MODEL_DIR", raising=False)
    model = tmp_path / "data/models/Jeff-1.1-Large"
    model.mkdir(parents=True)
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    assert locate_jeff_tokenizer(tmp_path) == model.resolve()
    external = tmp_path / "custom/Jeff"
    external.mkdir(parents=True)
    (external / "tokenizer.json").write_text("{}", encoding="utf-8")
    (external / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    monkeypatch.setenv("H200_MODEL_DIR", str(external))
    assert locate_jeff_tokenizer(tmp_path) == external.resolve()


def test_missing_tokenizer_explains_which_existing_gpu_command_to_run(tmp_path, monkeypatch):
    monkeypatch.delenv("H200_MODEL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="run_h200.sh pilot"):
        locate_jeff_tokenizer(tmp_path)


def test_checkpoint_autodetection_requires_complete_compatible_files(tmp_path):
    root = tmp_path
    incomplete = root / "data/runs/partial"
    incomplete.mkdir(parents=True)
    (incomplete / "manifest.json").write_text('{"loss_method":"s2"}', encoding="utf-8")
    older = root / "data/runs/s2/resume-0000100"
    newer = root / "data/runs/s2/resume-0001350"
    unrelated = root / "data/runs/s3/resume-0015000"
    _complete_checkpoint(older, update=100, epoch=0)
    _complete_checkpoint(newer, update=1350, epoch=2)
    _complete_checkpoint(unrelated, method="s3", update=1500, epoch=3)
    found = checkpoints(root, "s2")
    assert len(found) == 2
    assert found[0]["global_update"] == 1350
    assert choose_checkpoint(root, "s2", None) == newer.resolve()
    with pytest.raises(ValueError, match="method"):
        choose_checkpoint(root, "s2", unrelated)
    assert choose_checkpoint(root, "s3", None) == unrelated.resolve()
    _complete_checkpoint(root / "data/runs/s2/another", update=1350, epoch=2)
    with pytest.raises(ValueError, match="Multiple equally"):
        choose_checkpoint(root, "s2", None)


def test_readiness_check_is_readonly_and_explains_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("H200_MODEL_DIR", raising=False)
    result = check(tmp_path, "s2", tmp_path / "missing", include_counterfactual=False)
    assert result["status"] == "CASE_RAG_READINESS_CHECK"
    assert not any(result["official_artifacts"].values())
    assert result["tokenizer_directory"] is None
    assert result["available_checkpoints"] == []
    assert not (tmp_path / "data").exists()
