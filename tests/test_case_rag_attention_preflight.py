"""Prevent another 31B weight load with an incompatible Case-RAG backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_case_rag_batch_jeff as batch
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION


def checkpoint(tmp_path: Path, backend: str = "axolotl_hybrid_fa2_sdpa") -> Path:
    folder = tmp_path / "data/runs/s2-seed20261112-e2/resume-0001350"
    folder.mkdir(parents=True)
    payload = {
        "loss_method": "s2",
        "official_train_case_count": 43180,
        "model_revision": JEFF_MODEL_REVISION,
        "source": "train_oof_multipos_k16_epoch03.jsonl",
        "attention_backend": backend,
    }
    (folder / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    for name in (
        "head_config.json",
        "head.safetensors",
        "probe_logits.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
    ):
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic")
    return folder


def test_attention_mismatch_fails_without_31b_load(tmp_path, monkeypatch):
    ckpt = checkpoint(tmp_path)
    monkeypatch.setenv("TCMDISCERN_GEMMA4_ATTENTION", "sdpa")
    loader_calls = []
    import scripts.train_jeff_lora as original

    monkeypatch.setattr(original, "load_original_checkpoint", lambda *_: loader_calls.append(1))
    with pytest.raises(ValueError, match="Checkpoint requires attention backend"):
        batch.load_checkpoint(tmp_path / "model", ckpt, "s2")
    assert loader_calls == []


def test_matched_original_loader_is_used_and_backend_kept(monkeypatch, tmp_path):
    ckpt = checkpoint(tmp_path)
    monkeypatch.setattr(
        batch,
        "preflight_checkpoint_contract",
        lambda *args, **kwargs: {"status": "CASE_RAG_CHECKPOINT_PREFLIGHT_PASS"},
    )
    import scripts.train_jeff_lora as original

    decider = SimpleNamespace(attention_backend="axolotl_hybrid_fa2_sdpa")
    loaded = []
    monkeypatch.setattr(
        original,
        "load_original_checkpoint",
        lambda path: loaded.append(path) or SimpleNamespace(decider=decider),
    )
    monkeypatch.setattr(
        batch,
        "restore_checkpoint",
        lambda received, path: (
            {
                "method": "s2",
                "source_run_id": "s2-seed20261112-e2",
            }
            if received is decider and path == ckpt
            else pytest.fail("Unexpected Jeff/checkpoint instance")
        ),
    )
    result_decider, info = batch.load_checkpoint(tmp_path / "model", ckpt, "s2")
    assert result_decider is decider
    assert info["method"] == "s2"
    assert loaded == [tmp_path / "model"]


def test_metadata_matched_backend_checks_happen_without_gpu(tmp_path, monkeypatch):
    ckpt = checkpoint(tmp_path)
    monkeypatch.setenv("TCMDISCERN_GEMMA4_ATTENTION", "hybrid")
    monkeypatch.setattr(
        batch,
        "inspect_checkpoint",
        lambda path: {
            "weights_present": True,
            "decision_head_present": True,
            "release_metadata_matches": True,
            "tokenizer_present": True,
        },
    )
    import tcmdiscern.jeff_axolotl_attention as axolotl
    import tcmdiscern.jeff_release_attention as release

    monkeypatch.setattr(
        release, "inspect_release_attention", lambda path: {"release_jeff_bidirectional": None}
    )
    kernel_calls = []
    monkeypatch.setattr(
        axolotl,
        "preflight_hybrid_kernel",
        lambda path: kernel_calls.append(path) or {"kernel": "preflight-ok"},
    )
    (tmp_path / "data/jeff").mkdir(parents=True)
    (tmp_path / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl").touch()
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    report = batch.preflight_checkpoint_contract(
        tmp_path / "model",
        ckpt,
        "s2",
        require_baseline_parity=True,
        check_kernel=True,
    )
    assert report["status"] == "CASE_RAG_CHECKPOINT_PREFLIGHT_PASS"
    assert report["checkpoint_attention_backend"] == "axolotl_hybrid_fa2_sdpa"
    assert report["large_weights_loaded"] is False
    assert kernel_calls == [tmp_path / "model"]


def test_h200_check_executes_kernel_preflight_without_loading_model(tmp_path, monkeypatch):
    from scripts import run_case_rag_h200 as entry

    ckpt = checkpoint(tmp_path)
    invoked = []
    monkeypatch.setattr(
        entry,
        "check",
        lambda root, method, model_dir, *, include_counterfactual: {
            "status": "CASE_RAG_READINESS_CHECK"
        },
    )
    monkeypatch.setattr(entry, "choose_checkpoint", lambda root, method, explicit: ckpt)
    monkeypatch.setattr(
        batch,
        "preflight_checkpoint_contract",
        lambda model_dir, checkpoint, method, **kwargs: (
            invoked.append(kwargs)
            or {
                "status": "CASE_RAG_CHECKPOINT_PREFLIGHT_PASS",
                "large_weights_loaded": False,
            }
        ),
    )
    report = entry.run(
        root=tmp_path,
        method="s2",
        checkpoint=ckpt,
        model_dir=tmp_path / "model",
        mode="check",
        limit=None,
        include_counterfactual=False,
        run_id=None,
        require_baseline_parity=True,
    )
    assert report["checkpoint_preflight"]["status"] == "CASE_RAG_CHECKPOINT_PREFLIGHT_PASS"
    assert invoked == [{"require_baseline_parity": True, "check_kernel": True}]


def test_parity_reference_missing_fails_before_weights(tmp_path, monkeypatch):
    ckpt = checkpoint(tmp_path)
    monkeypatch.setenv("TCMDISCERN_GEMMA4_ATTENTION", "hybrid")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="full-Dev logits missing"):
        batch.preflight_checkpoint_contract(
            tmp_path / "model", ckpt, "s2", require_baseline_parity=True
        )
