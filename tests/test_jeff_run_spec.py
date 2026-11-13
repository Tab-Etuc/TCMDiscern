"""Fail-closed Jeff H200 training budget and checkpoint-resume contract tests.

These tests require no GPU, PEFT, or patient data.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

import tcmdiscern.jeff_train_loop as training
from tcmdiscern.jeff_schedule import epoch_order, total_optimizer_steps


def args(**overrides):
    params = {
        "model_dir": "/not-read-by-configuration-check",
        "method": "s3",
        "run_id": "synthetic-123",
        "steps": 10,
        "epochs": 1,
        "accumulation": 16,
        "seed": 20261112,
        "rank": 32,
        "alpha": 64,
        "lora_lr": 2e-5,
        "head_lr": 1e-4,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "warmup_steps": 0,
        "checkpoint_every": 4,
        "resume": None,
    }
    params.update(overrides)
    return Namespace(**params)


def test_full_training_run_budget_and_params_validate_before_any_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    spec = args()
    target = training.validate_run_args(spec)
    assert target == tmp_path / "data/runs/synthetic-123"
    assert not target.exists()
    fixed = training.fixed_run_settings(spec)
    assert fixed["official_train_case_count"] == 43180
    assert fixed["model_revision"]
    assert fixed["rank"] == 32 and fixed["alpha"] == 64
    assert fixed["source"] == "train_oof_multipos_k16_epoch03.jsonl"
    assert total_optimizer_steps(43180, 16, 1) == 2699
    assert len(epoch_order(43180, epoch=0, seed=20261112)) == 43180


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": "../../danger"},
        {"run_id": ""},
        {"method": "unknown"},
        {"steps": 0},
        {"epochs": -1},
        {"accumulation": 0},
        {"rank": 0},
        {"alpha": 0},
        {"warmup_steps": -1},
        {"checkpoint_every": 0},
        {"lora_lr": 0.0},
        {"head_lr": -0.1},
        {"weight_decay": -0.01},
        {"clip_norm": 0},
        {"steps": 43181, "epochs": 1, "accumulation": 1},
        {"steps": 10, "warmup_steps": 10},
    ],
)
def test_invalid_training_budget_rejected_before_model_load(tmp_path, monkeypatch, overrides):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        training.validate_run_args(args(**overrides))


def test_resume_must_use_same_run_id_and_existing_completed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    run = tmp_path / "data/runs/synthetic-123"
    path = run / "resume-0000004"
    path.mkdir(parents=True)
    result = training.validate_run_args(args(resume=path))
    assert result == run
    with pytest.raises(ValueError, match="same run_id"):
        other = tmp_path / "data/runs/other/resume-0000004"
        other.mkdir(parents=True)
        training.validate_run_args(args(resume=other))
    with pytest.raises(FileExistsError):
        training.validate_run_args(args(resume=None))
    with pytest.raises(ValueError, match="Invalid resume"):
        out = run / "not-checkpoint"
        out.mkdir()
        training.validate_run_args(args(resume=out))


def test_one_step_cli_is_safe_default_and_s1_uses_paired_candidate_source():
    spec = args(steps=1, epochs=1, accumulation=1, method="s1")
    settings = training.fixed_run_settings(spec)
    assert settings["source"] == "train_oof_multipos_k16_epoch03.jsonl"
    assert settings["target_updates"] == 1
    assert settings["lora_lr"] == 2e-5
    assert settings["head_lr"] == 1e-4
