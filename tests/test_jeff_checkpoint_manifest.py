"""Checkpoint metadata must match the exact original Train run configuration.

These tests require no Jeff weights, torch, CUDA, PEFT or patient records.
They specifically prevent a resume checkpoint from being rejected due to
mismatched model revision field names at H200 Dev evaluation time.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from tcmdiscern.jeff_checkpoint import validate_checkpoint_manifest
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION
from tcmdiscern.jeff_train_loop import fixed_run_settings


def sample_args(method: str) -> Namespace:
    return Namespace(
        run_id=f"{method}-reloaded",
        method=method,
        seed=20261112,
        rank=32,
        alpha=64,
        epochs=1,
        steps=2,
        accumulation=2,
        lora_lr=2e-5,
        head_lr=1e-4,
        weight_decay=0.01,
        clip_norm=1.0,
        warmup_steps=0,
    )


@pytest.mark.parametrize("method", ["s1", "s2", "s3", "s4"])
def test_resume_manifest_directly_accepted_by_fresh_dev_loader(method: str) -> None:
    manifest = fixed_run_settings(sample_args(method))
    manifest.update(global_update=1, cursor_epoch=0, cursor_offset=2)
    assert manifest["official_train_case_count"] == 43180
    assert manifest["model_revision"] == JEFF_MODEL_REVISION
    assert "frozen_model_revision" not in manifest
    validate_checkpoint_manifest(manifest)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"official_train_case_count": 43179}, "original Train"),
        ({"model_revision": "incorrect-model-revision"}, "different Jeff revision"),
        ({"loss_method": "unknown"}, "loss method"),
        ({"loss_method": "s1", "source": "train_oof_k16_epoch02.jsonl"}, "disagree"),
        ({"loss_method": "s3", "source": "train_oof_k16_epoch02.jsonl"}, "disagree"),
    ],
)
def test_corrupt_or_mismatched_checkpoint_manifest_rejected(change: dict, error: str) -> None:
    manifest = fixed_run_settings(sample_args("s3"))
    manifest.update(change)
    with pytest.raises(ValueError, match=error):
        validate_checkpoint_manifest(manifest)


def test_invalid_manifest_type_rejected() -> None:
    with pytest.raises(TypeError, match="JSON object"):
        validate_checkpoint_manifest(None)
