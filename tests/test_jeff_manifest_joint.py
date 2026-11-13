"""Prevent 31B disease LoRA runs from failing only after expensive training."""

from __future__ import annotations

import pytest

from tcmdiscern.jeff_checkpoint import validate_checkpoint_manifest
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION


@pytest.mark.parametrize(
    ("mode", "source", "width"),
    [
        (None, "train_oof_multipos_k16_epoch03.jsonl", None),
        ("disease16", "train_disease_k16.jsonl", 16),
        ("pair16", "train_pair_k16.jsonl", 16),
        ("global", "train_global_k5.jsonl", 5),
        ("conditioned", "train_conditioned_k5.jsonl", 5),
        ("pair", "train_pair_k5.jsonl", 5),
    ],
)
def test_checkpoint_metadata_accepts_its_actual_train_task(mode, source, width):
    payload = {
        "run_id": "unit",
        "official_train_case_count": 43180,
        "model_revision": JEFF_MODEL_REVISION,
        "loss_method": "s1",
        "source": source,
    }
    if mode is not None:
        payload.update(study_mode=mode, study_k=width)
    validate_checkpoint_manifest(payload)


def test_multipos_syndrome_checkpoint_still_accepted():
    validate_checkpoint_manifest(
        {
            "official_train_case_count": 43180,
            "model_revision": JEFF_MODEL_REVISION,
            "loss_method": "s3",
            "source": "train_oof_multipos_k16_epoch03.jsonl",
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"study_mode": "disease16", "study_k": 16, "source": "train_pair_k16.jsonl"},
        {"study_mode": "disease16", "study_k": 5, "source": "train_disease_k16.jsonl"},
        {"study_mode": "pair16", "study_k": 16, "source": "train_disease_k16.jsonl"},
        {"study_mode": "disease16", "study_k": 16, "loss_method": "s3"},
        {"study_mode": "typo", "study_k": 16, "source": "train_disease_k16.jsonl"},
        {"study_mode": None, "source": "train_disease_k16.jsonl"},
    ],
)
def test_checkpoint_manifest_rejects_incompatible_task(changes):
    payload = {
        "official_train_case_count": 43180,
        "model_revision": JEFF_MODEL_REVISION,
        "loss_method": "s1",
        "source": "train_oof_multipos_k16_epoch03.jsonl",
    }
    payload.update(changes)
    with pytest.raises(ValueError):
        validate_checkpoint_manifest(payload)
