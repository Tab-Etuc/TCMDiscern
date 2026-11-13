"""Paired supervision and true parameter-change guards before H200 long runs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_gpu_core import validate_update_magnitudes
from tcmdiscern.jeff_train_loop import (
    TRAIN_FILE,
    expected_case_exposures,
    load_train,
)


@pytest.mark.parametrize(
    ("rows", "accumulation", "updates", "expected"),
    [
        (43180, 64, 1, 64),
        (43180, 64, 16, 1024),
        (43180, 64, 675, 43180),
        (43180, 64, 1350, 86360),
        (43180, 1, 1, 1),
    ],
)
def test_expected_exposure_count_includes_last_short_accumulation(
    rows, accumulation, updates, expected
):
    assert expected_case_exposures(rows, accumulation, updates) == expected


def sample_case() -> dict:
    return {
        "row": {
            "id": "train-000000",
            "state": {"chief_complaint": "口渴", "description": "倦怠", "detection": "舌紅"},
            "question": "辨證題目",
            "qtype": "choice",
            "options": [{"id": f"證型{i}", "description": f"證型{i}"} for i in range(16)],
        },
        "target_index": 3,
        "target_letter": "D",
        "positive_indices": [3, 7],
        "positive_mask": [i in (3, 7) for i in range(16)],
        "target_distribution": [0.5 if i in (3, 7) else 0.0 for i in range(16)],
    }


def test_all_syndrome_methods_share_exact_candidate_source_and_shuffle(monkeypatch):
    import tcmdiscern.jeff_train_loop as loop

    assert len(set(TRAIN_FILE.values())) == 1
    frozen = sample_case()
    calls = []

    def loader(path, *, multi, expected_count):
        calls.append((path.name, multi, expected_count))
        return [deepcopy(frozen)]

    monkeypatch.setattr(loop, "load_frozen_train", loader)
    variants = {method: load_train(method)[0] for method in ("s1", "s2", "s3")}
    assert len(calls) == 3
    assert all(multi and count == 43180 for _, multi, count in calls)
    assert set(variants["s1"]) == {"row", "target_index", "target_letter"}
    assert variants["s1"]["row"] == variants["s2"]["row"] == variants["s3"]["row"]
    assert variants["s1"]["target_index"] == variants["s2"]["target_index"] == 3

    for epoch in (0, 1):
        shifted = {
            method: reshuffle_training_example(item, epoch=epoch, seed=20261112)
            for method, item in variants.items()
        }
        assert shifted["s1"]["row"] == shifted["s2"]["row"] == shifted["s3"]["row"]
        assert shifted["s1"]["target_index"] == shifted["s2"]["target_index"]
        assert shifted["s2"]["positive_indices"] == shifted["s3"]["positive_indices"]


@pytest.mark.parametrize(
    "bad",
    [
        {"lora_B": 0.0, "head_delta_weight": 2e-5, "head_delta_bias": 2e-5},
        {"lora_B": 2e-5, "head_delta_weight": 0.0, "head_delta_bias": 2e-5},
        {"lora_B": 2e-5, "head_delta_weight": 2e-5, "head_delta_bias": float("nan")},
        {"lora_B": float("inf"), "head_delta_weight": 2e-5, "head_delta_bias": 2e-5},
        {"lora_B": 2e-5, "head_delta_weight": 2e-5},
    ],
)
def test_first_update_rejects_no_learning_or_numerical_damage(bad):
    with pytest.raises(RuntimeError, match="failed to change"):
        validate_update_magnitudes(bad)


def test_first_update_requires_both_lora_and_head_changes():
    good = {
        "lora_B": 2e-5,
        "head_delta_weight": 1e-4,
        "head_delta_bias": 1e-4,
    }
    assert validate_update_magnitudes(good) == good
