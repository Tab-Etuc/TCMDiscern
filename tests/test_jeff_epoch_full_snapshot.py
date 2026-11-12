"""Frozen, privacy-preserving full Train two-epoch shuffle regression result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def test_all_train_epoch_shuffles_preserved_supervision() -> None:
    report = json.loads(
        (ROOT / "jeff_full_epoch_shuffle_2026-11-12.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "JEFF_ALL_TRAIN_TWO_EPOCH_RESHUFFLE_PASS"
    assert report["test_data_loaded"] is False
    assert report["gpu_model_training_performed"] is False
    assert report["total_training_question_instances_verified"] == 172720

    s1 = report["s1"]
    multi = report["s2_s3_s4_shared"]
    for snapshot in (s1, multi):
        assert snapshot["original_train_records"] == 43180
        assert snapshot["epochs_checked"] == 2
        assert snapshot["question_instances_checked"] == 86360
        assert snapshot["all_gold_and_positive_masks_validated"]
        assert snapshot["source_records_modified"] is False
        assert snapshot["gold_position_changed_between_epochs_n"] > 30000

    assert s1["positive_count_by_original_record"] == {"1": 43180}
    assert multi["positive_count_by_original_record"] == {"1": 42360, "2": 816, "3": 4}
