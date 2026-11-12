"""Aggregate-only regression snapshots of completed Jeff CPU preflight.

All tests run after cloning the repository, without original patient records
or full 31B model weights. The raw 43,180-row training JSONL stays ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_shared_multipos_training_set_retains_all_43180_cases() -> None:
    report = load("jeff_multipos_k16_epoch03_2026-11-12.json")
    qa = load("jeff_multipos_k16_verified_epoch03_2026-11-12.json")
    assert report["status"] == "MULTIPOSITIVE_K16_TRAIN_ROWS_EXPORTED_NO_MODEL_TRAINING"
    assert report["oof_policy"] == "exact_text_grouped"
    assert report["all_original_train_rows_preserved"]
    assert report["dev_test_labels_used"] is False
    assert report["model_training_completed"] is False
    assert report["data"]["rows"] == qa["total_verified_questions"] == 43180
    assert report["data"]["k"] == 16
    assert report["data"]["observed_positive_counts_by_original_row"] == {
        "1": 42360,
        "2": 816,
        "3": 4,
    }
    assert report["data"]["distinct_text_groups_by_number_of_observed_labels"] == {
        "1": 38347,
        "2": 251,
        "3": 1,
    }
    assert qa["original_rows_with_more_than_one_observed_gold"] == 820
    assert qa["all_original_observed_labels_present"]
    assert qa["same_candidates_for_s2_s3_s4"]
    assert qa["train_gold_not_in_jeff_prompt"]
    assert qa["dev_test_loaded"] is False
    assert sum(qa["original_gold_position_counts"].values()) == 43180
    assert sum(qa["observed_positive_slot_counts"].values()) == 44004


def test_pinned_tokenizer_rechecked_all_multipos_prompts() -> None:
    report = load("jeff_multipos_k16_tokenizer_2026-11-12.json")
    stats = report["stats"]
    assert report["status"] == "JEFF_MULTIPOS_K16_PINNED_TOKENIZER_COMPLETE"
    assert report["max_input_tokens"] == 8192
    assert report["model_weights_loaded"] is False
    assert report["model_training_performed"] is False
    assert stats["rows"] == 43180
    assert stats["median"] == 658
    assert stats["maximum"] == 3577
    assert stats["over_4096"] == stats["over_8192"] == 0


def test_jeff_cpu_autograd_matches_independent_numpy_gradient() -> None:
    report = load("jeff_cpu_autograd_loss_2026-11-12.json")
    assert report["status"] == "CPU_TORCH_AUTOGRAD_LOSS_PARITY_VERIFIED"
    assert report["device"] == "cpu"
    assert report["all_passed"]
    assert report["validated"] == 7
    assert report["model_weights_loaded"] is False
    assert report["jeff_head_logits_verified"] is False
    assert report["h200_step_zero_training_verified"] is False
    assert {row["method"] for row in report["comparisons"]} == {"s1", "s2", "s3", "s4"}
    assert all(row["passed"] for row in report["comparisons"])
    assert all(row["max_abs_gradient_difference"] < 1e-6 for row in report["comparisons"])


def test_dev_scoring_dry_run_reproduces_the_original_bm25_baseline() -> None:
    report = load("jeff_k16_eval_dryrun_2026-11-12.json")
    assert report["status"] == "K16_JEFF_EVALUATION_WIRING_SMOKE_PASS"
    assert report["real_jeff_inference"] is False
    assert report["trained_checkpoint_scored"] is False
    score = report["results"]
    assert score["n_cases"] == 5486
    assert score["label_classes"] == 148
    assert score["top1_correct_n"] == 4366
    assert score["gold_in_top16_n"] == 5409
    assert score["gold_absent_top16_n"] == 77
    assert abs(score["top1_accuracy"] - 0.7958439664600802) < 1e-12
    assert abs(score["macro_f1"] - 0.5484957707533203) < 1e-12
    assert score["net_correct_change"] == 0


def test_negative_exposure_and_duplicate_group_statistics() -> None:
    report = load("jeff_multipos_exposure_2026-11-12.json")
    assert report["status"] == "LOCAL_JEFF_K16_NEGATIVE_EXPOSURE_AUDITED"
    assert report["original_train_rows"] == 43180
    assert report["removed_train_rows"] == 0
    assert report["positive_exposures"] == 44004
    assert report["negative_exposures"] == 646876
    assert report["expected_total_option_exposures"] == 16 * 43180
    assert report["positive_exposures"] + report["negative_exposures"] == 16 * 43180
    assert report["rows_with_multiple_observed_gold"] == 820
    assert report["duplicate_groups"]["unique_clinical_inputs"] == 38599
    assert report["duplicate_groups"]["all_rows_retained"]
    assert len(report["per_syndrome"]) == 148
    assert report["model_training_completed"] is False
