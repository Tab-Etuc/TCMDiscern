"""Snapshot checks for corrected grouped-OFF K16 training supervision.

Only reads aggregate JSON committed to experiments/, so CI never needs raw
patient data or the confidential per-case training JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def snapshot(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_grouped_oof_has_no_exact_clinical_text_leakage() -> None:
    original = snapshot("bm25_train_oof_2026-11-11.json")
    grouped = snapshot("bm25_train_text_grouped_2026-11-11.json")
    assert original["total_train_rows"] == grouped["total_train_rows"] == 43180
    assert (
        original["cross_fold_exact_input_text"]["query_rows_with_identical_text_in_another_fold"]
        == 6054
    )
    assert grouped["fold_policy"] == "exact_text_grouped"
    assert (
        grouped["cross_fold_exact_input_text"]["query_rows_with_identical_text_in_another_fold"]
        == 0
    )
    assert grouped["excluded_records"] == 0
    assert sum(grouped["fold_query_rows"]) == 43180
    assert abs(grouped["bm25_oof_candidate_micro_recall"]["16"] - 0.9867068087077351) < 1e-12


def test_new_training_epoch_is_safe_and_previous_epochs_are_frozen() -> None:
    historical = snapshot("jeff_train_examples_epoch01_2026-11-11.json")
    revised = snapshot("jeff_train_examples_epoch02_2026-11-11.json")
    qa = snapshot("jeff_grouped_filtered_k16_verification_2026-11-12.json")

    assert historical["original_train_count"] == revised["original_train_count"] == 43180
    assert historical.get("oof_fold_policy", "stratified") == "stratified"
    assert revised["oof_fold_policy"] == "exact_text_grouped"
    assert revised["removed_train_rows"] == 0
    assert revised["ambiguous_negative_policy"] == (
        "exclude_alternative_gold_labels_for_identical_clinical_text"
    )
    assert len(revised["files"]) == 1
    assert revised["files"][0]["k_counts"] == {"16": 43180}
    assert revised["files"][0]["cases_with_confirmed_contradiction_prevented"] == 793
    assert revised["files"][0]["confirmed_ambiguous_negatives_replaced"] == 797

    assert qa["original_train_cases"] == 43180
    assert qa["original_train_cases_removed"] == 0
    assert qa["remaining_confirmed_conflicting_hard_negatives"] == 0
    assert sum(qa["answer_positions"].values()) == 43180
    assert qa["training_prompts_and_labels_verified"]
    assert qa["model_lora_training_completed"] is False
    assert qa["semantic_synonyms_reviewed"] is False


def test_verified_jeff_tokenizer_has_no_overlong_train_or_dev_records() -> None:
    report = snapshot("jeff_k16_tokenizer_2026-11-11.json")
    assert report["status"] == "VERIFIED_PINNED_JEFF_K16_TOKEN_COUNTS"
    assert report["max_context_tokens"] == 8192
    assert report["results"]["train_k16_epoch01"]["rows"] == 43180
    assert report["results"]["dev_k16"]["rows"] == 5486
    for item in report["results"].values():
        assert item["gt_4096"] == 0
        assert item["gt_8192"] == 0
