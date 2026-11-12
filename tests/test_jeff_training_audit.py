"""Pre-training risk audit and Jeff 1.1 model-contract regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.preflight_jeff_training import inspect_checkpoint, preflight
from tcmdiscern.jeff_config import (
    JEFF_BACKBONE_DTYPE,
    JEFF_BASE_MODEL,
    JEFF_BIDIRECTIONAL,
    JEFF_HEAD_DTYPE,
    JEFF_HEAD_SOFTCAP,
    JEFF_HIDDEN_DIM,
    JEFF_MAX_INPUT_TOKENS,
    JEFF_MODEL_ID,
    JEFF_PROMPT_VERSION,
    JEFF_UPSTREAM_CODE_TAG,
    PRIMARY_JEFF_K,
)

ROOT = Path(__file__).resolve().parents[1]


def risk_report() -> dict:
    return json.loads(
        (ROOT / "experiments/jeff_training_risk_audit_2026-11-11.json").read_text(encoding="utf-8")
    )


def test_jeff_model_contract_pinned_to_actual_1_1_large() -> None:
    assert JEFF_MODEL_ID == "jgeuter/Jeff-1.1-Large"
    assert JEFF_BASE_MODEL == "google/gemma-4-31B-it"
    assert JEFF_UPSTREAM_CODE_TAG == "v1.3.0"
    assert JEFF_PROMPT_VERSION == "direct-options-v1"
    assert JEFF_BIDIRECTIONAL is None
    assert PRIMARY_JEFF_K == 16
    assert JEFF_MAX_INPUT_TOKENS == 8192
    assert JEFF_HIDDEN_DIM == 5376
    assert JEFF_HEAD_SOFTCAP == 30.0
    assert JEFF_HEAD_DTYPE == "float32"
    assert JEFF_BACKBONE_DTYPE == "bfloat16"


def test_original_train_imbalance_is_explicitly_audited() -> None:
    report = risk_report()
    assert report["deleted_train_or_dev_rows"] == 0
    assert report["scope"] == "TCM-SD official Train/Dev; Test not opened"
    distribution = report["train_class_imbalance"]
    assert distribution["train_cases"] == 43180
    assert distribution["label_count"] == 148
    assert distribution["minimum_support"] == 7
    assert distribution["maximum_support"] == 7380
    assert sum(distribution["support_buckets_number_of_classes"].values()) == 148
    assert distribution["dev_classes_with_at_most_5_cases"] == 70


def test_hard_negative_exposure_is_critically_unbalanced() -> None:
    report = risk_report()["oof_train_label_retrieval"]
    assert report["original_train_cases"] == 43180
    assert sum(report["gold_rank_distribution"].values()) == 43180
    assert report["gold_not_in_bm25_top_16_n"] == 595
    assert report["gold_in_bm25_top_16_n"] == 42585
    assert report["training_negative_examples_count"] == 43180 * 15
    assert report["classes_with_negative_exposure_over_10_times_positive"] == 135


def test_exact_label_conflicts_and_dev_overlap_are_visible_but_unchanged() -> None:
    report = risk_report()
    duplicate = report["train_exact_text_duplication"]
    assert duplicate["identical_clinical_text_groups_with_conflicting_labels"] == 252
    assert duplicate["rows_belonging_to_conflicting_label_groups"] == 820

    overlap = report["train_dev_full_text_overlap"]
    assert overlap["dev_cases_identical_to_one_or_more_train_rows"] == 529
    assert overlap["dev_overlap_gold_not_among_train_same_text_labels"] == 20
    same = overlap["overlap_dev_cohort"]
    different = overlap["nonoverlap_dev_cohort"]
    assert same["cases"] + different["cases"] == 5486
    assert same["bm25_top1_accuracy"] > different["bm25_top1_accuracy"]
    assert overlap["unmodified_official_dev"]["bm25_top1_correct"] == 4366
    assert report["dev_k16_rare_class_coverage"]["number_of_classes_with_recall_0"] == 3


def test_label_normalization_limit_is_not_misrepresented() -> None:
    value = risk_report()["label_lexical_normalization"]
    assert value["label_count"] == 148
    assert value["colliding_canonical_forms_after_unicode_nfkc_and_punctuation_removal"] == 0
    assert value["semantic_synonym_check"].startswith("NOT_PERFORMED")


def test_checkpoint_preflight_does_not_claim_missing_weights_are_ready(tmp_path) -> None:
    checkpoint = inspect_checkpoint(tmp_path)
    assert checkpoint["weights_present"] is False
    assert checkpoint["decision_head_present"] is False
    assert checkpoint["release_metadata_matches"] is None


def test_machine_preflight_never_claims_to_have_run_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("scripts.preflight_jeff_training.inspect_gpu", list)
    monkeypatch.setattr(
        "scripts.preflight_jeff_training.installed_versions",
        lambda: {
            "torch": None,
            "transformers": None,
            "safetensors": None,
            "huggingface_hub": None,
            "accelerate": None,
            "peft": None,
        },
    )
    result = preflight(tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["training_executed"] is False
    assert result["checkpoint"]["weights_present"] is False
    assert any("No GPU" in b or "NVIDIA GPU unavailable" in b for b in result["blockers"])
    assert any("model weights are unavailable" in b for b in result["blockers"])
    assert any("H200" in w for w in result["warnings"])
