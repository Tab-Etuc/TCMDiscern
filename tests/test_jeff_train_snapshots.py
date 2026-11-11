"""Regression checks for the published Train-only Jeff/OOF aggregate snapshots.

No patient-level files are required, so the tests work in CI after clone.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_oof_folds_cover_exactly_every_original_train_row() -> None:
    report = load("bm25_train_oof_2026-11-11.json")
    assert report["total_train_rows"] == 43180
    assert report["train_label_classes"] == 148
    assert report["original_rows_deleted"] == report["excluded_records"] == 0
    assert report["folds"] == 5
    assert report["fold_query_rows"] == [8636] * 5
    assert report["fold_index_rows"] == [34544] * 5
    assert report["max_candidates_per_row"] == 50
    assert report["dev_cases_loaded"] == report["test_cases_loaded"] == 0
    assert report["train_used_for_index_only_in_other_folds"]


def test_oof_micro_macro_are_monotone_and_duplicate_audit_is_preserved() -> None:
    report = load("bm25_train_oof_2026-11-11.json")
    micro = report["bm25_oof_candidate_micro_recall"]
    macro = report["bm25_oof_candidate_macro_recall"]
    assert list(micro) == list(macro) == ["1", "3", "5", "10", "16", "17", "35", "50"]
    assert list(micro.values()) == sorted(micro.values())
    assert list(macro.values()) == sorted(macro.values())
    assert abs(micro["5"] - 0.9590319592403891) < 1e-12
    assert abs(macro["5"] - 0.7762025760596106) < 1e-12
    assert report["cross_fold_exact_input_text"] == {
        "query_rows_with_identical_text_in_another_fold": 6054,
        "such_rows_without_same_label_in_other_fold": 381,
    }
    assert report["original_clinical_text_containing_gold_literal"] == 1


def test_all_three_train_formats_keep_original_rows() -> None:
    report = load("jeff_train_examples_epoch00_2026-11-11.json")
    qa = load("jeff_training_quality_2026-11-11.json")
    assert report["original_train_count"] == 43180
    assert report["removed_train_rows"] == 0
    assert report["model_training"] == qa["jeff_model_inference"] == "NOT_RUN"
    assert report["raw_train_cases_containing_literal_gold_syndrome"] == 1
    assert len(report["files"]) == 3
    assert sum(x["cases"] for x in report["files"]) == 129540
    assert report["files"][0]["k_counts"] == {"5": 43180}
    assert report["files"][1]["k_counts"] == {"16": 43180}
    assert set(report["files"][2]["k_counts"]) == {str(k) for k in range(2, 17)}
    assert sum(report["files"][2]["k_counts"].values()) == 43180
    assert all(x["gold_label_field_excluded_from_evidence"] for x in report["files"])


def test_complete_oof_k1_50_and_fold_class_breakdown() -> None:
    report = load("bm25_train_oof_all_k_2026-11-11.json")
    rows = report["recall_all_k"]
    folds = report["folds"]
    labels = report["per_syndrome"]
    assert report["excluded_or_deduplicated_rows"] == 0
    assert report["dev_test_used"] is False
    assert report["official_train_rows"] == 43180
    assert [row["k"] for row in rows] == list(range(1, 51))
    assert [row["micro_recall"] for row in rows] == sorted(row["micro_recall"] for row in rows)
    assert [row["macro_recall"] for row in rows] == sorted(row["macro_recall"] for row in rows)
    assert (
        rows[4]["micro_recall"]
        == load("bm25_train_oof_2026-11-11.json")["bm25_oof_candidate_micro_recall"]["5"]
    )
    assert len(folds) == 5
    assert len(labels) == 148
    assert sum(fold["query_rows"] for fold in folds) == 43180
    assert sum(label["train_support"] for label in labels) == 43180
    assert sum(report["first_hit_rank_distribution"].values()) == 43180
    for fold in folds:
        assert fold["query_rows"] == 8636
        assert len(fold["recall_curve"]) == 50
        assert fold["labels"] == 148
    for row in rows:
        assert row["retrieved_gold_n"] + row["not_in_top_k_n"] == 43180
        assert row["retrieved_gold_n"] / 43180 == row["micro_recall"]
        assert (
            sum(fold["recall_curve"][row["k"] - 1]["correct_candidate_cases"] for fold in folds)
            == row["retrieved_gold_n"]
        )


def test_full_train_qa_confirms_129540_rows_and_tokenizer_gap() -> None:
    qa = load("jeff_training_quality_2026-11-11.json")
    assert qa["status"] == "JEFF_TRAIN_DATA_VALIDATED_ALL_129540_ROWS"
    assert qa["official_train_rows"] == 43180
    assert qa["evaluated_training_examples"] == 129540
    assert qa["all_checks_passed"]
    assert qa["dataset_deleted_rows"] == 0
    assert qa["test_dev_loaded"] is False
    assert qa["tokenized_context_length_checked"] is False
    assert len(qa["files"]) == 3
    for data in qa["files"]:
        assert data["cases"] == 43180
        assert data["train_targets_outside_jeff_prompt"]
        assert data["gold_excluded_from_hard_negatives"]
        assert data["original_clinical_text_containing_gold_literal"] == 1
        assert data["jeff_user_json_char_max"] >= data["jeff_user_json_char_p99"]
        assert data["jeff_user_json_char_p99"] >= data["jeff_user_json_char_median"]
