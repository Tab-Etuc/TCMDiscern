"""Regression tests for the official Train/Dev annotation ambiguity audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_jeff_multi_annotation import (
    annotation_stats,
    dev_retrieval_and_label_bounds,
)

REPORT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "jeff_multi_annotation_bounds_2026-11-12.json"
)


def test_train_dev_multi_annotation_frozen_report() -> None:
    audit = json.loads(REPORT.read_text(encoding="utf-8"))
    assert audit["status"] == "MULTI_ANNOTATION_AND_K16_DETERMINISTIC_BOUND_AUDITED"
    assert audit["model_predictions_measured"] is False
    assert audit["test_data_loaded"] is False
    assert audit["original_rows_removed"] == 0
    tr = audit["train"]
    dev = audit["dev"]
    assert tr["rows"] == 43180
    assert tr["distinct_clinical_inputs"] == 38599
    assert tr["groups_by_distinct_observed_gold_count"] == {"1": 38347, "2": 251, "3": 1}
    assert tr["ambiguous_groups"] == 252
    assert tr["rows_in_ambiguous_groups"] == 820
    assert tr["same_input_different_gold_unavoidable_errors_n"] == 268
    assert dev["rows"] == 5486
    assert dev["ambiguous_groups"] == 29
    assert dev["rows_in_ambiguous_groups"] == 64


def test_k16_candidate_and_deterministic_single_answer_bounds() -> None:
    audit = json.loads(REPORT.read_text(encoding="utf-8"))
    result = audit["dev_k16"]
    assert result["k"] == 16
    assert result["bm25_top1_correct"] == 4366
    assert result["gold_in_top_k_n"] == 5409
    assert result["gold_absent_from_top_k_n"] == 77
    assert result["single_answer_deterministic_same_prompt_upper_bound_n"] == 5385
    assert result["additional_inevitable_errors_from_conflicting_identical_inputs_n"] == 24
    assert (
        result["single_answer_deterministic_same_prompt_upper_bound_n"] / 5486
        == result["single_answer_deterministic_same_prompt_upper_bound_rate"]
    )
    shared = audit["dev_shared_input_with_train"]
    assert shared["dev_rows_same_clinical_input_in_train"] == 529
    assert shared["dev_rows_gold_not_among_observed_train_same_text_labels"] == 20


def test_same_visible_input_with_conflicting_labels_limits_single_answer() -> None:
    common = {"chief_complaint": "相同主訴", "description": "相同病情", "detection": "相同四診"}
    cases = [
        dict(common, norm_syndrome="甲"),
        dict(common, norm_syndrome="乙"),
    ]
    predictions = [
        SimpleNamespace(case_id=f"dev-{i:06d}", gold_labels=(gold,), candidate_labels=("甲", "乙"))
        for i, gold in enumerate(("甲", "乙"))
    ]
    stat = annotation_stats(cases)
    assert stat["ambiguous_groups"] == 1
    assert stat["rows_in_ambiguous_groups"] == 2
    assert stat["same_input_different_gold_unavoidable_errors_n"] == 1
    bound = dev_retrieval_and_label_bounds(cases, predictions, k=2)
    assert bound["gold_in_top_k_n"] == 2
    assert bound["single_answer_deterministic_same_prompt_upper_bound_n"] == 1
    assert bound["additional_inevitable_errors_from_conflicting_identical_inputs_n"] == 1
