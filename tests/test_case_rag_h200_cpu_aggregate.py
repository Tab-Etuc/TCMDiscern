"""Ensure the imported H200 full-Dev CPU aggregate stays internally consistent."""

import json
from pathlib import Path

import pytest

REPORT = Path(__file__).resolve().parents[1] / "experiments/case_rag_cpu_first_2026-11-17.json"


@pytest.fixture(scope="module")
def result():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_h200_aggregate_is_complete_and_test_sealed(result):
    assert result["status"] == "CASE_RAG_CPU_DIAGNOSTICS_COMPLETED"
    assert result["official_test_used"] is False
    assert (result["train_cases"], result["dev_cases"], result["classes"]) == (
        43180,
        5486,
        148,
    )
    assert result["clean_no_exact_train_text_n"] == 4957
    assert result["missing_case_rag_global2"] == []
    assert {"s1_global2", "s2_global2", "s3_global2"} <= set(result["case_rag_logits_available"])
    assert len(result["metrics"]) >= 11
    assert len(result["paired_error_decomposition"]) >= 18
    assert sum(part["dev_cases"] for part in result["train_frequency_gold_bins"]) == 5486


def test_all_reported_pairs_reconcile_with_original_correct_counts(result):
    metrics = result["metrics"]
    n = result["dev_cases"]
    for label, pair in result["paired_error_decomposition"].items():
        candidate, reference = label.split("_vs_", 1)
        if reference == "base":
            reference = candidate.split("_")[0] + "_base"
        elif reference not in ("bm25", "source_label"):
            reference = candidate.split("_")[0] + "_" + reference
        assert candidate in metrics and reference in metrics, label
        left_correct = metrics[candidate]["correct"]
        right_correct = metrics[reference]["correct"]
        assert pair["rescue"] - pair["break"] == left_correct - right_correct
        assert pair["both_correct"] + pair["reference_only_correct"] == right_correct
        assert pair["both_correct"] + pair["candidate_only_correct"] == left_correct
        assert (
            pair["both_correct"]
            + pair["reference_only_correct"]
            + pair["candidate_only_correct"]
            + pair["both_wrong"]
        ) == n
        assert (
            abs(pair["oracle_pair_top1_percent"] - 100 * (right_correct + pair["rescue"]) / n)
            < 0.0001
        )


def test_s3_original_and_global2_oracle_are_both_preserved(result):
    oracle = result["oracle_ceilings_descriptive_only"]
    assert oracle["bm25_or_s3_baseline_top1_percent"] == 88.4251
    assert oracle["bm25_or_s3_global2_top1_percent"] == 87.4043
    assert result["metrics"]["s3_global2"]["correct"] == 4578


def test_imported_result_keeps_only_aggregate_metrics(result):
    assert set(result).isdisjoint(
        {"patient_records", "raw_patient_text", "case_texts", "gold_per_case"}
    )
    expected_metric_names = {
        "top1_percent",
        "correct",
        "macro_f1_148_percent",
        "tail_37_macro_f1_percent",
        "tail_gold_cases",
        "tail_top1_percent",
        "no_exact_train_text_n",
        "no_exact_train_text_top1_percent",
    }
    for entry in result["metrics"].values():
        assert set(entry) <= expected_metric_names
