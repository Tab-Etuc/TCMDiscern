"""Guard the immutable 35-K official Dev summary against reporting regressions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "experiments/k1_35_cpu_selector_dev_2026-11-11.json"
BM25 = ROOT / "experiments/bm25_dev_official_2026-11-11.json"


def test_exhaustive_k_grid_and_original_data_policy() -> None:
    data = json.loads(GRID.read_text(encoding="utf-8"))
    original = json.loads(BM25.read_text(encoding="utf-8"))
    assert data["k_values"] == list(range(1, 36))
    assert data["k_count"] == 35
    assert data["excluded_records"] == original["excluded_cases"] == 0
    assert data["model_predictions_status"] == {
        "gemma_4_31b_it": "NOT_RUN",
        "gemma_lora": "NOT_RUN",
    }
    assert data["same_dev_cases_and_candidate_ranking_for_every_k"]
    assert data["same_classifier_weights_for_every_k"]


def test_all_k_have_real_5486_case_metrics_and_macro_f1() -> None:
    data = json.loads(GRID.read_text(encoding="utf-8"))
    original = json.loads(BM25.read_text(encoding="utf-8"))
    for name in ("bm25_rank1", "multinomial_nb", "complement_nb"):
        method = data["methods"][name]
        assert [x["k"] for x in method["results"]] == list(range(1, 36))
        for x in method["results"]:
            assert x["n"] == 5486
            assert x["correct_n"] / 5486 == x["top1"]
            assert (
                abs(x["oracle_micro_recall"] - original["micro"]["recall_at_each_k"][str(x["k"])])
                < 1e-12
            )
            assert (
                abs(x["oracle_macro_recall"] - original["macro"]["recall_at_each_k"][str(x["k"])])
                < 1e-12
            )
            assert 0 <= x["macro_f1"] <= 1
            assert x["correct_n"] <= x["oracle_covered_n"]


def test_per_class_metrics_retained_for_every_k_and_model() -> None:
    data = json.loads(GRID.read_text(encoding="utf-8"))
    for name in ("multinomial_nb", "complement_nb"):
        by_class = data["methods"][name]["per_class"]
        assert set(by_class) == {str(k) for k in range(1, 36)}
        for k in range(1, 36):
            records = by_class[str(k)]
            assert len(records) == 148
            assert sum(record["support"] for record in records) == 5486
            assert (
                abs(
                    sum(record["f1"] for record in records) / 148
                    - data["methods"][name]["results"][k - 1]["macro_f1"]
                )
                < 1e-12
            )
