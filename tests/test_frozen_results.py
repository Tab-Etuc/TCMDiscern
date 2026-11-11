"""Regression checks on the publishable aggregate-only frozen BM25 experiment."""

from __future__ import annotations

import json
from pathlib import Path

RECORD = Path(__file__).resolve().parents[1] / "experiments/bm25_dev_official_2026-11-11.json"


def load() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_official_split_has_zero_exclusions() -> None:
    result = load()
    assert result["official_split"] == {"train": 43180, "dev": 5486, "test": 5486}
    assert result["excluded_cases"] == 0
    assert result["deduplication"] == "NONE - original records retained unmodified"
    assert result["test_model_evaluation"] == "NOT_RUN"
    assert result["method"]["ranked_distinct_labels"] == 50


def test_micro_and_macro_recall_curves_are_valid() -> None:
    result = load()
    micro = result["micro"]["recall_at_each_k"]
    macro = result["macro"]["recall_at_each_k"]
    correct = result["micro"]["correct_counts_at_each_k"]
    assert len(micro) == len(macro) == len(correct) == 50
    for k in range(1, 51):
        key = str(k)
        assert correct[key] / 5486 == micro[key]
        assert 0 <= macro[key] <= 1
        if k > 1:
            assert micro[key] >= micro[str(k - 1)]
            assert macro[key] >= macro[str(k - 1)]
    assert sum(result["hit_rank_distribution"].values()) == 5486
    assert correct["5"] == 5237
    assert correct["17"] == 5412


def test_per_class_aggregates_cover_all_original_cases() -> None:
    result = load()
    by_class = result["syndrome_aggregates"]
    assert len(by_class) == 148
    assert sum(row["train_n"] for row in by_class) == 43180
    assert sum(row["dev_n"] for row in by_class) == 5486
    assert all(row["dev_n"] > 0 for row in by_class)
    assert set(by_class[0]["recall_at_selected_k"]) == {
        "1",
        "3",
        "5",
        "10",
        "12",
        "17",
        "25",
        "34",
        "35",
        "50",
    }


def test_frozen_k_policy_and_search_thresholds() -> None:
    result = load()
    assert result["next_gemma_study"]["primary_k"] == 17
    assert result["next_gemma_study"]["comparison_k"] == [5, 35]
    micro_k = {x["target_recall"]: x["minimum_k"] for x in result["micro"]["binary_search"]}
    macro_k = {x["target_recall"]: x["minimum_k"] for x in result["macro"]["binary_search"]}
    assert micro_k[0.95] == 5
    assert micro_k[0.98] == 12
    assert macro_k[0.9] == 17
    assert macro_k[0.95] == 34
    assert macro_k[0.98] is None
