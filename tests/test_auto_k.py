"""Correctness checks for monotone target search and retrieval ranking."""

from __future__ import annotations

import pytest

from scripts.auto_k_bm25 import summarize_ranks
from tcmdiscern.k_search import first_hit_rank, minimum_k, recall_at_k
from tcmdiscern.retrieval import CaseBM25, chinese_bigrams


def test_bigram_tokenizer_ignores_punctuation() -> None:
    assert chinese_bigrams("胸悶、腹痛! A12") == ["胸悶", "腹痛", "a12"]
    assert chinese_bigrams("痛") == ["痛"]


def test_ranked_labels_are_distinct_and_based_on_train_only() -> None:
    model = CaseBM25(
        ["胸悶胸痛胸悶", "脈浮畏寒", "咳嗽氣短"],
        ["氣滯", "風寒", "肺虛"],
        min_df=1,
    )
    result = model.rank(["胸悶胸痛", "脈浮", "XYZ"], max_k=3, batch_size=2)
    assert result[0][0] == "氣滯"
    assert result[1][0] == "風寒"
    assert result[2] == sorted(["氣滯", "風寒", "肺虛"])
    assert all(len(set(row)) == 3 for row in result)


def test_first_hit_rank() -> None:
    assert first_hit_rank(["A"], ["B", "A", "C"], max_k=3) == 2
    assert first_hit_rank(["A"], ["B", "C", "A"], max_k=2) == 3
    with pytest.raises(ValueError, match="gold"):
        first_hit_rank([], ["A"], max_k=2)


def test_binary_search_finds_smallest_target_k() -> None:
    ranks = [1, 4, 6, 51]
    result = minimum_k(ranks, target=0.5, max_k=10)
    assert result["minimum_k"] == 4
    assert result["status"] == "reached"
    assert recall_at_k(sorted(ranks), 3) < 0.5
    assert recall_at_k(sorted(ranks), 4) >= 0.5
    assert [step["k"] for step in result["steps"]] == [5, 2, 3, 4]


def test_binary_search_reports_unreachable_target() -> None:
    result = minimum_k([1, 4, 51, 51], target=0.95, max_k=50)
    assert result["minimum_k"] is None
    assert result["status"] == "unreachable"
    assert result["recall_at_max_k"] == 0.5


def test_recall_monotone_over_all_k() -> None:
    ranks = sorted([1, 4, 6, 51])
    scores = [recall_at_k(ranks, k) for k in range(1, 51)]
    assert scores == sorted(scores)


def test_macro_coverage_prevents_majority_label_domination() -> None:
    # 9 cases of A are easy, 1 case of B is only retrieved at rank 8.
    summary = summarize_ranks([1] * 9 + [8], ["A"] * 9 + ["B"], max_k=10, targets=[0.9])
    assert summary["recall_curve"]["1"] == 0.9
    assert summary["macro_recall_curve"]["1"] == 0.5
    assert summary["binary_searches"][0]["minimum_k"] == 1
    assert summary["macro_binary_searches"][0]["minimum_k"] == 8


def test_invalid_binary_search_inputs() -> None:
    with pytest.raises(ValueError, match="target"):
        minimum_k([1], target=0, max_k=10)
    with pytest.raises(ValueError, match="positive"):
        minimum_k([0], target=0.5, max_k=10)
