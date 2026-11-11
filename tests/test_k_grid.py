"""Tests for exhaustive K=1..35 evaluation, no hidden gold insertion."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_k_grid import metric_rows


def test_all_k_full_candidates_and_macro_f1() -> None:
    gold = np.asarray(["A", "B", "C"], dtype=object)
    candidates = [
        ("A", "B", "C"),
        ("A", "B", "C"),
        ("C", "A", "B"),
    ]
    # Reranker prefers the correct syndrome for all cases, where available.
    scores = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 4.0],
        ]
    )
    rows = metric_rows(
        gold=gold,
        candidate_names=candidates,
        classes=["A", "B", "C"],
        score_matrix=scores,
        max_k=3,
    )
    assert [row["k"] for row in rows] == [1, 2, 3]
    assert [row["oracle_covered_n"] for row in rows] == [2, 3, 3]
    assert [row["correct_n"] for row in rows] == [2, 3, 3]
    assert rows[2]["top1"] == 1.0
    assert rows[2]["macro_f1"] == 1.0
    assert rows[0]["macro_f1"] < 1.0


def test_bm25_rank1_unchanged_across_all_k() -> None:
    rows = metric_rows(
        gold=np.asarray(["A", "B"], dtype=object),
        candidate_names=[("A", "B"), ("A", "B")],
        classes=["A", "B"],
        max_k=2,
    )
    assert rows[0]["top1"] == rows[1]["top1"] == 0.5
    assert rows[0]["oracle_micro_recall"] == 0.5
    assert rows[1]["oracle_micro_recall"] == 1.0


def test_reject_duplicate_candidate_label() -> None:
    with pytest.raises(ValueError, match="Repeated"):
        metric_rows(
            gold=np.asarray(["A"], dtype=object),
            candidate_names=[("A", "A")],
            classes=["A", "B"],
            max_k=2,
        )


def test_reject_wrong_score_shape() -> None:
    with pytest.raises(ValueError, match="Score matrix shape"):
        metric_rows(
            gold=np.asarray(["A", "B"], dtype=object),
            candidate_names=[("B", "A"), ("B", "A")],
            classes=["A", "B"],
            score_matrix=np.ones((1, 2)),
            max_k=2,
        )
