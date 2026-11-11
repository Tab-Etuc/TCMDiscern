"""Unit tests for complete K-specific external LLM results (no model is faked as run)."""

from __future__ import annotations

import pytest

from scripts.score_llm_k_grid import score_predictions


def sample() -> tuple[list[dict], list[str], list[tuple[str, ...]]]:
    labels = ["A", "B"]
    options = [("B", "A"), ("B", "A")]
    rows = [
        {"case_id": "dev-000000", "k": 1, "predicted_label": "B"},
        {"case_id": "dev-000001", "k": 1, "predicted_label": "B"},
        {"case_id": "dev-000000", "k": 2, "predicted_label": "A", "latency_ms": 100},
        {"case_id": "dev-000001", "k": 2, "predicted_label": "B", "latency_ms": 200},
    ]
    return rows, labels, options


def test_exactly_one_prediction_per_dev_case_and_k() -> None:
    rows, labels, options = sample()
    result = score_predictions(rows, labels=labels, candidates=options, classes=["A", "B"], max_k=2)
    assert [r["k"] for r in result] == [1, 2]
    assert result[0]["top1"] == 0.5
    assert result[0]["oracle_recall"] == 0.5
    assert result[1]["top1"] == 1.0
    assert result[1]["macro_f1"] == 1.0
    assert result[1]["conditional_accuracy"] == 1.0
    assert result[1]["latency_ms"]["reported_n"] == 2
    assert result[1]["latency_ms"]["mean"] == 150


def test_incomplete_grid_rejected() -> None:
    rows, labels, options = sample()
    with pytest.raises(ValueError, match="missing 1"):
        score_predictions(rows[:-1], labels=labels, candidates=options, classes=["A", "B"], max_k=2)


def test_prediction_outside_current_candidate_pool_rejected() -> None:
    rows, labels, options = sample()
    rows[0]["predicted_label"] = "A"
    with pytest.raises(ValueError, match="outside"):
        score_predictions(rows, labels=labels, candidates=options, classes=["A", "B"], max_k=2)


def test_duplicate_case_and_k_rejected() -> None:
    rows, labels, options = sample()
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(
            rows + [rows[0]],
            labels=labels,
            candidates=options,
            classes=["A", "B"],
            max_k=2,
        )
