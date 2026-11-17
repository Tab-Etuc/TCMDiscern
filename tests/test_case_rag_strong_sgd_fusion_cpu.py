"""Safeguards for frozen, CPU-only SGD plus Jeff Case-RAG fusion."""

import numpy as np
import pytest

from scripts.evaluate_case_rag_strong_sgd_fusion_cpu import (
    class_balanced_weights,
    frozen_pairwise_margin1,
    historical_baseline_audit,
    row_normalize,
    run,
)


def test_historical_fusion_audit_distinguishes_exact_reproduction_and_h200_drift():
    exact = historical_baseline_audit(4667, 60.9597)
    assert exact["matched"]
    assert exact["delta_correct"] == 0
    assert exact["delta_macro_f1_pp"] == 0.0

    h200 = historical_baseline_audit(4673, 61.2118)
    assert not h200["matched"]
    assert h200["delta_correct"] == 6
    assert h200["delta_macro_f1_pp"] == 0.2521
    assert h200["this_environment_top1_percent"] == 85.1805
    assert "same-run" in h200["interpretation"] or "same" in h200["interpretation"]


@pytest.mark.parametrize(
    ("correct", "macro"),
    [(-1, 60.0), (5487, 60.0), (4667, float("nan"))],
)
def test_historical_audit_rejects_invalid_scores(correct, macro):
    with pytest.raises(ValueError, match="Invalid baseline measurement"):
        historical_baseline_audit(correct, macro)


def test_strict_global2_preflight_blocks_cpu_fit_when_logits_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing completed Global2 logits"):
        run(tmp_path, strict_global2=True)


def test_baseline_preflight_blocks_cpu_fit_when_logits_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="original full-Dev report/logits are missing"):
        run(tmp_path, strict_global2=False)


def test_weights_are_positive_deterministic_and_mean_one():
    labels = ["a"] * 40 + ["b"] * 10 + ["c"] * 2
    a = class_balanced_weights(labels)
    b = class_balanced_weights(labels)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b)
    assert np.all(a > 0)
    np.testing.assert_allclose(a.mean(), 1.0)
    assert a[-1] > a[0]


def test_frozen_pairwise_preserves_shortlist_and_score_robustness():
    cheap = np.zeros((3, 16), dtype=np.float64)
    jeff = np.zeros_like(cheap)
    cheap[0, 5] = 10.0
    cheap[1, 0] = 2.0
    cheap[1, 1] = 1.0  # weaker linear-model margin than Jeff's challenger
    jeff[1, 4] = 100.0
    cheap[2, 0] = 10.0
    jeff[2, 15] = 500.0  # outside the original Jeff top-10 gate
    support = np.ones_like(cheap)
    slots = frozen_pairwise_margin1(cheap, jeff, support)
    assert slots.tolist() == [5, 4, 0]


@pytest.mark.parametrize(
    ("cheap", "jeff", "support"),
    [
        (np.ones((2, 16)), np.ones((2, 15)), np.ones((2, 16))),
        (np.ones((2, 16)), np.ones((2, 16)), np.ones((2, 15))),
        (np.ones((2, 10)), np.ones((2, 10)), np.ones((2, 10))),
        (np.ones((2, 16)), np.ones((2, 16)), np.zeros((2, 16))),
    ],
)
def test_frozen_pairwise_rejects_invalid_input(cheap, jeff, support):
    with pytest.raises(ValueError):
        frozen_pairwise_margin1(cheap, jeff, support)


def test_frozen_pairwise_rejects_nonfinite_logits_and_scores():
    cheap = np.ones((2, 16))
    jeff = np.ones((2, 16))
    support = np.ones((2, 16))
    jeff[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        frozen_pairwise_margin1(cheap, jeff, support)
    jeff[0, 0] = 1.0
    cheap[1, 2] = float("inf")
    with pytest.raises(ValueError, match="Invalid SGD"):
        frozen_pairwise_margin1(cheap, jeff, support)


def test_row_normalize_is_shift_invariant():
    values = np.arange(32, dtype=np.float64).reshape(2, 16)
    np.testing.assert_allclose(row_normalize(values), row_normalize(values + 1000))
