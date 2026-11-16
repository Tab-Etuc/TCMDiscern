"""Invariant tests for the S1/S2 ensemble CPU analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from scripts.analyze_s1_s2_ensemble import choose, create_methods, macro_f1, metrics


def test_macro_f1_matches_sklearn_for_fixed_148_classes():
    rng = np.random.default_rng(321)
    gold = rng.integers(0, 148, size=220)
    pred = rng.integers(0, 148, size=220)
    actual = macro_f1(gold, pred, np.arange(148))
    expected = f1_score(gold, pred, labels=np.arange(148), average="macro", zero_division=0)
    assert abs(actual - expected) < 1e-12


def test_conservative_gate_comparison_uses_bm25_anchor():
    z = np.array([[1.0, 8.0, 2.0, 0.0], [10.0, 12.0, 0.0, 0.0], [4.0, 3.0, 3.0, 2.0]])
    assert np.array_equal(choose(z, 3, 3.0), np.array([1, 0, 0]))


def test_fixed_and_ensemble_methods_keep_original_candidate_slots():
    rng = np.random.default_rng(12)
    a = rng.normal(0, 2, (31, 16))
    b = rng.normal(0, 2, (31, 16))
    # Log class frequencies for each of 31 candidate lists, provided by
    # the Train-only frequency table in production.
    prior = np.log(rng.integers(10, 300, size=(31, 16)))
    methods = create_methods(a, b, prior)
    assert len(methods) > 20
    assert all(v.shape == (31,) and (v >= 0).all() and (v < 16).all() for v in methods.values())
    assert np.array_equal(methods["bm25"], np.zeros(31, dtype=int))
    assert np.array_equal(methods["s1_raw"], a.argmax(1))
    assert np.array_equal(methods["s2_raw"], b.argmax(1))


def test_structural_consensus_falls_back_on_disagreement():
    n = 3
    z1 = np.zeros((n, 16))
    z2 = np.zeros((n, 16))
    # Case 0: both agree on B; Case 1: disagree; Case 2: BM25 both.
    z1[0, 1] = 8
    z2[0, 1] = 8
    z1[1, 1] = 8
    z2[1, 2] = 8
    z1[2, 0] = 8
    z2[2, 0] = 8
    got = create_methods(z1, z2, np.zeros((n, 16)))
    assert np.array_equal(got["both_gate_agree"], [1, 0, 0])
    assert np.array_equal(got["gate_agree_or_s1"], [1, 0, 0])


def test_metrics_keep_unobserved_classes_at_zero():
    gold = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    count = np.ones(148, dtype=int) * 100
    m = metrics(gold, pred, count)
    assert m["n"] == 4
    assert m["correct"] == 2
    assert abs(m["macro_f1"] - (2 / 3) / 148) < 1e-12
