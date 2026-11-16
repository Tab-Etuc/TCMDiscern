"""S2-only listwise calibrator structural tests."""

from __future__ import annotations

import numpy as np

from scripts.s2_listwise_calibration import option_features


def test_s2_features_shape_and_finiteness():
    rng = np.random.default_rng(17)
    z = rng.normal(0, 3, size=(12, 16))
    counts = rng.integers(1, 15000, size=(12, 16))
    x = option_features(z, counts)
    assert x.shape == (12, 16, 13)
    assert np.isfinite(x).all()


def test_s2_feature_calibration_is_logit_shift_invariant():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(13, 16))
    freq = rng.integers(1, 500, size=z.shape)
    original = option_features(z, freq)
    shifted = option_features(z + 47, freq)
    assert np.allclose(original, shifted, rtol=1.0e-9, atol=1.0e-9)


def test_s2_bm25_anchor_and_frequency_features_are_consistent():
    z = np.arange(16, dtype=float)[None, :]
    freq = np.arange(1, 17, dtype=int)[None, :]
    x = option_features(z, freq)
    assert x.shape[0] == 1
    assert x[0, 0, 7] == 1
    assert np.sum(x[0, :, 7]) == 1
    assert np.allclose(x[0, :, 3], np.log(freq[0]))
    assert x[0, 0, 0] == 0
    assert np.allclose(x[0, :, 4], np.log(freq[0] / freq[0, 0]))


def test_s2_all_equal_logits_remain_finite():
    z = np.full((3, 16), 9.3)
    freq = np.ones_like(z, dtype=int)
    x = option_features(z, freq)
    assert np.isfinite(x).all()
    assert np.allclose(x[:, :, 0], 0)
    assert np.allclose(x[:, :, 1], 0)
