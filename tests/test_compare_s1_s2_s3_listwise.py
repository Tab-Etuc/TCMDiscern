"""Invariants for fixed identical S1/S2/S3 listwise controls."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from scripts.compare_s1_s2_s3_listwise import crossfit_one, select_positions


def test_fixed_threshold_uses_bm25_top_one_when_no_clear_challenger():
    z = np.zeros((3, 16))
    z[0, 1] = 0.3
    z[1, 4] = 0.1
    z[2, 15] = 1.0
    np.testing.assert_array_equal(select_positions(z), [1, 0, 15])


def test_selection_rejects_invalid_16_slot_scores():
    with pytest.raises(ValueError, match="finite"):
        select_positions(np.zeros((3, 15)))
    z = np.zeros((3, 16))
    z[2, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_positions(z)


def test_group_crossfit_fits_complete_synthetic_predictions():
    rng = np.random.default_rng(20261116)
    n = 30
    candidates = np.tile(np.arange(16), (n, 1))
    gold = np.arange(n) % 16
    logits = rng.normal(size=(n, 16))
    counts = np.full(148, 60)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=3).split(gold))
    positions = crossfit_one(logits, candidates, gold, counts, folds)
    assert positions.shape == (n,)
    assert np.all((positions >= 0) & (positions < 16))
