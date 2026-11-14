"""CPU synthetic regression for Gemma 4 causal-prefix H200 preflight."""

from __future__ import annotations

import numpy as np
import pytest

from tcmdiscern.jeff_attention import (
    compare_causal_prefix,
    perturb_suffix_tokens,
)


def test_fixed_length_suffix_probe_keeps_prefix_identical():
    original = list(range(1, 160))
    changed, split = perturb_suffix_tokens(original)
    assert len(changed) == len(original)
    assert changed[:split] == original[:split]
    assert changed[split:] != original[split:]
    assert all(token in original for token in changed)


def test_fixed_length_suffix_probe_handles_repeated_suffix_without_new_vocab():
    original = list(range(1, 41)) + [7] * 50
    changed, split = perturb_suffix_tokens(original)
    assert changed[:split] == original[:split]
    assert changed != original
    assert len(changed) == len(original)


def test_same_length_causal_result_remains_stable_with_changed_suffix():
    rng = np.random.default_rng(20261114)
    identical_prefix = rng.normal(size=(1, 40, 24)).astype(np.float32)
    original = np.concatenate(
        [identical_prefix.copy(), rng.normal(size=(1, 40, 24))],
        axis=1,
    )
    altered = original.copy()
    altered[:, 40:, :] = rng.normal(size=(1, 40, 24))
    report = compare_causal_prefix(original[:, :40, :], altered)
    assert report["mismatched_features"] == 0
    altered[:, 18, 7] += 12
    with pytest.raises(RuntimeError, match="attention backend changed"):
        compare_causal_prefix(original[:, :40, :], altered)


def test_causal_prefix_stays_identical_after_appending_tokens():
    rng = np.random.default_rng(20261112)
    prefix = rng.normal(size=(1, 32, 24)).astype(np.float32)
    extended = np.concatenate([prefix.copy(), rng.normal(size=(1, 16, 24))], axis=1)
    status = compare_causal_prefix(prefix, extended)
    assert status["mismatched_features"] == 0
    assert status["short_sequence_tokens"] == 32


def test_noncausal_attention_backend_is_rejected_before_gpu_training():
    prefix = np.ones((1, 32, 24), dtype=np.float32)
    extended = np.ones((1, 48, 24), dtype=np.float32)
    extended[:, 5, 3] = 2.0
    with pytest.raises(RuntimeError, match="attention backend changed earlier hidden"):
        compare_causal_prefix(prefix, extended)


def test_bad_shapes_and_nonfinite_states_are_rejected():
    short = np.ones((1, 32, 24), dtype=np.float32)
    long = np.ones((1, 48, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="hidden width"):
        compare_causal_prefix(short, np.ones((1, 48, 25), dtype=np.float32))
    long[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Non-finite"):
        compare_causal_prefix(short, long)
