"""CPU synthetic regression for Gemma 4 causal-prefix H200 preflight."""

from __future__ import annotations

import numpy as np
import pytest

from tcmdiscern.jeff_attention import compare_causal_prefix


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
