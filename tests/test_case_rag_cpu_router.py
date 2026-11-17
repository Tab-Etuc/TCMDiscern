"""No-GPU tests for grouped case evidence routing and alignment."""

from __future__ import annotations

from collections import Counter

import numpy as np

from scripts.analyze_case_rag_cpu_router import (
    _softmax_logits,
    group_crossfit_router,
    prediction_features,
)


def test_softmax_is_stable_and_normalized():
    logits = np.array([[10000.0, 9999.0, -10000.0], [0.0, -100.0, 0.0]])
    probabilities = _softmax_logits(logits)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
    assert probabilities[0, 0] > probabilities[0, 1]


def test_features_use_model_and_retrieval_data_without_gold():
    options = [{"id": f"syndrome-{i}"} for i in range(16)]
    rows = [{"id": "dev-000000", "options": options}]
    logits = np.array([[0.0, *[1.0 + i for i in range(15)]]])
    frequency = Counter({option["id"]: i + 20 for i, option in enumerate(options)})
    provenance = [
        {
            "case_id": "dev-000000",
            "conditions": {
                "global2": {
                    "support_labels": ["syndrome-15", "syndrome-0"],
                    "support_scores": [120.0, 80.0],
                    "supported_options": 2,
                }
            },
        }
    ]
    x, jeff, bm25 = prediction_features(rows, logits, frequency, provenance)
    assert x.shape == (1, 17)
    assert np.isfinite(x).all()
    assert jeff.tolist() == ["syndrome-15"]
    assert bm25.tolist() == ["syndrome-0"]
    assert x[0, 8] == 0.0
    assert x[0, 9] == 1.0
    assert x[0, 10] == 1.0


def test_group_folds_keep_duplicate_cases_in_same_split():
    groups = np.repeat(np.arange(30), 4)
    labels = np.array(["a", "b"])
    gold = np.array([labels[i % 2] for i in range(120)])
    jeff = np.array(["a"] * 120)
    bm25 = np.array(["b"] * 120)
    rng = np.random.default_rng(20261117)
    features = rng.normal(size=(120, 17))
    probs, folds = group_crossfit_router(features, gold, jeff, bm25, groups)
    assert probs.shape == (120,)
    assert np.isfinite(probs).all()
    assert np.all((probs >= 0) & (probs <= 1))
    assert len(folds) == 5
    assert sum(fold["validation_cases"] for fold in folds) == 120
    assert sum(fold["validation_disagreements"] for fold in folds) == 120
