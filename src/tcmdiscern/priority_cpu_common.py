"""Frozen CPU-only fusion primitives for prioritized H200 research.

The 2026-11-16 SGD training and 148-class metrics are copied verbatim from
the already validated Case-RAG evaluation scripts. This module intentionally
depends only on NumPy/sklearn and the Python standard library, so the core
experiments never import GPU runtimes or optional Case-RAG components.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

FEATURES = {
    "analyzer": "char",
    "ngram_range": (1, 3),
    "min_df": 3,
    "max_features": 180000,
    "sublinear_tf": True,
    "lowercase": False,
    "dtype": np.float32,
}

SGD_OPTIONS = {
    "loss": "modified_huber",
    "alpha": 2e-5,
    "penalty": "l2",
    "max_iter": 15,
    "tol": 1e-3,
    "random_state": 20261116,
}


def class_balanced_weights(labels: list[str]) -> np.ndarray:
    """Exact frozen 2026-11-16 SGD weighting: gamma=.25, smoothing=5."""
    frequency = Counter(labels)
    mean = len(labels) / len(frequency)
    weights = np.asarray(
        [np.clip((mean / (frequency[label] + 5.0)) ** 0.25, 0.25, 4.0) for label in labels],
        dtype=np.float64,
    )
    return weights / weights.mean()


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Fixed per-row z-score normalization with a stable zero-variance floor."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Invalid SGD/Jeff scores")
    return (matrix - matrix.mean(axis=1, keepdims=True)) / np.maximum(
        matrix.std(axis=1, keepdims=True), 1e-6
    )


def metrics(
    gold: list[str], labels: list[str], classes: list[str], tail: set[str], clean: np.ndarray
) -> dict:
    """Frozen 148-class accuracy/F1 and fixed Train-only tail37 metrics."""
    matches = np.asarray([a == b for a, b in zip(gold, labels, strict=True)])
    in_tail = np.asarray([name in tail for name in gold])
    return {
        "top1_percent": round(100 * float(matches.mean()), 4),
        "correct": int(matches.sum()),
        "macro_f1_148_percent": round(
            100 * f1_score(gold, labels, labels=classes, average="macro", zero_division=0), 4
        ),
        "tail_37_macro_f1_percent": round(
            100 * f1_score(gold, labels, labels=sorted(tail), average="macro", zero_division=0), 4
        ),
        "tail_gold_cases": int(in_tail.sum()),
        "tail_top1_percent": (
            round(100 * float(matches[in_tail].mean()), 4) if in_tail.any() else None
        ),
        "no_exact_train_text_n": int(clean.sum()),
        "no_exact_train_text_top1_percent": round(100 * float(matches[clean].mean()), 4),
    }
