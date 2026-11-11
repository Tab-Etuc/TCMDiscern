"""Train-only, stratified 5-fold BM25 hard-negative mining for Jeff.

All official Train rows are retained. Every query is searched against
the other four folds only. No Dev/Test rows or labels are ever loaded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .retrieval import CaseBM25


@dataclass(frozen=True)
class OOFRankings:
    """One BM25 ranking per original Train row, in original input order."""

    fold_by_row: tuple[int, ...]
    rankings: tuple[tuple[str, ...], ...]
    fold_seconds: tuple[float, ...]
    fold_train_sizes: tuple[int, ...]
    fold_query_sizes: tuple[int, ...]


def assign_stratified_folds(
    labels: Sequence[str], *, folds: int = 5, seed: int = 20261111
) -> tuple[int, ...]:
    """Assign every original row to exactly one query fold.

    This is inside Train only. Repeated cases and patient IDs are preserved.
    The rarest label needs at least folds rows, or each fold cannot retain
    all labels in its four-fold index.
    """
    if folds < 2:
        raise ValueError("Number of folds must be >=2")
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Labels must be nonempty strings")
    supports = Counter(labels)
    rare = min(supports.values())
    if rare < folds:
        raise ValueError(f"Rarest syndrome has {rare} cases, cannot stratify into {folds} folds")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = [-1] * len(labels)
    row_indices = np.arange(len(labels))
    for fold, (index_ids, query_ids) in enumerate(cv.split(row_indices, labels)):
        if np.intersect1d(index_ids, query_ids).size:
            raise RuntimeError("Fold contains query documents in its own index")
        for index in query_ids:
            if result[int(index)] != -1:
                raise RuntimeError("Train row assigned to two query folds")
            result[int(index)] = fold
    if any(value == -1 for value in result):
        raise RuntimeError("One or more Train rows were not assigned")
    return tuple(result)


def assign_grouped_folds(
    labels: Sequence[str],
    clinical_texts: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 20261111,
) -> tuple[int, ...]:
    """Keep every original case; assign identical clinical text to one query fold.

    No rows are deleted, and official Train/Dev/Test is untouched.
    """
    if not labels or len(labels) != len(clinical_texts):
        raise ValueError("Aligned nonempty clinical texts and labels required")
    if folds < 2 or min(Counter(labels).values()) < folds:
        raise ValueError("Not enough labeled examples for requested folds")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    ids = np.arange(len(labels))
    assignments = [-1] * len(labels)
    for fold, (index_ids, query_ids) in enumerate(splitter.split(ids, labels, clinical_texts)):
        if {clinical_texts[i] for i in index_ids}.intersection(
            clinical_texts[i] for i in query_ids
        ):
            raise RuntimeError("Exact duplicate clinical texts span the BM25 fold boundary")
        if len({labels[int(i)] for i in index_ids}) != len(set(labels)):
            raise RuntimeError("One or more labels absent from group-aware BM25 index")
        for index in query_ids:
            if assignments[int(index)] != -1:
                raise RuntimeError("Original Train row assigned twice")
            assignments[int(index)] = fold
    if any(fold < 0 for fold in assignments):
        raise RuntimeError("Some original Train cases lack a query fold")
    return tuple(assignments)


def rank_train_oof(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 20261111,
    max_k: int = 50,
    batch_size: int = 24,
    min_df: int = 2,
    fold_policy: str = "stratified",
    on_fold_done: Callable[[int, int, int, float], None] | None = None,
) -> OOFRankings:
    """Build five independent, four-fold BM25 indexes and query held-out fold.

    Corpus positions and query positions must be disjoint. Full original
    Train records remain available to training; no deduplication is done.
    """
    if len(texts) != len(labels) or not texts:
        raise ValueError("Train texts/labels must be nonempty and aligned")
    if not 1 <= max_k <= len(set(labels)):
        raise ValueError("max_k exceeds the available unique labels")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if fold_policy == "stratified":
        fold_by_row = assign_stratified_folds(labels, folds=folds, seed=seed)
    elif fold_policy == "exact_text_grouped":
        fold_by_row = assign_grouped_folds(labels, texts, folds=folds, seed=seed)
    else:
        raise ValueError("Unknown OOF fold policy")
    ranks: list[tuple[str, ...] | None] = [None] * len(texts)
    seconds: list[float] = []
    train_sizes: list[int] = []
    query_sizes: list[int] = []

    for fold in range(folds):
        train_ids = [i for i, f in enumerate(fold_by_row) if f != fold]
        query_ids = [i for i, f in enumerate(fold_by_row) if f == fold]
        if len(train_ids) + len(query_ids) != len(texts):
            raise RuntimeError("OOF fold size mismatch")
        if set(train_ids).intersection(query_ids):
            raise RuntimeError("A query row would be indexed in its own fold")
        start = perf_counter()
        retriever = CaseBM25(
            [texts[i] for i in train_ids],
            [labels[i] for i in train_ids],
            min_df=min_df,
        )
        if len(retriever.label_names) != len(set(labels)):
            raise RuntimeError("One or more labels are absent in the OOF index")
        ranked = retriever.rank([texts[i] for i in query_ids], max_k=max_k, batch_size=batch_size)
        if len(ranked) != len(query_ids):
            raise RuntimeError("BM25 did not return a ranking for each OOF query")
        for i, options in zip(query_ids, ranked, strict=True):
            if ranks[i] is not None or len(options) != max_k or len(set(options)) != max_k:
                raise RuntimeError("Invalid or duplicated OOF candidate rankings")
            ranks[i] = tuple(options)
        elapsed = round(perf_counter() - start, 3)
        seconds.append(elapsed)
        train_sizes.append(len(train_ids))
        query_sizes.append(len(query_ids))
        if on_fold_done is not None:
            on_fold_done(fold, len(train_ids), len(query_ids), elapsed)

    if any(row is None for row in ranks):
        raise RuntimeError("Missing a Train OOF ranking")
    return OOFRankings(
        fold_by_row=fold_by_row,
        rankings=tuple(row for row in ranks if row is not None),
        fold_seconds=tuple(seconds),
        fold_train_sizes=tuple(train_sizes),
        fold_query_sizes=tuple(query_sizes),
    )


def rank_statistics(
    labels: Sequence[str], oof: OOFRankings, *, ks: Sequence[int] = (1, 5, 16, 17, 35, 50)
) -> dict:
    """Only aggregate metrics; do not publish clinical texts or patient IDs."""
    if len(labels) != len(oof.rankings):
        raise ValueError("OOF results and labels are not aligned")
    if any(k < 1 or k > len(oof.rankings[0]) for k in ks):
        raise ValueError("K exceeds OOF ranking length")
    supports = Counter(labels)
    pos = []
    for gold, options in zip(labels, oof.rankings, strict=True):
        try:
            pos.append(options.index(gold) + 1)
        except ValueError:
            pos.append(len(options) + 1)
    micro = {}
    macro = {}
    for k in sorted(set(ks)):
        micro[str(k)] = sum(v <= k for v in pos) / len(pos)
        macro[str(k)] = sum(
            sum(p <= k for p, gold in zip(pos, labels, strict=True) if gold == label) / n
            for label, n in supports.items()
        ) / len(supports)
    return {
        "total_train_rows": len(labels),
        "folds": len(oof.fold_seconds),
        "fold_query_rows": list(oof.fold_query_sizes),
        "fold_index_rows": list(oof.fold_train_sizes),
        "fold_seconds": list(oof.fold_seconds),
        "bm25_oof_candidate_micro_recall": micro,
        "bm25_oof_candidate_macro_recall": macro,
        "original_rows_deleted": 0,
        "train_label_classes": len(supports),
        "least_common_train_label": min(supports.values()),
        "most_common_train_label": max(supports.values()),
        "warning": (
            "Only query row itself is excluded by OOF. Exact/near duplicate "
            "clinical texts from other folds remain by the frozen no-dedup policy."
        ),
    }
