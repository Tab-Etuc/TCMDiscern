"""Find the smallest candidate count K meeting a monotone Recall@K target."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchStep:
    k: int
    recall: float
    decision: str


def first_hit_rank(gold: Iterable[str], candidates: Sequence[str], *, max_k: int) -> int:
    """Return the first gold rank, or max_k + 1 for no hit inside the candidate pool."""
    if max_k < 1:
        raise ValueError("max_k must be positive")
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must not be empty")
    for index, candidate in enumerate(candidates[:max_k], start=1):
        if candidate in gold_set:
            return index
    return max_k + 1


def recall_at_k(sorted_ranks: Sequence[int], k: int) -> float:
    """Fraction of rows whose first correct candidate is at rank <= k."""
    if not sorted_ranks or k < 1:
        raise ValueError("Nonempty ranks and positive K are required")
    return bisect_right(sorted_ranks, k) / len(sorted_ranks)


def minimum_k_by_metric(recall_fn: Callable[[int], float], *, target: float, max_k: int) -> dict:
    """Binary search a monotone, precomputed Recall@K metric.

    Use for micro or macro coverage, NOT for a non-monotone reranker Top-1 score.
    """
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    if max_k < 1:
        raise ValueError("max_k must be positive")

    steps: list[SearchStep] = []
    low, high, found = 1, max_k, None
    while low <= high:
        mid = (low + high) // 2
        recall = recall_fn(mid)
        if not 0 <= recall <= 1:
            raise ValueError("recall_fn must return values in [0, 1]")
        if recall >= target:
            found = mid
            high = mid - 1
            direction = "smaller_k"
        else:
            low = mid + 1
            direction = "larger_k"
        steps.append(SearchStep(mid, recall, direction))
    return {
        "target_recall": target,
        "status": "reached" if found is not None else "unreachable",
        "minimum_k": found,
        "max_k": max_k,
        "recall_at_max_k": recall_fn(max_k),
        "steps": [
            {"k": step.k, "recall": step.recall, "decision": step.decision} for step in steps
        ],
    }


def minimum_k(
    hit_ranks: Sequence[int],
    *,
    target: float,
    max_k: int,
) -> dict:
    """Find minimum K for micro Recall@K from validation first-hit ranks."""
    if not hit_ranks or any(not isinstance(rank, int) or rank < 1 for rank in hit_ranks):
        raise ValueError("hit_ranks must contain positive integers")
    sorted_ranks = sorted(hit_ranks)
    return minimum_k_by_metric(lambda k: recall_at_k(sorted_ranks, k), target=target, max_k=max_k)
