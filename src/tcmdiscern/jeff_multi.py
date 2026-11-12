"""Train-only multiple-observed-label Jeff/SemIf supervision.

S2 Masked CE, S3 Set Loss and S4 soft-target CE consume the *same*
option list, shuffle and labels. The annotation set is learned solely
from original Train rows with byte-equal canonical clinical_text().
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .datasets import clinical_text
from .jeff_config import PRIMARY_JEFF_K
from .jeff_format import LETTERS, decision_row


def observed_label_sets(cases: Sequence[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Every official Train row contributes one observed label; none are dropped."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in cases:
        label = str(row["norm_syndrome"]).strip()
        if not label:
            raise ValueError("Original Train row has no syndrome label")
        grouped[clinical_text(row)].add(label)
    return {text: tuple(sorted(labels)) for text, labels in grouped.items()}


def build_multi_example(
    case: Mapping[str, Any],
    *,
    case_id: str,
    case_index: int,
    observed_labels: Sequence[str],
    ranked_candidates: Sequence[str],
    epoch: int,
    seed: int,
    k: int = PRIMARY_JEFF_K,
) -> dict[str, Any]:
    """Create one 16-option training row with 1..3 observed positive labels.

    The label set, its positions, and a normalized soft-target distribution
    remain outside the prompt. The originally annotated gold stays available
    as target_index for the S2 Masked CE control group.
    """
    if k != PRIMARY_JEFF_K:
        raise ValueError("Primary multiple-positive protocol requires K=16")
    if epoch < 0 or case_index < 0:
        raise ValueError("epoch and case_index must be nonnegative")
    gold = str(case["norm_syndrome"]).strip()
    positive = tuple(sorted(observed_labels))
    if not positive or len(positive) > k or len(set(positive)) != len(positive):
        raise ValueError("Invalid set of observed positive labels")
    if gold not in positive:
        raise ValueError("The original gold label must appear in its observed label set")
    if any(not label for label in positive):
        raise ValueError("Observed positive labels must be nonempty")
    if len(set(ranked_candidates)) != len(ranked_candidates):
        raise ValueError("OOF BM25 candidate labels must be unique")

    positive_set = set(positive)
    negatives = [name for name in ranked_candidates if name not in positive_set]
    if len(negatives) < k - len(positive):
        raise ValueError("Not enough nonpositive BM25 hard negatives for 16 options")

    selected = [*positive, *negatives[: k - len(positive)]]
    rng = Random(seed + epoch * 1_000_003 + case_index * 131 + k * 10_007)
    rng.shuffle(selected)

    positions = sorted(i for i, label in enumerate(selected) if label in positive_set)
    gold_index = selected.index(gold)
    if len(positions) != len(positive):
        raise RuntimeError("Observed positives must occur exactly once each")

    mask = [i in positions for i in range(k)]
    soft_target = [1.0 / len(positive) if flag else 0.0 for flag in mask]
    return {
        "row": decision_row(case, case_id=case_id, candidates=selected),
        "target_index": gold_index,
        "target_letter": LETTERS[gold_index],
        "positive_indices": positions,
        "positive_mask": mask,
        "target_distribution": soft_target,
    }
