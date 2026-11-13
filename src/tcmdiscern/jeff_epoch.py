"""Deterministic epoch-wise Jeff option reshuffling without touching source cases.

The pre-generated S1 and S2/S3/S4 JSONL files store one initial shuffle.
The actual H200 training loader MUST reshuffle A..P per epoch and remap every
external target, including the multi-positive distribution. This module is
CPU-only and never changes a gold label or any of the 16 candidates.
"""

from __future__ import annotations

from copy import deepcopy
from random import Random

from .jeff_format import LETTERS, MAX_DIRECT_OPTIONS


def reshuffle_training_example(
    item: dict,
    *,
    epoch: int,
    seed: int,
) -> dict:
    """Return a copy with options and all supervision permuted consistently."""
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    copy = deepcopy(item)
    row = copy["row"]
    case_id = row["id"]
    if not isinstance(case_id, str) or not case_id.startswith("train-"):
        raise ValueError("Epoch shuffling is permitted only on Train examples")
    try:
        index = int(case_id.removeprefix("train-"))
    except ValueError as exc:
        raise ValueError("Invalid original Train case ID") from exc
    options = row["options"]
    width = len(options)
    if not 2 <= width <= MAX_DIRECT_OPTIONS:
        raise ValueError("Epoch shuffling requires 2..16 options")
    if len({opt["id"] for opt in options}) != width:
        raise ValueError("Duplicate candidate labels")
    old_gold = copy["target_index"]
    if not isinstance(old_gold, int) or isinstance(old_gold, bool) or not 0 <= old_gold < width:
        raise ValueError("Invalid original gold index")

    has_multi = "positive_indices" in copy
    if has_multi:
        required = {"positive_indices", "positive_mask", "target_distribution"}
        if not required <= copy.keys():
            raise ValueError("Incomplete multi-positive supervision")
        old_pos = copy["positive_indices"]
        old_mask = copy["positive_mask"]
        old_distribution = copy["target_distribution"]
        if not old_pos or sorted(set(old_pos)) != old_pos:
            raise ValueError("Invalid multi-positive positions")
        if len(old_mask) != width or len(old_distribution) != width:
            raise ValueError("Incomplete 16-slot mask/distribution")
        if any(old_mask[i] != (i in old_pos) for i in range(width)):
            raise ValueError("Positive mask and positions disagree")
        if abs(sum(old_distribution) - 1) > 1e-12:
            raise ValueError("The soft targets must sum to one")
    elif any(k in copy for k in ("positive_mask", "target_distribution")):
        raise ValueError("Incomplete multi-positive supervision")

    # The frozen S1 and S2/S3/S4 files were produced with different initial
    # option shuffles. Canonicalize by label identity before each epoch so
    # identical candidate sets receive exactly the same prompt order for the
    # same case, epoch and seed. This removes a position-order confound in
    # paired loss ablations without rewriting either frozen source file.
    perm = sorted(range(width), key=lambda i: options[i]["id"])
    Random(seed + epoch * 1_000_003 + index * 131 + width * 10_007).shuffle(perm)
    row["options"] = [options[i] for i in perm]
    new_gold = perm.index(old_gold)
    copy["target_index"] = new_gold
    copy["target_letter"] = LETTERS[new_gold]
    if "hard_negative_indices" in copy:
        original_hard = copy["hard_negative_indices"]
        if any(not isinstance(i, int) or i == old_gold or not 0 <= i < width for i in original_hard):
            raise ValueError("Invalid hard-negative positions")
        copy["hard_negative_indices"] = [perm.index(i) for i in original_hard]
    if has_multi:
        copy["positive_indices"] = sorted(perm.index(i) for i in old_pos)
        copy["positive_mask"] = [old_mask[i] for i in perm]
        copy["target_distribution"] = [old_distribution[i] for i in perm]
        if new_gold not in copy["positive_indices"]:
            raise RuntimeError("Original gold disappeared from observed positives")
    return copy
