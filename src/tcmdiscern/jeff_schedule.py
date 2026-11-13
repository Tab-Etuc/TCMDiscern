"""Deterministic and resumable Jeff option-shuffle training schedule.

The official Train always has 43,180 rows. For tests, a smaller expected
count may be passed explicitly. Only Train rows may enter the schedule.
Saved checkpoints resume *after* completed optimizer updates, never from
an accumulation with partial gradients.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from random import Random

from .jeff_epoch import reshuffle_training_example
from .jeff_format import LETTERS


def check_training_example(
    item: dict, *, row_index: int, multi: bool, expected_k: int = 16, study_mode: str | None = None
) -> None:
    row = item["row"]
    if row["id"] != f"train-{row_index:06d}":
        raise ValueError(f"Wrong Train row order at index {row_index}")
    options = row["options"]
    if len(options) != expected_k or len({x["id"] for x in options}) != expected_k:
        raise ValueError(f"Train row {row_index}: needs {expected_k} distinct candidates")
    fields = {"chief_complaint", "description", "detection"}
    if study_mode == "conditioned":
        fields.add("lcd_name")
    if set(row["state"]) != fields:
        raise ValueError("Disallowed or incomplete patient evidence fields")
    if item["target_index"] not in range(expected_k):
        raise ValueError("Invalid target position")
    if item["target_letter"] != LETTERS[item["target_index"]]:
        raise ValueError("Incorrect target letter")
    if multi:
        pos = item["positive_indices"]
        mask = item["positive_mask"]
        dist = item["target_distribution"]
        if (
            not 1 <= len(pos) <= 3
            or pos != sorted(set(pos))
            or any(i not in range(16) for i in pos)
            or item["target_index"] not in pos
        ):
            raise ValueError("Invalid positive set")
        if (
            len(mask) != 16
            or len(dist) != 16
            or [i for i, m in enumerate(mask) if m] != pos
            or abs(sum(dist) - 1) > 1e-12
        ):
            raise ValueError("Invalid multi-positive supervision")
    elif study_mode is not None:
        allowed = {"row", "target_index", "target_letter", "hard_negative_indices"}
        if not set(item) <= allowed or not {"row", "target_index", "target_letter"} <= set(item):
            raise ValueError("Invalid disease-study training metadata")
        hard = item.get("hard_negative_indices", [])
        if (
            len(set(hard)) != len(hard)
            or any(i not in range(expected_k) or i == item["target_index"] for i in hard)
        ):
            raise ValueError("Invalid disease-study hard negatives")
    elif set(item) != {"row", "target_index", "target_letter"}:
        raise ValueError("S1 must contain only single-target supervision")


def load_frozen_train(
    file: Path, *, multi: bool, expected_count: int = 43180,
    expected_k: int = 16, study_mode: str | None = None,
) -> list[dict]:
    """Parse and validate immutable Train-only K16 rows into host RAM once."""
    items = []
    with file.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= expected_count:
                raise ValueError("Train file has unexpected extra rows")
            item = json.loads(line)
            check_training_example(
                item, row_index=index, multi=multi, expected_k=expected_k,
                study_mode=study_mode,
            )
            items.append(item)
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} Train rows, got {len(items)}")
    return items


def epoch_order(n: int, *, epoch: int, seed: int) -> list[int]:
    if n <= 0 or epoch < 0:
        raise ValueError("Need positive n and nonnegative epoch")
    order = list(range(n))
    Random(seed + epoch * 1_000_003).shuffle(order)
    return order


def epoch_batches(
    items: Sequence[dict],
    *,
    epoch: int,
    seed: int,
    accumulation: int,
    start_offset: int = 0,
) -> Iterator[tuple[int, list[dict]]]:
    """Yield (next offset, whole accumulation batch) ready for a full update."""
    if not isinstance(accumulation, int) or isinstance(accumulation, bool) or accumulation < 1:
        raise ValueError("Gradient accumulation must be a positive integer")
    n = len(items)
    if not 0 <= start_offset <= n:
        raise ValueError("Checkpoint offset outside epoch")
    if start_offset < n and start_offset % accumulation:
        raise ValueError("Resume offset must be an optimizer update boundary")
    ordered = epoch_order(n, epoch=epoch, seed=seed)
    for offset in range(start_offset, n, accumulation):
        selected = ordered[offset : offset + accumulation]
        current = [
            reshuffle_training_example(items[index], epoch=epoch, seed=seed) for index in selected
        ]
        yield offset + len(selected), current


def checkpoint_due(*, update: int, cadence: int, at_epoch_end: bool, target_updates: int) -> bool:
    """Save at periodic updates, every complete epoch, and the final update."""
    if update < 1 or cadence < 1 or target_updates < update:
        raise ValueError("Invalid completed optimizer or checkpoint budget")
    return update % cadence == 0 or at_epoch_end or update == target_updates


def total_optimizer_steps(rows: int, accumulation: int, epochs: int) -> int:
    if rows <= 0 or accumulation <= 0 or epochs <= 0:
        raise ValueError("Invalid positive training budget")
    return ((rows + accumulation - 1) // accumulation) * epochs
