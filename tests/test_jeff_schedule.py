"""Train-only deterministic epoch schedule and exact resume-boundary tests."""

from __future__ import annotations

import json

import pytest

from tcmdiscern.jeff_format import LETTERS
from tcmdiscern.jeff_schedule import (
    epoch_batches,
    epoch_order,
    load_frozen_train,
    total_optimizer_steps,
)


def example(i: int, *, multi: bool = True) -> dict:
    opts = [{"id": f"label-{j}", "description": f"label-{j}"} for j in range(16)]
    item = {
        "row": {
            "id": f"train-{i:06d}",
            "state": {"chief_complaint": "症狀", "description": "發病", "detection": "脈"},
            "question": "最佳證型",
            "options": opts,
        },
        "target_index": 1,
        "target_letter": LETTERS[1],
    }
    if multi:
        item["positive_indices"] = [1, 4]
        item["positive_mask"] = [j in (1, 4) for j in range(16)]
        item["target_distribution"] = [0.5 if j in (1, 4) else 0 for j in range(16)]
    return item


def test_train_schedule_order_is_deterministic_and_complete() -> None:
    assert sorted(epoch_order(99, epoch=0, seed=21)) == list(range(99))
    assert epoch_order(99, epoch=0, seed=21) == epoch_order(99, epoch=0, seed=21)
    assert epoch_order(99, epoch=0, seed=21) != epoch_order(99, epoch=1, seed=21)
    assert total_optimizer_steps(20, 6, 3) == 12


@pytest.mark.parametrize("method_multi", [False, True])
def test_complete_non_divisible_epoch_and_resume_without_losing_rows(
    method_multi: bool,
) -> None:
    dataset = [example(i, multi=method_multi) for i in range(20)]
    all_batches = list(epoch_batches(dataset, epoch=0, seed=42, accumulation=6))
    assert [next_offset for next_offset, _ in all_batches] == [6, 12, 18, 20]
    all_ids = [row["row"]["id"] for _, batch in all_batches for row in batch]
    assert len(set(all_ids)) == len(all_ids) == 20
    for _, batch in all_batches:
        for row in batch:
            options = row["row"]["options"]
            assert options[row["target_index"]]["id"] == "label-1"
            if method_multi:
                assert {options[p]["id"] for p in row["positive_indices"]} == {"label-1", "label-4"}
                assert sum(row["target_distribution"]) == 1

    restored = list(epoch_batches(dataset, epoch=0, seed=42, accumulation=6, start_offset=12))
    assert restored == all_batches[2:]
    assert all_ids == [row["row"]["id"] for _, batch in all_batches[:2] for row in batch] + [
        row["row"]["id"] for _, batch in restored for row in batch
    ]
    assert all_ids != [
        row["row"]["id"]
        for _, batch in epoch_batches(dataset, epoch=1, seed=42, accumulation=6)
        for row in batch
    ]


def test_reject_resume_mid_gradient_accumulation() -> None:
    rows = [example(i) for i in range(21)]
    with pytest.raises(ValueError, match="update boundary"):
        list(epoch_batches(rows, epoch=0, seed=1, accumulation=6, start_offset=7))
    with pytest.raises(ValueError):
        list(epoch_batches(rows, epoch=0, seed=1, accumulation=0))
    with pytest.raises(ValueError):
        list(epoch_batches(rows, epoch=0, seed=1, accumulation=6, start_offset=25))


def test_load_train_fails_closed_on_missing_duplicate_or_dev_rows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [example(i) for i in range(3)]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert len(load_frozen_train(path, multi=True, expected_count=3)) == 3
    with pytest.raises(ValueError, match="Expected 4"):
        load_frozen_train(path, multi=True, expected_count=4)
    rows[1]["row"]["id"] = "dev-000001"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Wrong Train"):
        load_frozen_train(path, multi=True, expected_count=3)
