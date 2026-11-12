"""Epoch shuffling must preserve every S1/S2/S3/S4 training target."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_format import LETTERS
from tcmdiscern.jeff_multi import build_multi_example


def train_row(case_id: str = "train-000023") -> dict:
    return {
        "row": {
            "id": case_id,
            "state": {"chief_complaint": "咳嗽", "description": "痰多", "detection": "苔白"},
            "question": "哪個證型？",
            "options": [{"id": f"證型{i}", "description": f"證型{i}"} for i in range(16)],
        },
        "target_index": 3,
        "target_letter": "D",
    }


def test_s1_epoch_shuffling_keeps_original_gold_and_source_unchanged() -> None:
    original = train_row()
    saved = deepcopy(original)
    result = reshuffle_training_example(original, epoch=4, seed=37)
    assert original == saved
    assert result == reshuffle_training_example(original, epoch=4, seed=37)
    assert {x["id"] for x in result["row"]["options"]} == {
        x["id"] for x in original["row"]["options"]
    }
    assert result["row"]["options"][result["target_index"]]["id"] == "證型3"
    assert result["target_letter"] == LETTERS[result["target_index"]]
    changed = reshuffle_training_example(original, epoch=5, seed=37)
    assert result["row"]["options"] != changed["row"]["options"]


def test_multi_epoch_shuffle_reorders_all_targets_without_leakage() -> None:
    case = {
        "norm_syndrome": "甲證",
        "chief_complaint": "頭暈",
        "description": "睡眠差",
        "detection": "脈細",
    }
    positive = ["甲證", "乙證", "丙證"]
    ranks = [*positive, *(f"負例{i}" for i in range(50))]
    original = build_multi_example(
        case,
        case_id="train-000123",
        case_index=123,
        observed_labels=positive,
        ranked_candidates=ranks,
        epoch=3,
        seed=7,
    )
    snapshot = deepcopy(original)
    for epoch in (0, 1, 2, 10):
        shuffled = reshuffle_training_example(original, epoch=epoch, seed=700)
        ids = [x["id"] for x in shuffled["row"]["options"]]
        assert set(ids) == {x["id"] for x in original["row"]["options"]}
        assert ids[shuffled["target_index"]] == "甲證"
        assert {ids[i] for i in shuffled["positive_indices"]} == set(positive)
        assert sum(shuffled["target_distribution"]) == pytest.approx(1)
        for i, label in enumerate(ids):
            assert shuffled["positive_mask"][i] is (label in positive)
            assert shuffled["target_distribution"][i] == pytest.approx(
                1 / 3 if label in positive else 0
            )
        assert shuffled["target_letter"] == LETTERS[shuffled["target_index"]]
    assert original == snapshot
    assert reshuffle_training_example(original, epoch=10, seed=700) == reshuffle_training_example(
        original, epoch=10, seed=700
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"row": {"id": "dev-000000", "options": [{"id": "甲"}] * 16}, "target_index": 0},
        {"row": {"id": "train-000000", "options": [{"id": "甲"}]}, "target_index": 0},
        {"row": {"id": "train-000000", "options": [{"id": "甲"}] * 16}, "target_index": 2},
    ],
)
def test_rejects_invalid_rows(bad) -> None:
    with pytest.raises(ValueError):
        reshuffle_training_example(bad, epoch=1, seed=100)


def test_multi_incomplete_and_invalid_target_detection() -> None:
    example = train_row()
    example["positive_indices"] = [3, 5]
    with pytest.raises(ValueError, match="Incomplete"):
        reshuffle_training_example(example, epoch=1, seed=10)
    example["positive_mask"] = [i in [3, 5] for i in range(16)]
    example["target_distribution"] = [0.5 if i in [3, 5] else 0 for i in range(16)]
    result = reshuffle_training_example(example, epoch=1, seed=10)
    assert len(result["positive_indices"]) == 2
    example["target_distribution"][3] = 0.4
    with pytest.raises(ValueError, match="sum to one"):
        reshuffle_training_example(example, epoch=1, seed=10)
