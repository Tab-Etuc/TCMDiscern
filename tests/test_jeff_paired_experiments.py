"""Paired-loss controls: same original case/options must get identical prompt slots."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_weighting import (
    load_train_oof_weights,
    normalized_batch_weights,
)


def example(index: int, *, multi: bool, permutation: list[int]) -> dict:
    labels = [f"證型{j:02d}" for j in range(16)]
    shuffled = [labels[i] for i in permutation]
    gold = shuffled.index("證型01")
    item = {
        "row": {
            "id": f"train-{index:06d}",
            "state": {"chief_complaint": "咳嗽", "description": "痰多", "detection": "苔白膩"},
            "question": "選證型",
            "options": [{"id": s, "description": s} for s in shuffled],
        },
        "target_index": gold,
        "target_letter": "ABCDEFGHIJKLMNOP"[gold],
    }
    if multi:
        item["positive_indices"] = [gold]
        item["positive_mask"] = [i == gold for i in range(16)]
        item["target_distribution"] = [1.0 if i == gold else 0.0 for i in range(16)]
    return item


def test_canonical_epoch_shuffle_equalizes_s1_and_s3_identical_case():
    s1 = example(9, multi=False, permutation=list(range(16)))
    s3 = example(9, multi=True, permutation=list(reversed(range(16))))
    original_s1, original_s3 = deepcopy(s1), deepcopy(s3)
    for seed in (7, 20261112, 20261113):
        for epoch in (0, 1, 2):
            a = reshuffle_training_example(s1, epoch=epoch, seed=seed)
            b = reshuffle_training_example(s3, epoch=epoch, seed=seed)
            assert a["row"] == b["row"]
            assert a["target_index"] == b["target_index"]
            assert b["positive_indices"] == [b["target_index"]]
            assert b["positive_mask"][b["target_index"]]
    assert s1 == original_s1 and s3 == original_s3


def test_train_oof_recoverable_rank_weighting_is_paired_and_normalized(tmp_path):
    ranks = [1, 2, 16, 17, None]
    examples = [example(i, multi=True, permutation=list(range(16))) for i in range(5)]
    file = tmp_path / "oof.jsonl"
    with file.open("w", encoding="utf-8") as out:
        for i, rank in enumerate(ranks):
            gold = "證型01"
            candidate_list = [f"負例{j:02d}" for j in range(50)]
            if rank is not None:
                candidate_list[rank - 1] = gold
            out.write(
                json.dumps(
                    {
                        "case_id": f"train-{i:06d}",
                        "gold_labels": [gold],
                        "candidate_labels": candidate_list,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    weights, report = load_train_oof_weights(examples, source=file)
    assert weights == [1.0, 2.0, 2.0, 1.0, 1.0]
    assert report["upweighted_original_rows"] == 2
    assert report["rank_categories"] == {
        "rank1": 1,
        "rank2_to_16": 2,
        "rank17_to_50": 1,
        "rank_over50": 1,
    }
    scales = normalized_batch_weights(examples, scheme="recoverable_oof_x2", oof_weights=weights)
    assert scales == pytest.approx((1 / 7, 2 / 7, 2 / 7, 1 / 7, 1 / 7))
    uniform = normalized_batch_weights(examples, scheme="uniform")
    assert uniform == pytest.approx([1 / 5] * 5)
    assert sum(scales) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown"):
        normalized_batch_weights(examples, scheme="random")
    with pytest.raises(ValueError, match="Missing"):
        normalized_batch_weights(examples, scheme="recoverable_oof_x2")


def test_misaligned_train_oof_labels_fail_closed(tmp_path):
    one = [example(0, multi=True, permutation=list(range(16)))]
    file = tmp_path / "rank.jsonl"
    file.write_text(
        json.dumps(
            {
                "case_id": "train-000000",
                "gold_labels": ["錯誤"],
                "candidate_labels": [f"候選{i}" for i in range(50)],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gold label disagree"):
        load_train_oof_weights(one, source=file)
