"""Full official Train CPU audit of dynamic per-epoch Jeff option shuffling.

Checks all 43,180 S1 and 43,180 S2/S3/S4 shared examples across TWO
epochs each. All gold/positive indices and distributions must follow the
original candidate label identities, not their old A-P positions. No Dev/Test
files, GPU weights or patient data are written into public reports.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_schedule import load_frozen_train


def validate_epochs(path: Path, *, multi: bool, seed: int) -> dict:
    records = load_frozen_train(path, multi=multi, expected_count=43180)
    shifted_gold = 0
    changed_between_epochs = 0
    total = 0
    by_positive_count: Counter[int] = Counter()
    for record in records:
        original_options = record["row"]["options"]
        original_labels = tuple(option["id"] for option in original_options)
        original_gold_label = original_labels[record["target_index"]]
        gold_positions = []
        known_positives = (
            {original_labels[i] for i in record["positive_indices"]}
            if multi
            else {original_gold_label}
        )
        for epoch in (0, 1):
            new = reshuffle_training_example(record, epoch=epoch, seed=seed)
            labels = tuple(option["id"] for option in new["row"]["options"])
            if set(labels) != set(original_labels) or len(set(labels)) != 16:
                raise ValueError("Candidate set changed during epoch reshuffle")
            if labels[new["target_index"]] != original_gold_label:
                raise ValueError("Official original gold does not follow reshuffle")
            if labels != original_labels:
                shifted_gold += 1
            gold_positions.append(new["target_index"])
            if multi:
                positions = new["positive_indices"]
                if {labels[j] for j in positions} != known_positives:
                    raise ValueError("An originally observed gold label disappeared")
                if [j for j, m in enumerate(new["positive_mask"]) if m] != positions:
                    raise ValueError("Positive mask is not aligned to new option order")
                distribution = new["target_distribution"]
                if abs(sum(distribution) - 1.0) > 1e-12:
                    raise ValueError("Target distribution ceased to be normalized")
                for idx, label in enumerate(labels):
                    expected = 1.0 / len(known_positives) if label in known_positives else 0.0
                    if abs(distribution[idx] - expected) > 1e-12:
                        raise ValueError("Soft-target distribution was not permuted correctly")
        by_positive_count[len(known_positives)] += 1
        changed_between_epochs += gold_positions[0] != gold_positions[1]
        total += 1
    if total != 43180:
        raise ValueError("Incomplete original Train")
    if changed_between_epochs <= 0:
        raise ValueError("Training rows did not reshuffle between epochs")
    return {
        "original_train_records": total,
        "epochs_checked": 2,
        "question_instances_checked": 2 * total,
        "positive_count_by_original_record": {
            str(k): value for k, value in sorted(by_positive_count.items())
        },
        "gold_position_changed_between_epochs_n": changed_between_epochs,
        "reshuffled_candidate_lists_not_equal_source_n": shifted_gold,
        "all_gold_and_positive_masks_validated": True,
        "source_records_modified": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    s1 = validate_epochs(
        root / "data/jeff/train_oof_k16_epoch02.jsonl",
        multi=False,
        seed=20261112,
    )
    multipos = validate_epochs(
        root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl",
        multi=True,
        seed=20261112,
    )
    if s1["positive_count_by_original_record"] != {"1": 43180}:
        raise ValueError("S1 must always supervise exactly one positive option")
    if multipos["positive_count_by_original_record"] != {"1": 42360, "2": 816, "3": 4}:
        raise ValueError("S2/S3/S4 observed positive frequencies changed")
    report = {
        "date": "2026-11-12",
        "status": "JEFF_ALL_TRAIN_TWO_EPOCH_RESHUFFLE_PASS",
        "seed": 20261112,
        "s1": s1,
        "s2_s3_s4_shared": multipos,
        "total_training_question_instances_verified": 4 * 43180,
        "test_data_loaded": False,
        "gpu_model_training_performed": False,
    }
    destination = root / "experiments/jeff_full_epoch_shuffle_2026-11-12.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
