"""Full 43,180-case CPU audit of paired Jeff loss and OOF weighting controls.

Never opens Dev/Test, loads GPU model or emits patient text. Rebuilds the exact
runtime candidate permutations for every case at epochs 0 and 1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_schedule import load_frozen_train
from tcmdiscern.jeff_weighting import load_train_oof_weights

ROOT = Path(__file__).resolve().parents[1]
TRAIN_N = 43180


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all Train paired-loss controls")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "experiments/jeff_paired_controls_2026-11-12.json"
    if target.exists() and not args.check_only:
        raise FileExistsError("Frozen paired-loss report already exists")
    s1 = load_frozen_train(
        ROOT / "data/jeff/train_oof_k16_epoch02.jsonl", multi=False, expected_count=TRAIN_N
    )
    multipos = load_frozen_train(
        ROOT / "data/jeff/train_oof_multipos_k16_epoch03.jsonl",
        multi=True,
        expected_count=TRAIN_N,
    )
    weights, difficulty = load_train_oof_weights(
        multipos, source=ROOT / "data/candidates/bm25_train_text_grouped.jsonl"
    )
    if len(weights) != TRAIN_N or difficulty["original_train_rows"] != TRAIN_N:
        raise ValueError("Missing weighted original Train cases")
    positive_sizes = Counter()
    paired_identical = 0
    total_questions = 0
    changed_candidate_sets = 0
    for epoch in (0, 1):
        for i, (single, multi) in enumerate(zip(s1, multipos, strict=True)):
            if (
                single["row"]["id"] != multi["row"]["id"]
                or single["row"]["state"] != multi["row"]["state"]
            ):
                raise ValueError(f"Source cases disagree at index {i}")
            source_gold = single["row"]["options"][single["target_index"]]["id"]
            multi_gold = multi["row"]["options"][multi["target_index"]]["id"]
            if source_gold != multi_gold:
                raise ValueError(f"Original gold disagrees at {i}")
            s1_epoch = reshuffle_training_example(single, epoch=epoch, seed=20261112)
            multi_epoch = reshuffle_training_example(multi, epoch=epoch, seed=20261112)
            labels1 = [o["id"] for o in s1_epoch["row"]["options"]]
            labels_multi = [o["id"] for o in multi_epoch["row"]["options"]]
            if (
                labels1[s1_epoch["target_index"]] != source_gold
                or labels_multi[multi_epoch["target_index"]] != source_gold
            ):
                raise ValueError(f"Runtime shuffle lost the original gold at {i}")
            expected_positives = {
                multi["row"]["options"][j]["id"] for j in multi["positive_indices"]
            }
            actual_positives = {labels_multi[j] for j in multi_epoch["positive_indices"]}
            if actual_positives != expected_positives:
                raise ValueError(f"Multi-positive label changed at {i}")
            positive_sizes[len(multi["positive_indices"])] += 1
            if len(multi["positive_indices"]) == 1:
                if labels1 != labels_multi:
                    raise ValueError(
                        f"Same-case single-positive control has mismatched A..P order at {i}"
                    )
                if s1_epoch["target_index"] != multi_epoch["target_index"]:
                    raise ValueError(f"Same-case single-positive gold slot differs at {i}")
                paired_identical += 1
            elif set(labels1) != set(labels_multi):
                changed_candidate_sets += 1
            total_questions += 2
    if paired_identical != 42360 * 2:
        raise ValueError("Unexpected number of identical candidate/prompt paired controls")
    if changed_candidate_sets <= 0:
        raise ValueError("Expected observed multi-positive candidate differences")
    if difficulty["rank_categories"]["rank2_to_16"] <= 0:
        raise ValueError("No recoverable hard Train cases")
    report = {
        "status": "PAIRED_TRAIN_CONTROLS_AND_HARD_WEIGHTING_VALIDATED",
        "source_cases": TRAIN_N,
        "epochs_compared": 2,
        "question_instances_checked": total_questions,
        "single_positive_cases_same_full_prompt_order_each_epoch": paired_identical // 2,
        "single_positive_case_epoch_pairs_verified": paired_identical,
        "cases_with_multiple_observed_labels": (positive_sizes[2] + positive_sizes[3]) // 2,
        "candidate_sets_different_for_multi_observed_cases_count_across_epochs": changed_candidate_sets,
        "all_original_gold_labels_preserved": True,
        "S2_S3_S4_same_source_options": True,
        "oof_recoverable_weighting": difficulty,
        "dev_test_opened": False,
        "gpu_model_loaded": False,
    }
    if not args.check_only:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
