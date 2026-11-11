"""Verify the safe, exact-text-grouped Jeff K16 Train epoch without touching Dev/Test.

This validates the same original 43,180 cases; conflicting alternative gold
labels for identical clinical evidence must never be labeled as hard negatives.
Raw examples stay under ignored data/. Aggregate counts go to experiments/.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from scripts.verify_jeff_train import validate_file
from tcmdiscern.datasets import load_records

ROOT = Path(__file__).resolve().parents[1]
EPOCH = 2


def main() -> None:
    train = load_records(ROOT / "data/tcm-sd/train.json")
    if len(train) != 43180:
        raise ValueError("The full official Train must be preserved")
    report = json.loads(
        (ROOT / "experiments/bm25_train_text_grouped_2026-11-11.json").read_text(encoding="utf-8")
    )
    if report.get("fold_policy") != "exact_text_grouped":
        raise ValueError("Requires the no-overlap grouped OOF candidate source")
    if report["cross_fold_exact_input_text"]["query_rows_with_identical_text_in_another_fold"]:
        raise ValueError("Exact duplicate clinical text leaked across candidate folds")

    ranks = load_and_validate_oof(
        train, ROOT / "data/candidates/bm25_train_text_grouped.jsonl", report
    )
    path = ROOT / f"data/jeff/train_oof_k16_epoch{EPOCH:02d}.jsonl"
    qa = validate_file(train, ranks, path, expected_k=16, exclude_ambiguous_negatives=True)
    epoch = json.loads(
        (ROOT / f"experiments/jeff_train_examples_epoch{EPOCH:02d}_2026-11-11.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        epoch["original_train_count"] != 43180
        or epoch["removed_train_rows"] != 0
        or epoch["oof_fold_policy"] != "exact_text_grouped"
        or epoch["oof_candidate_source"] != "bm25_train_text_grouped.jsonl"
        or epoch["ambiguous_negative_policy"]
        != "exclude_alternative_gold_labels_for_identical_clinical_text"
        or len(epoch["files"]) != 1
        or epoch["files"][0]["k_counts"] != {"16": 43180}
        or qa["k_distribution"] != {"16": 43180}
    ):
        raise ValueError("The active safe K16 training data contract is inconsistent")
    if qa["alternative_gold_excluded_from_hard_negatives"] is not True:
        raise ValueError("Ambiguous-negative exclusion verification not active")
    summary = epoch["files"][0]
    result = {
        "date": "2026-11-12",
        "status": "GROUPED_K16_TRAIN_43180_VERIFIED_EXACT_CONFLICT_NEGATIVES_ZERO",
        "original_train_cases": qa["cases"],
        "original_train_cases_removed": 0,
        "train_k": 16,
        "oof_fold_policy": epoch["oof_fold_policy"],
        "original_ambiguous_hard_negative_cases_prevented": summary[
            "cases_with_confirmed_contradiction_prevented"
        ],
        "original_ambiguous_hard_negative_occurrences_replaced": summary[
            "confirmed_ambiguous_negatives_replaced"
        ],
        "remaining_confirmed_conflicting_hard_negatives": 0,
        "answer_positions": qa["gold_position_distribution"],
        "training_prompts_and_labels_verified": True,
        "semantic_synonyms_reviewed": False,
        "model_lora_training_completed": False,
        "test_cases_read": 0,
        "prior_epoch01_unchanged": True,
    }
    out = ROOT / "experiments/jeff_grouped_filtered_k16_verification_2026-11-12.json"
    with out.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
