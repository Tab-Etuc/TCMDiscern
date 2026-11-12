"""Independent full-Train QA for shared S2/S3/S4 multi-positive supervision.

Checks all 43,180 original cases, BM25 negative ordering, every observed
positive, the original gold, exact 16 candidates, shuffle labels, soft-target
normalization and Jeff/SemIf prompt separation. Never loads Dev or Test.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_format import LETTERS, patient_state, semif_messages
from tcmdiscern.jeff_multi import observed_label_sets


def verify(cases: list[dict], ranks: list[list[str]], source: Path) -> dict:
    if len(cases) != 43180 or len(ranks) != len(cases):
        raise ValueError("Full original Train and original OOF rankings are mandatory")
    label_sets = observed_label_sets(cases)
    observed_rows: Counter[int] = Counter()
    positions: Counter[int] = Counter()
    position_masks: Counter[int] = Counter()
    changed_from_base = 0
    total = 0
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(cases):
                raise ValueError("Extra multi-positive training records")
            original = cases[index]
            gold = str(original["norm_syndrome"])
            alternatives = set(label_sets[clinical_text(original)])
            result = json.loads(line)
            if set(result) != {
                "row",
                "target_index",
                "target_letter",
                "positive_indices",
                "positive_mask",
                "target_distribution",
            }:
                raise ValueError(f"Unexpected training supervision fields at row {index}")
            row = result["row"]
            if row["id"] != f"train-{index:06d}":
                raise ValueError(f"Source row IDs are out of order at {index}")
            if row["state"] != patient_state(original):
                raise ValueError("Patient evidence differs from the original Train case")
            if set(row["state"]) != {"chief_complaint", "description", "detection"}:
                raise ValueError("Disallowed clinical evidence fields")
            options = row["options"]
            if len(options) != 16:
                raise ValueError("Must have exactly 16 original Jeff decision-head slots")
            values = [o["id"] for o in options]
            if len(set(values)) != 16 or any(o["description"] != o["id"] for o in options):
                raise ValueError("Duplicate or modified candidate labels")

            positives = result["positive_indices"]
            if (
                not isinstance(positives, list)
                or len(positives) != len(alternatives)
                or any(not isinstance(i, int) or isinstance(i, bool) for i in positives)
                or positives != sorted(set(positives))
            ):
                raise ValueError("Invalid positive positions")
            expected_positions = [i for i, name in enumerate(values) if name in alternatives]
            if positives != expected_positions:
                raise ValueError(f"Observed gold labels missing/duplicated at row {index}")
            if not 1 <= len(positives) <= 3:
                raise ValueError("Unexpected number of observed gold labels")
            if not isinstance(result["target_index"], int) or isinstance(
                result["target_index"], bool
            ):
                raise TypeError("Original gold index must be an integer")
            position = result["target_index"]
            if not 0 <= position < 16 or values[position] != gold:
                raise ValueError("Original official gold label moved to an incorrect position")
            if result["target_letter"] != LETTERS[position]:
                raise ValueError("Original gold letter is inconsistent")

            other = [name for name in ranks[index] if name not in alternatives]
            expected_set = alternatives | set(other[: 16 - len(alternatives)])
            if set(values) != expected_set:
                raise ValueError(f"BM25 difficult negatives were changed at {index}")
            if gold not in expected_set:
                raise ValueError("Gold label missing")
            if len(alternatives) > 1:
                changed_from_base += 1

            mask = result["positive_mask"]
            dist = result["target_distribution"]
            if (
                not isinstance(mask, list)
                or not isinstance(dist, list)
                or len(mask) != 16
                or len(dist) != 16
                or any(type(m) is not bool for m in mask)
            ):
                raise ValueError("Positive mask/soft target must have 16 entries")
            if [i for i, enabled in enumerate(mask) if enabled] != positives:
                raise ValueError("Positive mask disagrees with positive indices")
            if any(
                not isinstance(x, (int, float)) or isinstance(x, bool) or not 0 <= x <= 1
                for x in dist
            ):
                raise ValueError("Invalid soft-target probabilities")
            if abs(sum(dist) - 1.0) > 1e-12:
                raise ValueError("Soft-target distribution must sum to exactly one")
            for i, value in enumerate(dist):
                expected_value = 1.0 / len(positives) if i in positives else 0.0
                if abs(value - expected_value) > 1e-12:
                    raise ValueError("Soft-target value does not match its positive mask")

            messages = semif_messages(row)
            obj = json.loads(messages[1]["content"])
            if set(obj) != {"evidence", "criterion", "options"}:
                raise ValueError("Jeff prompt was changed")
            if obj["evidence"] != row["state"]:
                raise ValueError("Model evidence mismatch")
            if [x["letter"] for x in obj["options"]] != list(LETTERS):
                raise ValueError("Jeff must expose A..P in slot order")
            for forbidden in (
                "target_index",
                "target_letter",
                "positive_mask",
                "positive_indices",
                "target_distribution",
                "norm_syndrome",
            ):
                if forbidden in messages[1]["content"]:
                    raise ValueError(f"Supervision key leaked into Jeff prompt: {forbidden}")
            observed_rows[len(positives)] += 1
            positions[position] += 1
            for i in positives:
                position_masks[i] += 1
            total += 1

    if total != len(cases):
        raise ValueError(f"Expected 43,180 training questions, found {total}")
    return {
        "total_verified_questions": total,
        "total_original_train_cases": len(cases),
        "original_train_cases_removed": 0,
        "observed_positive_counts_by_original_row": {
            str(m): observed_rows[m] for m in sorted(observed_rows)
        },
        "original_rows_with_more_than_one_observed_gold": changed_from_base,
        "observed_positive_slot_counts": {str(i): position_masks[i] for i in range(16)},
        "original_gold_position_counts": {str(i): positions[i] for i in range(16)},
        "all_original_observed_labels_present": True,
        "same_candidates_for_s2_s3_s4": True,
        "train_gold_not_in_jeff_prompt": True,
        "dev_test_loaded": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify every multi-positive Train K=16 row")
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()
    train = load_records(root / "data/tcm-sd/train.json")
    report = json.loads(
        (root / "experiments/bm25_train_text_grouped_2026-11-11.json").read_text(encoding="utf-8")
    )
    if report["fold_policy"] != "exact_text_grouped":
        raise ValueError("Expected Group-aware Train OOF")
    ranks = load_and_validate_oof(
        train, root / "data/candidates/bm25_train_text_grouped.jsonl", report
    )
    target = root / f"data/jeff/train_oof_multipos_k16_epoch{args.epoch:02d}.jsonl"
    summary = verify(train, ranks, target)
    if summary["original_rows_with_more_than_one_observed_gold"] != 820:
        raise ValueError("The 820 original rows with multiple observed labels must remain")

    dest = root / f"experiments/jeff_multipos_k16_verified_epoch{args.epoch:02d}_2026-11-12.json"
    result = {
        "date": "2026-11-12",
        "status": "ALL_MULTIPOSITIVE_TRAIN_ROWS_VALIDATED",
        "model_training": "NOT_RUN",
        "method": "S2 Masked CE / S3 Set Loss / S4 Soft CE shared 16-option data",
        "original_official_split": "Train only",
        **summary,
    }
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(dest), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
