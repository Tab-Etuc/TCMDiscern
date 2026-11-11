"""Validate the active Jeff K=16 Train/Dev data against original BM25.

All record-level patient data remain in git-ignored data/. Only aggregate
statistics are written to experiments/. No model inference is performed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from scripts.verify_jeff_train import validate_file
from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_config import PRIMARY_JEFF_K
from tcmdiscern.jeff_format import decision_row, semif_messages


def validate_primary_k(root: Path, *, epoch: int = 1) -> dict:
    """Check the full 43,180 Train and 5,486 Dev original records."""
    k = PRIMARY_JEFF_K
    if k != 16:
        raise ValueError("This frozen run is the K=16 Jeff direct-head configuration")
    train = load_records(root / "data/tcm-sd/train.json")
    oof_report = json.loads(
        (root / "experiments/bm25_train_oof_2026-11-11.json").read_text(encoding="utf-8")
    )
    ranks = load_and_validate_oof(train, root / "data/candidates/bm25_train_oof.jsonl", oof_report)
    train_path = root / "data/jeff" / f"train_oof_k16_epoch{epoch:02d}.jsonl"
    train_validation = validate_file(train, ranks, train_path, expected_k=k)

    train_report = json.loads(
        (root / "experiments" / f"jeff_train_examples_epoch{epoch:02d}_2026-11-11.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        train_report["original_train_count"] != 43180
        or len(train_report["files"]) != 1
        or train_report["files"][0]["k_counts"] != {"16": 43180}
        or train_report["removed_train_rows"] != 0
        or train_report["model_training"] != "NOT_RUN"
    ):
        raise ValueError("Main training epoch must contain K=16 only, 43,180 rows")
    if train_validation["cases"] != 43180 or train_validation["k_distribution"] != {"16": 43180}:
        raise ValueError("All 43,180 Train examples must be Jeff K=16")

    dev = load_records(root / "data/tcm-sd/dev.json")
    retrieved = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(dev) != 5486 or len(retrieved) != 5486:
        raise ValueError("Dev must be the full original 5,486 cases")
    candidates = {label for case in dev for label in [str(case["norm_syndrome"])]}
    support = Counter(str(case["norm_syndrome"]) for case in dev)
    if len(candidates) != 148:
        raise ValueError("The original Dev 148-class task must be unchanged")

    coverage = Counter()
    found = 0
    rows_read = 0
    dev_path = root / "data/jeff/dev_k16.jsonl"
    with dev_path.open(encoding="utf-8") as stream:
        for index, raw in enumerate(stream):
            rows_read += 1
            if index >= len(dev):
                raise ValueError("Dev output has extra rows")
            recorded = json.loads(raw)
            expected_id = f"dev-{index:06d}"
            gold = str(dev[index]["norm_syndrome"])
            rank = retrieved[index]
            if rank.case_id != expected_id or rank.gold_labels != (gold,):
                raise ValueError("Original Dev and BM25 candidate labels are misaligned")
            expected = decision_row(
                dev[index],
                case_id=expected_id,
                candidates=rank.candidate_labels[:k],
            )
            if recorded != expected:
                raise ValueError(f"Jeff K=16 Dev row differs from BM25 at {index}")
            messages = semif_messages(recorded)
            if "target_index" in messages[1]["content"]:
                raise ValueError("Dev gold target was included in the Jeff prompt")
            if gold in rank.candidate_labels[:k]:
                found += 1
                coverage[gold] += 1
    if rows_read != len(dev):
        raise ValueError("Dev output is incomplete")

    micro = found / len(dev)
    macro = sum(coverage[label] / n for label, n in support.items()) / len(support)
    old = json.loads(
        (root / "experiments/bm25_k1_35_dev_2026-11-11.json").read_text(encoding="utf-8")
    )["results"][k - 1]
    if (
        abs(micro - old["micro_recall"]) > 1e-12
        or abs(macro - old["macro_recall"]) > 1e-12
        or found != old["candidate_hit_n"]
    ):
        raise ValueError("Jeff Dev candidate coverage differs from frozen BM25 baseline")

    return {
        "study": "TCMDiscern",
        "status": "PRIMARY_JEFF_K16_DATA_VALIDATED_NO_MODEL_TRAINING",
        "date": "2026-11-11",
        "model": "jgeuter/Jeff-1.1-Large",
        "primary_k": k,
        "target_slots": 16,
        "requires_grouped_decision": False,
        "forward_passes_per_decision": 1,
        "train_epoch": epoch,
        "train_rows_verified": train_validation["cases"],
        "train_k_distribution": train_validation["k_distribution"],
        "train_answer_positions": train_validation["gold_position_distribution"],
        "train_original_source_gold_literal_count": train_validation[
            "original_clinical_text_containing_gold_literal"
        ],
        "train_original_rows_removed": 0,
        "dev_rows_verified": len(dev),
        "dev_candidates_containing_gold_n": found,
        "dev_micro_recall_at_k": micro,
        "dev_macro_recall_at_k": macro,
        "dev_gold_injected_into_candidates": False,
        "model_checkpoint_trained": False,
        "dev_prediction_accuracy_not_measured": True,
        "other_k_results_preserved": True,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate Jeff primary-K=16 Train/Dev records")
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()
    if args.epoch < 0:
        parser.error("epoch cannot be negative")
    report = validate_primary_k(project_root, epoch=args.epoch)
    out = project_root / "experiments/jeff_primary_k16_2026-11-11.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
