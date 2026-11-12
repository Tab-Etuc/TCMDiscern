"""Export verified multi-positive S2/S3/S4 Jeff training examples, Train only.

The same 43,180 rows and K=16 options serve all three losses. No Dev/Test
gold labels are read. Every row retains its original official gold label;
extra observed gold labels come ONLY from byte-equal Train clinical input.

Usage:
  uv run python scripts/export_jeff_multipos.py --epoch 3
  uv run python scripts/verify_jeff_multipos.py --epoch 3
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_config import PRIMARY_JEFF_K
from tcmdiscern.jeff_multi import build_multi_example, observed_label_sets


def export(
    cases: list[dict],
    ranks: list[list[str]],
    *,
    path: Path,
    epoch: int,
    seed: int,
) -> dict:
    """Safely publish full training JSONL without overwriting previous experiments."""
    if path.exists():
        raise FileExistsError(f"Frozen training set already exists: {path}")
    if len(cases) != 43180 or len(ranks) != len(cases):
        raise ValueError("Multi-positive export requires the complete official Train OOF")
    label_sets = observed_label_sets(cases)
    by_m: Counter[int] = Counter()
    by_position: Counter[int] = Counter()
    groups: Counter[int] = Counter(len(labels) for labels in label_sets.values())

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-jeff-multipos-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for index, (case, rank) in enumerate(zip(cases, ranks, strict=True)):
                positive = label_sets[clinical_text(case)]
                example = build_multi_example(
                    case,
                    case_id=f"train-{index:06d}",
                    case_index=index,
                    observed_labels=positive,
                    ranked_candidates=rank,
                    epoch=epoch,
                    seed=seed,
                )
                m = len(positive)
                by_m[m] += 1
                by_position[example["target_index"]] += 1
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        # Hard-link creation fails if destination already exists; publish atomically.
        os.link(temp_path, path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return {
        "file": f"data/jeff/{path.name}",
        "rows": sum(by_m.values()),
        "original_rows_removed": 0,
        "k": PRIMARY_JEFF_K,
        "observed_positive_counts_by_original_row": {str(m): by_m[m] for m in sorted(by_m)},
        "distinct_text_groups_by_number_of_observed_labels": {
            str(m): groups[m] for m in sorted(groups)
        },
        "original_gold_position_histogram": {str(i): by_position[i] for i in range(PRIMARY_JEFF_K)},
        "target_distribution_normalized_each_row": True,
        "same_candidates_for_s2_s3_s4": True,
        "original_official_label_preserved": True,
        "each_observed_label_included_once": True,
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export 43,180 Train-only K16 multi-positive examples"
    )
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if args.epoch < 0:
        parser.error("--epoch must be >= 0")

    candidates = project / "data/candidates/bm25_train_text_grouped.jsonl"
    oof_report = json.loads(
        (project / "experiments/bm25_train_text_grouped_2026-11-11.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        oof_report.get("fold_policy") != "exact_text_grouped"
        or oof_report["cross_fold_exact_input_text"][
            "query_rows_with_identical_text_in_another_fold"
        ]
        != 0
    ):
        raise ValueError("Requires Group-aware OOF BM25 with zero exact-text cross-fold overlap")

    cases = load_records(project / "data/tcm-sd/train.json")
    if len(cases) != 43180:
        raise ValueError("Expected 43,180 original Train rows")
    ranks = load_and_validate_oof(cases, candidates, oof_report)

    dest = project / f"data/jeff/train_oof_multipos_k16_epoch{args.epoch:02d}.jsonl"
    report_path = project / f"experiments/jeff_multipos_k16_epoch{args.epoch:02d}_2026-11-12.json"
    if dest.exists() or report_path.exists():
        parser.error("Output already exists. Select an unused epoch; do not overwrite.")
    stats = export(cases, ranks, path=dest, epoch=args.epoch, seed=args.seed)
    report = {
        "study": "TCMDiscern",
        "date": "2026-11-12",
        "status": "MULTIPOSITIVE_K16_TRAIN_ROWS_EXPORTED_NO_MODEL_TRAINING",
        "official_split": "Train only",
        "source": "TCM-SD 43,180 original Train rows",
        "oof_policy": "exact_text_grouped",
        "oof_candidate_source": candidates.name,
        "base_model": "jgeuter/Jeff-1.1-Large",
        "model_training_completed": False,
        "all_original_train_rows_preserved": True,
        "train_only_observed_label_sets": True,
        "dev_test_labels_used": False,
        "epoch": args.epoch,
        "seed": args.seed,
        "prompt": "Jeff/SemIf (state,question,options); targets remain outside prompt",
        "supervision": [
            "S2 Masked CE (original target_index and observed positive_indices)",
            "S3 Set Loss (observed positive_indices)",
            "S4 Uniform Soft-target CE (target_distribution)",
        ],
        "data": stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": report_path.name, **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
