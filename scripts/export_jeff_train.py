"""Export supervised Jeff/SemIf training examples from Train-only OOF BM25.

No Dev/Test use. Every official Train row is retained. Gold is in exactly
one *option* and in external target_index, never in evidence. Jeff uses
a 16-slot head, so supervised single-pass examples use K=2..16.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random

from tcmdiscern.datasets import load_records
from tcmdiscern.jeff_config import PRIMARY_JEFF_K
from tcmdiscern.jeff_format import MAX_DIRECT_OPTIONS, train_example
from tcmdiscern.oof import assign_stratified_folds

TRAIN_ROWS = 43180


def make_train_example(
    case: dict,
    *,
    train_index: int,
    candidates: list[str],
    gold: str,
    k: int,
    epoch: int,
    seed: int,
) -> dict:
    """Generate a repeatable epoch-specific shuffle without touching global RNG."""
    negatives = [candidate for candidate in candidates if candidate != gold]
    if len(negatives) < k - 1:
        raise ValueError(f"Row {train_index}: insufficient mined negative labels")
    rng = Random(seed + epoch * 1_000_003 + train_index * 131 + k * 10_007)
    return train_example(
        case,
        case_id=f"train-{train_index:06d}",
        gold_label=gold,
        hard_negatives=negatives,
        k=k,
        rng=rng,
    )


def load_and_validate_oof(cases: list[dict], source: Path, report: dict) -> list[list[str]]:
    """Check all rows, labels, 50 unique candidates, and their original OOF fold."""
    if len(cases) != TRAIN_ROWS:
        raise ValueError("Expected the full 43,180 original Train cases")
    if report.get("status") != "TRAIN_ONLY_OOF_BM25_COMPLETE":
        raise ValueError("OOF retrieval report is missing or incomplete")
    if report.get("total_train_rows") != TRAIN_ROWS or report.get("excluded_records") != 0:
        raise ValueError("OOF candidates do not follow the frozen original split")
    if report.get("max_candidates_per_row") != 50 or report.get("folds") != 5:
        raise ValueError("Expected 5 folds of 50-label BM25 candidates")

    labels = [str(row["norm_syndrome"]) for row in cases]
    assignments = assign_stratified_folds(labels, folds=report["folds"], seed=report["seed"])
    seen = []
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= TRAIN_ROWS:
                raise ValueError("OOF file contains more than 43,180 rows")
            row = json.loads(line)
            if row.get("case_id") != f"train-{index:06d}":
                raise ValueError(f"OOF row ID/order mismatch at index {index}")
            if row.get("gold_labels") != [labels[index]]:
                raise ValueError(f"OOF gold-label mismatch at index {index}")
            if row.get("fold") != assignments[index]:
                raise ValueError(f"OOF fold mismatch at index {index}")
            options = row.get("candidate_labels")
            if not isinstance(options, list) or len(options) != 50:
                raise ValueError(f"OOF row {index} must have 50 candidate labels")
            if len(set(options)) != 50 or not all(isinstance(x, str) and x for x in options):
                raise ValueError(f"OOF row {index} has duplicate or empty candidates")
            seen.append(options)
    if len(seen) != TRAIN_ROWS:
        raise ValueError(f"OOF file contains {len(seen)} rows, expected {TRAIN_ROWS}")
    return seen


def write_examples(
    cases: list[dict],
    ranks: list[list[str]],
    *,
    path: Path,
    k: int | None,
    epoch: int,
    seed: int,
) -> dict:
    """Write full 43,180-row training output without overwriting older epochs."""
    if path.exists():
        raise FileExistsError(f"Frozen training epoch output already exists: {path}")
    distribution: Counter[int] = Counter()
    positions: Counter[int] = Counter()
    with path.open("x", encoding="utf-8") as stream:
        for index, (case, candidates) in enumerate(zip(cases, ranks, strict=True)):
            if k is None:
                # Per-epoch randomized K mix within Jeff's direct 2..16 slots.
                mix_rng = Random(seed + epoch * 2_000_003 + index * 104729)
                sample_k = mix_rng.randint(2, MAX_DIRECT_OPTIONS)
            else:
                sample_k = k
            gold = str(case["norm_syndrome"])
            example = make_train_example(
                case,
                train_index=index,
                candidates=candidates,
                gold=gold,
                k=sample_k,
                epoch=epoch,
                seed=seed,
            )
            options = example["row"]["options"]
            position = example["target_index"]
            if options[position]["id"] != gold or len(options) != sample_k:
                raise ValueError(f"Invalid training target/options at original row {index}")
            if gold in [option["id"] for j, option in enumerate(options) if j != position]:
                raise ValueError("A gold syndrome appears more than once in candidates")
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
            distribution[sample_k] += 1
            positions[position] += 1
    if sum(distribution.values()) != TRAIN_ROWS:
        raise RuntimeError("Training exporter did not retain all original Train records")
    return {
        "file": f"data/jeff/{path.name}",
        "cases": TRAIN_ROWS,
        "k_counts": {str(i): distribution[i] for i in sorted(distribution)},
        "answer_position_counts_zero_based": {str(i): positions[i] for i in sorted(positions)},
        "epoch": epoch,
        "cases_removed": 0,
        "training_only": True,
        "gold_label_field_excluded_from_evidence": True,
        "source_clinical_text_preserved_unmodified": True,
        "gold_inserted_as_one_training_option": True,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export full Train into Jeff 2..16-choice examples"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=root / "data/candidates/bm25_train_oof.jsonl",
    )
    parser.add_argument(
        "--oof-report",
        type=Path,
        default=root / "experiments/bm25_train_oof_2026-11-11.json",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[PRIMARY_JEFF_K],
        help="Direct Jeff training K values (default: primary K=16)",
    )
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Also export an optional mixed-K=2..16 ablation set",
    )
    args = parser.parse_args()
    if args.epoch < 0:
        parser.error("--epoch must be >=0")
    if not args.ks or len(args.ks) != len(set(args.ks)):
        parser.error("--ks must contain distinct K values")
    if any(not 2 <= k <= MAX_DIRECT_OPTIONS for k in args.ks):
        parser.error("Jeff direct training supports only K=2..16")

    cases = load_records(root / "data/tcm-sd/train.json")
    report = json.loads(args.oof_report.read_text(encoding="utf-8"))
    ranks = load_and_validate_oof(cases, args.candidates, report)
    output_dir = root / "data/jeff"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [f"train_oof_k{k:02d}_epoch{args.epoch:02d}.jsonl" for k in args.ks]
    if args.mixed:
        names.append(f"train_oof_mixed_k02_16_epoch{args.epoch:02d}.jsonl")
    report_path = (
        root / "experiments" / f"jeff_train_examples_epoch{args.epoch:02d}_2026-11-11.json"
    )
    if report_path.exists() or any((output_dir / name).exists() for name in names):
        parser.error("Epoch output already exists. Choose a new --epoch; do not overwrite.")
    summaries = []
    for k in args.ks:
        dest = output_dir / f"train_oof_k{k:02d}_epoch{args.epoch:02d}.jsonl"
        summary = write_examples(cases, ranks, path=dest, k=k, epoch=args.epoch, seed=args.seed)
        summaries.append(summary)
        print(f"Exported K={k}: {summary['cases']} Train rows -> {dest}", flush=True)
    if args.mixed:
        dest = output_dir / f"train_oof_mixed_k02_16_epoch{args.epoch:02d}.jsonl"
        summary = write_examples(cases, ranks, path=dest, k=None, epoch=args.epoch, seed=args.seed)
        summaries.append(summary)
        print(f"Exported mixed K=2..16: {summary['cases']} Train rows -> {dest}", flush=True)

    aggregate = {
        "study": "TCMDiscern",
        "date": "2026-11-11",
        "status": "JEFF_TRAIN_ROWS_EXPORTED_NO_MODEL_TRAINING",
        "official_split": "TCM-SD Train only",
        "original_train_count": TRAIN_ROWS,
        "removed_train_rows": 0,
        "oof_folds": report["folds"],
        "oof_seed": report["seed"],
        "shuffling_epoch": args.epoch,
        "shuffling_seed": args.seed,
        "prompt_version": "Jeff direct-options-v1",
        "training_row_format": "row + target_index + target_letter (target excluded from prompt)",
        "raw_train_cases_containing_literal_gold_syndrome": sum(
            str(row["norm_syndrome"])
            in " ".join(
                str(row.get(field) or "")
                for field in ("chief_complaint", "description", "detection")
            )
            for row in cases
        ),
        "training_options": "1 gold + K-1 ranked Train-OOF BM25 hard negatives",
        "index_self_leakage": False,
        "dev_test_loaded": False,
        "model_training": "NOT_RUN",
        "files": summaries,
        "notes": [
            "Gold is inserted only for supervised Train multiple-choice examples.",
            "Model inference on original Dev/Test must never insert gold into candidates.",
            "Repeated patients/texts from other folds are retained by the frozen data policy.",
            "A new epoch should regenerate candidate order and vary K for mixed training.",
            "K 17..35 uses Jeff's published wide-group inference; direct head has 16 slots.",
            "Published example rows and hard negatives contain clinical data and stay in git-ignored data/.",
        ],
    }
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(aggregate, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "number_of_exports": len(summaries),
                "cases_per_export": TRAIN_ROWS,
                "train_rows_removed": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
