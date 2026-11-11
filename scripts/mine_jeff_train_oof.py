"""Mine hard-negative syndrome candidates with 5-fold Train-only BM25.

Reads TCM-SD train.json ONLY; never opens Dev or Test. Keeps all original
Train rows, including repeated and conflicting-label cases. Writes patient
data exclusively into git-ignored data/ and aggregate metrics to experiments/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_config import PRIMARY_JEFF_K
from tcmdiscern.oof import rank_statistics, rank_train_oof

EXPECTED_TRAIN_ROWS = 43180
EXPECTED_CLASSES = 148


def cross_fold_exact_text_counts(
    texts: list[str], labels: list[str], folds: tuple[int, ...]
) -> dict[str, int]:
    """Audit retained identical texts across Train query/index folds; never delete."""
    # Each text maps to the folds / labels of all rows containing that exact text.
    present: dict[str, dict[int, set[str]]] = {}
    for text, label, fold in zip(texts, labels, folds, strict=True):
        present.setdefault(text, {}).setdefault(fold, set()).add(label)
    repeated = 0
    conflicts = 0
    for text, label, fold in zip(texts, labels, folds, strict=True):
        others = {
            other_label
            for other_fold, other_labels in present[text].items()
            if other_fold != fold
            for other_label in other_labels
        }
        if others:
            repeated += 1
            if label not in others:
                conflicts += 1
    return {
        "query_rows_with_identical_text_in_another_fold": repeated,
        "such_rows_without_same_label_in_other_fold": conflicts,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Mine 5-fold OOF BM25 negatives on official Train")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--fold-policy",
        choices=("stratified", "exact_text_grouped"),
        default="stratified",
        help="Only groups identical original Train texts; no rows are removed",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    if args.fold_policy == "exact_text_grouped":
        args.output = args.output or root / "data/candidates/bm25_train_text_grouped.jsonl"
        args.report = args.report or root / "experiments/bm25_train_text_grouped_2026-11-11.json"
    else:
        args.output = args.output or root / "data/candidates/bm25_train_oof.jsonl"
        args.report = args.report or root / "experiments/bm25_train_oof_2026-11-11.json"
    if not args.output.resolve().is_relative_to((root / "data").resolve()):
        parser.error("OOF candidate files with clinical labels must stay inside data/")
    if not args.report.resolve().is_relative_to((root / "experiments").resolve()):
        parser.error("Aggregate research report must stay inside experiments/")
    if args.output.exists() or args.report.exists():
        parser.error("Output already exists; frozen OOF experiments cannot be overwritten")

    cases = load_records(root / "data/tcm-sd/train.json")
    if len(cases) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} original Train cases")
    labels = [str(case["norm_syndrome"]) for case in cases]
    texts = [clinical_text(case) for case in cases]
    if len(set(labels)) != EXPECTED_CLASSES:
        raise ValueError("Expected all 148 TCM-SD syndrome labels")
    print(
        f"Original Train {len(cases)} rows / {len(set(labels))} classes, "
        f"{args.folds} folds / seed={args.seed}, zero deduplication",
        flush=True,
    )

    oof = rank_train_oof(
        texts,
        labels,
        folds=args.folds,
        seed=args.seed,
        max_k=args.max_k,
        batch_size=args.batch_size,
        fold_policy=args.fold_policy,
        on_fold_done=lambda fold, index_n, query_n, elapsed: print(
            f"OOF fold {fold + 1}/{args.folds}: "
            f"indexed {index_n}, queried {query_n}, {elapsed:.3f}s",
            flush=True,
        ),
    )
    stats = rank_statistics(labels, oof, ks=(1, 3, 5, 10, 16, 17, 35, 50))
    stats.update(
        {
            "study": "TCMDiscern",
            "date": "2026-11-11",
            "status": "TRAIN_ONLY_OOF_BM25_COMPLETE",
            "source_dataset": "official TCM-SD Train",
            "official_splits_unchanged": True,
            "query_fold_count": args.folds,
            "fold_policy": args.fold_policy,
            "seed": args.seed,
            "max_candidates_per_row": args.max_k,
            "retriever": "CaseBM25 CJK bigram; k1=1.5; b=0.75; min_df=2; per-class max",
            "train_used_for_index_only_in_other_folds": True,
            "dev_cases_loaded": 0,
            "test_cases_loaded": 0,
            "gold_label_field_used_to_build_queries": False,
            "original_clinical_text_containing_gold_literal": sum(
                gold in text for gold, text in zip(labels, texts, strict=True)
            ),
            "duplicate_policy": "retain all official Train rows, including repeated texts",
            "cross_fold_exact_input_text": cross_fold_exact_text_counts(
                texts, labels, oof.fold_by_row
            ),
            "excluded_records": 0,
            "primary_training_k": PRIMARY_JEFF_K,
            "optional_ablation_ks": [5],
            "later_eval_ks": list(range(1, 36)),
            "training_checkpoint": "NOT_RUN",
        }
    )
    if stats["total_train_rows"] != EXPECTED_TRAIN_ROWS:
        raise ValueError("OOF did not cover all original Train rows")
    if sum(stats["fold_query_rows"]) != EXPECTED_TRAIN_ROWS:
        raise ValueError("OOF folds do not sum to full Train")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for index, (label, fold, candidates) in enumerate(
            zip(labels, oof.fold_by_row, oof.rankings, strict=True)
        ):
            if len(candidates) != args.max_k or len(set(candidates)) != args.max_k:
                raise ValueError(f"Invalid OOF options for Train row {index}")
            stream.write(
                json.dumps(
                    {
                        "case_id": f"train-{index:06d}",
                        "fold": fold,
                        "gold_labels": [label],
                        "candidate_labels": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(stats, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "report": str(args.report),
                "candidate_rows": len(oof.rankings),
                "fold_times_s": stats["fold_seconds"],
                "oof_micro_recall_at_5": stats["bm25_oof_candidate_micro_recall"]["5"],
                "oof_macro_recall_at_5": stats["bm25_oof_candidate_macro_recall"]["5"],
                "excluded": 0,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
