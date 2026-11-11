"""Full aggregate OOF BM25 Recall@1..50, class and fold breakdown.

Reuses the *already generated* 43,180 Train OOF candidates; never
reruns BM25 or reads Dev/Test; patient-level JSONL stays under data/.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from tcmdiscern.datasets import load_records
from tcmdiscern.oof import assign_stratified_folds

MAX_K = 50


def summarize(labels: list[str], candidates: list[list[str]], fold_by_row: tuple[int, ...]) -> dict:
    if len(labels) != len(candidates) or len(labels) != len(fold_by_row):
        raise ValueError("Train labels, OOF candidates and fold membership must align")
    if len(set(labels)) != 148:
        raise ValueError("Expected 148 original Train syndrome classes")
    if any(len(options) != MAX_K or len(set(options)) != MAX_K for options in candidates):
        raise ValueError("All OOF rankings must contain 50 distinct labels")

    positions: list[int] = []
    by_class: dict[str, list[int]] = defaultdict(list)
    by_fold: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for gold, options, fold in zip(labels, candidates, fold_by_row, strict=True):
        try:
            rank = options.index(gold) + 1
        except ValueError:
            rank = MAX_K + 1
        positions.append(rank)
        by_class[gold].append(rank)
        by_fold[fold][gold].append(rank)

    def coverage(ranks: list[int], k: int) -> float:
        return sum(r <= k for r in ranks) / len(ranks)

    all_classes = sorted(by_class)
    curve: list[dict] = []
    for k in range(1, MAX_K + 1):
        found = sum(r <= k for r in positions)
        curve.append(
            {
                "k": k,
                "retrieved_gold_n": found,
                "new_hits_at_k": sum(r == k for r in positions),
                "not_in_top_k_n": len(positions) - found,
                "micro_recall": found / len(positions),
                "macro_recall": sum(coverage(by_class[label], k) for label in all_classes)
                / len(all_classes),
            }
        )

    fold_details = []
    for fold, groups in sorted(by_fold.items()):
        fold_cases = sum(len(values) for values in groups.values())
        if len(groups) != 148:
            raise ValueError(f"Fold {fold} has missing syndrome label")
        fold_curve = []
        for k in range(1, MAX_K + 1):
            hits = sum(sum(rank <= k for rank in values) for values in groups.values())
            fold_curve.append(
                {
                    "k": k,
                    "correct_candidate_cases": hits,
                    "micro_recall": hits / fold_cases,
                    "macro_recall": sum(coverage(groups[label], k) for label in all_classes)
                    / len(all_classes),
                }
            )
        fold_details.append(
            {
                "fold": fold,
                "query_rows": fold_cases,
                "labels": len(groups),
                "recall_curve": fold_curve,
            }
        )

    per_label = [
        {
            "label": label,
            "train_support": len(by_class[label]),
            "first_hit_counts": {
                str(k): sum(rank == k for rank in by_class[label]) for k in range(1, MAX_K + 2)
            },
            "recall_at_selected_k": {
                str(k): coverage(by_class[label], k) for k in (1, 3, 5, 10, 16, 17, 35, 50)
            },
        }
        for label in all_classes
    ]

    return {
        "dataset": "TCM-SD",
        "experiment": "Train OOF case-BM25 syndrome candidate coverage",
        "date": "2026-11-11",
        "status": "COMPLETED_TRAIN_ONLY_5FOLD",
        "official_train_rows": len(labels),
        "original_split_retained": True,
        "excluded_or_deduplicated_rows": 0,
        "dev_test_used": False,
        "max_k": MAX_K,
        "classes": 148,
        "recall_all_k": curve,
        "folds": fold_details,
        "first_hit_rank_distribution": {str(k): Counter(positions)[k] for k in range(1, MAX_K + 2)},
        "per_syndrome": per_label,
        "interpretation": [
            "Train OOF candidate recall is not a final Jeff model score.",
            "Duplicate clinical texts can remain across Train folds by user decision.",
            "Each query row is excluded from its own BM25 index.",
            "Gold labels were not inserted into OOF retrieval candidate ranking.",
            "148 syndrome labels appear in each of the five held-out folds.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = load_records(root / "data/tcm-sd/train.json")
    earlier = json.loads(
        (root / "experiments/bm25_train_oof_2026-11-11.json").read_text(encoding="utf-8")
    )
    ranks = load_and_validate_oof(cases, root / "data/candidates/bm25_train_oof.jsonl", earlier)
    golds = [str(row["norm_syndrome"]) for row in cases]
    assignments = assign_stratified_folds(golds, folds=earlier["folds"], seed=earlier["seed"])
    result = summarize(golds, ranks, assignments)
    for k in earlier["bm25_oof_candidate_micro_recall"]:
        value = result["recall_all_k"][int(k) - 1]
        if abs(value["micro_recall"] - earlier["bm25_oof_candidate_micro_recall"][k]) > 1e-12:
            raise ValueError(f"Micro OOF Recall@{k} differs from first run")
        if abs(value["macro_recall"] - earlier["bm25_oof_candidate_macro_recall"][k]) > 1e-12:
            raise ValueError(f"Macro OOF Recall@{k} differs from first run")

    path = root / "experiments/bm25_train_oof_all_k_2026-11-11.json"
    csv_path = root / "experiments/bm25_train_oof_k1_35_2026-11-11.csv"
    if path.exists() or csv_path.exists():
        raise FileExistsError("A frozen OOF statistical snapshot already exists")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "k",
            "retrieved_gold_n",
            "new_hits_at_k",
            "not_in_top_k_n",
            "micro_recall",
            "macro_recall",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["recall_all_k"][:35])
    print(
        json.dumps(
            {
                "all_k_report": str(path),
                "csv_1_to_35": str(csv_path),
                "train_cases": len(cases),
                "folds": len(result["folds"]),
                "per_syndrome": len(result["per_syndrome"]),
                "k5_micro": result["recall_all_k"][4]["micro_recall"],
                "k17_macro": result["recall_all_k"][16]["macro_recall"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
