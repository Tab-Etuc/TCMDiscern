"""Run a train-only BM25 retrieval baseline and binary-search Recall@K on Dev.

Only reads TCM-SD train.json and dev.json. Test is deliberately unavailable.
Saved reports and candidates are under git-ignored data/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.k_search import first_hit_rank, minimum_k, minimum_k_by_metric, recall_at_k
from tcmdiscern.retrieval import CaseBM25


def summarize_ranks(ranks: list[int], labels: list[str], max_k: int, targets: list[float]) -> dict:
    if not ranks or len(ranks) != len(labels):
        raise ValueError("Nonempty, aligned ranks and labels are required")
    ordered = sorted(ranks)
    groups: dict[str, list[int]] = defaultdict(list)
    for label, rank in zip(labels, ranks, strict=True):
        groups[label].append(rank)
    groups = {label: sorted(values) for label, values in groups.items()}
    macro = {
        str(k): sum(recall_at_k(values, k) for values in groups.values()) / len(groups)
        for k in range(1, max_k + 1)
    }
    return {
        "count": len(ranks),
        "number_of_gold_labels": len(groups),
        "recall_curve": {str(k): recall_at_k(ordered, k) for k in range(1, max_k + 1)},
        "macro_recall_curve": macro,
        "binary_searches": [minimum_k(ranks, target=target, max_k=max_k) for target in targets],
        "macro_binary_searches": [
            minimum_k_by_metric(lambda k: macro[str(k)], target=target, max_k=max_k)
            for target in targets
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="TCM-SD Dev BM25 automatic K search")
    parser.add_argument("--data", type=Path, default=root / "data/tcm-sd")
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.80, 0.85, 0.90, 0.95])
    parser.add_argument("--limit-dev", type=int, default=None, help="Smoke test only")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/reports/bm25_binary_k_dev.json",
    )
    args = parser.parse_args()

    if args.limit_dev is not None and args.limit_dev < 1:
        parser.error("--limit-dev must be positive")
    if len(set(args.targets)) != len(args.targets):
        parser.error("--targets must not contain duplicates")
    if args.limit_dev is not None and args.output == root / "data/reports/bm25_binary_k_dev.json":
        parser.error("Smoke test requires a distinct --output; official Dev must remain whole")

    train = load_records(args.data / "train.json")
    dev = load_records(args.data / "dev.json")
    if args.limit_dev is not None:
        dev = dev[: args.limit_dev]

    print(f"Training BM25 on {len(train)} cases; evaluating {len(dev)} Dev cases", flush=True)
    start = perf_counter()
    retriever = CaseBM25(
        [clinical_text(record) for record in train],
        [str(record["norm_syndrome"]) for record in train],
    )
    print(f"BM25 index ready, {len(retriever.label_names)} unique labels", flush=True)
    ranked_labels = retriever.rank(
        [clinical_text(record) for record in dev],
        max_k=args.max_k,
        batch_size=args.batch_size,
    )
    elapsed = perf_counter() - start
    ranks = [
        first_hit_rank([str(row["norm_syndrome"])], options, max_k=args.max_k)
        for row, options in zip(dev, ranked_labels, strict=True)
    ]

    # Frozen official benchmark: every original Dev case is evaluated.
    # Duplicate/overlap counts are documented separately by scripts/audit_datasets.py.
    gold_labels = [str(row["norm_syndrome"]) for row in dev]
    report = {
        "dataset": "TCM-SD",
        "evaluated_split": "dev",
        "index_split": "train",
        "method": "case_bm25_cjk_bigram_per_syndrome_max",
        "label_count": len(retriever.label_names),
        "index_size": len(train),
        "dev_subset": args.limit_dev,
        "max_k": args.max_k,
        "retrieval_seconds": round(elapsed, 2),
        "duplicate_handling": "retain_all_original_train_and_dev_rows",
        "selection_policy": {
            "primary_gemma_k": 17,
            "comparison_k": [5, 35],
            "final_gemma_k_to_be_selected_on_dev": True,
        },
        "original_split": summarize_ranks(ranks, gold_labels, args.max_k, args.targets),
        "evaluation_notes": [
            "Dev gold labels are used only for evaluation after retrieval, never in the index.",
            "This is a case-level BM25 baseline, NOT Hybrid or an Embedding result.",
            "All original Train and Dev cases are retained; duplication is audit-only.",
            "Recall@K is monotone. Binary search minimizes K subject to a target, not Top-1 accuracy.",
            "The best Gemma reranking K cannot be inferred from retrieval Recall@K alone.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    candidate_path = root / (
        "data/candidates/bm25_dev.jsonl"
        if args.limit_dev is None
        else "data/candidates/bm25_dev_smoke.jsonl"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as output:
        for i, (row, candidates) in enumerate(zip(dev, ranked_labels, strict=True)):
            output.write(
                json.dumps(
                    {
                        "case_id": f"dev-{i:06d}",
                        "gold_labels": [str(row["norm_syndrome"])],
                        "candidate_labels": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "index_size": len(train),
        "dev_cases": len(dev),
        "recall_at_1": report["original_split"]["recall_curve"]["1"],
        "macro_recall_at_5": report["original_split"]["macro_recall_curve"].get("5"),
        "recall_at_10": report["original_split"]["recall_curve"].get("10"),
        "recall_at_25": report["original_split"]["recall_curve"].get("25"),
        "recall_at_max_k": report["original_split"]["recall_curve"][str(args.max_k)],
        "binary_search": report["original_split"]["binary_searches"],
        "macro_binary_search": report["original_split"]["macro_binary_searches"],
        "seconds": report["retrieval_seconds"],
        "saved_report": str(args.output),
        "saved_candidates": str(candidate_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
