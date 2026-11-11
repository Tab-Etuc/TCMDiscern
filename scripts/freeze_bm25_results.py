"""Export reproducible, aggregate-only evidence from the frozen BM25 Dev run."""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy
import scipy
import sklearn

from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.k_search import first_hit_rank

SELECTED_K = (1, 3, 5, 10, 12, 17, 25, 34, 35, 50)


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ValueError(description)


def export(root: Path) -> dict:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    predictions = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    report = json.loads((root / "data/reports/bm25_binary_k_dev.json").read_text())
    audit = json.loads((root / "data/reports/tcmsd_audit.json").read_text())

    require(
        len(train) == 43180 and len(dev) == len(predictions) == 5486,
        "Official Train/Dev case count mismatch",
    )
    require(
        report["index_split"] == "train" and report["evaluated_split"] == "dev",
        "Only original Train index and Dev evaluation are allowed",
    )
    require(report["max_k"] == 50, "Unexpected candidate pool size")
    require(
        "exact_text_and_user_disjoint_subset" not in report,
        "Regenerate report under the frozen no-dedup protocol",
    )

    train_support = Counter(str(x["norm_syndrome"]) for x in train)
    label_ranks: dict[str, list[int]] = defaultdict(list)
    ranks: list[int] = []
    for i, (row, p) in enumerate(zip(dev, predictions, strict=True)):
        label = str(row["norm_syndrome"])
        require(p.case_id == f"dev-{i:06d}", "Candidate ID/order mismatch")
        require(p.gold_labels == (label,), "Candidate gold does not match official Dev")
        require(
            len(p.candidate_labels) == len(set(p.candidate_labels)) == 50,
            "Candidate list must contain 50 unique labels",
        )
        require(p.predicted_label is None, "Unexpected reranker prediction in BM25 output")
        require(
            set(p.candidate_labels).issubset(train_support),
            "Candidate label is absent from Train",
        )
        rank = first_hit_rank([label], p.candidate_labels, max_k=50)
        label_ranks[label].append(rank)
        ranks.append(rank)
    require(len(label_ranks) == len(train_support) == 148, "Expected 148 classes")
    origin = report["original_split"]
    for k in range(1, 51):
        found = sum(r <= k for r in ranks)
        require(
            abs(found / len(dev) - origin["recall_curve"][str(k)]) < 1e-12,
            f"Micro Recall@{k} mismatch",
        )
        mean = (
            sum(sum(r <= k for r in values) / len(values) for values in label_ranks.values()) / 148
        )
        require(
            abs(mean - origin["macro_recall_curve"][str(k)]) < 1e-12,
            f"Macro Recall@{k} mismatch",
        )

    return {
        "study": "TCMDiscern",
        "date": "2026-11-11",
        "status": "DEV_RETRIEVAL_ONLY",
        "source": "https://github.com/Borororo/ZY-BERT",
        "dataset_license": "CC BY-NC-SA 4.0 (as stated in upstream README)",
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "dependency_lock": "uv.lock",
            "gpu_used": False,
        },
        "official_input_sizes_bytes": {
            "train": (root / "data/tcm-sd/train.json").stat().st_size,
            "dev": (root / "data/tcm-sd/dev.json").stat().st_size,
            "test": (root / "data/tcm-sd/test.json").stat().st_size,
            "syndrome_vocab": (root / "data/tcm-sd/syndrome_vocab.txt").stat().st_size,
            "syndrome_knowledge": (root / "data/tcm-sd/syndrome_knowledge.json").stat().st_size,
        },
        "official_split": {"train": 43180, "dev": 5486, "test": 5486},
        "deduplication": "NONE - original records retained unmodified",
        "excluded_cases": 0,
        "test_model_evaluation": "NOT_RUN",
        "known_overlap_audit": {
            "exact_text": audit["cross_split_text_overlap"],
            "train_patient_reuse": audit["train_patient_id_reuse_in_evaluation_rows"],
        },
        "method": {
            "index": "43,180 original Train cases",
            "query": "5,486 original Dev cases",
            "input_fields": ["chief_complaint", "description", "detection"],
            "excluded_fields": ["norm_syndrome", "syndrome", "user_id", "lcd_id", "lcd_name"],
            "retriever": "BM25 CJK character bigram",
            "k1": 1.5,
            "b": 0.75,
            "min_df": 2,
            "aggregation": "maximum matching Train case score per unique syndrome",
            "ranked_distinct_labels": 50,
            "classes": 148,
            "embedding": None,
            "hybrid": None,
            "gemma_lora": None,
            "elapsed_index_and_dev_seconds": report["retrieval_seconds"],
        },
        "micro": {
            "definition": "all 5,486 Dev cases weighted equally",
            "correct_counts_at_each_k": {str(k): sum(r <= k for r in ranks) for k in range(1, 51)},
            "recall_at_each_k": origin["recall_curve"],
            "binary_search": origin["binary_searches"],
        },
        "macro": {
            "definition": "mean of per-syndrome recall across 148 classes in Dev",
            "recall_at_each_k": origin["macro_recall_curve"],
            "binary_search": origin["macro_binary_searches"],
        },
        "hit_rank_distribution": {str(k): sum(r == k for r in ranks) for k in range(1, 52)},
        "syndrome_aggregates": [
            {
                "name": name,
                "train_n": train_support[name],
                "dev_n": len(values),
                "recall_at_selected_k": {
                    str(k): sum(rank <= k for rank in values) / len(values) for k in SELECTED_K
                },
                "missed_in_top50": sum(rank == 51 for rank in values),
            }
            for name, values in sorted(label_ranks.items())
        ],
        "next_gemma_study": {
            "primary_k": 17,
            "comparison_k": [5, 35],
            "selection": "official Dev Top-1, Macro-F1 and latency",
            "test": "reserve for final one-time evaluation",
        },
        "limitations": [
            "Known duplicate texts and repeated user IDs remain in the official splits.",
            "Micro/Macro Recall are retrieval metrics, not Gemma diagnostic accuracy.",
            "Binary search applies to monotone Recall@K, not reranker Top-1.",
            "No embedding, hybrid or Gemma evaluations have been run here.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Freeze public aggregate-only experiment summary")
    parser.add_argument(
        "--output", type=Path, default=root / "experiments/bm25_dev_official_2026-11-11.json"
    )
    args = parser.parse_args()
    record = export(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dev_cases": 5486,
                "micro_recall_at_17": record["micro"]["recall_at_each_k"]["17"],
                "macro_recall_at_17": record["macro"]["recall_at_each_k"]["17"],
                "excluded_cases": record["excluded_cases"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
