"""Score all K=1..35 from a real Gemma/LLM prediction JSONL file.

The LLM runner is responsible for producing K-specific predictions. This
script enforces that all 35 * 5486 predictions exist and that every answer
is among the first K retrieved labels. No automatic backfill or substitution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions


def score_predictions(
    predictions: list[dict],
    *,
    labels: list[str],
    candidates: list[tuple[str, ...]],
    classes: list[str],
    max_k: int,
) -> list[dict]:
    """Validate one prediction per (K, Dev row), then compute aggregate metrics."""
    if len(labels) != len(candidates):
        raise ValueError("Labels and candidate rows must be aligned")
    if not 1 <= max_k <= 35:
        raise ValueError("K must be within 1..35")
    n = len(labels)
    if not n or any(len(row) < max_k for row in candidates):
        raise ValueError("Missing Dev candidates")

    output: dict[tuple[int, int], tuple[str, float | None]] = {}
    for line, row in enumerate(predictions, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Line {line}: expected JSON object")
        k = row.get("k")
        case_id = row.get("case_id")
        label = row.get("predicted_label")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > max_k:
            raise ValueError(f"Line {line}: K out of range")
        if not isinstance(case_id, str) or not case_id.startswith("dev-"):
            raise ValueError(f"Line {line}: case_id format invalid")
        try:
            index = int(case_id[4:])
        except ValueError as exc:
            raise ValueError(f"Line {line}: case_id is not numeric") from exc
        if case_id != f"dev-{index:06d}" or not 0 <= index < n:
            raise ValueError(f"Line {line}: invalid Dev case_id")
        if not isinstance(label, str) or label not in candidates[index][:k]:
            raise ValueError(f"Line {line}: prediction is outside the K candidates")
        key = (k, index)
        if key in output:
            raise ValueError(f"Line {line}: duplicate (k, case_id) prediction")
        latency = row.get("latency_ms")
        if latency is not None and (
            not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not np.isfinite(latency)
            or latency < 0
        ):
            raise ValueError(f"Line {line}: invalid latency")
        output[key] = (label, float(latency) if latency is not None else None)

    if len(output) != max_k * n:
        missing = max_k * n - len(output)
        raise ValueError(f"Incomplete K grid: missing {missing} predictions")

    all_classes = sorted(classes)
    if len(set(all_classes)) != len(all_classes):
        raise ValueError("Duplicate class names")
    if not set(labels).issubset(all_classes):
        raise ValueError("Dev gold not in classification vocabulary")
    results: list[dict] = []
    for k in range(1, max_k + 1):
        chosen = [output[(k, i)][0] for i in range(n)]
        covered = sum(labels[i] in candidates[i][:k] for i in range(n))
        correct = sum(prediction == labels[i] for i, prediction in enumerate(chosen))
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            chosen,
            labels=all_classes,
            zero_division=0,
        )
        latencies = [v for i in range(n) if (v := output[(k, i)][1]) is not None]
        results.append(
            {
                "k": k,
                "cases": n,
                "gold_covered_n": covered,
                "oracle_recall": covered / n,
                "correct_n": correct,
                "top1": correct / n,
                "macro_f1": float(np.mean(f1)),
                "conditional_accuracy": correct / covered if covered else None,
                "latency_ms": {
                    "reported_n": len(latencies),
                    "mean": mean(latencies) if latencies else None,
                    "median": median(latencies) if latencies else None,
                    "p95": float(np.percentile(latencies, 95)) if latencies else None,
                },
                "per_class": [
                    {
                        "label": name,
                        "support": int(support[i]),
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1": float(f1[i]),
                    }
                    for i, name in enumerate(all_classes)
                ],
            }
        )
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate K-specific Gemma predictions on Dev")
    parser.add_argument("predictions", type=Path, help="JSONL from real K-specific inference")
    parser.add_argument("--model", required=True, help="Exact model/checkpoint identifier")
    parser.add_argument("--method", required=True, choices=("listwise", "pointwise", "other"))
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments/k1_35_gemma_dev_results.json",
    )
    args = parser.parse_args()
    dev = load_records(root / "data/tcm-sd/dev.json")
    cached = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(dev) != len(cached) or len(dev) != 5486:
        raise ValueError("Official full Dev split is required")
    labels = [str(row["norm_syndrome"]) for row in dev]
    candidates = []
    for i, saved in enumerate(cached):
        if saved.case_id != f"dev-{i:06d}" or saved.gold_labels != (labels[i],):
            raise ValueError("Cached candidate/gold mismatch")
        candidates.append(saved.candidate_labels)

    with args.predictions.open(encoding="utf-8") as stream:
        outputs = [json.loads(line) for line in stream if line.strip()]
    classes = sorted(
        {
            line.strip()
            for line in (root / "data/tcm-sd/syndrome_vocab.txt")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        }
    )
    if len(classes) != 148:
        raise ValueError("Expected 148 classes")
    results = score_predictions(
        outputs,
        labels=labels,
        candidates=candidates,
        classes=classes,
        max_k=35,
    )
    summary = {
        "model": args.model,
        "rerank_method": args.method,
        "status": "EVALUATED_35_K_VALUES_FROM_EXTERNAL_MODEL_PREDICTIONS",
        "dataset": "TCM-SD official original Dev, 5486 cases",
        "excluded_cases": 0,
        "test_evaluation": "NOT_RUN",
        "k_values": list(range(1, 36)),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    by_top1 = max(results, key=lambda x: (x["top1"], -x["k"]))
    by_macro = max(results, key=lambda x: (x["macro_f1"], -x["k"]))
    print(f"All 35 K values measured on {len(dev)} cases: {args.model}")
    print(f"Best Dev Top-1: K={by_top1['k']}, {by_top1['top1']:.4%}")
    print(f"Best Dev Macro-F1: K={by_macro['k']}, {by_macro['macro_f1']:.4%}")
    print(f"Saved aggregate results: {args.output}")


if __name__ == "__main__":
    main()
