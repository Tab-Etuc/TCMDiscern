"""Aggregate prioritized H200 experiments into a reproducible paper-facing table.

Scores are original TCM-SD Dev; the official Test remains sealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/h200_priority_experiments_summary.json"


def collect(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    from scripts.run_h200_priority_experiments import EXPERIMENTS

    baseline_files = (
        ("bm25", "experiments/bm25_dev_official_2026-11-11.json"),
        ("s1_original", "experiments/jeff_s1-seed20261112-e2-epoch2_k16_dev.json"),
        ("s2_original", "experiments/jeff_s2-seed20261112-e2-epoch2_k16_dev.json"),
        ("s3_original", "experiments/jeff_s3-seed20261112-e2-epoch2_k16_dev.json"),
    )
    training_results = []
    for item in EXPERIMENTS:
        run_id = item.run_id
        folder = root / "data/runs" / run_id
        original = {
            "priority": item.priority,
            "run_id": run_id,
            "components": item.components,
            "negative_sampling": item.negatives,
            "seed": item.seed,
        }
        training = folder / "summary.json"
        if training.is_file():
            summary = json.loads(training.read_text(encoding="utf-8"))
            original["training"] = {
                "global_updates": summary.get("global_update"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "peak_cuda_bytes": summary.get("peak_cuda_bytes"),
                "input_tokens": summary.get("input_tokens"),
                "loss": summary.get("mean_training_loss"),
            }
        evaluations = {}
        for epoch in (1, 2):
            report = root / "experiments" / f"jeff_{run_id}-epoch{epoch}_k16_dev.json"
            if report.is_file():
                details = json.loads(report.read_text(encoding="utf-8"))
                if details.get("kind") != "FULL_DEV_FINETUNED" or details.get("rows") != 5486:
                    raise ValueError(f"Incomplete evaluation: {report}")
                evaluations[f"epoch{epoch}"] = {
                    "top1_accuracy": details["scores"]["top1_accuracy"],
                    "macro_f1": details["scores"]["macro_f1"],
                    "mean_forward_seconds": details.get("mean_wall_seconds_per_decision"),
                    "tokens": details.get("tokens"),
                    "report": str(report.relative_to(root)),
                }
        original["dev"] = evaluations
        training_results.append(original)
    cpu_path = root / "experiments/h200_priority_cpu_ablation.json"
    cpu = json.loads(cpu_path.read_text(encoding="utf-8")) if cpu_path.is_file() else None
    result = {
        "status": "H200_PRIORITY_AGGREGATE",
        "official_test_used": False,
        "dataset": {"train": 43180, "dev": 5486, "classes": 148},
        "original_dev_reports": {
            name: (root / relative).is_file() for name, relative in baseline_files
        },
        "cpu_ablation": (
            {"methods": cpu["methods"], "budget_curve": cpu["budget_curve"]}
            if cpu else None
        ),
        "gpu_training_and_dev": training_results,
        "selective_gpu_latency": (
            json.loads((root / "experiments/h200_priority_real_gpu_latency.json").read_text(encoding="utf-8"))
            if (root / "experiments/h200_priority_real_gpu_latency.json").is_file() else None
        ),
        "complete_gpu_runs": sum(bool(t["dev"].get("epoch2")) for t in training_results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"H200_PRIORITY_SUMMARY_WRITTEN completed={result['complete_gpu_runs']}/{len(training_results)}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    collect(output=args.output)
