"""Immutable aggregate-only Dev report and preregistered model selection.

The same 3 predeclared seeds are averaged for each S1/S2/S3 epoch. The
lowest-variance/most favorable seed is never cherry-picked for Test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .plan import DEV_ROWS, SEEDS, registered_exploratory


def _read_scores(path: Path, *, expected_method: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed full-Dev Jeff report: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("scores", {})
    checkpoint = report.get("checkpoint", {})
    if (
        report.get("kind") != "FULL_DEV_FINETUNED"
        or report.get("rows") != DEV_ROWS
        or report.get("trained_lora") is not True
        or not isinstance(checkpoint, dict)
        or checkpoint.get("method") != expected_method
        or metrics.get("n_cases") != DEV_ROWS
        or metrics.get("label_classes") != 148
    ):
        raise ValueError(f"Dev report was not produced by the registered method: {path}")
    for key in ("top1_accuracy", "macro_f1"):
        score = metrics.get(key)
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid {key} in {path.name}")
    if abs(metrics["top1_accuracy"] * DEV_ROWS - metrics.get("top1_correct_n", -1)) > 1e-6:
        raise ValueError(f"Accuracy denominator or numerator corrupted: {path}")
    return {
        "top1_accuracy": float(metrics["top1_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "top1_correct_n": int(metrics["top1_correct_n"]),
        "bm25_to_jeff_rescued_cases": metrics.get("bm25_to_jeff_rescued_cases"),
        "bm25_to_jeff_broken_cases": metrics.get("bm25_to_jeff_broken_cases"),
    }


def summarize(root: Path, *, include_exploratory: bool = False) -> dict:
    """Read only aggregate Dev reports, select method+epoch by mean Top-1."""
    experiments = root / "experiments"
    primary = []
    for method in ("s1", "s2", "s3"):
        for epoch in (1, 2):
            trials = []
            for seed in SEEDS:
                run_id = f"{method}-seed{seed}-e2"
                filename = experiments / f"jeff_{run_id}-epoch{epoch}_k16_dev.json"
                trials.append({"seed": seed, **_read_scores(filename, expected_method=method)})
            primary.append(
                {
                    "method": method,
                    "epoch": epoch,
                    "seed_trials": trials,
                    "mean_top1": sum(x["top1_accuracy"] for x in trials) / len(trials),
                    "mean_macro_f1": sum(x["macro_f1"] for x in trials) / len(trials),
                }
            )
    chosen = max(
        primary,
        key=lambda row: (
            row["mean_top1"],
            row["mean_macro_f1"],
            -row["epoch"],
            row["method"] == "s1",
            row["method"] == "s3",
        ),
    )
    selected_run = f"{chosen['method']}-seed{SEEDS[0]}-e2"
    selected_step = chosen["epoch"] * 675
    extras = []
    if include_exploratory:
        for spec in registered_exploratory():
            path = experiments / f"jeff_{spec.run_id}-epoch1_k16_dev.json"
            extras.append(
                {
                    "method": spec.method,
                    "weighting": spec.weighting,
                    "seed": spec.seed,
                    **_read_scores(path, expected_method=spec.method),
                }
            )
    baseline = experiments / "jeff_zero_shot_k16_dev.json"
    if not baseline.is_file():
        raise FileNotFoundError("Missing full zero-shot Jeff Dev baseline")
    zero_shot = json.loads(baseline.read_text(encoding="utf-8"))
    if zero_shot.get("kind") != "FULL_DEV_ZERO_SHOT" or zero_shot.get("rows") != DEV_ROWS:
        raise ValueError("Zero-shot Dev result is not a complete 5,486-case evaluation")
    return {
        "status": "H200_FULL_DEV_RESULTS_SELECTED_NO_TEST_EVALUATION",
        "official_dev_cases": DEV_ROWS,
        "primary_k": 16,
        "model_selection_rule": "highest 3-seed mean Dev Top1; mean Macro-F1 tie break",
        "fixed_test_seed": SEEDS[0],
        "primary_trials": primary,
        "zero_shot_scores": zero_shot["scores"],
        "optional_exploratory": extras,
        "selected": {
            "method": chosen["method"],
            "epoch": chosen["epoch"],
            "mean_dev_top1": chosen["mean_top1"],
            "mean_dev_macro_f1": chosen["mean_macro_f1"],
            "checkpoint": f"data/runs/{selected_run}/resume-{selected_step:07d}",
        },
        "test_used_for_model_selection": False,
        "test_evaluated": False,
    }


def save_selection(root: Path, *, include_exploratory: bool = False) -> Path:
    path = root / "experiments/jeff_h200_dev_selection.json"
    if path.exists():
        raise FileExistsError("Final Dev selection already exists; never silently overwrite")
    report = summarize(root, include_exploratory=include_exploratory)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
