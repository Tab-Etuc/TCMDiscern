"""Freeze representative Dev-selected disease / pair Jeff checkpoints.

Select the epoch by three-seed mean Top-1 (Macro-F1 tie-break), then take the
predeclared first seed. Both classes are scored against Train∪Dev labels, so
novel Dev categories are included in Macro-F1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tcmdiscern.h200.plan import SEEDS

ROOT = Path(__file__).resolve().parents[1]
MODES = ("disease", "pair")
EPOCHS = (1, 2)


def run_id(mode: str, seed: int) -> str:
    if mode not in MODES:
        raise ValueError("Invalid joint16 method")
    return f"joint16-{mode}-seed{seed}-e2"


def summarize(root: Path = ROOT) -> dict:
    outcome = {}
    for mode in MODES:
        comparisons = []
        for epoch in EPOCHS:
            trials = []
            for seed in SEEDS:
                rid = f"{run_id(mode, seed)}-epoch{epoch}"
                file = root / "experiments" / f"joint16_dev_{mode}_{rid}.json"
                if not file.is_file():
                    raise FileNotFoundError(f"Missing official Dev report {file}")
                data = json.loads(file.read_text(encoding="utf-8"))
                scores = data.get("scores", {})
                if (
                    data.get("kind") != "FULL_EVALUATION"
                    or data.get("split") != "dev"
                    or data.get("mode") != mode
                    or data.get("run_id") != rid
                    or scores.get("n_cases") != 5486
                ):
                    raise ValueError(f"Invalid full Dev run: {file}")
                if any(
                    not isinstance(scores.get(key), (float, int))
                    or not math.isfinite(scores[key])
                    or not 0 <= scores[key] <= 1
                    for key in ("top1_accuracy", "macro_f1")
                ):
                    raise ValueError("Invalid score value")
                trials.append({
                    "seed": seed,
                    "accuracy": scores["top1_accuracy"],
                    "macro_f1": scores["macro_f1"],
                })
            comparisons.append({
                "epoch": epoch,
                "seed_scores": trials,
                "mean_accuracy": sum(t["accuracy"] for t in trials) / len(trials),
                "mean_macro_f1": sum(t["macro_f1"] for t in trials) / len(trials),
            })
        selected = max(comparisons, key=lambda x: (
            x["mean_accuracy"], x["mean_macro_f1"], -x["epoch"],
        ))
        outcome[mode] = {
            "selected_epoch": selected["epoch"],
            "fixed_test_seed": SEEDS[0],
            "checkpoint": (
                f"data/runs/{run_id(mode, SEEDS[0])}/"
                f"resume-{selected['epoch'] * 675:07d}"
            ),
            "comparisons": comparisons,
        }
    return {
        "status": "JOINT16_DEV_SELECTED_TEST_UNOPENED",
        "selection_rule": "three-seed mean Top-1 then Macro-F1, fixed first seed on Test",
        "selected": outcome,
        "test_used": False,
    }


def main() -> None:
    report = summarize()
    destination = ROOT / "experiments/joint16_dev_selection.json"
    with destination.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
