"""Select locked disease-study Dev checkpoints before the final official Test.

Method selection uses the three-seed mean of Macro-F1, with Accuracy as
tie-breaker. Within each chosen method, the single checkpoint is selected
on that method's Dev Macro-F1 and Accuracy. Test is never read here.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from tcmdiscern.h200.plan import SEEDS

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("global", "conditioned", "conditioned_hard", "pair")


def run_id(method: str, seed: int) -> str:
    return f"disease-{method}-seed{seed}-e2"


def infer_mode(method: str) -> str:
    return "conditioned" if method == "conditioned_hard" else method


def load_dev_scores(root: Path = ROOT) -> dict:
    groups = {}
    for method in METHODS:
        runs = []
        for seed in SEEDS:
            name = run_id(method, seed)
            mode = infer_mode(method)
            source = root / "experiments" / f"disease_study_dev_{mode}_{name}.json"
            if not source.is_file():
                raise FileNotFoundError(f"Missing final Dev evaluation: {source}")
            report = json.loads(source.read_text(encoding="utf-8"))
            if (
                report.get("status") != "COMPLETE"
                or report.get("split") != "dev"
                or report.get("mode") != mode
                or report.get("run_id") != name
                or report.get("scores", {}).get("n") != 5486
            ):
                raise ValueError(f"Invalid full Dev scoring report: {source}")
            scores = report["scores"]
            runs.append({
                "run_id": name, "seed": seed,
                "top1": scores["top1"], "macro_f1": scores["macro_f1"],
            })
        best = max(runs, key=lambda row: (row["macro_f1"], row["top1"], -row["seed"]))
        groups[method] = {
            "mean_top1": mean(r["top1"] for r in runs),
            "std_top1": stdev(r["top1"] for r in runs),
            "mean_macro_f1": mean(r["macro_f1"] for r in runs),
            "std_macro_f1": stdev(r["macro_f1"] for r in runs),
            "runs": runs,
            "representative_checkpoint": best["run_id"],
        }
    best_method = max(
        METHODS,
        key=lambda mode: (
            groups[mode]["mean_macro_f1"], groups[mode]["mean_top1"],
            -METHODS.index(mode),
        ),
    )
    return {
        "status": "DEV_SELECTED_TEST_SEALED",
        "selection_rule": "three-seed mean Macro-F1, then mean Top-1; checkpoint chosen by Dev Macro-F1 then Top-1",
        "methods": groups,
        "overall_best_method": best_method,
        "overall_best_run": groups[best_method]["representative_checkpoint"],
        "test_evaluated": False,
    }


def main() -> None:
    report = load_dev_scores()
    dest = ROOT / "experiments/disease_study_dev_selection.json"
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
