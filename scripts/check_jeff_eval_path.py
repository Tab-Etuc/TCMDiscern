"""CPU evaluation-path smoke test: Jeff K16 scorer must reproduce BM25 baseline.

We feed fake logits that always pick BM25 rank 1, then require exact
Top-1/Macro-F1 match to the independently frozen BM25 Dev experiment.
This validates evaluation wiring only; there are NO Jeff model predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_scoring import score_k16


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dev = load_records(root / "data/tcm-sd/dev.json")
    candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(dev) != len(candidates) or len(dev) != 5486:
        raise ValueError("Official Dev must be full length 5,486")
    for i, (case, row) in enumerate(zip(dev, candidates, strict=True)):
        if row.case_id != f"dev-{i:06d}" or row.gold_labels != (case["norm_syndrome"],):
            raise ValueError(f"Dev retrieval labels/order do not align at row {i}")

    # No model loaded; fake deterministic logits A > B..P.
    logits = np.zeros((len(dev), 16), dtype=np.float32)
    logits[:, 0] = 1.0
    classes = sorted({str(case["norm_syndrome"]) for case in dev})
    score = score_k16(candidates, logits, all_classes=classes)
    frozen = json.loads(
        (root / "experiments/bm25_k1_35_dev_2026-11-11.json").read_text(encoding="utf-8")
    )["results"][15]

    if score["top1_correct_n"] != 4366 or score["gold_in_top16_n"] != 5409:
        raise ValueError("Baseline count differs from frozen BM25 result")
    if abs(score["top1_accuracy"] - frozen["bm25_top1_accuracy"]) > 1e-12:
        raise ValueError("End-to-end scoring changed the BM25 accuracy")
    if abs(score["macro_f1"] - frozen["bm25_top1_macro_f1"]) > 1e-12:
        raise ValueError("End-to-end scoring changed the BM25 Macro-F1")
    if score["bm25_to_jeff_rescued_cases"] or score["bm25_to_jeff_broken_cases"]:
        raise ValueError("Dry run should exactly reproduce the original BM25 predictions")

    report = {
        "date": "2026-11-12",
        "status": "K16_JEFF_EVALUATION_WIRING_SMOKE_PASS",
        "method": "fake rank-1 logits, exactly reproduces original BM25",
        "real_jeff_inference": False,
        "trained_checkpoint_scored": False,
        "complete_official_dev": True,
        "test_cases_loaded": False,
        "results": score,
    }
    path = root / "experiments/jeff_k16_eval_dryrun_2026-11-12.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
