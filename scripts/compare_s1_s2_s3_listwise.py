"""Same-budget, cross-fitted listwise calibration for frozen S1/S2/S3 Dev logits.

Uses the S2 exploratory policy rare_k16_d0.2 unchanged for all three models.
This is a comparative diagnostic on previously inspected Dev, not a new
independent model-selection result. Never reads the official Test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_s1_s2_ensemble import ROOT, S1, S2, metrics, paired_bootstrap, prepare
from scripts.analyze_s1_s2_s3 import S3, load_s3, pair_counts
from scripts.s2_listwise_calibration import option_features

OUTPUT = ROOT / "experiments/s1_s2_s3_cpu/listwise_control.json"
SEED = 20261115
THRESHOLD = 0.2
N_FOLDS = 5


def select_positions(scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    if scores.ndim != 2 or scores.shape[1] != 16 or not np.isfinite(scores).all():
        raise ValueError("Expected finite cross-fitted 16-slot scores")
    pos = np.argmax(scores, axis=1)
    delta = scores[np.arange(len(scores)), pos] - scores[:, 0]
    return np.where((pos > 0) & (delta >= threshold), pos, 0)


def crossfit_one(
    logits: np.ndarray,
    candidates: np.ndarray,
    gold: np.ndarray,
    counts: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    n, k = logits.shape
    if candidates.shape != (n, k):
        raise ValueError("Mismatched candidate/logit shapes")
    features = option_features(logits, counts[candidates])
    observed_gold = (candidates == gold[:, None]).astype(int)
    scores = np.full((n, k), np.nan, dtype=np.float64)
    for fit, valid in folds:
        training_features = features[fit].reshape(-1, features.shape[-1])
        training_labels = observed_gold[fit].ravel()
        weighting = np.clip(np.sqrt(100.0 / counts[gold[fit]]), 0.5, 3.0)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, max_iter=400, random_state=SEED),
        )
        model.fit(
            training_features,
            training_labels,
            logisticregression__sample_weight=np.repeat(weighting, k),
        )
        scores[valid] = model.decision_function(
            features[valid].reshape(-1, features.shape[-1])
        ).reshape(-1, k)
    if not np.isfinite(scores).all():
        raise ValueError("Cross-fitted scores missing or non-finite")
    return select_positions(scores)


def main() -> None:
    gold, cand, s1, s2, counts, groups, unseen = prepare(S1, S2)
    references = [row["case_id"] for row in json_lines(ROOT / "data/candidates/bm25_dev.jsonl")]
    s3 = load_s3(S3, len(gold), references)
    n = len(gold)
    strata = np.where(counts[gold] < 10, 0, np.where(counts[gold] < 100, 1, 2))
    folds = list(
        StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
            np.arange(n), strata, groups
        )
    )
    if any(set(groups[fit]) & set(groups[valid]) for fit, valid in folds):
        raise ValueError("Same clinical text appears on both sides of calibration")

    predictions = {}
    summaries = {}
    for name, logits in (("s1", s1), ("s2", s2), ("s3", s3)):
        positions = crossfit_one(logits, cand, gold, counts, folds)
        pred = cand[np.arange(n), positions]
        predictions[name] = pred
        summaries[name] = {
            "all_dev": metrics(gold, pred, counts),
            "no_exact_train_text": metrics(gold, pred, counts, unseen),
            "rescue_vs_bm25": int(np.sum((cand[:, 0] != gold) & (pred == gold))),
            "break_vs_bm25": int(np.sum((cand[:, 0] == gold) & (pred != gold))),
        }
        print(
            f"{name}: Top1 {summaries[name]['all_dev']['top1']:.4%}, "
            f"Macro {summaries[name]['all_dev']['macro_f1']:.4%}, "
            f"Tail {summaries[name]['all_dev']['tail_f1']:.4%}"
        )
    comparisons = {}
    for first, second in (("s1", "s2"), ("s1", "s3"), ("s2", "s3")):
        comparisons[f"{second}_minus_{first}"] = {
            **pair_counts(gold, predictions[first], predictions[second]),
            "bootstrap_95_delta_pp": paired_bootstrap(
                gold,
                predictions[first],
                predictions[second],
                counts,
                groups,
                replicates=2000,
            ),
        }
    report = {
        "study": "S1/S2/S3 identical fixed listwise calibration and group cross-fit",
        "data": {
            "official_dev_rows": n,
            "exact_train_text_overlap_dev_rows": n - len(unseen),
            "candidate_count": cand.shape[1],
            "official_test_opened": False,
        },
        "configuration": {
            "fold_count": N_FOLDS,
            "group": "normalized complete clinical text",
            "seed": SEED,
            "classifier": "StandardScaler + LogisticRegression(C=0.05,max_iter=400)",
            "training_weight": "clip(sqrt(100/train_class_count),0.5,3)",
            "rank_k": 16,
            "margin": THRESHOLD,
            "features": "S2 calibration option_features, no gold in inference features",
            "origin": "All hyperparameters taken unchanged from the previously explored S2 listwise calibration",
        },
        "scores": summaries,
        "comparisons": comparisons,
        "limitations": [
            "The model is cross-fitted on Dev and does not represent a deployable independent Train-fitted calibrator.",
            "S2-derived settings were previously selected on the same Dev, so relative performance is exploratory.",
            "S1/S2/S3 LoRA trained with one completed seed; no confirmatory test or independent 3-seed selection.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved:", OUTPUT)


def json_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
