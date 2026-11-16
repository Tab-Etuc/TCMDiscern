"""S2-only listwise evidence calibration using saved 16-slot Jeff logits.

Exploratory five-fold group cross-fit on previously inspected Dev.
The learned classifier scores candidate correctness using S2 raw logits,
BM25 slot rank and Train-only class frequency. Each outer held-out row
is unseen by the model. Strong S1 and S2 fixed Gate are referenced.
Official Test remains sealed; no training of 31B weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .analyze_s1_s2_ensemble import metrics, paired_bootstrap, prepare
else:
    from analyze_s1_s2_ensemble import metrics, paired_bootstrap, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/s1_s2_cpu/s2_listwise_group_oof.json"
SEED = 20261115


def option_features(raw, count):
    """No single-feature uses gold or Dev labels."""
    n, k = raw.shape
    logfreq = np.log(count).astype(float)
    zmean = raw.mean(axis=1, keepdims=True)
    zsd = np.maximum(raw.std(axis=1, keepdims=True), 1e-5)
    rank = np.arange(k)
    order = np.argsort(-raw, axis=1, kind="stable")
    inv = np.empty((n, k), dtype=int)
    np.put_along_axis(inv, order, np.tile(rank, (n, 1)), axis=1)
    centered = (raw - zmean) / zsd
    strong = np.exp((raw - raw.max(axis=1, keepdims=True)) / 2)
    probs = strong / strong.sum(axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + 1e-20), axis=1, keepdims=True)
    margin_top = np.sort(raw, axis=1)[:, -1] - np.sort(raw, axis=1)[:, -2]
    parts = [
        np.broadcast_to(raw - raw[:, [0]], (n, k)),
        centered,
        np.broadcast_to(raw - raw.max(axis=1, keepdims=True), (n, k)),
        logfreq,
        logfreq - logfreq[:, [0]],
        np.broadcast_to(np.log1p(rank), (n, k)),
        np.broadcast_to(1.0 / (rank + 1), (n, k)),
        np.broadcast_to((rank == 0).astype(float), (n, k)),
        np.log1p(inv),
        np.exp((raw - raw.max(axis=1, keepdims=True)) / 2),
        np.broadcast_to(entropy, (n, k)),
        np.broadcast_to(np.log1p(margin_top)[:, None], (n, k)),
        np.broadcast_to(np.log1p(zsd), (n, k)),
    ]
    feat = np.stack(parts, axis=2)
    if feat.shape != (n, k, len(parts)) or not np.isfinite(feat).all():
        raise ValueError("Bad S2 features")
    return feat


def main():
    gold, candidates, z1, z2, count, groups, unseen = prepare(
        ROOT / "data/jeff/dev_s1-seed20261112-e2-epoch2_k16_5486.jsonl",
        ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl",
    )
    n, k = z2.shape
    features = option_features(z2, count[candidates])
    labels = (candidates == gold[:, None]).astype(int)
    if not np.array_equal(
        labels.sum(axis=1), np.any(candidates == gold[:, None], axis=1).astype(int)
    ):
        raise ValueError("Candidate hit check failed")
    # Reuse fixed preselected S1/S2 gate policies; do not re-search them.
    lp = np.log(count[candidates])

    def fixed(z):
        score = z - 0.6 * lp
        pos = np.argmax(score[:, :10], axis=1)
        choose = (pos > 0) & (score[np.arange(n), pos] - score[:, 0] >= 3.0)
        return candidates[np.arange(n), np.where(choose, pos, 0)]

    b = candidates[:, 0]
    g1 = fixed(z1)
    g2 = fixed(z2)
    strata = np.where(count[gold] < 10, 0, np.where(count[gold] < 100, 1, 2))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.arange(n), strata, groups))
    if any(set(groups[fit]) & set(groups[val]) for fit, val in folds):
        raise AssertionError("Dev text-group leakage")
    variants = {}
    predictions = {}
    for weight in ("uniform", "rare"):
        # Different objectives: original top1 versus soft macro-tail protection.
        score = np.full((n, k), np.nan, dtype=float)
        for fit, val in folds:
            xfit = features[fit].reshape(-1, features.shape[-1])
            yfit = labels[fit].ravel()
            if weight == "uniform":
                w = np.ones(len(fit), dtype=float)
            else:
                w = np.clip(np.sqrt(100.0 / count[gold[fit]]), 0.5, 3.0)
            sample_weight = np.repeat(w, k)
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.05, max_iter=400, random_state=SEED)
            )
            clf.fit(xfit, yfit, logisticregression__sample_weight=sample_weight)
            score[val] = clf.decision_function(
                features[val].reshape(-1, features.shape[-1])
            ).reshape(-1, k)
        if not np.isfinite(score).all():
            raise ValueError("Missing cross-fitted scores")
        for kk in (5, 10, 16):
            xx = score.copy()
            xx[:, kk:] = -np.inf
            pos = np.argmax(xx, axis=1)
            delta = xx[np.arange(n), pos] - xx[:, 0]
            for threshold in (0.0, 0.2, 0.5, 1.0):
                take = np.where((pos > 0) & (delta >= threshold), pos, 0)
                pred = candidates[np.arange(n), take]
                name = f"{weight}_k{kk}_d{threshold:g}"
                predictions[name] = pred
                variants[name] = {
                    "overall": metrics(gold, pred, count),
                    "no_train_text_overlap": metrics(gold, pred, count, unseen),
                    "rescued_vs_bm25": int(np.sum((b != gold) & (pred == gold))),
                    "broken_vs_bm25": int(np.sum((b == gold) & (pred != gold))),
                }
    baselines = {
        "bm25": metrics(gold, b, count),
        "s1_prior_gate": metrics(gold, g1, count),
        "s2_prior_gate": metrics(gold, g2, count),
    }
    paired = {}
    for name in ("rare_k16_d0.2", "rare_k10_d0.2", "rare_k16_d0", "uniform_k16_d0"):
        proposed = predictions[name]
        paired[name] = {
            "rescues_vs_s2_gate": int(np.sum((g2 != gold) & (proposed == gold))),
            "damages_vs_s2_gate": int(np.sum((g2 == gold) & (proposed != gold))),
            "cluster_paired_bootstrap_95_vs_s2_gate": paired_bootstrap(
                gold, g2, proposed, count, groups, replicates=1200
            ),
        }
    ranked_acc = sorted(
        variants.items(),
        key=lambda x: (x[1]["overall"]["top1"], x[1]["overall"]["macro_f1"]),
        reverse=True,
    )
    ranked_macro = sorted(
        variants.items(),
        key=lambda x: (x[1]["overall"]["macro_f1"], x[1]["overall"]["top1"]),
        reverse=True,
    )
    s2base = baselines["s2_prior_gate"]
    beats_s2 = [
        (k, v)
        for k, v in ranked_macro
        if v["overall"]["top1"] > s2base["top1"] and v["overall"]["macro_f1"] > s2base["macro_f1"]
    ]
    report = {
        "note": "Exploratory group-cross-fitted Dev S2-only ranking calibrator, model never sees held-out fold labels; Dev previously inspected",
        "n": n,
        "feature_count": features.shape[-1],
        "train_only_class_counts": True,
        "test_touched": False,
        "baselines": baselines,
        "threshold_policy": "Fixed small threshold grid described before evaluating results, but selected variants on same previously-inspected Dev are exploratory.",
        "variants": variants,
        "paired_comparisons": paired,
        "top_accuracy": [k for k, _ in ranked_acc[:10]],
        "top_macro": [k for k, _ in ranked_macro[:10]],
        "beat_s2_both": [k for k, _ in beats_s2],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Baselines")
    for name, x in baselines.items():
        print(name, round(x["top1"] * 100, 3), round(x["macro_f1"] * 100, 3))
    print("Top accuracy")
    for name, x in ranked_acc[:10]:
        v = x["overall"]
        print(
            name,
            round(v["top1"] * 100, 3),
            round(v["macro_f1"] * 100, 3),
            x["rescued_vs_bm25"],
            x["broken_vs_bm25"],
        )
    print("Top Macro")
    for name, x in ranked_macro[:10]:
        v = x["overall"]
        print(
            name,
            round(v["top1"] * 100, 3),
            round(v["macro_f1"] * 100, 3),
            x["rescued_vs_bm25"],
            x["broken_vs_bm25"],
        )
    print("Beat S2 both", len(beats_s2))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
