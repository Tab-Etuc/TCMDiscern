"""Frozen Jeff/SGD contribution ablations and selective compute budget curve.

Every metric is exploratory on the original 5,486 Dev. Inputs are restricted
to published Train/Dev and already saved full-Dev logits. No new threshold
search, no access to Test and no Jeff GPU forwards.
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from scripts.evaluate_case_rag_paper_dev import metrics
from scripts.evaluate_case_rag_strong_sgd_fusion_cpu import (
    FEATURES,
    SGD_OPTIONS,
    class_balanced_weights,
    row_normalize,
)
from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.datasets import clinical_text, load_records, normalized_input
from tcmdiscern.jeff_scoring import read_model_logits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/h200_priority_cpu_ablation.json"
BUDGETS = (0, 5, 10, 15, 20, 30, 40, 50, 75, 100)


def arbitrate(
    scores: np.ndarray, logits: np.ndarray, supports: np.ndarray,
    *, prior_mult: float = 0.6, k: int = 10, gate_delta: float = 3.0,
    use_gate: bool = True, use_pairwise: bool = True,
) -> np.ndarray:
    n = len(scores)
    linear = row_normalize(scores)
    prior = logits - prior_mult * np.log(supports)
    jeff_norm = row_normalize(prior)
    best_sgd = scores.argmax(axis=1)
    best_jeff = prior[:, :k].argmax(axis=1)
    if use_gate:
        diff = prior[np.arange(n), best_jeff] - prior[:, 0]
        best_jeff = np.where((best_jeff != 0) & (diff >= gate_delta), best_jeff, 0)
    if not use_pairwise:
        return best_jeff
    gain_sgd = linear[np.arange(n), best_sgd] - linear[np.arange(n), best_jeff]
    gain_jeff = jeff_norm[np.arange(n), best_jeff] - jeff_norm[np.arange(n), best_sgd]
    return np.where(gain_jeff > gain_sgd, best_jeff, best_sgd)


def run(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Official Train/Dev count mismatch")
    gold = [str(r["norm_syndrome"]) for r in dev]
    train_gold = [str(r["norm_syndrome"]) for r in train]
    counts = Counter(train_gold)
    labels = sorted(counts)
    if len(labels) != 148 or set(gold) != set(labels):
        raise ValueError("Incorrect fixed 148-class label roster")
    with (root / "data/jeff/dev_k16.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 5486:
        raise ValueError("Incomplete natural candidate positions")
    choices = np.asarray([[str(option["id"]) for option in r["options"]] for r in rows])
    if choices.shape != (5486, 16) or any(len(set(row)) != 16 for row in choices):
        raise ValueError("Candidate uniqueness/width mismatch")
    vocab = {label: i for i, label in enumerate(labels)}
    indices = np.asarray([[vocab[c] for c in row] for row in choices], dtype=np.int32)
    support = np.asarray([[counts[c] for c in row] for row in choices], dtype=np.float64)
    linear = TfidfVectorizer(**FEATURES)
    x_train = linear.fit_transform([clinical_text(r) for r in train])
    x_dev = linear.transform([clinical_text(r) for r in dev])
    model = SGDClassifier(**SGD_OPTIONS)
    model.fit(x_train, train_gold, sample_weight=class_balanced_weights(train_gold))
    if list(model.classes_) != labels:
        raise ValueError("Label order mismatch")
    scores = np.take_along_axis(model.decision_function(x_dev), indices, axis=1)
    frequencies = sorted(labels, key=lambda x: (counts[x], x))
    tail = set(frequencies[:37])
    seen = {normalized_input(r, full=True) for r in train}
    clean = np.asarray([normalized_input(r, full=True) not in seen for r in dev])
    if int(clean.sum()) != 4957:
        raise ValueError("Exact Train/Dev overlap changed")
    def assess(indices: np.ndarray) -> dict:
        predicted = choices[np.arange(len(choices)), indices].tolist()
        return metrics(gold, predicted, labels, tail, clean)

    cheap = scores.argmax(axis=1)
    results = {"bm25": assess(np.zeros(5486, dtype=int)), "sgd_only": assess(cheap)}
    logits: dict[str, np.ndarray] = {}
    for method in ("s1", "s2", "s3"):
        _, p = frozen_paths(root, method)
        logits[method] = read_model_logits(p, num_cases=5486)
        results[f"{method}_raw"] = assess(logits[method].argmax(axis=1))
        results[f"{method}_prior_gate"] = assess(arbitrate(scores, logits[method], support, use_pairwise=False))
        results[f"{method}_full_pairwise"] = assess(arbitrate(scores, logits[method], support))
    zero_file = root / "data/jeff/dev_zero_shot_k16_full.jsonl"
    if zero_file.is_file():
        zero = read_model_logits(zero_file, num_cases=5486)
        results["jeff_zero_shot_raw"] = assess(zero.argmax(axis=1))
        results["sgd_plus_jeff_zero_shot"] = assess(arbitrate(scores, zero, support))
    s1 = logits["s1"]
    fixed = arbitrate(scores, s1, support)
    ablations = {
        "without_prior": arbitrate(scores, s1, support, prior_mult=0),
        "without_top10_gate": arbitrate(scores, s1, support, use_gate=False),
        "without_pairwise_arbitration": arbitrate(scores, s1, support, use_pairwise=False),
        "top16_gate": arbitrate(scores, s1, support, k=16),
        "without_prior_without_gate": arbitrate(scores, s1, support, prior_mult=0, use_gate=False),
        "sgd_only": cheap,
    }
    for name, prediction in ablations.items():
        results[name] = assess(prediction)
    margin = np.sort(row_normalize(scores), axis=1)[:, -1] - np.sort(row_normalize(scores), axis=1)[:, -2]
    # Stable tie ordering by original Dev row; this only sets a compute budget,
    # and never consults gold when deciding which rows call Jeff.
    order = np.argsort(margin, kind="stable")
    curve = []
    for budget in BUDGETS:
        invoked = np.zeros(len(dev), dtype=bool)
        invoked[order[:int(len(dev) * budget / 100)]] = True
        selected = np.where(invoked, fixed, cheap)
        curve.append({
            "max_jeff_invocation_budget_percent": budget,
            "actual_jeff_invocations": int(invoked.sum()),
            "actual_fraction": round(float(invoked.mean()), 6),
            "metrics": assess(selected),
            "decisions_changed_from_sgd": int(np.sum(selected != cheap)),
        })
    # The identical rank ordering is used by the GPU-only selective-call
    # benchmark. IDs are public synthetic case indices, without patient text.
    target_indices = root / "data/h200/priority_budget_20pct_indices.json"
    target_indices.parent.mkdir(parents=True, exist_ok=True)
    indices_20 = order[: int(len(dev) * 0.20)].tolist()
    target_indices.write_text(
        json.dumps(
            {"dev_cases": len(dev), "selected_indices": indices_20,
             "policy": "lowest SGD top2 margin", "official_test_used": False},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "H200_PRIORITY_CPU_ABLATION_COMPLETED",
        "scope": "exploratory original official Dev; prior method selection viewed Dev",
        "dev_cases": len(dev), "train_cases": len(train),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "official_test_used": False,
        "gpu_forward_calls": 0,
        "budget_ordering": "lowest per-case normalized SGD top2 margin first",
        "budget_policy": "when selected, apply frozen S1 prior+gate+pairwise; otherwise SGD",
        "selected_20pct_indices": str(target_indices.relative_to(root)),
        "methods": results, "budget_curve": curve,
        "training_modes": ["head_only", "lora_only", "both"],
        "metric_note": "148-class macro-F1 and fixed 37-class Train-tail; no repeated Dev tuning",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("H200_PRIORITY_CPU_ABLATION_COMPLETED")
    for key in ("sgd_only", "s1_full_pairwise", "without_prior", "without_top10_gate"):
        x = results[key]
        print(f"{key}: Top1={x['top1_percent']:.4f}% MacroF1={x['macro_f1_148_percent']:.4f}%")
    for row in curve:
        x = row["metrics"]
        print(f"budget={row['max_jeff_invocation_budget_percent']:3d}% Top1={x['top1_percent']:.4f}% F1={x['macro_f1_148_percent']:.4f}%")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(output=args.output)
