"""CPU-only frozen SGD+Jeff fusion baseline with optional H200 Case-RAG logits.

Audit the historical 2026-11-16 fixed plain-SGD/Jeff S1 margin-1
baseline (4667/5486, macro-F1 60.9597%) before testing Global2. Record
cross-environment SGD drift instead of rejecting valid matched-arm evaluation.
The only intervention is replacing original Jeff logits with frozen
Global2 decision logits. No threshold search, new GPU work or Test usage.

H200: .venv/bin/python -m scripts.evaluate_case_rag_strong_sgd_fusion_cpu --strict-global2

Only aggregated scores are written; original private patient records remain in data/.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from scripts.evaluate_case_rag_paper_dev import metrics, paired
from scripts.prepare_case_rag_paper_ablation import ROOT, TAG
from scripts.run_case_rag_jeff import read_condition_logits
from tcmdiscern.case_rag import transitive_group_ids
from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.datasets import clinical_text, load_records, normalized_input
from tcmdiscern.jeff_scoring import read_model_logits

METHODS = ("s1", "s2", "s3")
N_CASES = 5486
FEATURES = {
    "analyzer": "char",
    "ngram_range": (1, 3),
    "min_df": 3,
    "max_features": 180000,
    "sublinear_tf": True,
    "lowercase": False,
    "dtype": np.float32,
}
SGD_OPTIONS = {
    "loss": "modified_huber",
    "alpha": 2e-5,
    "penalty": "l2",
    "max_iter": 15,
    "tol": 1e-3,
    "random_state": 20261116,
}
FROZEN_THRESHOLD = 1.0
FROZEN_PRIOR_MULTIPLIER = 0.6
FROZEN_GATE_DELTA = 3.0
FROZEN_GATE_TOP_K = 10


def class_balanced_weights(labels: list[str]) -> np.ndarray:
    """Exact frozen 2026-11-16 SGD weighting: gamma=.25, smoothing=5."""
    frequency = Counter(labels)
    mean = len(labels) / len(frequency)
    weights = np.asarray(
        [np.clip((mean / (frequency[label] + 5.0)) ** 0.25, 0.25, 4.0) for label in labels],
        dtype=np.float64,
    )
    return weights / weights.mean()


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Invalid SGD/Jeff scores")
    return (matrix - matrix.mean(axis=1, keepdims=True)) / np.maximum(
        matrix.std(axis=1, keepdims=True), 1e-6
    )


def frozen_pairwise_margin1(
    cheap_scores: np.ndarray, jeff_logits: np.ndarray, train_supports: np.ndarray
) -> np.ndarray:
    """Exact fixed arbitration protocol, top-16 candidates, no gold access."""
    if (
        cheap_scores.ndim != 2
        or cheap_scores.shape != jeff_logits.shape
        or cheap_scores.shape != train_supports.shape
        or cheap_scores.shape[1] != 16
    ):
        raise ValueError("Expected aligned [N,16] SGD, Jeff and Train-support matrices")
    if not np.isfinite(train_supports).all() or (train_supports < 1).any():
        raise ValueError("Expected positive finite Train class counts")
    if not np.isfinite(jeff_logits).all():
        raise ValueError("Nonfinite Jeff logits")
    cheap = row_normalize(cheap_scores)
    prior = jeff_logits - FROZEN_PRIOR_MULTIPLIER * np.log(train_supports)
    jeff = row_normalize(prior)
    index = np.arange(len(cheap))
    cheap_pos = np.argmax(cheap_scores, axis=1)
    challenger = np.argmax(prior[:, :FROZEN_GATE_TOP_K], axis=1)
    shift = prior[index, challenger] - prior[:, 0]
    jeff_pos = np.where((challenger != 0) & (shift >= FROZEN_GATE_DELTA), challenger, 0)
    cheap_gap = cheap[index, cheap_pos] - cheap[index, jeff_pos]
    jeff_gap = jeff[index, jeff_pos] - jeff[index, cheap_pos]
    return np.where(jeff_gap > FROZEN_THRESHOLD * cheap_gap, jeff_pos, cheap_pos)


def historical_baseline_audit(correct: int, macro_f1_148_percent: float) -> dict:
    """Explicitly distinguish same-environment pairing from past Dev scores."""
    if not 0 <= correct <= N_CASES or not np.isfinite(macro_f1_148_percent):
        raise ValueError("Invalid baseline measurement")
    historical_correct = 4667
    historical_f1 = 60.9597
    matched = correct == historical_correct and abs(macro_f1_148_percent - historical_f1) <= 0.005
    return {
        "matched": matched,
        "historical_correct": historical_correct,
        "historical_top1_percent": round(100 * historical_correct / N_CASES, 4),
        "historical_macro_f1_148_percent": historical_f1,
        "this_environment_correct": correct,
        "this_environment_top1_percent": round(100 * correct / N_CASES, 4),
        "this_environment_macro_f1_148_percent": macro_f1_148_percent,
        "delta_correct": correct - historical_correct,
        "delta_macro_f1_pp": round(macro_f1_148_percent - historical_f1, 4),
        "interpretation": (
            "Historical full-Dev baseline reproduced."
            if matched
            else "Historical scores differ under the current run. "
            "Compare each Global2 arm ONLY with its same-run original Base arm. "
            "Investigate dependencies and input fingerprints before claiming exact reproduction."
        ),
    }


def _read_frozen_inputs(root: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    with (root / "data/jeff/dev_k16.jsonl").open(encoding="utf-8") as file:
        original = [json.loads(line) for line in file]
    with (root / "data/case_rag" / TAG / "global2.jsonl").open(encoding="utf-8") as file:
        evidence = [json.loads(line) for line in file]
    if (
        len(train) != 43180
        or len(dev) != N_CASES
        or len(original) != N_CASES
        or len(evidence) != N_CASES
    ):
        raise ValueError("Frozen official Train/Dev or Global2 prompt count differs")
    for index, (baseline, augmented) in enumerate(zip(original, evidence, strict=True)):
        case_id = f"dev-{index:06d}"
        if (
            baseline["id"] != case_id
            or augmented["id"] != case_id
            or baseline["options"] != augmented["options"]
            or len(baseline["options"]) != 16
            or len({option["id"] for option in baseline["options"]}) != 16
        ):
            raise ValueError(f"Mismatch in frozen K16 candidate order: {case_id}")
    return train, dev, original, evidence


def run(root: Path = ROOT, *, strict_global2: bool = False) -> dict:
    # Fail before fitting the SGD if any requested H200 logits are missing.
    if strict_global2:
        for method in METHODS:
            path = (
                root / "data/case_rag" / TAG / "scores" / f"paper-{method}-full" / "global2.jsonl"
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing completed Global2 logits: {path}")
    for method in METHODS:
        _, baseline_path = frozen_paths(root, method)
        if not baseline_path.is_file() or baseline_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing original Jeff logits: {baseline_path}")
    train, dev, original, augmented = _read_frozen_inputs(root)
    train_y = [str(row["norm_syndrome"]) for row in train]
    gold = [str(row["norm_syndrome"]) for row in dev]
    frequency = Counter(train_y)
    classes = sorted(frequency)
    if len(classes) != 148 or set(gold) != set(classes):
        raise ValueError("Official syndrome class list changed")
    tail = set(sorted(classes, key=lambda name: (frequency[name], name))[:37])
    options = np.asarray([[str(option["id"]) for option in row["options"]] for row in original])
    class_index = {name: i for i, name in enumerate(classes)}
    if not set(options.flatten()).issubset(class_index):
        raise ValueError("BM25 shortlist contains unknown Train labels")
    shortlist_indices = np.asarray(
        [[class_index[label] for label in row] for row in options], dtype=np.int32
    )
    support = np.asarray([[frequency[label] for label in row] for row in options], dtype=np.float64)
    print(
        "[CPU SGD] Environment:"
        f" Python={platform.python_version()}"
        f" NumPy={np.__version__}"
        f" scikit-learn={sklearn.__version__}"
        f" SciPy={scipy.__version__}",
        flush=True,
    )
    print("[CPU SGD] Fitting original full-Train char TF-IDF and frozen SGD...", flush=True)
    vectorizer = TfidfVectorizer(**FEATURES)
    x_train = vectorizer.fit_transform([clinical_text(row) for row in train])
    x_dev = vectorizer.transform([clinical_text(row) for row in dev])
    model = SGDClassifier(**SGD_OPTIONS)
    model.fit(x_train, train_y, sample_weight=class_balanced_weights(train_y))
    if list(model.classes_) != classes:
        raise ValueError("SGD label vocabulary differs from frozen 148-class protocol")
    sgd_scores = np.take_along_axis(model.decision_function(x_dev), shortlist_indices, axis=1)
    del x_train, x_dev, vectorizer, model
    clean_train_text = {normalized_input(row, full=True) for row in train}
    clean = np.asarray([normalized_input(row, full=True) not in clean_train_text for row in dev])
    if int(clean.sum()) != 4957:
        raise ValueError("Train/Dev exact overlap changed")
    clusters, inverse = np.unique(transitive_group_ids(dev), return_inverse=True)
    predictions: dict[str, np.ndarray] = {}
    used_score_files: dict[str, str] = {}
    for method in METHODS:
        _, path = frozen_paths(root, method)
        jeff = read_model_logits(path, num_cases=N_CASES)
        predictions[f"{method}_base"] = options[
            np.arange(N_CASES), frozen_pairwise_margin1(sgd_scores, jeff, support)
        ]
    # The fixed fusion policy is unchanged, but SGDClassifier can produce
    # slightly different scores across sklearn/compiled numerical environments.
    # Log differences; keep all Base/Global2 arms paired in this SAME process
    # with the SAME fitted SGD. Never silently claim a historical reproduction.
    s1_correct = int(np.sum(predictions["s1_base"] == gold))
    s1_score = metrics(gold, predictions["s1_base"].tolist(), classes, tail, clean)
    reproduction = historical_baseline_audit(s1_correct, s1_score["macro_f1_148_percent"])
    if reproduction["matched"]:
        print(
            f"[CPU SGD] Historical S1 reproduced: {s1_correct}/5486 "
            f"Macro-F1={s1_score['macro_f1_148_percent']:.4f}%",
            flush=True,
        )
    else:
        print(
            "[CPU SGD] HISTORICAL_BASELINE_DRIFT: "
            f"historical={reproduction['historical_correct']}/5486 "
            f"F1={reproduction['historical_macro_f1_148_percent']:.4f}%, "
            f"this_environment={s1_correct}/5486 "
            f"F1={s1_score['macro_f1_148_percent']:.4f}%; "
            "continue matched Base/Global2 paired comparisons with this same fitted SGD",
            flush=True,
        )
    for method in METHODS:
        path = root / "data/case_rag" / TAG / "scores" / f"paper-{method}-full" / "global2.jsonl"
        if not path.is_file():
            if strict_global2:
                raise FileNotFoundError(f"Missing completed Global2 logits: {path}")
            continue
        scores = read_condition_logits(path, augmented)
        predictions[f"{method}_global2"] = options[
            np.arange(N_CASES), frozen_pairwise_margin1(sgd_scores, scores, support)
        ]
        used_score_files[method] = str(path.relative_to(root))
    if strict_global2 and len(used_score_files) != len(METHODS):
        raise FileNotFoundError("All three completed Global2 logits required")
    statistics: dict[str, dict] = {}
    for label, pred in predictions.items():
        scored = metrics(gold, pred.tolist(), classes, tail, clean)
        record = {"metrics": scored}
        if label.endswith("_global2"):
            method = label.split("_")[0]
            reference = predictions[f"{method}_base"]
            record["paired_vs_same_fusion_original"] = paired(
                gold, reference.tolist(), pred.tolist(), inverse, len(clusters)
            )
            baseline_correct = int(np.sum(reference == gold))
            change = scored["correct"] - baseline_correct
            if (
                change
                != record["paired_vs_same_fusion_original"]["rescued"]
                - record["paired_vs_same_fusion_original"]["broken"]
            ):
                raise ValueError("Global2 paired accounting mismatch")
        statistics[label] = record
    return {
        "status": "CASE_RAG_STRONG_SGD_CPU_FUSION_COMPLETED",
        "study_scope": "Dev exploratory, historical 2026-11-16 fixed pairwise rule",
        "no_fusion_parameter_tuning": True,
        "official_test_used": False,
        "new_gpu_forwards": 0,
        "dev_cases": N_CASES,
        "train_cases": len(train),
        "original_s1_baseline_verified": reproduction["matched"],
        "historical_s1_reproduction": reproduction,
        "numerical_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "score_comparison_policy": (
            "Evaluate Global2 against the same-method original Base within this run "
            "using identical SGD scores, candidate order and fixed arbitration. "
            "Historical score differences are separate environment diagnostics."
        ),
        "global2_logit_sources": used_score_files,
        "fixed_sgd_parameters": {
            "loss": "modified_huber",
            "alpha": SGD_OPTIONS["alpha"],
            "max_iter": SGD_OPTIONS["max_iter"],
            "gamma": 0.25,
            "smoothing": 5.0,
            "arbitration": "pairwise_margin1",
            "prior_multiplier": FROZEN_PRIOR_MULTIPLIER,
            "jeff_top10_gate_delta": FROZEN_GATE_DELTA,
        },
        "methods": statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-global2", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/case_rag_strong_sgd_fusion_cpu_2026-11-17.json",
    )
    args = parser.parse_args()
    report = run(strict_global2=args.strict_global2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report["status"], "Global2:", list(report["global2_logit_sources"]))
    for method, content in report["methods"].items():
        score = content["metrics"]
        print(
            f"{method:18s} Top-1={score['top1_percent']:.4f}% "
            f"Macro-F1={score['macro_f1_148_percent']:.4f}% "
            f"Tail-F1={score['tail_37_macro_f1_percent']:.4f}% "
            f"No-Train-duplicate={score['no_exact_train_text_top1_percent']:.4f}%"
        )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
