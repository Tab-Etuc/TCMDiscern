"""Exploratory CPU-only BM25/Jeff disagreement routing, 5-group-fold evaluation.

Loads frozen 5486 Dev logits and Train retrieval provenance; never opens
official Test or loads any Jeff weights. Trained gates use only *other folds*
of Dev, grouped by patient/text identity, and therefore remain exploratory
because Dev already informed prior research decisions. Report every policy.

Supports original Jeff S1/S2/S3 immediately; if archived H200 Global2 logits
are available in paper_5486_20261116/scores, automatically tests those too.

Usage:
    .venv/bin/python -m scripts.analyze_case_rag_cpu_router
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_case_rag_cpu_first import (
    METHODS,
    _prediction_from_logits,
    paired_errors,
)
from scripts.evaluate_case_rag_paper_dev import metrics
from scripts.prepare_case_rag_paper_ablation import ROOT, TAG
from scripts.run_case_rag_jeff import read_condition_logits
from tcmdiscern.case_rag import transitive_group_ids
from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_scoring import read_model_logits


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    """Normalize Jeff's fixed 16-way head output without refitting temperatures."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(np.clip(shifted, -100, 0))
    return weights / weights.sum(axis=1, keepdims=True)


def prediction_features(
    rows: list[dict],
    logits: np.ndarray,
    frequency: Counter,
    provenance: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract inference-available evidence only, aligned to frozen candidate IDs."""
    n = len(rows)
    if logits.shape != (n, 16) or len(provenance) != n:
        raise ValueError("Case-RAG logits/provenance shape mismatch")
    probabilities = _softmax_logits(logits)
    indices = np.argmax(logits, axis=1)
    ranked = np.sort(logits, axis=1)
    jeff = _prediction_from_logits(rows, logits)
    bm25 = np.asarray([row["options"][0]["id"] for row in rows], dtype=str)
    features = []
    for i, row in enumerate(rows):
        if provenance[i]["case_id"] != row["id"]:
            raise ValueError(f"Provenance case ID changed at row {i}")
        top = jeff[i]
        ref = bm25[i]
        evidence = provenance[i]["conditions"]["global2"]
        labels = evidence["support_labels"]
        scores = evidence["support_scores"]
        if len(labels) != 2 or len(scores) != 2:
            raise ValueError(f"Two frozen Global2 neighbors required at row {i}")
        if any(label not in frequency for label in (top, ref, *labels)):
            raise ValueError(f"Unknown candidate/source label at row {i}")
        if any(not np.isfinite(score) for score in scores):
            raise ValueError(f"Bad source BM25 scores at row {i}")
        jeff_rank = int(indices[i])
        f_jeff = frequency[top]
        f_bm25 = frequency[ref]
        source_one, source_two = labels
        features.append(
            [
                float(logits[i, jeff_rank] - logits[i, 0]),  # Jeff vs BM25 candidate
                float(ranked[i, -1] - ranked[i, -2]),  # Jeff choice margin
                float(probabilities[i, 0]),  # model probability for BM25 slot
                float(probabilities[i, jeff_rank]),  # model confidence of its choice
                np.log1p(f_bm25),
                np.log1p(f_jeff),
                np.log1p(f_bm25) - np.log1p(f_jeff),
                float(jeff_rank) / 15,
                float(source_one == ref),
                float(source_two == ref),
                float(source_one == top),
                float(source_two == top),
                float(source_one == source_two),
                np.log1p(max(scores[0], 0)),
                np.log1p(max(scores[1], 0)),
                np.log1p(max(scores[0], 0)) - np.log1p(max(scores[1], 0)),
                float(evidence["supported_options"]) / 2,
            ]
        )
    array = np.asarray(features, dtype=np.float64)
    if array.shape != (n, 17) or not np.isfinite(array).all():
        raise ValueError("Feature extraction produced invalid values")
    return array, jeff, bm25


def group_crossfit_router(
    features: np.ndarray,
    gold: np.ndarray,
    jeff: np.ndarray,
    bm25: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[dict]]:
    """Train only on fit folds; emit a held-out BM25 preference for each row.

    Where both candidates predict the wrong class, their routing target is
    undefined and excluded from *fit*, but remains in held-out evaluation.
    """
    n = len(gold)
    if len(features) != n or len(jeff) != n or len(bm25) != n or len(groups) != n:
        raise ValueError("Crossfit case dimension mismatch")
    if np.unique(groups).size < n_splits:
        raise ValueError("Insufficient independent case groups")
    chance = np.full(n, np.nan, dtype=float)
    fold_records = []
    splits = GroupKFold(n_splits=n_splits)
    for fold, (fit, valid) in enumerate(splits.split(features, groups=groups)):
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Case group leaked between fit and held-out fold")
        eligible = fit[
            (jeff[fit] != bm25[fit]) & ((jeff[fit] == gold[fit]) ^ (bm25[fit] == gold[fit]))
        ]
        target = (bm25[eligible] == gold[eligible]).astype(np.int8)
        if len(eligible) < 10 or len(np.unique(target)) != 2:
            raise ValueError(f"Fold {fold}: insufficient duel supervision")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=20261117),
        )
        clf.fit(features[eligible], target)
        probs = clf.predict_proba(features[valid])[:, 1]
        chance[valid] = probs
        fold_records.append(
            {
                "fold": fold,
                "fit_cases": len(fit),
                "validation_cases": len(valid),
                "fit_decisive_duels": len(eligible),
                "fit_bm25_correct_duels": int(target.sum()),
                "validation_disagreements": int(np.sum(jeff[valid] != bm25[valid])),
            }
        )
    if not np.isfinite(chance).all() or np.any((chance < 0) | (chance > 1)):
        raise ValueError("Crossfit predictions incomplete")
    return chance, fold_records


def run(root: Path = ROOT, *, strict_global2: bool = False) -> dict:
    data_dir = root / "data"
    train = load_records(data_dir / "tcm-sd/train.json")
    dev = load_records(data_dir / "tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Unexpected TCM-SD Train/Dev sample counts")
    freq = Counter(str(r["norm_syndrome"]) for r in train)
    classes = sorted(freq)
    if len(classes) != 148:
        raise ValueError("Unexpected syndrome classes")
    tail = set(sorted(classes, key=lambda x: (freq[x], x))[:37])
    gold = np.asarray([str(row["norm_syndrome"]) for row in dev], dtype=str)
    train_text = {clinical_text(row) for row in train}
    clean = np.asarray([clinical_text(r) not in train_text for r in dev])
    ids = np.asarray(transitive_group_ids(dev), dtype=str)
    if int(clean.sum()) != 4957:
        raise ValueError("Exact Train/Dev clinical-text drift")
    folder = data_dir / "case_rag" / TAG
    with (folder / "baseline.jsonl").open(encoding="utf-8") as stream:
        original_rows = [json.loads(line) for line in stream]
    with (folder / "provenance.jsonl").open(encoding="utf-8") as stream:
        provenance = [json.loads(line) for line in stream]
    with (folder / "global2.jsonl").open(encoding="utf-8") as stream:
        global_rows = [json.loads(line) for line in stream]
    if (
        len(original_rows) != len(dev)
        or len(provenance) != len(dev)
        or len(global_rows) != len(dev)
    ):
        raise ValueError("Frozen full Dev input alignment failed")
    with (data_dir / "jeff/dev_k16.jsonl").open(encoding="utf-8") as stream:
        for index, (row, line, source_row) in enumerate(
            zip(original_rows, stream, global_rows, strict=True)
        ):
            if row["id"] != f"dev-{index:06d}" or row != json.loads(line):
                raise ValueError(f"Jeff input differs at case {index}")
            if row["options"] != source_row["options"]:
                raise ValueError(f"Global2 candidates differ at case {index}")
    models: dict[str, dict] = {}
    for method in METHODS:
        study_path, reference = frozen_paths(root, method)
        baseline_logits = read_model_logits(reference, num_cases=5486)
        report = json.loads(study_path.read_text(encoding="utf-8"))
        models[f"{method}_base"] = {"rows": original_rows, "logits": baseline_logits}
        if report["scores"]["top1_correct_n"] != int(
            np.sum(_prediction_from_logits(original_rows, baseline_logits) == gold)
        ):
            raise ValueError(f"{method} original logits do not match the reported correct count")
        path = folder / "scores" / f"paper-{method}-full" / "global2.jsonl"
        if path.is_file():
            logits = read_condition_logits(path, global_rows)
            models[f"{method}_global2"] = {"rows": global_rows, "logits": logits}
    missing = [method for method in METHODS if f"{method}_global2" not in models]
    if strict_global2 and missing:
        raise FileNotFoundError(
            f"Completed H200 Global2 logits missing for {missing}; "
            "sync original score JSONLs before CPU routing"
        )
    results = {}
    for model_name, data in models.items():
        feats, jeff, bm25 = prediction_features(data["rows"], data["logits"], freq, provenance)
        probabilities, folds = group_crossfit_router(feats, gold, jeff, bm25, ids)
        methods = {
            "jeff": jeff,
            "bm25": bm25,
        }
        for threshold in (0.5, 0.65, 0.8):
            mask = (jeff != bm25) & (probabilities >= threshold)
            methods[f"group_crossfit_duel_p{threshold:g}"] = np.where(mask, bm25, jeff)
        results[model_name] = {
            "trained_on": "other grouped Dev folds (exploratory, Dev previously inspected)",
            "feature_count": feats.shape[1],
            "folds": folds,
            "rows": len(gold),
            "metrics": {
                name: {
                    **metrics(gold.tolist(), pred.tolist(), classes, tail, clean),
                    "switches_vs_jeff": int(np.sum(pred != jeff)),
                    "paired_vs_jeff": paired_errors(gold, jeff, pred),
                }
                for name, pred in methods.items()
            },
        }
    return {
        "status": "CASE_RAG_CPU_ROUTER_EXPLORATORY_COMPLETE",
        "scope": "exploratory-only GroupKFold within previously inspected Dev",
        "model_count": len(models),
        "model_names": sorted(models),
        "num_dev_cases": 5486,
        "missing_global2_models": missing,
        "clean_after_excluding_train_identical": 4957,
        "num_gold_tail_cases": int(np.isin(gold, list(tail)).sum()),
        "official_test_used": False,
        "new_gpu_forwards": 0,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-global2", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/case_rag_cpu_router_2026-11-17.json",
    )
    args = parser.parse_args()
    report = run(strict_global2=args.strict_global2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{report['status']} models={len(report['models'])}")
    for model, content in report["models"].items():
        print(f"=== {model} ===")
        for name, metric in content["metrics"].items():
            print(
                f"  {name:27s} Top-1={metric['top1_percent']:.3f}% "
                f"Macro-F1={metric['macro_f1_148_percent']:.3f}% "
                f"Tail-F1={metric['tail_37_macro_f1_percent']:.3f}% "
                f"switches={metric['switches_vs_jeff']}"
            )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
