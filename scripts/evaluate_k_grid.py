"""Run all K=1..35 on the full original TCM-SD Dev split.

Compares the BM25's raw top-ranked syndrome against lightweight CPU
candidate rerankers trained exclusively on original Train. This is an
auxiliary baseline; it does NOT execute Gemma or any LoRA.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.retrieval import chinese_bigrams

K_MIN = 1
K_MAX = 35


def metric_rows(
    *,
    gold: np.ndarray,
    candidate_names: list[tuple[str, ...]],
    classes: list[str],
    score_matrix: np.ndarray | None = None,
    max_k: int = K_MAX,
) -> list[dict]:
    """Compute all K values from one candidate pool and one fixed scoring model."""
    n = len(gold)
    if n == 0 or len(candidate_names) != n:
        raise ValueError("Gold and candidates must be nonempty and aligned")
    if not 1 <= max_k <= min(len(x) for x in candidate_names):
        raise ValueError("Invalid maximum K for candidate rows")

    class_ids = {label: i for i, label in enumerate(classes)}
    if len(class_ids) != len(classes):
        raise ValueError("The class vocabulary contains duplicate labels")
    if any(label not in class_ids for label in gold):
        raise ValueError("A gold label is not in the class vocabulary")
    try:
        candidate_ids = np.asarray(
            [[class_ids[label] for label in row[:max_k]] for row in candidate_names],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown candidate label: {exc}") from exc
    if any(len(set(row)) != len(row) for row in candidate_names):
        raise ValueError("Repeated candidate labels")

    if score_matrix is not None:
        if score_matrix.shape != (n, len(classes)):
            raise ValueError("Score matrix shape does not match cases/classes")
        if not np.isfinite(score_matrix).all():
            raise ValueError("Score matrix contains non-finite values")
        ranked_scores = np.take_along_axis(score_matrix, candidate_ids, axis=1)
    else:
        ranked_scores = None

    gold_ids = np.asarray([class_ids[label] for label in gold], dtype=np.int32)
    matches = candidate_ids == gold_ids[:, None]
    results: list[dict] = []
    for k in range(1, max_k + 1):
        # The same fixed classifier is evaluated with increasing candidate sets;
        # the classifier itself is never retrained for each K.
        if ranked_scores is None:
            selected_position = np.zeros(n, dtype=np.int32)
        else:
            selected_position = np.argmax(ranked_scores[:, :k], axis=1)
        predicted = candidate_ids[np.arange(n), selected_position]
        correct = predicted == gold_ids
        coverage = matches[:, :k].any(axis=1)
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_ids,
            predicted,
            labels=np.arange(len(classes), dtype=np.int32),
            zero_division=0,
        )
        assert int(sum(support)) == n
        results.append(
            {
                "k": k,
                "n": n,
                "oracle_covered_n": int(coverage.sum()),
                "oracle_micro_recall": float(coverage.mean()),
                "oracle_macro_recall": float(
                    np.mean([float(coverage[gold_ids == i].mean()) for i in range(len(classes))])
                ),
                "correct_n": int(correct.sum()),
                "top1": float(correct.mean()),
                "macro_f1": float(np.mean(f1)),
                "conditional_accuracy_given_gold_in_candidates": (
                    float(correct.sum() / coverage.sum()) if coverage.any() else None
                ),
                "per_class": [
                    {
                        "label": label,
                        "support": int(support[i]),
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "f1": float(f1[i]),
                    }
                    for i, label in enumerate(classes)
                ],
            }
        )
    return results


def prepare_inputs(root: Path) -> tuple[list[dict], list[dict], list[tuple[str, ...]]]:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    predictions = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(train) != 43180 or len(dev) != len(predictions) or len(dev) != 5486:
        raise ValueError("Official Train/Dev split size mismatch")
    candidate_names = []
    for i, (source, saved) in enumerate(zip(dev, predictions, strict=True)):
        if saved.case_id != f"dev-{i:06d}":
            raise ValueError("BM25 Dev candidate order mismatch")
        if saved.gold_labels != (str(source["norm_syndrome"]),):
            raise ValueError("BM25 Dev gold does not match official Dev")
        if len(saved.candidate_labels) < K_MAX:
            raise ValueError("Cached candidate list has fewer than 35 labels")
        if saved.predicted_label is not None:
            raise ValueError("Candidate input must not include model predictions")
        candidate_names.append(saved.candidate_labels)
    return train, dev, candidate_names


def run(root: Path, *, max_k: int) -> dict:
    train, dev, candidate_names = prepare_inputs(root)
    classes = sorted({str(record["norm_syndrome"]) for record in train})
    if len(classes) != 148:
        raise ValueError("Expected 148 TCM-SD classes")

    gold = np.asarray([str(record["norm_syndrome"]) for record in dev], dtype=object)
    base = metric_rows(gold=gold, candidate_names=candidate_names, classes=classes, max_k=max_k)

    start = perf_counter()
    vectorizer = CountVectorizer(
        analyzer=chinese_bigrams,
        dtype=np.float32,
        min_df=2,
    )
    train_features = vectorizer.fit_transform([clinical_text(record) for record in train])
    dev_features = vectorizer.transform([clinical_text(record) for record in dev])
    feature_seconds = perf_counter() - start

    output = {
        "study": "TCMDiscern",
        "date": "2026-11-11",
        "status": "COMPLETED_CPU_AUXILIARY_BASELINES_ONLY",
        "dataset": "official_TCM_SD_train_43180_dev_5486_test_untouched",
        "excluded_records": 0,
        "k_values": list(range(1, max_k + 1)),
        "k_count": max_k,
        "same_dev_cases_and_candidate_ranking_for_every_k": True,
        "same_classifier_weights_for_every_k": True,
        "input_fields": ["chief_complaint", "description", "detection"],
        "candidate_source": "precomputed BM25 Bigram train-case-to-syndrome rankings",
        "methods": {},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu": "not used",
        },
        "timing_seconds": {"vectorize_train_and_dev": round(feature_seconds, 3)},
        "model_predictions_status": {
            "gemma_4_31b_it": "NOT_RUN",
            "gemma_lora": "NOT_RUN",
        },
        "method_limits": [
            "Models trained only on Train. Dev gold used only to evaluate.",
            "No cases removed or deduplicated in the official benchmark.",
            "MultinomialNB and ComplementNB are CPU baselines, not Gemma nor proxies for Gemma performance.",
            "Each classifier's scores are fixed across K. Listwise LLM prompts may behave differently.",
            "Macro-F1 is classification F1 and differs from macro retrieval recall.",
        ],
    }

    # Do not duplicate 148 per-class entries for the untrained rank-1
    # retriever at every K: the predictions are unchanged by K.
    output["methods"]["bm25_rank1"] = {
        "description": "Always choose the highest-ranked BM25 syndrome; independent of K",
        "results": [
            {key: value for key, value in row.items() if key != "per_class"} for row in base
        ],
        "per_class_k1": base[0]["per_class"],
    }

    for name, model in (
        ("multinomial_nb", MultinomialNB(alpha=1.0)),
        ("complement_nb", ComplementNB(alpha=1.0)),
    ):
        begin = perf_counter()
        model.fit(train_features, [str(row["norm_syndrome"]) for row in train])
        training_seconds = perf_counter() - begin
        begin = perf_counter()
        scores = model.predict_log_proba(dev_features)
        inference_seconds = perf_counter() - begin
        # The classes returned by sklearn must be aligned with candidate IDs.
        if list(model.classes_) != classes:
            raise ValueError(f"{name} class order does not match the official label vocabulary")
        begin = perf_counter()
        metrics = metric_rows(
            gold=gold,
            candidate_names=candidate_names,
            classes=classes,
            score_matrix=scores,
            max_k=max_k,
        )
        eval_seconds = perf_counter() - begin
        output["methods"][name] = {
            "description": "Train-only bag-of-CJK-bigrams classifier reranking BM25 candidate labels",
            "parameters": {"alpha": 1.0, "min_df": 2, "vectorizer": "CountVectorizer"},
            "timing_seconds": {
                "fit_model": round(training_seconds, 3),
                "dev_score_once": round(inference_seconds, 3),
                "all_k_evaluation": round(eval_seconds, 3),
            },
            "results": [
                {key: value for key, value in row.items() if key != "per_class"} for row in metrics
            ],
            "per_class": {str(row["k"]): row["per_class"] for row in metrics},
        }
        print(
            f"{name}: top1 K1={metrics[0]['top1']:.4f}, "
            f"K{max_k}={metrics[-1]['top1']:.4f}, "
            f"best macro-F1 K={max(metrics, key=lambda x: x['macro_f1'])['k']}",
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every K from 1 to 35 on official Dev")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--max-k", type=int, default=K_MAX)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "experiments/k1_35_cpu_selector_dev_2026-11-11.json",
    )
    args = parser.parse_args()
    if not 1 <= args.max_k <= 35:
        parser.error("--max-k must be within 1..35")
    print("Train 43,180; Dev 5,486; Test untouched; no deduplication", flush=True)
    report = run(root, max_k=args.max_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
