"""Score frozen 5486 Dev Case-RAG versus Jeff, BM25, and source-label controls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from scripts.prepare_case_rag_paper_ablation import ROOT, TAG
from scripts.run_case_rag_jeff import read_condition_logits
from tcmdiscern.case_rag import transitive_group_ids
from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_scoring import read_model_logits

EPOCH_RUN = "seed20261112-e2"
OUTPUT_ARMS = ("global2", "random2", "global2_labelhide", "global2_labelonly")


def expected_output_arms(method: str, output: Path) -> tuple[str, ...]:
    """Include optional full-Dev replication controls only once their files exist."""
    if method == "s2":
        return OUTPUT_ARMS
    if method not in {"s1", "s3"}:
        raise ValueError("Unsupported full-Dev checkpoint")
    controls = ("random2", "global2_labelhide", "global2_labelonly")
    return (
        "global2",
        *(arm for arm in controls if (output / f"{arm}.jsonl").is_file()),
    )


def _labels(rows: list[dict], matrix: np.ndarray) -> list[str]:
    if matrix.shape != (len(rows), 16):
        raise ValueError("Full Dev logits require exactly 16 candidates per patient")
    return [row["options"][int(np.argmax(matrix[i]))]["id"] for i, row in enumerate(rows)]


def metrics(
    gold: list[str], labels: list[str], classes: list[str], tail: set[str], clean: np.ndarray
) -> dict:
    matches = np.asarray([a == b for a, b in zip(gold, labels, strict=True)])
    in_tail = np.asarray([name in tail for name in gold])
    return {
        "top1_percent": round(100 * float(matches.mean()), 4),
        "correct": int(matches.sum()),
        "macro_f1_148_percent": round(
            100 * f1_score(gold, labels, labels=classes, average="macro", zero_division=0), 4
        ),
        "tail_37_macro_f1_percent": round(
            100 * f1_score(gold, labels, labels=sorted(tail), average="macro", zero_division=0), 4
        ),
        "tail_gold_cases": int(in_tail.sum()),
        "tail_top1_percent": (
            round(100 * float(matches[in_tail].mean()), 4) if in_tail.any() else None
        ),
        "no_exact_train_text_n": int(clean.sum()),
        "no_exact_train_text_top1_percent": round(100 * float(matches[clean].mean()), 4),
    }


def paired(
    gold: list[str], reference: list[str], predicted: list[str], inverse: np.ndarray, groups: int
) -> dict:
    correct = np.asarray([a == b for a, b in zip(predicted, gold, strict=True)])
    baseline = np.asarray([a == b for a, b in zip(reference, gold, strict=True)])
    rescue = int((correct & ~baseline).sum())
    damaged = int((~correct & baseline).sum())
    # Group bootstrap preserves duplicate-patient/duplicate-text dependencies.
    differences = np.bincount(
        inverse, weights=(correct.astype(int) - baseline.astype(int)), minlength=groups
    )
    sizes = np.bincount(inverse, minlength=groups)
    random = np.random.default_rng(20261117)
    values = []
    for _ in range(1000):
        sampled = random.integers(0, groups, size=groups)
        values.append(float(100 * differences[sampled].sum() / sizes[sampled].sum()))
    return {
        "delta_top1_pp": round(100 * float(correct.mean() - baseline.mean()), 4),
        "rescued": rescue,
        "broken": damaged,
        "exact_mcnemar_p": float(binomtest(min(rescue, damaged), rescue + damaged).pvalue)
        if rescue + damaged
        else 1.0,
        "cluster_bootstrap_95ci_pp": [
            round(float(value), 4) for value in np.percentile(values, [2.5, 97.5])
        ],
    }


def evaluate(method: str, *, root: Path = ROOT, strict: bool = False) -> dict:
    from scripts.run_case_rag_jeff import load_inputs

    if method not in {"s1", "s2", "s3"}:
        raise ValueError("Unsupported full-Dev checkpoint")
    folder = root / "data/case_rag" / TAG
    if not (folder / "paper_ablations.json").is_file():
        raise FileNotFoundError("Full Dev paper ablations are not prepared")
    published, ref = frozen_paths(root, method)
    study = json.loads(published.read_text(encoding="utf-8"))
    if (
        study["rows"] != 5486
        or study["checkpoint"]["method"] != method
        or study["attention_backend"] != "axolotl_hybrid_fa2_sdpa"
    ):
        raise ValueError("Frozen Jeff study metadata changed")
    saved = read_model_logits(ref, num_cases=5486)
    collection = load_inputs(folder)
    original = collection["baseline"]
    dev = load_records(root / "data/tcm-sd/dev.json")
    train = load_records(root / "data/tcm-sd/train.json")
    gold = [str(row["norm_syndrome"]) for row in dev]
    frequencies = Counter(str(row["norm_syndrome"]) for row in train)
    labels = sorted(frequencies)
    tail = set(sorted(labels, key=lambda x: (frequencies[x], x))[:37])
    exact = {clinical_text(row) for row in train}
    clean = np.asarray([clinical_text(row) not in exact for row in dev])
    clusters, inverse = np.unique(transitive_group_ids(dev), return_inverse=True)
    source_label = [
        row["state"]["retrieved_cases"][0]["candidate_syndrome"] for row in collection["global2"]
    ]
    bm25 = [row["options"][0]["id"] for row in original]
    baseline = _labels(original, saved)
    reference = metrics(gold, baseline, labels, tail, clean)
    if abs(reference["top1_percent"] / 100 - study["scores"]["top1_accuracy"]) > 1e-6:
        raise ValueError("Original full-Dev Top-1 and raw logits disagree")
    if abs(reference["macro_f1_148_percent"] / 100 - study["scores"]["macro_f1"]) > 1e-6:
        raise ValueError("Original full-Dev Macro-F1 and raw logits disagree")
    result = {
        "status": "CASE_RAG_PAPER_FULL_DEV_PENDING_GPU",
        "method": method,
        "cases": len(dev),
        "classes": len(labels),
        "tail_class_definition": "37 least frequent Train labels, fixed prior to Dev scoring",
        "tail_gold_classes": len(set(gold) & tail),
        "train_dev_exact_text_overlap_n": int((~clean).sum()),
        "dev_dependency_clusters": len(clusters),
        "reference_checkpoint": study["checkpoint"]["source_run_id"],
        "methods": {
            "baseline": reference,
            "bm25": metrics(gold, bm25, labels, tail, clean),
            "retrieved_source_label_cpu": metrics(gold, source_label, labels, tail, clean),
        },
        "comparisons": {},
        "inference_time": {
            "original_full_dev_mean_ms": round(1000 * study["mean_wall_seconds_per_decision"], 3),
            "original_timing_source": "original separate full-Dev Jeff evaluation",
            "arms": {},
        },
        "missing_arms": [],
        "test_used": False,
    }
    for name, predictions, reference_predictions in (
        ("retrieved_source_label_cpu_vs_bm25", source_label, bm25),
        ("retrieved_source_label_cpu_vs_baseline", source_label, baseline),
        ("bm25_vs_baseline", bm25, baseline),
    ):
        result["comparisons"][name] = paired(
            gold, reference_predictions, predictions, inverse, len(clusters)
        )
    output = folder / "scores" / f"paper-{method}-full"
    expected = expected_output_arms(method, output)
    model_labels = {}
    for arm in expected:
        path = output / f"{arm}.jsonl"
        if not path.is_file():
            result["missing_arms"].append(arm)
            continue
        if arm in collection:
            prompts = collection[arm]
        else:
            with (folder / f"{arm}.jsonl").open(encoding="utf-8") as stream:
                prompts = [json.loads(line) for line in stream]
        scores = read_condition_logits(path, prompts)
        timing = path.with_name(path.stem + ".timings.jsonl")
        if timing.is_file():
            from tcmdiscern.case_rag_runtime import _timing_summary

            result["inference_time"]["arms"][arm] = _timing_summary(timing, count=5486)
        decisions = _labels(prompts, scores)
        model_labels[arm] = decisions
        result["methods"][arm] = metrics(gold, decisions, labels, tail, clean)
        for name, other in (
            ("baseline", baseline),
            ("bm25", bm25),
            ("retrieved_source_label_cpu", source_label),
        ):
            result["comparisons"][f"{arm}_vs_{name}"] = paired(
                gold, other, decisions, inverse, len(clusters)
            )
    for control in ("random2", "global2_labelhide", "global2_labelonly"):
        if "global2" in model_labels and control in model_labels:
            result["comparisons"][f"global2_vs_{control}"] = paired(
                gold,
                model_labels[control],
                model_labels["global2"],
                inverse,
                len(clusters),
            )
    if not result["missing_arms"]:
        result["status"] = "CASE_RAG_PAPER_FULL_DEV_COMPLETED"
    elif strict:
        raise FileNotFoundError(f"Missing {method} GPU results: {result['missing_arms']}")
    public = root / f"experiments/case_rag_paper_full_dev_{method}_20261117.json"
    public.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=("s1", "s2", "s3"), default=["s2", "s1", "s3"]
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    for method in args.methods:
        report = evaluate(method, strict=args.strict)
        print(method, report["status"], "missing=", report["missing_arms"], flush=True)
        for arm, row in report["methods"].items():
            print(
                f"  {arm}: Top-1 {row['top1_percent']}%, Macro-F1 {row['macro_f1_148_percent']}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
