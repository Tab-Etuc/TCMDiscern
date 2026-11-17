"""CPU-first per-case error decomposition for the frozen 5486-case TCM-SD Dev.

Uses saved decision logits, original 16 BM25 candidates, and Train-retrieved
case labels. No model imports, training, CUDA, or official Test data.

Locally available original S1/S2/S3 logits are always analyzed. Once H200
Case-RAG score JSONLs are copied into their original scores directories,
the *same command* automatically extends the analysis to Global2 and the
label/content ablations.

Usage:
  .venv/bin/python -m scripts.analyze_case_rag_cpu_first
  .venv/bin/python -m scripts.analyze_case_rag_cpu_first --strict-global2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.evaluate_case_rag_paper_dev import metrics as published_metrics
from scripts.prepare_case_rag_paper_ablation import ROOT, TAG
from scripts.run_case_rag_jeff import read_condition_logits
from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_scoring import read_model_logits

METHODS = ("s1", "s2", "s3")
ARMS = ("global2", "random2", "global2_labelhide", "global2_labelonly")


def paired_errors(gold: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Count actual complementary correct cases, rather than independent rates."""
    if gold.shape != reference.shape or gold.shape != candidate.shape:
        raise ValueError("Pairwise predictions must have identical shapes")
    a = reference == gold
    b = candidate == gold
    rescued = int((~a & b).sum())
    broken = int((a & ~b).sum())
    pvalue = (
        float(binomtest(min(rescued, broken), rescued + broken).pvalue)
        if (rescued + broken)
        else 1.0
    )
    return {
        "both_correct": int((a & b).sum()),
        "reference_only_correct": broken,
        "candidate_only_correct": rescued,
        "both_wrong": int((~a & ~b).sum()),
        "rescue": rescued,
        "break": broken,
        "net_correct_change": rescued - broken,
        "delta_top1_pp": round(100 * (rescued - broken) / len(gold), 4),
        "mcnemar_exact_p_unadjusted": pvalue,
        "oracle_pair_top1_percent": round(100 * float((a | b).mean()), 4),
    }


def split_by_gold_frequency(gold: np.ndarray, counts: Counter, predictions: dict) -> list[dict]:
    """Predeclare descriptive bins from Train, not tuning after reading Dev."""
    output = []
    for name, low, high in (
        ("train_1_to_9", 1, 9),
        ("train_10_to_49", 10, 49),
        ("train_50_to_199", 50, 199),
        ("train_200_plus", 200, None),
    ):
        mask = np.array(
            [low <= counts[str(g)] and (high is None or counts[str(g)] <= high) for g in gold]
        )
        sample_n = int(mask.sum())
        output.append(
            {
                "train_frequency_bin": name,
                "dev_cases": sample_n,
                "top1_percent": {
                    key: round(100 * float(np.mean(pred[mask] == gold[mask])), 4)
                    if sample_n
                    else None
                    for key, pred in predictions.items()
                },
            }
        )
    if sum(a["dev_cases"] for a in output) != len(gold):
        raise ValueError("Some gold labels are absent from the official Train split")
    return output


def divergent_tail_choices(
    gold: np.ndarray, jeff: np.ndarray, competitor: np.ndarray, tail: set[str]
) -> dict:
    """When predictions disagree, inspect which one favored a rare syndrome."""
    ref_tail = np.isin(competitor, list(tail))
    model_tail = np.isin(jeff, list(tail))
    disagreement = jeff != competitor

    def summarize(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        return {
            "n": n,
            "competitor_correct": int(((competitor == gold) & mask).sum()),
            "jeff_correct": int(((jeff == gold) & mask).sum()),
            "neither_correct": int(((jeff != gold) & (competitor != gold) & mask).sum()),
            "competitor_win_rate_percent": round(
                100 * float(((competitor == gold) & mask).sum()) / n, 4
            )
            if n
            else None,
        }

    return {
        "competitor_tail_jeff_not_tail": summarize(disagreement & ref_tail & ~model_tail),
        "jeff_tail_competitor_not_tail": summarize(disagreement & model_tail & ~ref_tail),
        "both_non_tail_but_disagree": summarize(disagreement & ~model_tail & ~ref_tail),
        "both_tail_but_disagree": summarize(disagreement & model_tail & ref_tail),
    }


def fixed_frequency_override_diagnostics(
    gold: np.ndarray,
    models: dict[str, np.ndarray],
    frequency: Counter,
    classes: list[str],
    tail: set[str],
    clean: np.ndarray,
) -> list[dict]:
    """Diagnostic only: favor lexical evidence for rare predicted classes.

    These are *predeclared* Train-frequency cutoffs, not a learned or
    Dev-optimized routing policy. Always report all values without selecting
    the best on Dev.
    """
    output = []
    for model_name in ("s1_base", "s2_base", "s3_base", "s1_global2", "s2_global2", "s3_global2"):
        if model_name not in models:
            continue
        jeff = models[model_name]
        model_freq = np.asarray([frequency[label] for label in jeff])
        base_accuracy = float(np.mean(jeff == gold))
        for competitor_name in ("bm25", "source_label"):
            competitor = models[competitor_name]
            competitor_freq = np.asarray([frequency[label] for label in competitor])
            for maximum in (9, 49, 199):
                swap = (competitor_freq <= maximum) & (model_freq > maximum) & (jeff != competitor)
                predicted = np.where(swap, competitor, jeff)
                values = published_metrics(gold.tolist(), predicted.tolist(), classes, tail, clean)
                output.append(
                    {
                        "model": model_name,
                        "alternative": competitor_name,
                        "train_frequency_cutoff": maximum,
                        "switched_cases": int(swap.sum()),
                        "rescued": int(((predicted == gold) & (jeff != gold)).sum()),
                        "broken": int(((predicted != gold) & (jeff == gold)).sum()),
                        "delta_top1_pp": round(
                            100 * (float(np.mean(predicted == gold)) - base_accuracy), 4
                        ),
                        "top1_percent": values["top1_percent"],
                        "macro_f1_148_percent": values["macro_f1_148_percent"],
                        "tail_37_macro_f1_percent": values["tail_37_macro_f1_percent"],
                    }
                )
    return output


def _prediction_from_logits(rows: list[dict], matrix: np.ndarray) -> np.ndarray:
    if matrix.shape != (len(rows), 16):
        raise ValueError("Predictions must have 16 candidate logits")
    return np.asarray(
        [row["options"][int(np.argmax(matrix[i]))]["id"] for i, row in enumerate(rows)], dtype=str
    )


def run(root: Path = ROOT, *, strict_global2: bool = False) -> dict:
    """Strictly validate originals and optionally use complete Case-RAG logits."""
    data = root / "data"
    train = load_records(data / "tcm-sd/train.json")
    dev = load_records(data / "tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Official TCM-SD Train/Dev cardinality changed")
    frequency = Counter(str(r["norm_syndrome"]) for r in train)
    classes = sorted(frequency)
    if len(classes) != 148:
        raise ValueError("Original 148 syndrome label space changed")
    tail = set(sorted(classes, key=lambda x: (frequency[x], x))[:37])
    folder = data / "case_rag" / TAG
    with (folder / "baseline.jsonl").open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    with (folder / "global2.jsonl").open(encoding="utf-8") as file:
        source_rows = [json.loads(line) for line in file]
    if len(rows) != 5486 or len(source_rows) != len(rows):
        raise ValueError("Frozen case-rag prompts are not complete")
    with (data / "jeff/dev_k16.jsonl").open(encoding="utf-8") as file:
        for i, (row, original_line, source_row) in enumerate(
            zip(rows, file, source_rows, strict=True)
        ):
            if row["id"] != f"dev-{i:06d}" or row != json.loads(original_line):
                raise ValueError(f"Original Jeff Dev prompt changed at {i}")
            if row["options"] != source_row["options"] or row["id"] != source_row["id"]:
                raise ValueError(f"Case-RAG altered original BM25 candidates at {i}")
            if len(row["options"]) != 16 or len({v["id"] for v in row["options"]}) != 16:
                raise ValueError(f"Invalid BM25 candidate set {i}")
            if len(source_row["state"]["retrieved_cases"]) != 2:
                raise ValueError(f"Wrong retrieval budget at case {i}")
    gold = np.asarray([str(row["norm_syndrome"]) for row in dev], dtype=str)
    bm25 = np.asarray([row["options"][0]["id"] for row in rows], dtype=str)
    source = np.asarray(
        [row["state"]["retrieved_cases"][0]["candidate_syndrome"] for row in source_rows], dtype=str
    )
    train_texts = {clinical_text(row) for row in train}
    clean = np.asarray([clinical_text(row) not in train_texts for row in dev])
    if int(clean.sum()) != 4957:
        raise ValueError("Exact Train/Dev duplicate exclusion drift")
    models: dict[str, np.ndarray] = {"bm25": bm25, "source_label": source}
    existing = {}
    for method in METHODS:
        _, path = frozen_paths(root, method)
        logits = read_model_logits(path, num_cases=5486)
        models[f"{method}_base"] = _prediction_from_logits(rows, logits)
        folder_scores = folder / "scores" / f"paper-{method}-full"
        for arm in ARMS:
            path = folder_scores / f"{arm}.jsonl"
            if not path.is_file():
                continue
            # Reads complete original logits only; never accepts a short prefix.
            scores = read_condition_logits(
                path, source_rows if arm == "global2" else _read_arm(folder, arm)
            )
            models[f"{method}_{arm}"] = _prediction_from_logits(
                source_rows if arm == "global2" else _read_arm(folder, arm), scores
            )
            existing[f"{method}_{arm}"] = str(path.relative_to(root))
    if strict_global2 and any(f"{method}_global2" not in models for method in METHODS):
        raise FileNotFoundError("Full-Dev Case-RAG logits not synced for S1/S2/S3")
    labels = set(map(str, gold))
    if labels != set(classes):
        raise ValueError("Expected all 148 gold syndrome classes in Dev")
    if not set(map(str, source)) <= set(classes):
        raise ValueError("Source label is absent from original Train syndrome set")
    scores = {}
    for key, labels_pred in models.items():
        record = published_metrics(gold.tolist(), labels_pred.tolist(), classes, tail, clean)
        scores[key] = record
    # Verify original full-Dev results before using any errors for conclusions.
    for method in METHODS:
        study_path, _ = frozen_paths(root, method)
        published = json.loads(study_path.read_text(encoding="utf-8"))
        expected = published["scores"]
        observed = scores[f"{method}_base"]
        if observed["correct"] != expected["top1_correct_n"]:
            raise ValueError(f"{method}: baseline logits disagree with published full Dev")
        if abs(observed["macro_f1_148_percent"] / 100 - expected["macro_f1"]) > 1e-5:
            raise ValueError(f"{method}: baseline F1 disagrees with original full Dev")
    # Predefine all S3-focused contrasts and extend to S1/S2 if GPU scores exist.
    pairs = {}
    for method in METHODS:
        model = (
            models[f"{method}_global2"]
            if f"{method}_global2" in models
            else models[f"{method}_base"]
        )
        label = f"{method}_{'global2' if f'{method}_global2' in models else 'base'}"
        for competitor in ("bm25", "source_label"):
            pairs[f"{label}_vs_{competitor}"] = paired_errors(gold, models[competitor], model)
        if f"{method}_global2" in models:
            pairs[f"{method}_global2_vs_base"] = paired_errors(
                gold, models[f"{method}_base"], model
            )
            for control in ("random2", "global2_labelhide", "global2_labelonly"):
                key = f"{method}_{control}"
                if key in models:
                    pairs[f"{method}_global2_vs_{control}"] = paired_errors(
                        gold, models[key], model
                    )
    tail_conflicts = {}
    for method in METHODS:
        label = f"{method}_global2" if f"{method}_global2" in models else f"{method}_base"
        for competitor in ("bm25", "source_label"):
            tail_conflicts[f"{label}_vs_{competitor}"] = divergent_tail_choices(
                gold, models[label], models[competitor], tail
            )
    sample = {
        "status": "CASE_RAG_CPU_DIAGNOSTICS_COMPLETED",
        "official_test_used": False,
        "dev_cases": len(dev),
        "train_cases": len(train),
        "classes": len(classes),
        "tail_definition": "37 least frequent Train labels",
        "tail_gold_cases": int(np.isin(gold, list(tail)).sum()),
        "clean_no_exact_train_text_n": int(clean.sum()),
        "case_rag_logits_available": sorted(existing),
        "missing_case_rag_global2": [
            f"{method}_global2" for method in METHODS if f"{method}_global2" not in models
        ],
        "metrics": scores,
        "paired_error_decomposition": pairs,
        "tail_prediction_disagreement": tail_conflicts,
        "train_frequency_gold_bins": split_by_gold_frequency(gold, frequency, models),
        "fixed_frequency_override_diagnostics_dev_only": fixed_frequency_override_diagnostics(
            gold, models, frequency, classes, tail, clean
        ),
        "oracle_ceilings_descriptive_only": {
            "bm25_or_s3_baseline_top1_percent": pairs["s3_base_vs_bm25"][
                "oracle_pair_top1_percent"
            ],
            "source_or_s3_baseline_top1_percent": pairs["s3_base_vs_source_label"][
                "oracle_pair_top1_percent"
            ],
        },
        "scope_note": (
            "CPU diagnostic on previously inspected Dev; oracle is a feasibility upper bound, "
            "not a learnable model. Global2-specific case-level claims require H200 score JSONLs."
        ),
    }
    if "s3_global2_vs_bm25" in pairs:
        sample["oracle_ceilings_descriptive_only"]["bm25_or_s3_global2_top1_percent"] = pairs[
            "s3_global2_vs_bm25"
        ]["oracle_pair_top1_percent"]
    return sample


def _read_arm(folder: Path, arm: str) -> list[dict]:
    with (folder / f"{arm}.jsonl").open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    if len(rows) != 5486:
        raise ValueError(f"Frozen {arm} evidence input count changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-global2", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/case_rag_cpu_first_2026-11-17.json",
    )
    args = parser.parse_args()
    data = run(strict_global2=args.strict_global2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("STATUS:", data["status"])
    print("DEV:", data["dev_cases"], "TAIL_GOLD:", data["tail_gold_cases"])
    print("GLOBAL2_SAVED:", data["case_rag_logits_available"])
    print("SCORES:")
    for name, row in data["metrics"].items():
        print(
            f"  {name:28s} Top1={row['top1_percent']:7.3f}% Macro-F1={row['macro_f1_148_percent']:7.3f}% Tail-F1={row['tail_37_macro_f1_percent']:7.3f}%"
        )
    print("S3 ERROR COMPLEMENTARITY:")
    for name, row in data["paired_error_decomposition"].items():
        if name.startswith("s3_"):
            print(
                f"  {name}: rescue={row['rescue']}, break={row['break']}, pair-oracle={row['oracle_pair_top1_percent']:.3f}%"
            )
    print("REPORT:", args.output)


if __name__ == "__main__":
    main()
