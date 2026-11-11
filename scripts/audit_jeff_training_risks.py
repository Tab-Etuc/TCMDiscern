"""Research-quality risk audit before fine-tuning Jeff on official TCM-SD.

Only Train and Dev are read; the reserved Test file is never opened.
Original cases, labels and candidate rows remain unchanged. Only aggregate
statistics without raw clinical text or patient IDs are published.
"""

from __future__ import annotations

import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tcmdiscern.datasets import load_records, normalized_input
from tcmdiscern.evaluation import read_predictions

ROOT = Path(__file__).resolve().parents[1]
K = 16
EXPECTED_TRAIN = 43180
EXPECTED_DEV = 5486
EXPECTED_LABELS = 148


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _canonical(label: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKC", label)
        if not unicodedata.category(ch).startswith(("P", "Z"))
    ).casefold()


def _bucket_support(support: Counter[str]) -> dict[str, int]:
    buckets = {
        "1_to_10": (1, 10),
        "11_to_20": (11, 20),
        "21_to_50": (21, 50),
        "51_to_100": (51, 100),
        "101_to_500": (101, 500),
        "over_500": (501, float("inf")),
    }
    return {name: sum(a <= n <= b for n in support.values()) for name, (a, b) in buckets.items()}


def _summarize_group(indices: list[int], dev: list[dict], candidates: list) -> dict:
    correct_top1 = 0
    gold_in_k = 0
    for index in indices:
        gold = str(dev[index]["norm_syndrome"])
        ranking = candidates[index].candidate_labels
        correct_top1 += ranking[0] == gold
        gold_in_k += gold in ranking[:K]
    return {
        "cases": len(indices),
        "bm25_top1_correct": correct_top1,
        "bm25_top1_accuracy": _rate(correct_top1, len(indices)),
        "bm25_k16_gold_covered": gold_in_k,
        "bm25_recall_at_16": _rate(gold_in_k, len(indices)),
    }


def audit(root: Path = ROOT) -> dict:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    if len(train) != EXPECTED_TRAIN or len(dev) != EXPECTED_DEV:
        raise ValueError("Only full official Train and Dev are permitted")

    train_support: Counter[str] = Counter(str(c["norm_syndrome"]) for c in train)
    dev_support: Counter[str] = Counter(str(c["norm_syndrome"]) for c in dev)
    if set(train_support) != set(dev_support) or len(train_support) != EXPECTED_LABELS:
        raise ValueError("Expected all 148 common official syndrome labels")

    by_text: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in train:
        by_text[normalized_input(item, full=True)][str(item["norm_syndrome"])] += 1
    duplicate_groups = sum(sum(labels.values()) > 1 for labels in by_text.values())
    conflict_groups = sum(len(labels) > 1 for labels in by_text.values())
    conflict_rows = sum(sum(labels.values()) for labels in by_text.values() if len(labels) > 1)
    total_duplicate_rows = sum(
        sum(labels.values()) for labels in by_text.values() if sum(labels.values()) > 1
    )

    oof_path = root / "data/candidates/bm25_train_oof.jsonl"
    oof: list[dict] = []
    with oof_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= EXPECTED_TRAIN:
                raise ValueError("OOF index has extra rows")
            row = json.loads(line)
            if row["case_id"] != f"train-{index:06d}" or row["gold_labels"] != [
                str(train[index]["norm_syndrome"])
            ]:
                raise ValueError(f"OOF Train row {index} mismatch")
            if len(row["candidate_labels"]) != 50 or len(set(row["candidate_labels"])) != 50:
                raise ValueError("OOF candidate rows must contain 50 distinct labels")
            oof.append(row)
    if len(oof) != EXPECTED_TRAIN:
        raise ValueError("OOF is incomplete")

    rank_hist: Counter[str] = Counter()
    negative_exposure: Counter[str] = Counter()
    train_k16_gold_found = 0
    for record, row in zip(train, oof, strict=True):
        gold = str(record["norm_syndrome"])
        ranked = row["candidate_labels"]
        try:
            rank = ranked.index(gold) + 1
        except ValueError:
            rank = 51
        if rank == 1:
            rank_hist["1"] += 1
        elif rank <= 5:
            rank_hist["2_to_5"] += 1
        elif rank <= K:
            rank_hist["6_to_16"] += 1
        elif rank <= 50:
            rank_hist["17_to_50"] += 1
        else:
            rank_hist["above_50"] += 1
        train_k16_gold_found += rank <= K
        negatives = [name for name in ranked if name != gold][: K - 1]
        if len(negatives) != K - 1:
            raise ValueError("Missing K-1 OOF hard negatives")
        negative_exposure.update(negatives)

    if sum(rank_hist.values()) != EXPECTED_TRAIN:
        raise ValueError("OOF rank histogram does not sum to original Train")
    if sum(negative_exposure.values()) != EXPECTED_TRAIN * (K - 1):
        raise ValueError("Negatives must contain 15 labels per original Train row")

    dev_candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(dev_candidates) != EXPECTED_DEV:
        raise ValueError("Dev BM25 candidate ranking is incomplete")
    overlap: list[int] = []
    nonoverlap: list[int] = []
    mismatch_with_identical_text = 0
    raw_gold_label_literal = 0
    dev_k16_gold_found = 0
    by_syndrome_coverage: Counter[str] = Counter()
    for index, (record, pred) in enumerate(zip(dev, dev_candidates, strict=True)):
        gold = str(record["norm_syndrome"])
        if pred.case_id != f"dev-{index:06d}" or pred.gold_labels != (gold,):
            raise ValueError(f"BM25 Dev candidate/gold mismatch at index {index}")
        text = normalized_input(record, full=True)
        if text in by_text:
            overlap.append(index)
            if gold not in by_text[text]:
                mismatch_with_identical_text += 1
        else:
            nonoverlap.append(index)
        raw_gold_label_literal += gold in " ".join(
            str(record.get(k) or "") for k in ("chief_complaint", "description", "detection")
        )
        if gold in pred.candidate_labels[:K]:
            dev_k16_gold_found += 1
            by_syndrome_coverage[gold] += 1

    by_class_16 = {label: by_syndrome_coverage[label] / dev_support[label] for label in dev_support}
    top1 = _summarize_group(list(range(EXPECTED_DEV)), dev, dev_candidates)
    if dev_k16_gold_found != 5409 or top1["bm25_top1_correct"] != 4366:
        raise ValueError("Dev evaluation differs from the previously frozen BM25 baseline")
    duplicate_subset = _summarize_group(overlap, dev, dev_candidates)
    nonduplicate_subset = _summarize_group(nonoverlap, dev, dev_candidates)

    collision: dict[str, list[str]] = defaultdict(list)
    for label in train_support:
        collision[_canonical(label)].append(label)
    after_normalization_duplicate_groups = sum(len(group) > 1 for group in collision.values())

    most_common_count = train_support.most_common(1)[0][1]
    least_count = min(train_support.values())
    ratios = sorted(negative_exposure[label] / count for label, count in train_support.items())
    return {
        "date": "2026-11-11",
        "model": "jgeuter/Jeff-1.1-Large",
        "status": "CPU_DATA_RISK_AUDIT_COMPLETED_NO_LORA_TRAINING",
        "scope": "TCM-SD official Train/Dev; Test not opened",
        "original_splits_retained": True,
        "deleted_train_or_dev_rows": 0,
        "train_class_imbalance": {
            "train_cases": EXPECTED_TRAIN,
            "label_count": len(train_support),
            "minimum_support": least_count,
            "median_support": statistics.median(train_support.values()),
            "maximum_support": most_common_count,
            "most_common_class_fraction": most_common_count / EXPECTED_TRAIN,
            "max_to_min_class_ratio": most_common_count / least_count,
            "support_buckets_number_of_classes": _bucket_support(train_support),
            "dev_minimum_class_support": min(dev_support.values()),
            "dev_classes_with_at_most_5_cases": sum(n <= 5 for n in dev_support.values()),
        },
        "oof_train_label_retrieval": {
            "original_train_cases": EXPECTED_TRAIN,
            "k": K,
            "gold_rank_distribution": dict(rank_hist),
            "gold_in_bm25_top_16_n": train_k16_gold_found,
            "gold_in_bm25_top_16_rate": train_k16_gold_found / EXPECTED_TRAIN,
            "gold_not_in_bm25_top_16_n": EXPECTED_TRAIN - train_k16_gold_found,
            "gold_always_injected_in_training_examples": True,
            "training_negative_examples_count": sum(negative_exposure.values()),
            "hard_negative_exposure_min": min(negative_exposure.values()),
            "hard_negative_exposure_median": statistics.median(negative_exposure.values()),
            "hard_negative_exposure_max": max(negative_exposure.values()),
            "negative_to_positive_exposure_ratio_median": statistics.median(ratios),
            "negative_to_positive_exposure_ratio_max": max(ratios),
            "classes_with_negative_exposure_over_10_times_positive": sum(
                negative_exposure[label] > 10 * count for label, count in train_support.items()
            ),
            "classes_never_used_as_a_hard_negative": sum(
                negative_exposure[label] == 0 for label in train_support
            ),
        },
        "train_exact_text_duplication": {
            "identical_full_clinical_text_groups_with_two_or_more_rows": duplicate_groups,
            "rows_belonging_to_identical_full_clinical_text_groups": total_duplicate_rows,
            "identical_clinical_text_groups_with_conflicting_labels": conflict_groups,
            "rows_belonging_to_conflicting_label_groups": conflict_rows,
        },
        "train_dev_full_text_overlap": {
            "dev_cases_identical_to_one_or_more_train_rows": len(overlap),
            "dev_overlap_gold_not_among_train_same_text_labels": mismatch_with_identical_text,
            "overlap_dev_cohort": duplicate_subset,
            "nonoverlap_dev_cohort": nonduplicate_subset,
            "unmodified_official_dev": top1,
            "warning": "This is a descriptive subgroup sensitivity check; no cases removed from official metrics.",
        },
        "dev_k16_rare_class_coverage": {
            "dev_cases": EXPECTED_DEV,
            "k16_gold_found": dev_k16_gold_found,
            "number_of_classes_with_recall_0": sum(value == 0 for value in by_class_16.values()),
            "number_of_classes_with_recall_below_half": sum(
                value < 0.5 for value in by_class_16.values()
            ),
            "number_of_classes_with_recall_1": sum(value == 1 for value in by_class_16.values()),
            "macro_recall": sum(by_class_16.values()) / len(by_class_16),
            "dev_raw_clinical_text_containing_literal_gold_label": raw_gold_label_literal,
        },
        "label_lexical_normalization": {
            "label_count": EXPECTED_LABELS,
            "colliding_canonical_forms_after_unicode_nfkc_and_punctuation_removal": after_normalization_duplicate_groups,
            "semantic_synonym_check": "NOT_PERFORMED; official labels are used unchanged",
        },
        "model_training": "NOT_RUN",
        "critical_unverified_training_conditions": [
            "Released Jeff 1.1 Large checkpoint and head load at pinned revision",
            "32-bit DecisionHead gradient + bf16 backbone last hidden state gradient",
            "LoRA adapter is initialized as a zero-effect update on merged weights",
            "No full vocabulary [B,L,V] LM output is constructed during training",
            "Original tokenizer token lengths for all Train and Dev questions <=8192",
            "One GPU H200 memory and supported PyTorch/CUDA environment",
            "Checkpoint zero-step parity against Jeff pretrained model on fixed questions",
            "Held-out Dev classification Top-1, Macro-F1, calibration and latency",
        ],
    }


def main() -> None:
    report = audit()
    output = ROOT / "experiments/jeff_training_risk_audit_2026-11-11.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
