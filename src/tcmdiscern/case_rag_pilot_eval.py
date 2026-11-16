"""Report what a small TCM-SD Dev pilot can and cannot estimate.

Official Train/Dev labels are used only for *evaluation*. The report stores
aggregate counts; patient identities and clinical text never leave data/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from .datasets import normalized_input


def pilot_label_coverage(
    train: Sequence[dict],
    dev: Sequence[dict],
    dev_indices: Sequence[int],
) -> dict:
    """Audit gold support, Train overlaps, and a hard upper bound on fixed-148 F1.

    A random 300-case pilot is intentionally *not* class-balanced. Missing
    reference labels have zero F1 when all 148 are explicitly supplied to
    sklearn.metrics.f1_score. This is a mathematical ceiling, not model output.
    """
    if not dev_indices or len(dev_indices) != len(set(dev_indices)):
        raise ValueError("Evaluation requires nonempty unique Dev case indices")
    if any(not isinstance(i, int) or not 0 <= i < len(dev) for i in dev_indices):
        raise ValueError("Evaluation refers to an invalid Dev case index")
    train_classes = {str(row["norm_syndrome"]) for row in train}
    if len(train_classes) != 148:
        raise ValueError("Expected the official 148 Train syndrome labels")
    chosen = [dev[i] for i in dev_indices]
    counts = Counter(str(row["norm_syndrome"]) for row in chosen)
    if not set(counts).issubset(train_classes):
        raise ValueError("Dev gold contains a syndrome outside original Train labels")

    train_text_labels: dict[str, set[str]] = defaultdict(set)
    train_patient_ids: set[str] = set()
    for row in train:
        train_text_labels[normalized_input(row, full=True)].add(
            str(row["norm_syndrome"])
        )
        pid = str(row.get("user_id") or "").strip()
        if pid:
            train_patient_ids.add(pid)
    matched_text = 0
    conflicting = 0
    matched_patient = 0
    for row in chosen:
        labels = train_text_labels.get(normalized_input(row, full=True))
        if labels:
            matched_text += 1
            conflicting += str(row["norm_syndrome"]) not in labels
        uid = str(row.get("user_id") or "").strip()
        if uid and uid in train_patient_ids:
            matched_patient += 1
    present = len(counts)
    return {
        "cases": len(chosen),
        "class_count_reference": len(train_classes),
        "observed_gold_classes": present,
        "absent_gold_classes": len(train_classes) - present,
        "classes_with_one_gold_case": sum(v == 1 for v in counts.values()),
        "classes_with_two_or_fewer_gold_cases": sum(v <= 2 for v in counts.values()),
        "maximum_possible_fixed_148_macro_f1_percent": round(
            100 * present / len(train_classes), 4
        ),
        "train_dev_normalized_text_overlap_cases": matched_text,
        "train_dev_same_text_different_gold_cases": conflicting,
        "train_dev_patient_id_overlap_cases": matched_patient,
        "reference_split": "official Dev, sampled without replacement",
        "gold_usage": "metrics and data-audit only; never retrieval or model input",
        "interpretation": (
            "Small non-stratified pilot: within-sample paired Top-1/rescue-break "
            "is exploratory; fixed-148 macro-F1 cannot be directly compared to "
            "full-Dev macro-F1. Evaluate the locked full 5,486 Dev for 148-class "
            "and long-tail claims."
        ),
        "official_test_used": False,
    }
