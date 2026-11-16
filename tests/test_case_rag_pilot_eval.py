"""A 300-case random Dev pilot must not masquerade as full 148-class Dev."""
from __future__ import annotations

import pytest

from tcmdiscern.case_rag_pilot_eval import pilot_label_coverage


def row(label: str, i: int, *, text: str | None = None, uid: str = "") -> dict:
    return {
        "norm_syndrome": label,
        "chief_complaint": text or f"臨床觀察-{i}",
        "description": "有胃痛",
        "detection": "舌紅",
        "user_id": uid,
    }


def test_sparse_sample_reports_correct_fixed_macro_f1_ceiling_and_overlap():
    train = [row(f"證型{i}", i) for i in range(148)]
    train[0] = row("證型0", 0, text="重複臨床資料", uid="patient42")
    dev = [
        row("證型0", 1000, text="重複臨床資料", uid="patient42"),
        row("證型1", 1001),
        row("證型1", 1002),
        row("證型2", 1003),
        row("證型0", 1004, text="重複臨床資料"),
    ]
    report = pilot_label_coverage(train, dev, [0, 1, 2, 3, 4])
    assert report["cases"] == 5
    assert report["observed_gold_classes"] == 3
    assert report["absent_gold_classes"] == 145
    assert report["classes_with_one_gold_case"] == 1
    assert report["classes_with_two_or_fewer_gold_cases"] == 3
    assert report["maximum_possible_fixed_148_macro_f1_percent"] == round(100 * 3 / 148, 4)
    assert report["train_dev_normalized_text_overlap_cases"] == 2
    assert report["train_dev_same_text_different_gold_cases"] == 0
    assert report["train_dev_patient_id_overlap_cases"] == 1


def test_conflicting_labels_and_invalid_indices_are_visible():
    train = [row(f"證型{i}", i) for i in range(148)]
    dev = [row("證型2", 0)]
    info = pilot_label_coverage(train, dev, [0])
    assert info["train_dev_same_text_different_gold_cases"] == 1
    with pytest.raises(ValueError, match="unique"):
        pilot_label_coverage(train, dev, [0, 0])
    with pytest.raises(ValueError, match="invalid"):
        pilot_label_coverage(train, dev, [1])
