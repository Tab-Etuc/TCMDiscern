"""CPU evaluation scoring regression tests for frozen Jeff K=16 inference."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tcmdiscern.evaluation import Prediction
from tcmdiscern.jeff_scoring import read_model_logits, score_k16


def fake_dev() -> tuple[list[Prediction], list[str]]:
    labels = [f"證型{i:03d}" for i in range(148)]
    rows = [
        Prediction(
            case_id=f"dev-{i:06d}",
            gold_labels=(labels[i],),
            candidate_labels=tuple(labels[(i + j) % 148] for j in range(16)),
        )
        for i in range(148)
    ]
    return rows, labels


def test_full_148_class_identity_baseline_and_rescue() -> None:
    rows, labels = fake_dev()
    logits = np.zeros((148, 16), dtype=float)
    logits[:, 0] = 1
    result = score_k16(rows, logits, all_classes=labels)
    assert result["n_cases"] == 148
    assert result["top1_correct_n"] == 148
    assert result["macro_f1"] == 1
    assert result["bm25_top1_correct_n"] == 148
    assert result["gold_in_top16_n"] == 148
    assert result["bm25_to_jeff_rescued_cases"] == 0
    assert result["bm25_to_jeff_broken_cases"] == 0

    first = rows[0]
    replaced = [first.candidate_labels[1], first.candidate_labels[0], *first.candidate_labels[2:]]
    rows[0] = Prediction(first.case_id, first.gold_labels, tuple(replaced))
    logits[0, 0] = 0
    logits[0, 1] = 1
    result = score_k16(rows, logits, all_classes=labels)
    assert result["top1_correct_n"] == 148
    assert result["bm25_top1_correct_n"] == 147
    assert result["bm25_to_jeff_rescued_cases"] == 1
    assert result["net_correct_change"] == 1


def test_wrong_reranker_can_break_correct_bm25_choice() -> None:
    rows, labels = fake_dev()
    logits = np.zeros((148, 16), dtype=float)
    logits[:, 0] = 1
    logits[0, 0] = 0
    logits[0, 1] = 2
    result = score_k16(rows, logits, all_classes=labels)
    assert result["top1_correct_n"] == 147
    assert result["bm25_top1_correct_n"] == 148
    assert result["bm25_to_jeff_broken_cases"] == 1
    assert result["net_correct_change"] == -1


def test_reject_incomplete_or_unaligned_logit_batches() -> None:
    rows, classes = fake_dev()
    logits = np.ones((148, 16))
    with pytest.raises(ValueError, match="16-slot"):
        score_k16(rows, logits[:, :15], all_classes=classes)
    with pytest.raises(ValueError, match="finite"):
        invalid = logits.copy()
        invalid[0, 0] = np.nan
        score_k16(rows, invalid, all_classes=classes)
    with pytest.raises(ValueError, match="out of official order"):
        wrong = rows.copy()
        wrong[1] = Prediction("dev-999999", wrong[1].gold_labels, wrong[1].candidate_labels)
        score_k16(wrong, logits, all_classes=classes)
    with pytest.raises(ValueError, match="pre"):
        wrong = rows.copy()
        wrong[1] = Prediction(
            wrong[1].case_id,
            wrong[1].gold_labels,
            wrong[1].candidate_labels,
            predicted_label=wrong[1].candidate_labels[0],
        )
        score_k16(wrong, logits, all_classes=classes)


def test_logit_file_reader_matches_by_exact_case_id(tmp_path) -> None:
    path = tmp_path / "model_logits.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for i in range(148):
            stream.write(json.dumps({"case_id": f"dev-{i:06d}", "logits": [float(i)] * 16}) + "\n")
    matrix = read_model_logits(path, num_cases=148)
    assert matrix.shape == (148, 16)
    assert matrix[4, 0] == 4
    with pytest.raises(ValueError, match="More model predictions"):
        read_model_logits(path, num_cases=147)
    path.write_text(
        json.dumps({"case_id": "dev-000001", "logits": [0.0] * 16}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Out-of-order"):
        read_model_logits(path, num_cases=1)
