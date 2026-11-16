"""Small, self-contained checks for the three-loss comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_s1_s2_s3 import load_s3, pair_counts


def _sample(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_s3_accepts_ordered_complete_logits(tmp_path):
    source = tmp_path / "s3.jsonl"
    ids = ["dev-000000", "dev-000001"]
    rows = [
        {"case_id": case_id, "logits": [float(j + i) for j in range(16)]}
        for i, case_id in enumerate(ids)
    ]
    _sample(source, rows)
    logits = load_s3(source, 2, ids)
    assert logits.shape == (2, 16)
    np.testing.assert_allclose(logits[1], np.arange(16) + 1)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing", "Expected 2"),
        ("order", "case IDs/order"),
        ("nan", "finite"),
        ("short", "16-dimensional"),
    ],
)
def test_load_s3_rejects_corruption(tmp_path, change, error):
    source = tmp_path / "s3.jsonl"
    ids = ["dev-000000", "dev-000001"]
    rows = [{"case_id": case_id, "logits": [0.0] * 16} for case_id in ids]
    if change == "missing":
        rows.pop()
    elif change == "order":
        rows.reverse()
    elif change == "nan":
        rows[0]["logits"][3] = float("nan")
    elif change == "short":
        rows[0]["logits"].pop()
    _sample(source, rows)
    with pytest.raises(ValueError, match=error):
        load_s3(source, 2, ids)


def test_paired_mcnemar_counts_discordance():
    gold = np.array([0, 1, 2, 3, 4])
    first = np.array([0, 5, 2, 5, 5])
    second = np.array([5, 1, 2, 3, 5])
    result = pair_counts(gold, first, second)
    assert result["first_only_correct"] == 1
    assert result["second_only_correct"] == 2
    assert result["net_more_correct_second"] == 1
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_mcnemar_identical_predictions():
    y = np.array([0, 1, 2])
    p = np.array([0, 0, 2])
    result = pair_counts(y, p, p)
    assert result["net_more_correct_second"] == 0
    assert result["mcnemar_exact_two_sided_p"] == 1.0
