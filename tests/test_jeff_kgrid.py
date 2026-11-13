"""CPU-only gates for Jeff K-grid evaluation and official Dev integrity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.run_jeff_kgrid import (
    OFFICIAL_DEV,
    ROOT,
    decider_scores,
    evaluate,
    iter_k_rows,
    validate_dev,
)
from tcmdiscern.evaluation import Prediction
from tcmdiscern.jeff_scoring import score_k


def synthetic_data():
    classes = [f"證型{i:03d}" for i in range(148)]
    cases = [
        {
            "chief_complaint": "測試主訴",
            "description": "測試病況",
            "detection": "測試四診",
            "norm_syndrome": classes[i],
            "user_id": "private-id-must-be-excluded",
        }
        for i in range(2)
    ]
    predictions = [
        Prediction(
            case_id=f"dev-{i:06d}",
            gold_labels=(classes[i],),
            candidate_labels=tuple(classes[(i + j) % 148] for j in range(50)),
        )
        for i in range(2)
    ]
    return classes, cases, predictions


def test_kgrid_source_validation_and_gold_is_never_exposed():
    _, cases, predictions = synthetic_data()
    validate_dev(cases, predictions, expected=2)
    for k in (2, 5, 16, 17, 35):
        rows = list(iter_k_rows(cases, predictions, k=k, limit=2))
        assert len(rows) == 2
        for index, row in enumerate(rows):
            assert len(row["options"]) == k
            assert [option["id"] for option in row["options"]] == list(
                predictions[index].candidate_labels[:k]
            )
            assert set(row["state"]) == {"chief_complaint", "description", "detection"}
            assert "user_id" not in json.dumps(row, ensure_ascii=False)
            assert "norm_syndrome" not in json.dumps(row, ensure_ascii=False)
    wrong = predictions.copy()
    wrong[0] = Prediction(wrong[0].case_id, ("錯誤標籤",), wrong[0].candidate_labels)
    with pytest.raises(ValueError, match="gold mismatch"):
        validate_dev(cases, wrong, expected=2)


def test_general_scorer_handles_bm25_and_grouped_jeff():
    classes, _, predictions = synthetic_data()
    for k in (1, 5, 16, 17, 35):
        vectors = np.zeros((2, k))
        vectors[:, 0] = 1
        report = score_k(predictions, vectors, k=k, all_classes=classes)
        assert report["top1_correct_n"] == 2
        assert report["bm25_top1_correct_n"] == 2
        assert report["gold_in_topk_n"] == 2
        assert report["net_correct_change"] == 0
    with pytest.raises(ValueError, match="17-slot"):
        score_k(predictions, np.ones((2, 16)), k=17, all_classes=classes)


class FakeDecider:
    def __init__(self):
        self.last = {}

    def decide(self, row):
        k = len(row["options"])
        self.last = {"shortened": False, "thought": False, "output_tokens": 0}
        return [1 / k] * k, [float(k - i) for i in range(k)], 200 * (3 if k > 16 else 1), 0.1


def test_grouped_outputs_are_atomic_and_not_overwritten(tmp_path):
    _, cases, predictions = synthetic_data()
    destination = tmp_path / "k17.jsonl"
    vectors, tokens, elapsed = decider_scores(
        iter_k_rows(cases, predictions, k=17, limit=2),
        FakeDecider(),
        k=17,
        destination=destination,
    )
    assert len(vectors) == 2 and len(vectors[0]) == 17
    assert tokens == 1200 and elapsed >= 0
    saved = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert saved[0]["case_id"] == "dev-000000"
    assert saved[0]["scores"][0] > saved[0]["scores"][1]
    with pytest.raises(FileExistsError):
        decider_scores(
            iter_k_rows(cases, predictions, k=17, limit=1),
            FakeDecider(),
            k=17,
            destination=destination,
        )


@pytest.mark.skipif(
    not (ROOT / "data/tcm-sd/dev.json").is_file()
    or not (ROOT / "data/candidates/bm25_dev.jsonl").is_file(),
    reason="Private official TCM-SD Dev not available in this checkout",
)
def test_full_official_dev_dryrun_reproduces_bm25_for_multiple_k():
    result = evaluate(
        ks=[1, 5, 16, 17, 35],
        limit=OFFICIAL_DEV,
        model_dir=None,
        checkpoint=None,
        run_id="local-dryrun",
        dry_run=True,
    )
    assert len(result["reports"]) == 5
    assert all(r["top1_correct_n"] == r["bm25_top1_correct_n"] for r in result["reports"])
    assert all(
        r["top1_accuracy"] == result["reports"][0]["top1_accuracy"] for r in result["reports"]
    )
    assert result["reports"][0]["top1_accuracy"] == pytest.approx(0.7958, abs=0.0001)
