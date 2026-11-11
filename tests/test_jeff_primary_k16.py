"""Active Jeff K=16 research decision and frozen data-verification regression."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest

from scripts.auto_k_bm25 import PRIMARY_JEFF_K as RETRIEVAL_PRIMARY
from scripts.export_jeff_dev import PRIMARY_JEFF_K as DEV_PRIMARY
from scripts.export_jeff_train import PRIMARY_JEFF_K as TRAIN_PRIMARY
from scripts.mine_jeff_train_oof import PRIMARY_JEFF_K as OOF_PRIMARY
from tcmdiscern.jeff_config import (
    COMPARISON_KS,
    MAX_DIRECT_JEFF_K,
    MAX_EVALUATION_K,
    PRIMARY_JEFF_K,
)
from tcmdiscern.jeff_format import decision_row, semif_messages, train_example

ROOT = Path(__file__).resolve().parents[1]


def test_primary_jeff_k_shared_across_retrieval_and_train_and_dev() -> None:
    assert PRIMARY_JEFF_K == MAX_DIRECT_JEFF_K == 16
    assert PRIMARY_JEFF_K == TRAIN_PRIMARY == DEV_PRIMARY == OOF_PRIMARY == RETRIEVAL_PRIMARY
    assert MAX_EVALUATION_K == 35
    assert COMPARISON_KS == (5, 17, 35)


def test_k16_trains_direct_head_and_k17_needs_grouping() -> None:
    case = {
        "chief_complaint": "胸悶",
        "description": "痰多",
        "detection": "苔白膩",
    }
    negatives = [f"錯誤證型{i}" for i in range(20)]
    example = train_example(
        case,
        case_id="train-000000",
        gold_label="正解證型",
        hard_negatives=negatives,
        k=PRIMARY_JEFF_K,
        rng=Random(9),
    )
    options = example["row"]["options"]
    assert len(options) == 16
    assert options[example["target_index"]]["id"] == "正解證型"
    assert {item["id"] for item in options} == {"正解證型", *negatives[:15]}
    messages = semif_messages(example["row"])
    payload = json.loads(messages[1]["content"])
    assert [option["letter"] for option in payload["options"]] == list("ABCDEFGHIJKLMNOP")
    assert "target_index" not in messages[1]["content"]

    with pytest.raises(ValueError, match="2..16"):
        decision_row(case, case_id="dev-000000", candidates=negatives[:17])
    grouped = decision_row(
        case,
        case_id="dev-000000",
        candidates=negatives[:17],
        allow_grouped=True,
    )
    assert len(grouped["options"]) == 17


def test_full_k16_train_dev_frozen_verification_without_patient_data() -> None:
    report = json.loads(
        (ROOT / "experiments/jeff_primary_k16_2026-11-11.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "PRIMARY_JEFF_K16_DATA_VALIDATED_NO_MODEL_TRAINING"
    assert report["primary_k"] == report["target_slots"] == 16
    assert report["requires_grouped_decision"] is False
    assert report["forward_passes_per_decision"] == 1
    assert report["train_k_distribution"] == {"16": 43180}
    assert sum(report["train_answer_positions"].values()) == 43180
    assert report["train_rows_verified"] == 43180
    assert report["train_original_rows_removed"] == 0
    assert report["train_original_source_gold_literal_count"] == 1
    assert report["dev_rows_verified"] == 5486
    assert report["dev_candidates_containing_gold_n"] == 5409
    assert report["dev_gold_injected_into_candidates"] is False
    assert report["dev_prediction_accuracy_not_measured"] is True
    assert report["model_checkpoint_trained"] is False
    assert report["other_k_results_preserved"] is True
    old = json.loads(
        (ROOT / "experiments/bm25_k1_35_dev_2026-11-11.json").read_text(encoding="utf-8")
    )["results"]
    assert old[15]["candidate_hit_n"] == 5409
    assert old[16]["candidate_hit_n"] == 5412
    assert report["dev_micro_recall_at_k"] == old[15]["micro_recall"]
    assert report["dev_macro_recall_at_k"] == old[15]["macro_recall"]
