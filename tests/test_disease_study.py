"""Focused disease-conditioned K5 training/evaluation and H200 plan tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.disease_h200 import registered_runs, run_id, train_command
from scripts.prepare_disease_study import validate_existing
from tcmdiscern.disease_study import (
    K,
    disease_only_predictions,
    eval_item,
    gold_for,
    pair_key,
    pair_parts,
    prioritize_disease,
    score_predictions,
    study_decision,
    study_torch_loss,
    train_item,
)
from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_schedule import check_training_example


def case(disease="消渴", syndrome="氣陰兩虛", **extra):
    return {
        "chief_complaint": "口乾",
        "description": "多飲",
        "detection": "舌紅",
        "lcd_name": disease,
        "norm_syndrome": syndrome,
        **extra,
    }


def test_joint_pair_label_is_unambiguous_and_condition_is_model_visible():
    pair = pair_key("消渴", "氣陰兩虛")
    assert pair_parts(pair) == ("消渴", "氣陰兩虛")
    assert gold_for(case(), "pair") == pair
    ranked = [pair_key("消渴", f"證{i}") for i in range(K)]
    row = study_decision(case(), case_id="dev-000000", ranked=ranked, mode="pair")
    assert row["options"][0]["description"] == "中醫病名：消渴；證型：證0"
    assert "lcd_name" not in row["state"]
    conditioned = study_decision(
        case(), case_id="dev-000000",
        ranked=["氣陰兩虛", "濕熱", "陰虛", "陽虛", "氣虛"],
        mode="conditioned",
    )
    assert conditioned["state"]["lcd_name"] == "消渴"
    assert "norm_syndrome" not in json.dumps(conditioned, ensure_ascii=False)


def test_rank_prioritizes_only_train_observed_disease_labels():
    ranked = ["甲", "乙", "丙", "丁", "戊", "己"]
    result = prioritize_disease(ranked, disease="消渴", allow={"消渴": frozenset({"丁", "丙"})})
    assert result == ["丙", "丁", "甲", "乙", "戊", "己"]
    assert prioritize_disease(ranked, disease="未見病名", allow={}) == ranked


def test_train_hard_negatives_remap_correctly_after_epoch_shuffle():
    ranked = ["濕熱", "氣陰兩虛", "血瘀", "氣虛", "陰虛", "陽虛"]
    entry = train_item(
        case(), index=0, ranked=ranked, mode="conditioned",
        allow={"消渴": frozenset({"氣陰兩虛", "濕熱", "氣虛"})},
    )
    check_training_example(entry, row_index=0, multi=False, expected_k=K,
                           study_mode="conditioned")
    original = [o["id"] for o in entry["row"]["options"]]
    hard = {original[i] for i in entry["hard_negative_indices"]}
    assert hard == {"濕熱", "氣虛"}
    shuffled = reshuffle_training_example(entry, epoch=1, seed=19)
    moved = [o["id"] for o in shuffled["row"]["options"]]
    assert moved[shuffled["target_index"]] == "氣陰兩虛"
    assert {moved[i] for i in shuffled["hard_negative_indices"]} == hard


def test_train_pair_and_eval_prompts_use_separate_gold_metadata():
    query = case()
    gold = gold_for(query, "pair")
    ranked = [pair_key("消渴", f"證{i}") for i in range(7)]
    item = train_item(query, index=2, ranked=ranked, mode="pair")
    check_training_example(item, row_index=2, multi=False, expected_k=K, study_mode="pair")
    options = [o["id"] for o in item["row"]["options"]]
    assert options[item["target_index"]] == gold
    row, meta = eval_item(query, index=0, ranked=ranked, mode="pair", split="dev")
    assert meta["gold"] == gold
    assert meta["candidates"] == ranked[:K]
    assert "target_index" not in row and "gold" not in row
    assert gold not in row["state"]


def test_macro_f1_and_oracle_for_original_single_gold_rows():
    train = [case(syndrome="甲"), case(syndrome="乙")]
    data = [case(syndrome="甲"), case(syndrome="乙")]
    sources = [
        {"case_id": f"dev-{i:06d}", "gold": label, "candidates": ["甲", "乙", "丙", "丁", "戊"]}
        for i, label in enumerate(("甲", "乙"))
    ]
    score = score_predictions(data, sources, ["甲", "甲"], mode="global",
                              train_labels=["甲", "乙"])
    assert score["top1"] == 0.5
    assert score["oracle_at_5"] == 1.0
    assert 0 < score["macro_f1"] < 1
    assert disease_only_predictions(train, data) == ["乙", "乙"]


def test_cache_validation_checks_complete_row_count_and_k(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"row": {"id": "train-000000", "options": []},
                              "target_index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt cached prompts"):
        validate_existing(p, count=1, mode="global", kind="train", split="train")


def test_h200_registered_budget_and_baselines():
    run_specs = registered_runs()
    assert len(run_specs) == 12
    assert len({run_id(method, seed) for method, seed in run_specs}) == 12
    args = train_command("conditioned_hard", run_specs[0][1], Path("/models/Jeff"))
    assert args[args.index("--study-mode") + 1] == "conditioned"
    assert args[args.index("--pairwise-weight") + 1] == "0.2"


def test_pairwise_torch_loss_has_finite_gradients():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([1.2, -0.2, 1.3, 0.1, -0.4], requires_grad=True)
    example = {"target_index": 0, "hard_negative_indices": [2, 3]}
    baseline = study_torch_loss(logits, example, pairwise_weight=0, margin=0.2)
    enhanced = study_torch_loss(logits, example, pairwise_weight=0.2, margin=0.2)
    assert enhanced > baseline
    enhanced.backward()
    assert torch.isfinite(logits.grad).all()
