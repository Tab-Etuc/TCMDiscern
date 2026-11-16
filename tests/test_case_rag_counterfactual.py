"""Counterfactual retrieval label/provenance integrity and scoring regression."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import run_case_rag_jeff as runner
from tcmdiscern.case_rag_counterfactual import (
    ARMS,
    HIDDEN_LABEL,
    SOURCE_GUARD,
    assert_only_allowed_changes,
    variants,
)
from tcmdiscern.jeff_format import decision_row


def sample_prompts(case_id: str = "dev-000000"):
    original = decision_row(
        {
            "chief_complaint": "頭暈三日",
            "description": "遇冷加重",
            "detection": "脈浮苔薄",
        },
        case_id=case_id,
        candidates=[f"證型{j}" for j in range(16)],
    )
    paired = deepcopy(original)
    paired["state"]["retrieval_note"] = "來源為歷史病例"
    paired["state"]["retrieved_cases"] = [
        {
            "candidate_syndrome": f"證型{i}",
            "chief_complaint": f"真實個案{i}症狀",
            "description": f"完整病史{i}",
            "detection": f"舌苔{i}",
        }
        for i in range(2)
    ]
    return original, paired


def test_counterfactual_controls_keep_patient_and_support_text_identical():
    original, paired = sample_prompts()
    controls = variants(original, paired)
    assert tuple(controls) == ARMS
    assert_only_allowed_changes(controls)
    for arm in ARMS:
        assert controls[arm]["options"] == original["options"]
        assert controls[arm]["state"]["chief_complaint"] == original["state"]["chief_complaint"]
        assert "norm_syndrome" not in json.dumps(controls[arm], ensure_ascii=False)
    assert [r["candidate_syndrome"] for r in controls["labelswap2"]["state"]["retrieved_cases"]] == [
        "證型1", "證型0"
    ]
    assert all(
        r["candidate_syndrome"] == HIDDEN_LABEL
        for r in controls["labelhide2"]["state"]["retrieved_cases"]
    )
    assert controls["orderflip2"]["state"]["retrieved_cases"] == (
        paired["state"]["retrieved_cases"][::-1]
    )
    assert controls["doubleflip2"]["state"]["retrieved_cases"] == (
        controls["labelswap2"]["state"]["retrieved_cases"][::-1]
    )
    assert controls["sourceguard2"]["state"]["retrieval_note"] == SOURCE_GUARD
    assert original["state"] == controls["baseline"]["state"]


def test_counterfactual_validator_rejects_changed_patient_and_option():
    original, paired = sample_prompts()
    altered = variants(original, paired)
    altered["labelswap2"]["state"]["chief_complaint"] = "掉包的病人"
    with pytest.raises(ValueError, match="patient observations"):
        assert_only_allowed_changes(altered)
    altered = variants(original, paired)
    altered["orderflip2"]["options"].reverse()
    with pytest.raises(ValueError, match="diagnostic decision"):
        assert_only_allowed_changes(altered)
    paired["state"]["retrieved_cases"][1]["candidate_syndrome"] = "證型3"
    with pytest.raises(ValueError, match="original Top-2"):
        variants(original, paired)


def test_counterfactual_runner_accepts_all_stress_arms(tmp_path):
    original, paired = sample_prompts()
    controls = variants(original, paired)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": 1, "conditions": list(ARMS),
        "counterfactual": True,
        "evidence_budget": 2, "all_within_8192": True,
    }), encoding="utf-8")
    for arm, row in controls.items():
        (tmp_path / f"{arm}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    loaded = runner.load_inputs(tmp_path)
    assert tuple(loaded) == ARMS
    assert loaded["paired2"] == [paired]
    assert loaded["labelswap2"][0]["options"] == loaded["baseline"][0]["options"]
    tampered = deepcopy(controls["labelhide2"])
    tampered["state"]["chief_complaint"] = "偽造患者"
    (tmp_path / "labelhide2.jsonl").write_text(
        json.dumps(tampered, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="patient observations"):
        runner.load_inputs(tmp_path)


def test_source_attribution_score_quantifies_swap_bias(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    dataset = tmp_path / "data/tcm-sd"
    dataset.mkdir(parents=True)
    train = [
        {"chief_complaint": f"訓練症狀{j}", "description": f"病史{j}",
         "detection": f"四診{j}", "norm_syndrome": f"證型{j}"}
        for j in range(148)
    ]
    dev = [
        {"chief_complaint": f"評估症狀{j}", "description": f"現病史{j}",
         "detection": f"檢查{j}", "norm_syndrome": f"證型{j % 2}"}
        for j in range(4)
    ]
    (dataset / "train.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train) + "\n"
    )
    (dataset / "dev.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dev) + "\n"
    )
    inputs = {}
    for j in range(4):
        original, paired = sample_prompts(f"dev-{j:06d}")
        for arm, row in variants(original, paired).items():
            inputs.setdefault(arm, []).append(row)
    scores = tmp_path / "scores"
    scores.mkdir()
    for arm in ARMS:
        records = []
        for row in inputs[arm]:
            probabilities = [0.0] * 16
            probabilities[1 if arm in ("labelswap2", "doubleflip2") else 0] = 10.0
            records.append(json.dumps({"case_id": row["id"], "logits": probabilities}))
        (scores / f"{arm}.jsonl").write_text("\n".join(records) + "\n")
    report = runner.score_counterfactual_outputs(inputs, scores)
    assert report["samples"] == 4
    assert report["source_interventions_vs_paired2"]["labelswap2"][
        "prediction_flip_rate_percent"
    ] == 100
    assert report["source_interventions_vs_paired2"]["labelswap2"][
        "top2_pair_flip_rate_percent"
    ] == 100
    assert report["source_interventions_vs_paired2"]["orderflip2"][
        "prediction_flip_rate_percent"
    ] == 0
    assert report["methods"]["paired2"]["top1_percent"] == 50
    assert report["source_interventions_vs_paired2"]["labelswap2"]["rescued_vs_paired2"] == 2
    assert report["label_order_factorial"]["mean_label_swap_effect_on_top2_logit_margin"] == -20.0
    assert report["label_order_factorial"]["mean_order_effect_on_top2_logit_margin"] == 0.0
    assert report["label_order_factorial"]["mean_abs_label_order_interaction"] == 0.0
