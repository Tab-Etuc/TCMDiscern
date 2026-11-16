"""Fixed-budget clinical evidence extraction and Jeff scoring regression."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import run_case_rag_jeff as runner
from tcmdiscern.case_rag_excerpt import (
    EXCERPT_ARMS,
    evidence_case,
    excerpt,
    validate_excerpts,
)
from tcmdiscern.jeff_format import decision_row
from tcmdiscern.retrieval import chinese_bigrams


def test_query_excerpts_recover_relevant_verbatim_content():
    source = "其他病史" * 32 + "，舌紅脈細鼻涕清稀，反覆寒戰，" + "無關資料" * 32
    query = "鼻涕清稀 舌紅脈細 反覆寒戰"
    original = excerpt(source, query, width=54, mode="prefix")
    selected = excerpt(source, query, width=54, mode="query")
    control = excerpt(source, query, width=54, mode="random", seed=1)
    target_terms = set(chinese_bigrams(query))
    assert selected in source
    assert len(selected) <= 54
    assert len(target_terms & set(chinese_bigrams(selected))) > len(
        target_terms & set(chinese_bigrams(original))
    )
    assert control in source


def test_excerpts_are_source_grounded_and_gold_independent():
    history = {
        "chief_complaint": "胃痛與咳嗽",
        "description": "普通背景" * 40 + "，舌尖紅且口乾舌燥，" + "普通背景" * 20,
        "detection": "脈細苔薄，偶見潮熱夜汗，反覆胃痛，無其他主訴",
    }
    query = "患者胃痛，口乾舌燥，夜汗"
    for mode in ("prefix", "query", "random", "full"):
        record = evidence_case(history, "證型甲", query, mode=mode, seed=3)
        prior = evidence_case(history, "證型甲", query, mode="prefix", seed=3)
        validate_excerpts(prior, record, history, mode=mode)
        assert record["candidate_syndrome"] == "證型甲"
        assert all(
            record[field] in history[field] for field in (
                "chief_complaint", "description", "detection",
            )
        )


def test_excerpt_runner_rejects_changed_support_syndrome(tmp_path):
    original = decision_row({
        "chief_complaint": "患者胃痛", "description": "餐後不適", "detection": "脈滑",
    }, case_id="dev-000021", candidates=[f"證型{i}" for i in range(16)])
    paired = deepcopy(original)
    paired["state"]["retrieval_note"] = "過去病例"
    paired["state"]["retrieved_cases"] = [
        {"candidate_syndrome": f"證型{i}", "chief_complaint": f"主訴{i}",
         "description": f"既往史{i}", "detection": f"四診{i}"}
        for i in range(2)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": 1, "conditions": list(EXCERPT_ARMS),
        "evidence_budget": 2, "all_within_8192": True,
        "excerpt_audit": True,
    }))
    for arm in EXCERPT_ARMS:
        row = deepcopy(original if arm == "baseline" else paired)
        (tmp_path / f"{arm}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
    assert runner.load_inputs(tmp_path)["paired2"] == [paired]
    wrong = deepcopy(paired)
    wrong["state"]["retrieved_cases"][0]["candidate_syndrome"] = "證型錯誤"
    (tmp_path / "querywindow2.jsonl").write_text(
        json.dumps(wrong, ensure_ascii=False) + "\n"
    )
    with pytest.raises(ValueError, match="source case label"):
        runner.load_inputs(tmp_path)


def test_excerpt_score_controls_score_only_with_synthetic_logits(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    dataset = tmp_path / "data/tcm-sd"
    dataset.mkdir(parents=True)
    train = [
        {"chief_complaint": f"訓練症狀{i}", "description": f"病史{i}",
         "detection": f"脈{i}", "norm_syndrome": f"證型{i}"}
        for i in range(148)
    ]
    dev = [
        {"chief_complaint": f"患者{i}", "description": f"現病史{i}",
         "detection": f"四診{i}", "norm_syndrome": "證型1"}
        for i in range(4)
    ]
    for name, rows in (("train.json", train), ("dev.json", dev)):
        (dataset / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        )
    inputs = {arm: [] for arm in EXCERPT_ARMS}
    for i in range(4):
        base = decision_row(
            dev[i], case_id=f"dev-{i:06d}",
            candidates=[f"證型{j}" for j in range(16)],
        )
        for arm in EXCERPT_ARMS:
            variant = deepcopy(base)
            if arm != "baseline":
                variant["state"]["retrieval_note"] = "歷史醫案"
                variant["state"]["retrieved_cases"] = [
                    {"candidate_syndrome": f"證型{j}", "chief_complaint": f"原始{j}",
                     "description": f"病史{j}", "detection": f"脈{j}"}
                    for j in range(2)
                ]
            inputs[arm].append(variant)
    logits_dir = tmp_path / "logits"
    logits_dir.mkdir()
    for arm in EXCERPT_ARMS:
        lines = []
        for row in inputs[arm]:
            vals = [0.0] * 16
            vals[1 if arm == "querywindow2" else 0] = 10.0
            lines.append(json.dumps({"case_id": row["id"], "logits": vals}))
        (logits_dir / f"{arm}.jsonl").write_text("\n".join(lines) + "\n")
    report = runner.score_excerpt_outputs(inputs, logits_dir)
    assert report["methods"]["querywindow2"]["top1_percent"] == 100
    assert report["methods"]["paired2"]["top1_percent"] == 0
    assert report["vs_paired2_fixed_prefix"]["querywindow2"]["rescued"] == 4
    assert report["vs_paired2_fixed_prefix"]["querywindow2"]["broken"] == 0
