"""Regression tests for matched two-case RAG budget and CPU scoring."""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy

import pytest

from scripts import run_case_rag_jeff as runner
from tcmdiscern.case_rag import CONDITIONS_2, CaseRAGIndex
from tcmdiscern.jeff_format import decision_row


def sample(label: str, i: int) -> dict:
    return {
        "chief_complaint": f"症狀{i}咳嗽胃痛",
        "description": f"病史{i}食後不適",
        "detection": f"脈象{i}舌苔",
        "norm_syndrome": label,
        "user_id": f"patient-{i}",
    }


def test_two_case_modes_are_gold_blind_and_budget_matched():
    labels = [f"證型{j}" for j in range(16)]
    train = [sample(label, j * 10 + k) for j, label in enumerate(labels) for k in range(5)]
    index = CaseRAGIndex(train)
    query = sample("未傳入的正確答案", 990)
    for condition in CONDITIONS_2:
        row, detail = index.make_row(
            query, labels, case_id="dev-000011", condition=condition, seed=17,
        )
        assert [opt["id"] for opt in row["options"]] == labels
        assert "未傳入的正確答案" not in str(row)
        assert detail["condition"] == condition
        if condition == "baseline":
            assert not detail["support_labels"]
            continue
        assert len(detail["support_groups"]) == len(set(detail["support_groups"])) == 2
        if condition in ("balanced2", "paired2", "random2"):
            assert Counter(detail["support_labels"]) == Counter({labels[0]: 1, labels[1]: 1})


def test_two_case_paired_first_pair_matches_four_case_paired():
    labels = [f"證型{j}" for j in range(16)]
    train = [sample(label, j * 10 + k) for j, label in enumerate(labels) for k in range(5)]
    index = CaseRAGIndex(train)
    query = sample("unused", 990)
    _, a = index.make_row(query, labels, case_id="dev-000001", condition="paired2")
    _, b = index.make_row(query, labels, case_id="dev-000001", condition="paired4")
    assert a["source_train_positions"] == b["source_train_positions"][:2]


def _source_root(tmp_path):
    train = [sample(f"證型{i}", 2000 + i) for i in range(148)]
    dev = [sample("證型1", i + 1) for i in range(4)]
    (tmp_path / "data/tcm-sd").mkdir(parents=True)
    (tmp_path / "data/tcm-sd/train.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train) + "\n", encoding="utf-8"
    )
    (tmp_path / "data/tcm-sd/dev.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in dev) + "\n", encoding="utf-8"
    )
    return dev


def test_runner_scores_two_case_primary_comparison_without_real_model(tmp_path, monkeypatch):
    # Pure synthetic scoring validates pairing statistics; no real Jeff score inferred.
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    dev = _source_root(tmp_path)
    folder = tmp_path / "prompts"
    folder.mkdir()
    options = [f"證型{i}" for i in range(16)]
    prompts = [
        decision_row(row, case_id=f"dev-{i:06d}", candidates=options)
        for i, row in enumerate(dev)
    ]
    (folder / "manifest.json").write_text(json.dumps({
        "samples": 4, "conditions": list(CONDITIONS_2),
        "evidence_budget": 2, "all_within_8192": True,
    }), encoding="utf-8")
    for arm in CONDITIONS_2:
        records = []
        for original in prompts:
            row = deepcopy(original)
            if arm != "baseline":
                row["state"]["retrieved_cases"] = [{} for _ in range(2)]
            records.append(json.dumps(row, ensure_ascii=False))
        (folder / f"{arm}.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    folder_scores = tmp_path / "scores"
    folder_scores.mkdir()
    for arm in CONDITIONS_2:
        records = []
        for original in prompts:
            logits = [0.0] * 16
            logits[1 if arm == "paired2" else 0] = 1.0
            records.append(json.dumps({"case_id": original["id"], "logits": logits}))
        (folder_scores / f"{arm}.jsonl").write_text("\n".join(records) + "\n")
    summary = runner.score_outputs(folder, folder_scores)
    assert summary["conditions"]["paired2"]["top1_percent"] == 100
    assert summary["conditions"]["balanced2"]["top1_percent"] == 0
    assert summary["paired2_vs_controls"]["balanced2"]["rescued_vs_comparator"] == 4
    assert summary["paired2_vs_controls"]["balanced2"]["broken_vs_comparator"] == 0


def test_frozen_s2_baseline_parity_gate(tmp_path):
    full = tmp_path / "full.jsonl"
    local = tmp_path / "local.jsonl"
    entries = [
        {"case_id": f"dev-{i:06d}", "logits": [
            1.0 if j == i else 0.0 for j in range(16)
        ]}
        for i in range(4)
    ]
    full.write_text("\n".join(json.dumps(row) for row in entries) + "\n")
    sampled = [entries[1], entries[3]]
    local.write_text("\n".join(json.dumps(row) for row in sampled) + "\n")
    checked = runner.check_reference_baseline(
        [{"id": sampled[0]["case_id"]}, {"id": sampled[1]["case_id"]}],
        local, full, total_cases=4,
    )
    assert checked["decision_disagreements_vs_saved_s2"] == 0
    sampled[1]["logits"] = [100.0 if j == 0 else 0.0 for j in range(16)]
    local.write_text("\n".join(json.dumps(row) for row in sampled) + "\n")
    mismatched = runner.check_reference_baseline(
        [{"id": sampled[0]["case_id"]}, {"id": sampled[1]["case_id"]}],
        local, full, total_cases=4,
    )
    assert mismatched["decision_disagreements_vs_saved_s2"] == 1


def test_manifest_rejects_budget_mismatch(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": 1, "conditions": list(CONDITIONS_2),
        "evidence_budget": 4, "all_within_8192": True,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="budget"):
        runner.load_inputs(tmp_path)
