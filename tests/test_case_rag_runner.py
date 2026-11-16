"""CPU regression tests for the Case-RAG Jeff scoring harness."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.run_case_rag_jeff import load_inputs, read_condition_logits
from tcmdiscern.case_rag import CONDITIONS
from tcmdiscern.jeff_format import decision_row


def fake_row():
    labels = [f"證型{i}" for i in range(16)]
    query = {
        "chief_complaint": "喉嚨乾燥",
        "description": "偶爾咳嗽",
        "detection": "舌紅",
    }
    return decision_row(query, case_id="dev-000000", candidates=labels)


def prepare(tmp_path):
    original = fake_row()
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "samples": 1, "all_within_8192": True,
            "conditions": list(CONDITIONS),
        }),
        encoding="utf-8",
    )
    for mode in CONDITIONS:
        row = deepcopy(original)
        if mode != "baseline":
            row["state"]["retrieval_note"] = "已標註歷史醫案"
            row["state"]["retrieved_cases"] = [
                {"candidate_syndrome": row["options"][j%2]["id"]}
                for j in range(4)
            ]
        (tmp_path / f"{mode}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return original


def test_load_inputs_keeps_one_frozen_candidate_order(tmp_path):
    original = prepare(tmp_path)
    rows = load_inputs(tmp_path)
    assert len(rows) == len(CONDITIONS)
    assert rows["baseline"] == [original]
    assert all(rows[name][0]["options"] == original["options"] for name in CONDITIONS)


def test_rejects_modified_candidate_options(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "paired4.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["options"][0]["id"] = "新插入的候選"
    path.write_text(json.dumps(record, ensure_ascii=False)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed the decision task"):
        load_inputs(tmp_path)


def test_rejects_missing_retrieved_case(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "random4.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["state"]["retrieved_cases"] = record["state"]["retrieved_cases"][:-1]
    path.write_text(json.dumps(record, ensure_ascii=False)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong case count"):
        load_inputs(tmp_path)


def test_read_condition_logits_checks_id_size_and_finiteness(tmp_path):
    original = prepare(tmp_path)
    path = tmp_path / "logits.jsonl"
    score = {"case_id": original["id"], "logits": [float(i) for i in range(16)]}
    path.write_text(json.dumps(score)+"\n", encoding="utf-8")
    matrix = read_condition_logits(path, [original])
    assert matrix.shape == (1, 16)
    assert int(matrix.argmax()) == 15
    path.write_text(
        json.dumps({"case_id": "dev-000002", "logits": score["logits"]})+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case ID"):
        read_condition_logits(path, [original])
    path.write_text(
        json.dumps({"case_id": original["id"], "logits": [0] * 15})+"\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sixteen"):
        read_condition_logits(path, [original])
