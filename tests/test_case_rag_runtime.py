"""Validate the GPU runner's I/O contract using synthetic data, without GPU."""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from scripts.run_case_rag_jeff import load_inputs, read_condition_logits
from tcmdiscern.case_rag import CONDITIONS
from tcmdiscern.jeff_format import decision_row


def make_folder(tmp_path):
    rows = [
        decision_row(
            {
                "chief_complaint": "胃脘疼痛",
                "description": "餐後加重",
                "detection": "舌紅",
            },
            case_id=f"dev-{i:06d}",
            candidates=[f"證型{j}" for j in range(16)],
        )
        for i in (1, 9)
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "samples": 2,
        "conditions": list(CONDITIONS),
        "all_within_8192": True,
    }), encoding="utf-8")
    for condition in CONDITIONS:
        output = []
        for row in rows:
            item = deepcopy(row)
            if condition != "baseline":
                item["state"]["retrieved_cases"] = [{} for _ in range(4)]
            output.append(json.dumps(item, ensure_ascii=False))
        (tmp_path / f"{condition}.jsonl").write_text(
            "\n".join(output) + "\n", encoding="utf-8"
        )
    return rows


def test_model_runner_accepts_five_identical_candidate_sets(tmp_path):
    original = make_folder(tmp_path)
    condition_rows = load_inputs(tmp_path)
    assert all(len(condition_rows[k]) == 2 for k in CONDITIONS)
    assert condition_rows["baseline"] == original
    assert [entry["id"] for entry in load_inputs(tmp_path, limit=1)["paired4"]] == [
        "dev-000001"
    ]


def test_model_runner_refuses_changed_candidate_order(tmp_path):
    make_folder(tmp_path)
    path = tmp_path / "paired4.jsonl"
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payloads[0]["options"].reverse()
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in payloads), encoding="utf-8")
    with pytest.raises(ValueError, match="changed the decision task"):
        load_inputs(tmp_path)


def test_score_parser_verifies_16_logits_and_case_identity(tmp_path):
    inputs = make_folder(tmp_path)
    logits = [
        {"case_id": inputs[0]["id"], "logits": [0.0] * 16},
        {"case_id": inputs[1]["id"], "logits": [1.0] * 16},
    ]
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in logits), encoding="utf-8")
    np.testing.assert_allclose(read_condition_logits(path, inputs), np.array([
        [0.0] * 16, [1.0] * 16
    ]))
    logits[1]["case_id"] = "dev-999999"
    path.write_text("\n".join(json.dumps(r) for r in logits), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong Jeff logits"):
        read_condition_logits(path, inputs)
    logits[1]["case_id"] = inputs[1]["id"]
    logits[1]["logits"] = [float("nan")] * 16
    path.write_text("\n".join(json.dumps(r) for r in logits), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_condition_logits(path, inputs)
