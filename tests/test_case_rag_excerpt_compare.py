"""CPU regression of matched original/focused Case-RAG outcome comparisons."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts import compare_case_rag_evidence_styles as evaluator
from tcmdiscern.case_rag import CONDITIONS, CONDITIONS_2
from tcmdiscern.jeff_format import decision_row


def write_lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in values) + "\n", encoding="utf-8")


@pytest.mark.parametrize("budget", [2, 4])
def test_matched_excerpts_only_compare_same_checkpoint_and_sources(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    prefix_tag = "budget2_0300_20261116" if budget == 2 else "pilotclean_0300_20261116"
    focus_tag = f"focused{budget}_0300_20261116"
    arms = CONDITIONS_2 if budget == 2 else CONDITIONS
    labels = [f"證型{i}" for i in range(16)]
    train = [
        {
            "chief_complaint": "頭暈",
            "description": f"患者{i}經常頭痛",
            "detection": "舌紅",
            "user_id": f"train-{i}",
            "norm_syndrome": f"證型{i}",
        }
        for i in range(148)
    ]
    dev = [
        {
            "chief_complaint": "頭暈",
            "description": f"患者{i}經常頭痛",
            "detection": "舌紅",
            "user_id": f"dev-{i}",
            "norm_syndrome": "證型1",
        }
        for i in range(4)
    ]
    write_lines(tmp_path / "data/tcm-sd/train.json", train)
    write_lines(tmp_path / "data/tcm-sd/dev.json", dev)
    raw = [decision_row(row, case_id=f"dev-{i:06d}", candidates=labels) for i, row in enumerate(dev)]
    for tag, is_focus in ((prefix_tag, False), (focus_tag, True)):
        folder = tmp_path / "data/case_rag" / tag
        folder.mkdir(parents=True)
        manifest = {
            "samples": 4,
            "conditions": list(arms),
            "seed": 20261116,
            "evidence_budget": budget,
            "all_within_8192": True,
            "input_tokens": {arm: {"mean": 750 if is_focus else 740} for arm in arms},
        }
        if is_focus:
            manifest["source_tag"] = prefix_tag
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "provenance.jsonl").write_text("{}\n" * 4, encoding="utf-8")
        for arm in arms:
            prompts = []
            for case in raw:
                row = deepcopy(case)
                if arm != "baseline":
                    row["state"]["retrieved_cases"] = [
                        {"candidate_syndrome": labels[i % 2], "description": (
                            "患者真實記錄摘錄" if is_focus else "患者記錄開頭"
                        )} for i in range(budget)
                    ]
                prompts.append(row)
            write_lines(folder / f"{arm}.jsonl", prompts)
        name = "focus" if is_focus else "orig"
        output = folder / "scores" / name
        output.mkdir(parents=True)
        (output / "run_manifest.json").write_text(json.dumps({
            "source": tag, "checkpoint_method": "s2", "checkpoint_run_id": "frozen",
            "checkpoint": "/s2", "model_dir": "/released", "cases": 4,
            "conditions": list(arms),
        }), encoding="utf-8")
        for arm in arms:
            logits = []
            for case in raw:
                output_label = (
                    1 if is_focus and arm == f"paired{budget}" else 0
                )
                logits.append({
                    "case_id": case["id"],
                    "logits": [1.0 if j == output_label else 0.0 for j in range(16)],
                })
            write_lines(output / f"{arm}.jsonl", logits)
    result = evaluator.compare(budget, reference_run="orig", focused_run="focus")
    assert result["conditions"][f"paired{budget}"]["rescued"] == 4
    assert result["conditions"][f"paired{budget}"]["broken"] == 0
    assert result["conditions"][f"paired{budget}"]["delta_top1_pp"] == 100
    assert result["baseline_predictions_equal"]
    focus_identity = tmp_path / "data/case_rag" / focus_tag / "scores/focus/run_manifest.json"
    data = json.loads(focus_identity.read_text(encoding="utf-8"))
    data["checkpoint"] = "/different"
    focus_identity.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Different Jeff checkpoints"):
        evaluator.compare(budget, reference_run="orig", focused_run="focus")
