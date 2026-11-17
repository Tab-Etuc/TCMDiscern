"""No-GPU end-to-end tests for real selective-call checkpointing."""
from __future__ import annotations

import json

import numpy as np

from scripts import run_h200_priority_real_gpu_latency as module


class FakeDecider:
    def __init__(self) -> None:
        self.calls = 0
        self.last = {"thought": False, "shortened": False, "output_tokens": 0}

    def decide(self, row):
        self.calls += 1
        return ([1 / 16] * 16, list(range(16)), 240, 0)


class FakeSession:
    def __init__(self) -> None:
        self.decider = FakeDecider()
        self.loads = 0

    def original(self):
        self.loads += 1
        return type("Released", (), {"decider": self.decider})()


def test_partial_selective_gpu_resume_runs_only_remaining(monkeypatch, tmp_path):
    selection = list(range(int(5486 * 0.20)))
    pick_file = tmp_path / "data/h200/priority_budget_20pct_indices.json"
    pick_file.parent.mkdir(parents=True, exist_ok=True)
    pick_file.write_text(
        json.dumps({
            "dev_cases": 5486, "selected_indices": selection,
            "policy": "lowest SGD top2 margin",
        }), encoding="utf-8"
    )
    dev = tmp_path / "data/jeff/dev_k16.jsonl"
    dev.parent.mkdir(parents=True, exist_ok=True)
    with dev.open("w", encoding="utf-8") as f:
        for i in range(5486):
            f.write(json.dumps({
                "id": f"dev-{i:06d}",
                "options": [{"id": f"label-{j}"} for j in range(16)],
            }) + "\n")
    checkpoint = tmp_path / "data/runs/s1-seed20261112-e2/resume-0001350"
    checkpoint.mkdir(parents=True, exist_ok=True)
    previous_count = 17
    partial = tmp_path / "data/h200/priority_real_gpu_latency_partial.jsonl"
    with partial.open("w", encoding="utf-8") as f:
        for i in selection[:previous_count]:
            f.write(json.dumps({
                "row_index": i, "model_run_id": module.RUN_ID,
                "top1_match": True, "tokens": 240, "seconds": 0.015,
                "max_abs_logits_delta": 0.0,
            }) + "\n")
    fake_logits = np.tile(np.arange(16, dtype=np.float64), (5486, 1))
    monkeypatch.setattr(module, "frozen_paths", lambda root, method: (None, "ignored"))
    monkeypatch.setattr(module, "read_model_logits", lambda path, num_cases: fake_logits)
    monkeypatch.setattr(module, "restore_checkpoint", lambda decider, path: None)

    session = FakeSession()
    report = module.run(
        session, root=tmp_path,
        output=tmp_path / "experiments/h200_priority_real_gpu_latency.json",
    )
    assert report["model_calls"] == len(selection)
    assert report["reused_partial_calls"] == previous_count
    assert report["executed_current_session_calls"] == len(selection) - previous_count
    assert session.decider.calls == len(selection) - previous_count
    assert session.loads == 1
    assert report["same_checkpoint_saved_logit_argmax_mismatches"] == 0
    assert len(module.read_partial(partial, selection)) == len(selection)

    class NoGPU:
        def original(self):
            raise AssertionError("Cached complete report should not load model")

    second = module.run(
        NoGPU(), root=tmp_path,
        output=tmp_path / "experiments/h200_priority_real_gpu_latency.json",
    )
    assert second == report
