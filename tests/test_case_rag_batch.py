"""CPU tests for one-load multi-suite Case-RAG execution and true prompt reuse."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_case_rag_batch_jeff as batch


def _collection(prefix: str, *, different: bool = False):
    baseline = [{"id": f"dev-{i:06d}", "state": {"chief_complaint": prefix}}
                for i in range(3)]
    paired = deepcopy(baseline)
    for row in paired:
        row["state"]["retrieved_cases"] = [{"candidate_syndrome": "A"}]
    if different:
        paired[0]["state"]["retrieved_cases"][0]["candidate_syndrome"] = "B"
    return {"baseline": baseline, "paired2": paired}


class FakeDecider:
    def __init__(self):
        self.calls = 0
        self.last = {}

    def decide(self, row):
        self.calls += 1
        self.last = {"shortened": False, "thought": None, "output_tokens": 0}
        return [1 / 16] * 16, [float(self.calls)] * 16, 32, None


def _synthetic_checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    for name in (
        "manifest.json", "head_config.json", "probe_logits.json",
        "adapter/adapter_config.json",
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "loss_method": "s2", "run_id": "synthetic",
        }) + "\n", encoding="utf-8")
    for name in ("head.safetensors", "adapter/adapter_model.safetensors"):
        (root / name).write_bytes(b"fake-weights")
    return root


def test_batch_planner_shares_only_identical_whole_prompts():
    first = _collection("原始病例")
    second = _collection("原始病例")
    modified = _collection("原始病例", different=True)
    suites = {"a": first, "b": second, "c": modified}
    plan = batch.plan_identical_inferences(suites)
    assert len(plan) == 6
    assert sum(entry["reuse"] is not None for entry in plan) == 3
    assert plan[2]["reuse"] == {"tag": "a", "condition": "baseline"}
    assert plan[3]["reuse"] == {"tag": "a", "condition": "paired2"}
    assert plan[5]["reuse"] is None


def test_batch_reuses_completed_logits_without_fake_extra_forwards(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    collections = {
        "first": _collection("原始病例"),
        "second": _collection("原始病例"),
    }
    for tag in collections:
        (tmp_path / "data/case_rag" / tag).mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    checkpoint = _synthetic_checkpoint(tmp_path / "checkpoint")
    fake = FakeDecider()
    monkeypatch.setattr(
        batch, "load_checkpoint",
        lambda model_dir, checkpoint, expected_method: (
            fake, {"method": "s2", "source_run_id": "synthetic"}
        ),
    )
    monkeypatch.setattr(
        batch, "score_outputs",
        lambda folder, output, limit: {"status": "SYNTHETIC_ONLY"},
    )
    plan = batch.plan_identical_inferences(collections)
    report = batch.execute(
        collections, plan,
        model_dir=Path("model"), checkpoint=checkpoint,
        expected_method="s2", run_id="verified",
    )
    assert fake.calls == 6
    assert report["total_new_forwards"] == 6
    assert report["deduplicated_arm_count"] == 2
    a = tmp_path / "data/case_rag/first/scores/verified"
    b = tmp_path / "data/case_rag/second/scores/verified"
    assert (a / "baseline.jsonl").stat().st_ino == (b / "baseline.jsonl").stat().st_ino
    assert (a / "paired2.jsonl").stat().st_ino == (b / "paired2.jsonl").stat().st_ino
    assert (a / "baseline.timings.jsonl").stat().st_ino == (
        b / "baseline.timings.jsonl"
    ).stat().st_ino
    assert (a / "paired2.timings.jsonl").stat().st_ino == (
        b / "paired2.timings.jsonl"
    ).stat().st_ino
    score = json.loads((
        tmp_path / "experiments/case_rag_batch_second_verified.json"
    ).read_text())
    assert score["status"] == "SYNTHETIC_ONLY"
    # Re-run after a completed suite: accepted without any repeated model work.
    earlier = fake.calls
    second = batch.execute(
        collections, plan, model_dir=Path("model"), checkpoint=checkpoint,
        expected_method="s2", run_id="verified",
    )
    assert second["total_new_forwards"] == 0
    assert fake.calls == earlier



def test_batch_rejects_edited_prompt_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    cases = {"first": _collection("原始病例")}
    folder = tmp_path / "data/case_rag/first"
    folder.mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    checkpoint = _synthetic_checkpoint(tmp_path / "checkpoint")
    model_loads = []
    decider = FakeDecider()

    def _load(model_dir, checkpoint_dir, expected_method):
        model_loads.append(1)
        return decider, {"method": "s2", "source_run_id": "synthetic"}

    monkeypatch.setattr(batch, "load_checkpoint", _load)
    monkeypatch.setattr(batch, "score_outputs", lambda folder, output, limit: {
        "status": "SYNTHETIC_ONLY",
    })
    original = batch.plan_identical_inferences(cases)
    batch.execute(
        cases, original, model_dir=Path("model"), checkpoint=checkpoint,
        expected_method="s2", run_id="locked",
    )
    assert len(model_loads) == 1
    cases["first"]["paired2"][0]["state"]["retrieved_cases"][0][
        "candidate_syndrome"
    ] = "另一個證型"
    with pytest.raises(ValueError, match="prompt_sha256"):
        batch.execute(
            cases, batch.plan_identical_inferences(cases),
            model_dir=Path("model"), checkpoint=checkpoint,
            expected_method="s2", run_id="locked",
        )
    assert len(model_loads) == 1


def test_batch_restores_shared_timing_if_link_was_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    suites = {
        "a": _collection("同一患者"),
        "b": _collection("同一患者"),
    }
    for tag in suites:
        (tmp_path / "data/case_rag" / tag).mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    checkpoint = _synthetic_checkpoint(tmp_path / "checkpoint")
    fake = FakeDecider()
    monkeypatch.setattr(batch, "load_checkpoint", lambda model_dir, checkpoint_dir, method: (
        fake, {"method": "s2", "source_run_id": "synthetic"},
    ))
    monkeypatch.setattr(batch, "score_outputs", lambda folder, output, limit: {
        "status": "SYNTHETIC_ONLY",
    })
    plan = batch.plan_identical_inferences(suites)
    batch.execute(
        suites, plan, model_dir=Path("model"), checkpoint=checkpoint,
        expected_method="s2", run_id="same",
    )
    missing = tmp_path / "data/case_rag/b/scores/same/paired2.timings.jsonl"
    missing.unlink()
    started = fake.calls
    result = batch.execute(
        suites, plan, model_dir=Path("model"), checkpoint=checkpoint,
        expected_method="s2", run_id="same",
    )
    assert result["total_new_forwards"] == 0
    assert fake.calls == started
    assert missing.is_file()

def test_batch_rejects_modified_checkpoint_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    folder = tmp_path / "data/case_rag/first"
    folder.mkdir(parents=True)
    checkpoint = _synthetic_checkpoint(tmp_path / "checkpoint")
    destination = folder / "scores/bad-model"
    destination.mkdir(parents=True)
    (destination / "run_manifest.json").write_text(json.dumps({"checkpoint": "different"}))
    with pytest.raises(ValueError, match="changed"):
        batch._verify_run_identity(
            folder, destination, _collection("baseline"),
            Path("model"), checkpoint,
            {"method": "s2", "source_run_id": "synthetic"},
        )
