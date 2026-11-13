"""Regression suite: one-load GH200 pilot and in-process stage state machine.

Synthetic stubs deliberately avoid allocating 31B, PyTorch, or patient data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import run_jeff_pilot_once as pilot
from scripts.joint16_h200 import valid_eval_report, valid_training_report
from tcmdiscern.h200.workflow import H200Runner, Stage


def test_pilot_cli_uses_registered_16_real_updates(tmp_path):
    args = pilot.pilot_training_args(tmp_path, root=tmp_path)
    assert args.run_id == "pilot-s3-16updates"
    assert args.steps == 16
    assert args.accumulation == 64
    assert args.method == "s3"
    assert args.resume is None


def test_pilot_resumes_last_complete_optimizer_boundary(tmp_path):
    run_dir = tmp_path / "data/runs" / pilot.RUN_ID
    target = run_dir / "resume-0000004"
    target.mkdir(parents=True)
    for name in (
        "training_state.json",
        "manifest.json",
        "probe_logits.json",
        "head.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
    ):
        (target / name).touch()
    (target / "adapter").mkdir()
    (target / "adapter/adapter_model.safetensors").touch()
    from tcmdiscern.h200.plan import SEEDS
    from tcmdiscern.jeff_config import JEFF_MODEL_REVISION

    (target / "training_state.json").write_text(
        json.dumps(
            {
                "run_id": pilot.RUN_ID,
                "loss_method": "s3",
                "seed": SEEDS[0],
                "target_updates": 16,
                "accumulation": 64,
                "model_revision": JEFF_MODEL_REVISION,
                "global_update": 4,
            }
        ),
        encoding="utf-8",
    )
    args = pilot.pilot_training_args(tmp_path, root=tmp_path)
    assert args.resume == target
    (target / "rng.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete optimizer checkpoint"):
        pilot.pilot_training_args(tmp_path, root=tmp_path)


def test_integrated_pilot_initializes_one_model_and_reloads_saved_adapter(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "model").mkdir()
    original = SimpleNamespace()
    adapter = SimpleNamespace(peft_config={"default": {}})
    decider = SimpleNamespace(base=original)
    jeff = SimpleNamespace(decider=decider)
    calls = {"load": 0, "train": 0, "reset": 0, "reload": 0}

    def fake_load(model_dir):
        calls["load"] += 1
        return jeff

    def fake_train(args, *, released):
        assert released is jeff
        calls["train"] += 1
        decider.base = adapter
        folder = tmp_path / "data/runs" / pilot.RUN_ID
        final = folder / "resume-0000016"
        final.mkdir(parents=True)
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "status": "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
                    "run_id": pilot.RUN_ID,
                    "loss_method": "s3",
                    "global_update": 16,
                    "accumulation": 64,
                    "source_count_verified": 43180,
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "training_state.json",
            "manifest.json",
            "probe_logits.json",
            "head.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng.pt",
        ):
            (final / name).touch()
        (final / "adapter").mkdir()
        (final / "adapter/adapter_model.safetensors").touch()
        return {"global_update": 16}

    def fake_reset(decider, model_dir):
        assert decider.base is adapter
        calls["reset"] += 1
        decider.base = original

    def fake_restore(decider, saved):
        assert decider.base is original
        assert saved.name == "resume-0000016"
        calls["reload"] += 1
        decider.base = adapter

    def fake_decide(rows, decider, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(
                json.dumps(
                    {
                        "case_id": f"dev-{i:06d}",
                        "logits": [float(j) for j in range(16)],
                    }
                )
                + "\n"
                for i in range(2)
            ),
            encoding="utf-8",
        )
        return 2, 100, 0.5

    import scripts.train_jeff_lora as trainer

    monkeypatch.setattr(trainer, "load_original_checkpoint", fake_load)
    monkeypatch.setattr(pilot, "train", fake_train)
    monkeypatch.setattr(pilot, "reset_released_backbone", fake_reset)
    monkeypatch.setattr(pilot, "restore_checkpoint", fake_restore)
    monkeypatch.setattr(pilot, "run_decider", fake_decide)
    monkeypatch.setattr(pilot, "read_two_dev_rows", lambda root: [{"id": "one"}, {"id": "two"}])

    report = pilot.run_one_load(tmp_path / "model", root=tmp_path)
    assert report["model_weight_loads"] == 1
    assert report["optimizer_steps"] == 16
    assert calls == {"load": 1, "train": 1, "reset": 1, "reload": 1}
    assert (tmp_path / "data/h200/pilot_one_load.json").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        pilot.run_one_load(tmp_path / "model", root=tmp_path)


def test_inline_gpu_stage_verifies_and_skips_completed(tmp_path):
    calls = []
    runner = H200Runner(
        root=tmp_path,
        python="/virtualenv/python",
        invoke=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected fork")),
    )
    report = tmp_path / "done.json"

    def perform():
        calls.append(1)
        report.write_text('{"rows":2}', encoding="utf-8")

    stage = Stage(
        name="one-load",
        command=("-m", "example"),
        expected=report,
        verify=lambda: json.loads(report.read_text())["rows"] == 2,
        perform=perform,
    )
    runner.stage(stage)
    runner.stage(stage)
    assert calls == [1]
    assert runner.status()[0]["status"] == "complete"
    report.unlink()
    with pytest.raises(FileNotFoundError):
        runner.stage(stage)


def test_existing_smoke_logits_must_match_current_attention(monkeypatch, tmp_path):
    output = tmp_path / "smoke.jsonl"
    output.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": f"dev-{i:06d}",
                    "logits": [0.0] * 16,
                }
            )
            + "\n"
            for i in range(2)
        ),
        encoding="utf-8",
    )

    def changed_backend(rows, decider, dest):
        dest.write_text(
            "".join(
                json.dumps(
                    {
                        "case_id": f"dev-{i:06d}",
                        "logits": [5.0] * 16,
                    }
                )
                + "\n"
                for i in range(2)
            ),
            encoding="utf-8",
        )
        return 2, 20, 0.2

    monkeypatch.setattr(pilot, "run_decider", changed_backend)
    with pytest.raises(RuntimeError, match="disagree with the active causal"):
        pilot.score_two_if_missing([{}, {}], object(), output)
    assert output.is_file()


def test_joint_gpu_report_checks_full_raw_logits_not_just_json(tmp_path):
    private = tmp_path / "data/joint16/dev_disease_pilot_logits.jsonl"
    private.parent.mkdir(parents=True)
    private.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": f"dev-{i:06d}",
                    "logits": [float(j) for j in range(16)],
                }
            )
            + "\n"
            for i in range(2)
        ),
        encoding="utf-8",
    )
    report = tmp_path / "joint_report.json"
    report.write_text(
        json.dumps(
            {
                "kind": "SMOKE",
                "mode": "disease",
                "split": "dev",
                "run_id": "pilot",
                "k": 16,
                "attention_backend": "eager",
                "performance": {"n_predictions": 2},
                "logits_file": "data/joint16/dev_disease_pilot_logits.jsonl",
            }
        ),
        encoding="utf-8",
    )
    assert valid_eval_report(
        report, task="disease", split="dev", run_id="pilot", limit=2, root=tmp_path
    )
    private.write_text('{"case_id":"dev-000000","logits":[1]}\n', encoding="utf-8")
    assert not valid_eval_report(
        report, task="disease", split="dev", run_id="pilot", limit=2, root=tmp_path
    )


def test_joint_gpu_training_report_requires_complete_optimizer_snapshot(tmp_path):
    run_dir = tmp_path / "data/runs" / "joint16-pilot-disease"
    checkpoint = run_dir / "resume-0000001"
    checkpoint.mkdir(parents=True)
    for name in (
        "training_state.json",
        "manifest.json",
        "probe_logits.json",
        "head.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
    ):
        (checkpoint / name).touch()
    (checkpoint / "adapter").mkdir()
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    adapter.touch()
    summary = run_dir / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
                "study_mode": "disease16",
                "loss_method": "s1",
                "run_id": "joint16-pilot-disease",
                "global_update": 1,
                "target_updates": 1,
                "accumulation": 1,
                "source_count_verified": 43180,
            }
        ),
        encoding="utf-8",
    )
    assert valid_training_report(
        summary,
        task="disease",
        run_id="joint16-pilot-disease",
        steps=1,
        accumulation=1,
    )
    adapter.unlink()
    assert not valid_training_report(
        summary,
        task="disease",
        run_id="joint16-pilot-disease",
        steps=1,
        accumulation=1,
    )


def test_integrated_pilot_completion_checks_all_files(tmp_path):
    runner = H200Runner(root=tmp_path)
    report = tmp_path / "data/h200/pilot_one_load.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "status": "JEFF_SYNDROME_PILOT_ONE_LOAD_PASS",
                "model_weight_loads": 1,
                "optimizer_steps": 16,
                "fresh_adapter_reload": True,
                "test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    assert not runner._valid_integrated_pilot()
