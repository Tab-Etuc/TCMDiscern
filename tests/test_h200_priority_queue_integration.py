"""CPU-only integration tests of the high-cost H200 experiment queue.

The fake GPU session never imports Jeff or allocates VRAM. It exercises the
actual queue ordering, early-exit branches, checkpoint identities and resume
files to catch failures before the multi-day 31B experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_h200_priority_experiments as queue
from scripts import run_h200_priority_real_gpu_latency as latency
from tcmdiscern.jeff_axolotl_attention import requested_attention_backend
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION
from tcmdiscern.jeff_train_loop import fixed_run_settings


def fake_selected(tmp_path: Path) -> list[int]:
    values = list(range(int(5486 * 0.20)))
    path = tmp_path / "data/h200/priority_budget_20pct_indices.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dev_cases": 5486,
                "policy": "lowest SGD top2 margin",
                "selected_indices": values,
            }
        ) + "\n",
        encoding="utf-8",
    )
    return values


def fake_latency_report(indices: list[int]) -> dict:
    return {
        "status": "H200_PRIORITY_GPU_LATENCY_COMPLETED",
        "official_test_used": False,
        "model_run_id": "s1-seed20261112-e2",
        "total_dev_cases": 5486,
        "model_calls": len(indices),
        "selected_indices": indices,
        "same_checkpoint_saved_logit_argmax_mismatches": 0,
        "p50_seconds_per_call": 0.1,
        "p95_seconds_per_call": 0.3,
        "mean_seconds_per_call": 0.2,
    }


@pytest.mark.parametrize("batch_size", [1, 2])
def test_second_seed_uses_original_s1_micro_batch(tmp_path, monkeypatch, batch_size):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    original = tmp_path / "data/runs/s1-seed20261112-e2/summary.json"
    original.parent.mkdir(parents=True)
    original.write_text(
        json.dumps(
            {
                "loss_method": "s1", "seed": 20261112,
                "target_updates": 1350, "accumulation": 64,
                "source": "train_oof_multipos_k16_epoch03.jsonl",
                "rank": 32, "alpha": 64, "warmup_steps": 40,
                "epochs": 2,
                "model_revision": JEFF_MODEL_REVISION,
                "lora_lr": 2e-5, "head_lr": 1e-4,
                "weight_decay": 0.01, "clip_norm": 1.0,
                "checkpoint_every": 300,
                "attention_backend": requested_attention_backend(),
                "micro_batch_size": batch_size,
            }
        ),
        encoding="utf-8",
    )
    args = queue.train_args(queue.Experiment("seed", method="s1", seed=20261113), tmp_path)
    assert args[args.index("--micro-batch-size") + 1] == str(batch_size)


def test_existing_summary_with_wrong_settings_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    task = queue.Experiment("s1-head-only", components="head")
    path = tmp_path / "data/runs" / task.run_id / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"global_update": 1350, "loss_method": "s1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="contract mismatch"):
        queue.train_args(task, tmp_path)


def test_partial_gpu_latency_resumes_in_original_order(tmp_path):
    indices = [2, 5, 7]
    path = tmp_path / "progress.jsonl"
    path.write_text(
        json.dumps(
            {
                "row_index": 2, "model_run_id": latency.RUN_ID,
                "top1_match": True, "tokens": 100, "seconds": 0.5,
                "max_abs_logits_delta": 0.02,
            }
        ) + "\n",
        encoding="utf-8",
    )
    assert len(latency.read_partial(path, indices)) == 1
    with pytest.raises(ValueError, match="does not match"):
        latency.read_partial(path, [5, 2, 7])


def test_reject_stale_gpu_latency_cache(tmp_path):
    indices = [2, 5]
    output = tmp_path / "latency.json"
    output.write_text(json.dumps(fake_latency_report(indices)), encoding="utf-8")
    assert latency.completed_report(output, indices) is not None
    with pytest.raises(RuntimeError, match="another policy"):
        latency.completed_report(output, [2, 7])


def setup_fake_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "STATE", tmp_path / "data/h200/priority_experiment_queue.json")
    for method in ("s1", "s2", "s3"):
        source = (
            tmp_path / "data/jeff"
            / f"dev_{method}-seed20261112-e2-epoch2_k16_5486.jsonl"
        )
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        report = (
            tmp_path / "experiments"
            / f"jeff_{method}-seed20261112-e2-epoch2_k16_dev.json"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.touch()
    original = tmp_path / "data/runs/s1-seed20261112-e2/resume-0001350"
    original.mkdir(parents=True)
    for filename in (
        "training_state.json", "manifest.json", "probe_logits.json",
        "head.safetensors", "optimizer.pt", "scheduler.pt", "rng.pt",
    ):
        (original / filename).touch()
    adapter = original / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").touch()
    (original / "training_state.json").write_text(
        json.dumps(
            {
                "loss_method": "s1", "seed": 20261112, "target_updates": 1350,
                "accumulation": 64, "source": "train_oof_multipos_k16_epoch03.jsonl",
                "rank": 32, "alpha": 64, "warmup_steps": 40, "epochs": 2,
                "model_revision": JEFF_MODEL_REVISION, "lora_lr": 2e-5,
                "head_lr": 1e-4, "weight_decay": 0.01, "clip_norm": 1.0,
                "checkpoint_every": 300,
                "attention_backend": requested_attention_backend(),
                "micro_batch_size": 1,
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "original-model"
    model.mkdir()
    selected = fake_selected(tmp_path)
    calls = []

    def mock_cpu(module, output, *, python, args=(), always_run=False):
        calls.append(("cpu", module))
        if "ablation" in module:
            fake_selected(tmp_path)

    monkeypatch.setattr(queue, "run_cpu_stage", mock_cpu)
    import scripts.summarize_h200_priority_experiments as summary

    monkeypatch.setattr(summary, "collect", lambda: calls.append(("summary", None)))
    return model, selected, calls


def test_when_training_is_done_missing_latency_still_runs_gpu(monkeypatch, tmp_path):
    model, selected, calls = setup_fake_queue(monkeypatch, tmp_path)
    monkeypatch.setattr(queue, "pending_tasks", lambda **kwargs: [])

    class Session:
        model_weight_loads = 1

        def __init__(self, model_dir):
            calls.append(("gpu-session", str(model_dir)))

    import tcmdiscern.h200.session as resident

    monkeypatch.setattr(resident, "JeffGPUSession", Session)

    def fake_gpu_run(session):
        calls.append(("gpu-latency", None))
        assert session.model_weight_loads == 1
        return fake_latency_report(selected)

    monkeypatch.setattr(latency, "run", fake_gpu_run)
    queue.run_queue(model_dir=model, skip_seeds=False, dry_run=False)
    assert ("gpu-latency", None) in calls
    assert calls.index(("gpu-latency", None)) < calls.index(("summary", None))
    state = json.loads(queue.STATE.read_text(encoding="utf-8"))
    assert state["status"] == "completed"


def test_cached_gpu_latency_skips_loading_31b_when_no_training(monkeypatch, tmp_path):
    model, indices, calls = setup_fake_queue(monkeypatch, tmp_path)
    monkeypatch.setattr(queue, "pending_tasks", lambda **kwargs: [])
    cache = tmp_path / "experiments/h200_priority_real_gpu_latency.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(fake_latency_report(indices)), encoding="utf-8")
    import tcmdiscern.h200.session as resident

    class ForbiddenSession:
        def __init__(self, model_dir):
            raise AssertionError("Idle GPU must remain unloaded")

    monkeypatch.setattr(resident, "JeffGPUSession", ForbiddenSession)
    queue.run_queue(model_dir=model, skip_seeds=False, dry_run=False)
    assert ("summary", None) in calls


def test_smoke_precedes_full_training_and_two_epochs(monkeypatch, tmp_path):
    model, _indices, calls = setup_fake_queue(monkeypatch, tmp_path)
    task = queue.Experiment("s1-head-only", components="head")
    monkeypatch.setattr(queue, "EXPERIMENTS", (task,))
    monkeypatch.setattr(queue, "complete_optimizer_checkpoint", lambda p: p.is_dir())
    monkeypatch.setattr(
        queue,
        "valid_report",
        lambda run_id, epoch=2: (
            tmp_path / "experiments" / f"jeff_{run_id}-epoch{epoch}_k16_dev.json"
        ).is_file(),
    )
    monkeypatch.setattr(latency, "run", lambda session: calls.append(("gpu-latency", None)))

    class Session:
        model_weight_loads = 1

        def __init__(self, model_dir):
            calls.append(("gpu-session", None))

        def train_cli(self, command):
            args = command[2:]
            run_id = args[args.index("--run-id") + 1]
            steps = int(args[args.index("--steps") + 1])
            parsed = queue_runner_args().parse_args(args)
            summary = fixed_run_settings(parsed)
            summary.update(
                status="JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
                trainable_components=parsed.trainable_components,
                negative_sampling=parsed.negative_sampling,
                global_update=steps,
                source_count_verified=43180,
                expected_case_exposures=64 if steps == 1 else 86360,
                microbatches=64 if steps == 1 else 86360,
                first_gradient={"all_gradients_finite": True},
                first_optimizer_update_verified={"active_component_max_delta": 0.001},
            )
            folder = tmp_path / "data/runs" / run_id
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for step in (steps,) if steps == 1 else (675, 1350):
                chk = folder / f"resume-{step:07d}"
                chk.mkdir()
                (chk / "training_state.json").write_text(json.dumps(summary), encoding="utf-8")
            calls.append(("train", run_id))

        def syndrome_dev(self, *, root, checkpoint, run_id):
            path = tmp_path / "experiments" / f"jeff_{run_id}_k16_dev.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            calls.append(("eval", run_id))

    import tcmdiscern.h200.session as resident

    monkeypatch.setattr(resident, "JeffGPUSession", Session)
    queue.run_queue(model_dir=model, skip_seeds=True, dry_run=False)
    tail = [event for event in calls if event[0] in ("gpu-latency", "train", "eval")]
    assert tail == [
        ("gpu-latency", None),
        ("train", "priority-smoke-s1-head-only"),
        ("train", "priority-s1-head-only"),
        ("eval", "priority-s1-head-only-epoch1"),
        ("eval", "priority-s1-head-only-epoch2"),
    ]


def queue_runner_args():
    from scripts.run_jeff_finetune import build_parser

    return build_parser()
