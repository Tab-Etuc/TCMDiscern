"""Micro-batch throughput controls preserve original H200 research budgets.

All assertions here are CPU-only; the real padded Gemma4/Jeff forward and
backward are separately exercised by scripts.smoke_tiny_jeff on GH200.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from scripts.run_jeff_finetune import build_parser
from tcmdiscern.h200.plan import SEEDS, Experiment
from tcmdiscern.h200.workflow import H200Runner
from tcmdiscern.jeff_losses import numpy_loss_gradient
from tcmdiscern.jeff_resume import verify_resume_settings
from tcmdiscern.jeff_train_loop import (
    fixed_run_settings,
    prepared_microbatches,
    validate_run_args,
)


def spec(batch_size=1, accumulation=64):
    return build_parser().parse_args(
        [
            "--model-dir",
            "/unused",
            "--method",
            "s3",
            "--run-id",
            "cpu-microbatch-contract",
            "--steps",
            "2",
            "--epochs",
            "1",
            "--accumulation",
            str(accumulation),
            "--micro-batch-size",
            str(batch_size),
        ]
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_microbatch_redistribution_preserves_each_case_and_weight(batch_size):
    items = [
        {"row": {"id": f"train-{index:06d}", "tokens": [index] * (index * 9 + 7)}}
        for index in range(1, 65)
    ]
    weights = [index / sum(range(1, 65)) for index in range(1, 65)]
    groups = prepared_microbatches(
        items,
        weights,
        micro_batch_size=batch_size,
        encode=lambda row: row["tokens"],
    )
    flat = [entry for group in groups for entry in group]
    assert all(1 <= len(group) <= batch_size for group in groups)
    assert Counter(item["row"]["id"] for item, _, _ in flat) == Counter(
        item["row"]["id"] for item in items
    )
    assert {item["row"]["id"]: weight for item, weight, _ in flat} == {
        item["row"]["id"]: weight for item, weight in zip(items, weights, strict=True)
    }
    assert abs(sum(weight for _, weight, _ in flat) - 1.0) < 1e-12
    if batch_size > 1:
        lengths = [len(tokens) for _, _, tokens in flat]
        assert lengths == sorted(lengths)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("method", ["s1", "s2", "s3", "s4"])
def test_identical_loss_and_logit_gradients_under_grouping(batch_size, method):
    rng = np.random.default_rng(20261114)
    items = [
        {
            "row": {"id": str(index), "tokens": [7] * (10 + index)},
            "target_index": 2,
            "positive_indices": [2, 5],
        }
        for index in range(7)
    ]
    weights = [(index + 1) / sum(range(1, 8)) for index in range(7)]
    logits = [rng.normal(size=16) for _ in items]
    reference_loss = 0.0
    reference_gradient = np.zeros((7, 16))
    for index, (item, weight) in enumerate(zip(items, weights, strict=True)):
        loss, grad = numpy_loss_gradient(
            logits[index],
            positive_indices=[item["target_index"]] if method == "s1" else item["positive_indices"],
            gold_index=item["target_index"],
            method=method,
        )
        reference_loss += loss * weight
        reference_gradient[index] = grad * weight
    groups = prepared_microbatches(
        items,
        weights,
        micro_batch_size=batch_size,
        encode=lambda row: row["tokens"],
    )
    batched_loss = 0.0
    batched_gradient = np.zeros_like(reference_gradient)
    for group in groups:
        for item, weight, _ in group:
            index = int(item["row"]["id"])
            loss, grad = numpy_loss_gradient(
                logits[index],
                positive_indices=[item["target_index"]]
                if method == "s1"
                else item["positive_indices"],
                gold_index=item["target_index"],
                method=method,
            )
            batched_loss += loss * weight
            batched_gradient[index] = grad * weight
    assert abs(batched_loss - reference_loss) < 1e-12
    assert np.allclose(batched_gradient, reference_gradient, atol=1e-12)


def test_length_bucket_falls_back_to_single_for_extreme_train_prompt():
    lengths = [300, 310, 900, 920, 1600, 1650, 7200]
    items = [
        {"row": {"id": f"train-{i:06d}", "tokens": [i + 3] * size}}
        for i, size in enumerate(lengths)
    ]
    groups = prepared_microbatches(
        items,
        [1.0 / len(items)] * len(items),
        micro_batch_size=2,
        encode=lambda row: row["tokens"],
    )
    assert sum(len(group) for group in groups) == len(items)
    assert len([group for group in groups if len(group) > 1]) >= 1
    assert all(
        len(group) == 1 or max(len(entry[2]) for entry in group) * len(group) <= 2048
        for group in groups
    )
    assert any(len(group) == 1 and len(group[0][2]) == 7200 for group in groups)


def test_batch1_legacy_checkpoint_compatible_but_accelerated_resume_is_frozen():
    normal = fixed_run_settings(spec())
    faster = fixed_run_settings(spec(batch_size=2))
    assert "micro_batch_size" not in normal
    assert faster["micro_batch_size"] == 2
    verify_resume_settings(dict(normal), normal)
    verify_resume_settings(dict(faster), faster)
    with pytest.raises(ValueError, match="micro_batch_size"):
        verify_resume_settings(dict(faster), normal)
    with pytest.raises(ValueError, match="micro_batch_size"):
        verify_resume_settings(dict(normal), faster)


@pytest.mark.parametrize("invalid", [0, -1, 9, 65])
def test_invalid_microbatch_rejected_before_gpu(tmp_path, monkeypatch, invalid):
    import tcmdiscern.jeff_train_loop as training

    monkeypatch.setattr(training, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="micro_batch_size"):
        validate_run_args(spec(batch_size=invalid))


def test_microbatch_cannot_exceed_original_gradient_accumulation(tmp_path, monkeypatch):
    import tcmdiscern.jeff_train_loop as training

    monkeypatch.setattr(training, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="micro_batch_size"):
        validate_run_args(spec(batch_size=4, accumulation=2))


def test_registered_runs_can_opt_in_without_changing_original_seed(monkeypatch):
    monkeypatch.setenv("TCMDISCERN_MICRO_BATCH_SIZE", "2")
    run = Experiment("s3", SEEDS[0], epochs=2)
    command = run.train_arguments(model_dir="/release")
    assert command[command.index("--micro-batch-size") + 1] == "2"
    assert command[command.index("--accumulation") + 1] == "64"
    assert command[command.index("--seed") + 1] == str(SEEDS[0])
    assert command[command.index("--steps") + 1] == "1350"


def test_h200_rejects_changed_microbatch_before_loading_model(tmp_path, monkeypatch):
    experiment = Experiment("s3", SEEDS[0], epochs=2)
    run_dir = tmp_path / "data/runs" / experiment.run_id
    checkpoint = run_dir / "resume-0000300"
    checkpoint.mkdir(parents=True)
    for name in (
        "manifest.json",
        "probe_logits.json",
        "head.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
    ):
        (checkpoint / name).touch()
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    adapter.parent.mkdir()
    adapter.touch()
    (checkpoint / "training_state.json").write_text(
        json.dumps(
            {
                "run_id": experiment.run_id,
                "loss_method": "s3",
                "target_updates": 1350,
                "seed": SEEDS[0],
                "example_weighting": "uniform",
                "global_update": 300,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TCMDISCERN_MICRO_BATCH_SIZE", "2")
    runner = H200Runner(root=tmp_path)
    with pytest.raises(ValueError, match="micro_batch_size"):
        runner._preflight_resume_protocol(full=False)


def test_completed_results_cannot_mix_batch_protocols(tmp_path, monkeypatch):
    run = Experiment("s3", SEEDS[0], epochs=2)
    path = tmp_path / "data/runs" / run.run_id / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"micro_batch_size": 1}), encoding="utf-8")
    monkeypatch.setenv("TCMDISCERN_MICRO_BATCH_SIZE", "2")
    with pytest.raises(ValueError, match="micro_batch_size"):
        H200Runner(root=tmp_path)._preflight_resume_protocol(full=False)


def test_registered_run_defaults_to_fast_micro_batch_size_two(monkeypatch):
    monkeypatch.delenv("TCMDISCERN_MICRO_BATCH_SIZE", raising=False)
    command = Experiment("s3", SEEDS[0], epochs=2).train_arguments(model_dir="/release")
    assert command[command.index("--micro-batch-size") + 1] == "2"
