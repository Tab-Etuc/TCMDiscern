"""Priority H200 runner and training-ablation CPU safeguards."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_case_rag_strong_sgd_fusion_cpu import frozen_pairwise_margin1
from scripts.run_h200_priority_cpu_ablation import arbitrate
from scripts.run_h200_priority_experiments import Experiment, train_args
from scripts.run_jeff_finetune import build_parser
from tcmdiscern.jeff_train_loop import fixed_run_settings, validate_run_args


def make_args(tmp_path: Path, *, component: str = "both", negative: str = "hard"):
    return build_parser().parse_args([
        "--model-dir", str(tmp_path),
        "--method", "s1", "--run-id", "priority-test-example",
        "--steps", "1", "--epochs", "1", "--accumulation", "64",
        "--warmup-steps", "0", "--trainable-components", component,
        "--negative-sampling", negative,
    ])


@pytest.mark.parametrize("component", ("head", "lora", "both"))
def test_component_runs_have_distinct_resume_contracts(tmp_path: Path, component: str):
    args = make_args(tmp_path, component=component)
    manifest = fixed_run_settings(args)
    assert manifest.get("trainable_components", "both") == component


def test_random_negatives_record_source_in_checkpoint_contract(tmp_path: Path):
    args = make_args(tmp_path, negative="random")
    manifest = fixed_run_settings(args)
    assert manifest["source"] == "train_oof_randomneg_k16_epoch03.jsonl"
    assert manifest["negative_sampling"] == "random"


def test_random_negatives_not_allowed_for_s2(tmp_path: Path):
    args = make_args(tmp_path, negative="random")
    args.method = "s2"
    with pytest.raises(ValueError, match="requires syndrome S1"):
        validate_run_args(args)


def test_prioritized_commands_have_ablation_flags(tmp_path: Path, monkeypatch):
    import scripts.run_h200_priority_experiments as queue

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    task = Experiment("s1-head-only", components="head")
    command = train_args(task, tmp_path)
    args = build_parser().parse_args(command[2:])
    assert args.trainable_components == "head"
    assert args.negative_sampling == "hard"
    assert args.steps == 1350
    assert args.micro_batch_size == 1


def test_fixed_pairwise_matches_existing_frozen_reference():
    rng = np.random.default_rng(20261117)
    sgd = rng.normal(size=(19, 16))
    jeff = rng.normal(scale=3, size=(19, 16))
    supports = rng.integers(1, 1000, size=(19, 16))
    expected = frozen_pairwise_margin1(sgd, jeff, supports)
    actual = arbitrate(sgd, jeff, supports)
    np.testing.assert_array_equal(actual, expected)


def test_gate_without_pairwise_only_uses_jeff_scores():
    rng = np.random.default_rng(20261117)
    sgd = rng.normal(size=(10, 16))
    jeff = rng.normal(size=(10, 16))
    support = np.ones((10, 16))
    prediction = arbitrate(sgd, jeff, support, use_pairwise=False)
    assert prediction.shape == (10,)
    assert np.all((prediction >= 0) & (prediction < 16))
