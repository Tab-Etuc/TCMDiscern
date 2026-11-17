"""Catch H200 source-package drift and unnecessary optional RAG dependencies."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import run_h200_priority_experiments as queue
from scripts import run_h200_priority_real_gpu_latency as latency
from tcmdiscern.jeff_axolotl_attention import requested_attention_backend
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION
from tcmdiscern.priority_cpu_common import (
    FEATURES,
    SGD_OPTIONS,
    class_balanced_weights,
    metrics,
    row_normalize,
)

ROOT = Path(__file__).resolve().parents[1]


def test_core_cpu_module_imports_when_optional_case_rag_is_unavailable():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; sys.modules["tcmdiscern.case_rag"] = None; from scripts.run_h200_priority_cpu_ablation import run; print("CORE_READY")',
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert process.stdout.strip() == "CORE_READY"


def test_priority_launcher_forces_current_checkout_src_over_old_pythonpath(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "possibly-old-install")
    process = subprocess.run(
        ["bash", str(ROOT / "run_h200_priority_experiments.sh"), "--dry-run"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "[PREFLIGHT] Python imports current checkout:" in process.stdout
    assert str(ROOT / "src/tcmdiscern/__init__.py") in process.stdout
    assert '"official_test_used": false' in process.stdout


def test_cpu_common_matches_historical_frozen_protocol():
    from scripts import evaluate_case_rag_paper_dev as historical_paper
    from scripts import evaluate_case_rag_strong_sgd_fusion_cpu as historical_sgd

    assert FEATURES == historical_sgd.FEATURES
    assert SGD_OPTIONS == historical_sgd.SGD_OPTIONS
    cases = ["甲", "乙", "甲", "丙", "甲", "乙"]
    np.testing.assert_array_equal(class_balanced_weights(cases), historical_sgd.class_balanced_weights(cases))
    scores = np.asarray([[2.0, 0.0, -2.0], [1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(row_normalize(scores), historical_sgd.row_normalize(scores))
    actual = metrics(
        ["甲", "乙", "甲", "丙"], ["甲", "甲", "丙", "丙"],
        ["甲", "乙", "丙"], {"丙"}, np.asarray([True, False, True, True]),
    )
    expected = historical_paper.metrics(
        ["甲", "乙", "甲", "丙"], ["甲", "甲", "丙", "丙"],
        ["甲", "乙", "丙"], {"丙"}, np.asarray([True, False, True, True]),
    )
    assert actual == expected


def test_legacy_h200_checkpoint_and_model_autodiscovery(tmp_path, monkeypatch):
    git_root = tmp_path / "TCMDiscern-git"
    legacy = tmp_path / "TCMDiscern-main"
    git_root.mkdir()
    checkpoint = legacy / "data/runs/s1-seed20261112-e2/resume-0001350"
    checkpoint.mkdir(parents=True)
    summary = checkpoint.parent / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "loss_method": "s1",
                "seed": 20261112,
                "target_updates": 1350,
                "accumulation": 64,
                "source": "train_oof_multipos_k16_epoch03.jsonl",
                "rank": 32,
                "alpha": 64,
                "warmup_steps": 40,
                "epochs": 2,
                "model_revision": JEFF_MODEL_REVISION,
                "lora_lr": 2e-5,
                "head_lr": 1e-4,
                "weight_decay": 0.01,
                "clip_norm": 1.0,
                "checkpoint_every": 300,
                "micro_batch_size": 2,
                "attention_backend": requested_attention_backend(),
            }
        ),
        encoding="utf-8",
    )
    model = legacy / "data/models/Jeff-1.1-Large"
    model.mkdir(parents=True)
    monkeypatch.setattr(queue, "ROOT", git_root)
    monkeypatch.delenv("H200_MODEL_DIR", raising=False)

    assert queue.baseline_microbatch() == 2
    assert latency.original_s1_checkpoint(git_root) == checkpoint
    assert queue.default_model_dir() == model
