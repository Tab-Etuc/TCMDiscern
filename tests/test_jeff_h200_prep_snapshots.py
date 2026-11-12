"""No-GPU regression checks for the independent H200 smoke preparations.

Reports are aggregates of synthetic CPU tests. Every check verifies that
a simulated model was used and no 31B Jeff score is being claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.train_jeff_lora import LORA_TARGETS
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def test_cpu_peft_all_seven_author_projection_targets_match() -> None:
    report = json.loads(
        (ROOT / "jeff_toy_peft_all_projections_2026-11-12.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "TINY_CPU_PEFT_AND_HEAD_GRADIENT_CHECKPOINT_PASS"
    assert set(report["target_modules"]) == set(LORA_TARGETS)
    assert all(v == 1 for v in report["target_modules"].values())
    assert report["training_real_jeff"] is False
    assert report["original_jeff_head_loaded"] is False
    assert report["all_four_losses_have_nonzero_lora_and_head_gradient"]
    assert report["step0_max_abs_logit_difference"] == 0.0
    assert report["adapter_checkpoint_reload_matches"]
    assert {label for label in report["losses"]} == {"s1", "s2", "s3", "s4"}


def test_cpu_resume_restores_optimizer_scheduler_rng_and_exact_logits() -> None:
    report = json.loads((ROOT / "jeff_cpu_resume_2026-11-12.json").read_text(encoding="utf-8"))
    assert report["status"] == "PEFT_CPU_RESTART_MATCHES_UNINTERRUPTED_TRAINING"
    assert report["patient_data_loaded"] is False
    assert report["actual_h200_test"] is False
    assert report["replayed_second_update_matches"]
    assert report["step2_max_logit_difference"] == 0.0
    assert report["resumed_optimizer_state"]
    assert report["resumed_scheduler_state"]
    assert report["resumed_torch_rng"]


def test_model_revision_for_training_is_pinned() -> None:
    assert JEFF_MODEL_REVISION == "970c21af246a44c5527f7109fca129ab6015c941"
