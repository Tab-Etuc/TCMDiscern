"""Driver-profile compatibility: GH200 R550 uses isolated CUDA 12.8 PyTorch."""
from __future__ import annotations

import pytest

from scripts import preflight_jeff_training as preflight


def test_cuda_profile_matches_driver_without_host_changes():
    assert preflight.torch_for_driver("550.90.07") == ("2.11.0+cu128", "12.8")
    assert preflight.torch_for_driver("580.95.05") == ("2.13.0", "13.0")
    with pytest.raises(ValueError, match="R525"):
        preflight.torch_for_driver("524.10.00")


def test_gh200_cuda12_preflight_accepts_isolated_pinned_runtime(monkeypatch, tmp_path):
    versions = {
        **preflight.PINNED,
        "torch": "2.11.0+cu128",
        "peft": "0.21.1",
    }
    monkeypatch.setattr(preflight, "installed_versions", lambda: versions)
    monkeypatch.setattr(
        preflight, "inspect_gpu",
        lambda: [{
            "name": "NVIDIA GH200 480GB",
            "memory_mib": 97871,
            "driver_version": "550.90.07",
        }],
    )
    monkeypatch.setattr(
        preflight, "inspect_checkpoint",
        lambda _: {
            "weights_present": True,
            "decision_head_present": True,
            "release_metadata_matches": True,
        },
    )
    monkeypatch.setattr(
        preflight, "torch_cuda_runtime",
        lambda: {"cuda": "12.8", "available": True, "bf16": True},
    )
    report = preflight.preflight(tmp_path)
    assert report["blockers"] == []
    assert report["status"] == "PRELIMINARY_PASS_REQUIRES_GPU_SMOKE"
    assert report["selected_torch_version"] == "2.11.0+cu128"
    assert report["selected_cuda_runtime"] == "12.8"


def test_incompatible_runtime_remains_blocked(monkeypatch, tmp_path):
    monkeypatch.setattr(
        preflight, "installed_versions",
        lambda: {**preflight.PINNED, "torch": "2.7.0", "peft": "0.21.1"},
    )
    monkeypatch.setattr(
        preflight, "inspect_gpu",
        lambda: [{"name": "GH200", "memory_mib": 97871, "driver_version": "550.90.07"}],
    )
    monkeypatch.setattr(
        preflight, "inspect_checkpoint",
        lambda _: {
            "weights_present": True,
            "decision_head_present": True,
            "release_metadata_matches": True,
        },
    )
    monkeypatch.setattr(
        preflight, "torch_cuda_runtime",
        lambda: {"cuda": "12.8", "available": True, "bf16": True},
    )
    report = preflight.preflight(tmp_path)
    assert report["status"] == "BLOCKED"
    assert any("PyTorch" in blocker for blocker in report["blockers"])
