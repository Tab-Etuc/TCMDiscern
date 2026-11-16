"""Tests for H200 split-checkout frozen artifacts and fail-fast lookup."""

from __future__ import annotations

from pathlib import Path

import pytest

from tcmdiscern.case_rag_artifacts import (
    artifact_roots,
    checkpoint_path,
    frozen_paths,
)


def paths(root: Path, method: str) -> tuple[Path, Path]:
    return (
        root / f"experiments/jeff_{method}-seed20261112-e2-epoch2_k16_dev.json",
        root / f"data/jeff/dev_{method}-seed20261112-e2-epoch2_k16_5486.jsonl",
    )


def write_stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test", encoding="utf-8")


def test_frozen_full_dev_finds_original_training_root(tmp_path):
    checkout = tmp_path / "TCMDiscern-git"
    original = tmp_path / "TCMDiscern-main"
    checkout.mkdir()
    originals = paths(original, "s2")
    for path in originals:
        write_stub(path)
    assert frozen_paths(checkout, "s2") == originals


def test_frozen_full_dev_supports_partially_migrated_assets(tmp_path):
    checkout = tmp_path / "TCMDiscern-git"
    original = tmp_path / "TCMDiscern-main"
    checkout.mkdir()
    report, _ = paths(checkout, "s2")
    _, logits = paths(original, "s2")
    write_stub(report)
    write_stub(logits)
    assert frozen_paths(checkout, "s2") == (report, logits)


def test_frozen_full_dev_prefers_complete_current_checkout(tmp_path):
    checkout = tmp_path / "TCMDiscern-git"
    original = tmp_path / "TCMDiscern-main"
    for path in (*paths(checkout, "s2"), *paths(original, "s2")):
        write_stub(path)
    assert frozen_paths(checkout, "s2") == paths(checkout, "s2")


def test_missing_original_full_dev_reports_all_attempts(tmp_path):
    checkout = tmp_path / "TCMDiscern-git"
    checkout.mkdir()
    with pytest.raises(FileNotFoundError, match="TCMDiscern-main"):
        frozen_paths(checkout, "s1")


def test_explicit_legacy_root_can_be_outside_sibling(tmp_path, monkeypatch):
    checkout = tmp_path / "TCMDiscern-git"
    outside = tmp_path / "training-archive"
    checkout.mkdir()
    for path in paths(outside, "s3"):
        write_stub(path)
    monkeypatch.setenv("TCMDISCERN_LEGACY_ROOT", str(outside))
    assert frozen_paths(checkout, "s3") == paths(outside, "s3")
    assert outside in artifact_roots(checkout)


def test_checkpoint_search_uses_legacy_training_run(tmp_path):
    checkout = tmp_path / "TCMDiscern-git"
    original = tmp_path / "TCMDiscern-main"
    checkout.mkdir()
    write_stub(original / "data/runs/s2/manifest.json")

    def choose(candidate, method, explicit):
        path = candidate / "data/runs" / method
        if not (path / "manifest.json").exists():
            raise FileNotFoundError("Missing original checkpoint")
        return path

    assert checkpoint_path(checkout, "s2", choose) == original / "data/runs/s2"


def test_reject_unrecognized_checkpoint_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        frozen_paths(tmp_path, "unknown")


def test_all_three_sources_are_checked_before_expensive_token_audit(tmp_path, monkeypatch):
    from scripts import run_case_rag_paper_h200 as paper

    checkout = tmp_path / "TCMDiscern-git"
    original = tmp_path / "TCMDiscern-main"
    checkout.mkdir()
    for method in ("s2", "s1"):
        for path in paths(original, method):
            write_stub(path)
    called = []
    monkeypatch.setattr(paper, "_load_inputs", lambda root: {"baseline": []})
    monkeypatch.setattr(
        paper,
        "_verify_frozen_prompts",
        lambda *args: called.append("tokenized"),
    )
    with pytest.raises(FileNotFoundError, match="S3:"):
        paper.preflight(checkout, checkout / "data/models/Jeff-1.1-Large")
    assert called == []
