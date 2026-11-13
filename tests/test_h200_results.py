"""End-to-end aggregate-only H200 selection tests, with synthetic Dev reports."""

from __future__ import annotations

import json

import pytest

from tcmdiscern.h200.plan import DEV_ROWS, SEEDS
from tcmdiscern.h200.results import save_selection, summarize


def test_selection_uses_paired_seed_mean_not_luckiest_seed(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    for method in ("s1", "s2", "s3"):
        for epoch in (1, 2):
            for index, seed in enumerate(SEEDS):
                score = (
                    (0.80 + 0.01 * index)
                    if method == "s1"
                    else (0.81 + 0.01 * index)
                    if epoch == 1
                    else (0.83 + 0.004 * index)
                )
                # Accuracy must have an exact integer numerator to pass validation.
                correct = round(score * DEV_ROWS)
                score = correct / DEV_ROWS
                report = {
                    "kind": "FULL_DEV_FINETUNED",
                    "rows": DEV_ROWS,
                    "trained_lora": True,
                    "checkpoint": {"method": method},
                    "scores": {
                        "n_cases": DEV_ROWS,
                        "label_classes": 148,
                        "top1_correct_n": correct,
                        "top1_accuracy": score,
                        "macro_f1": score - 0.17,
                    },
                }
                name = f"jeff_{method}-seed{seed}-e2-epoch{epoch}_k16_dev.json"
                (experiments / name).write_text(json.dumps(report), encoding="utf-8")
    (experiments / "jeff_zero_shot_k16_dev.json").write_text(
        json.dumps(
            {
                "kind": "FULL_DEV_ZERO_SHOT",
                "rows": DEV_ROWS,
                "scores": {"n_cases": DEV_ROWS, "top1_accuracy": 0.70},
            }
        ),
        encoding="utf-8",
    )
    report = summarize(tmp_path)
    assert report["selected"]["method"] == "s3"
    assert report["selected"]["epoch"] == 2
    assert report["selected"]["checkpoint"].endswith("s3-seed20261112-e2/resume-0001350")
    assert report["test_evaluated"] is False
    assert len(report["primary_trials"]) == 6
    path = save_selection(tmp_path)
    with pytest.raises(FileExistsError):
        save_selection(tmp_path)
    assert json.loads(path.read_text())["selected"] == report["selected"]


def test_missing_or_wrong_checkpoint_report_blocks_selection(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)
    for method in ("s1", "s2", "s3"):
        for epoch in (1, 2):
            for seed in SEEDS:
                name = f"jeff_{method}-seed{seed}-e2-epoch{epoch}_k16_dev.json"
                (experiments / name).write_text(
                    json.dumps(
                        {
                            "kind": "SMOKE_NOT_A_BENCHMARK",
                            "rows": DEV_ROWS,
                            "trained_lora": True,
                            "checkpoint": {"method": method},
                            "scores": {
                                "n_cases": DEV_ROWS,
                                "label_classes": 148,
                                "top1_accuracy": 0.0,
                                "top1_correct_n": 0,
                                "macro_f1": 0.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
    with pytest.raises(ValueError, match="not produced by"):
        summarize(tmp_path)
