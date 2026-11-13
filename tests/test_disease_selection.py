"""Dev-only model selection must use complete three-seed reports and no Test."""
from __future__ import annotations

import json

import pytest

from scripts.select_disease_study import METHODS, load_dev_scores, run_id
from tcmdiscern.h200.plan import SEEDS


def test_select_disease_study_from_three_complete_seeds(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for offset, method in enumerate(METHODS):
        mode = "conditioned" if method == "conditioned_hard" else method
        for i, seed in enumerate(SEEDS):
            name = run_id(method, seed)
            report = {
                "status": "COMPLETE", "split": "dev", "mode": mode,
                "run_id": name,
                "scores": {"n": 5486, "top1": 0.5 + offset * 0.02 + i * 0.001,
                           "macro_f1": 0.3 + offset * 0.02 + i * 0.001},
            }
            (folder / f"disease_study_dev_{mode}_{name}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
    scores = load_dev_scores(tmp_path)
    assert scores["overall_best_method"] == "pair"
    assert scores["overall_best_run"] == run_id("pair", SEEDS[-1])
    assert scores["test_evaluated"] is False


def test_selection_refuses_incomplete_train_evaluation(tmp_path):
    (tmp_path / "experiments").mkdir()
    with pytest.raises(FileNotFoundError):
        load_dev_scores(tmp_path)
