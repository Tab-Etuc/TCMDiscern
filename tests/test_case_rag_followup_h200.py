"""CPU regressions for one-backbone S3/S1 full-Dev follow-up."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_case_rag_followup_h200 as followup


def _fake_original_preflight(tmp_path: Path):
    collection = {
        name: [{"id": "dev-000000"}] for name in ("baseline", "global2", *followup.CONTROLS)
    }
    stages = []
    for method in ("s2", "s1", "s3"):
        checkpoint = tmp_path / method
        checkpoint.mkdir(exist_ok=True)
        directory = (
            tmp_path / "data/case_rag" / followup.paper.TAG / "scores" / f"paper-{method}-full"
        )
        directory.mkdir(parents=True)
        (directory / "run_manifest.json").write_text(
            json.dumps({"source": followup.paper.TAG}), encoding="utf-8"
        )
        stages.append(
            {
                "method": method,
                "checkpoint": checkpoint,
                "identity": {"source": followup.paper.TAG},
                "pending": [],
            }
        )
    return collection, stages


def test_followup_plan_checks_only_missing_arms_and_never_loads_backbone(tmp_path, monkeypatch):
    collection, stages = _fake_original_preflight(tmp_path)
    monkeypatch.setattr(followup.paper, "preflight", lambda root, model_dir: (collection, stages))
    monkeypatch.setattr(
        followup.paper,
        "_saved_progress",
        lambda folder, arm, rows: len(rows) if (folder / f"{arm}.jsonl").is_file() else 0,
    )
    s3 = followup.paper._scores_path(tmp_path, "s3")
    (s3 / "global2_labelhide.jsonl").write_text("complete", encoding="utf-8")

    result = followup.run(
        tmp_path,
        tmp_path / "Jeff",
        check_only=True,
        session_factory=lambda path: pytest.fail("Preflight must not allocate GPU"),
    )
    assert result["status"] == "CASE_RAG_FOLLOWUP_PREFLIGHT_PASSED"
    assert result["planned_new_forwards"] == len(followup.CONTROLS) * 2 - 1
    assert result["new_backbone_loads"] == 0
    assert result["methods"] == ["s3", "s1"]


def test_one_shared_backbone_switches_s3_to_s1_once_and_resumes(tmp_path, monkeypatch):
    collection, stages = _fake_original_preflight(tmp_path)
    monkeypatch.setattr(followup.paper, "preflight", lambda root, model_dir: (collection, stages))
    monkeypatch.setattr(
        followup.paper,
        "_saved_progress",
        lambda folder, arm, rows: len(rows) if (folder / f"{arm}.jsonl").is_file() else 0,
    )
    actions = []

    class Session:
        def __init__(self, model):
            self.model_weight_loads = 0
            actions.append("session")

        def original(self):
            if self.model_weight_loads == 0:
                self.model_weight_loads = 1
                actions.append("load31b")
            else:
                actions.append("switch")
            return SimpleNamespace(decider=object())

    monkeypatch.setattr(
        followup,
        "restore_checkpoint",
        lambda decider, checkpoint: (
            actions.append("restore_" + checkpoint.name) or {"method": checkpoint.name}
        ),
    )

    import tcmdiscern.case_rag_runtime as runtime

    def fake_forward(rows, decider, path):
        path.write_text("complete", encoding="utf-8")
        actions.append(path.parent.name + "/" + path.stem)
        return {"source": "new_inference", "new_forwards": len(rows)}

    monkeypatch.setattr(runtime, "run_resumable", fake_forward)
    monkeypatch.setattr(
        followup.paper,
        "read_condition_logits",
        lambda path, rows: path.read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        followup,
        "evaluate",
        lambda method, *, root, strict: {
            "methods": {
                name: {
                    "top1_percent": 80,
                    "macro_f1_148_percent": 40,
                    "tail_37_macro_f1_percent": 20,
                }
                for name in followup.CONTROLS
            }
        },
    )
    result = followup.run(tmp_path, tmp_path / "Jeff", session_factory=Session)
    assert result["status"] == "CASE_RAG_FOLLOWUP_COMPLETED"
    assert result["backbone_loads"] == 1
    assert result["new_forwards_planned"] == 6
    assert actions.count("load31b") == 1
    assert actions.count("switch") == 1
    assert actions.index("restore_s3") < actions.index("restore_s1")
    assert len([a for a in actions if a.startswith("paper-")]) == 6

    actions.clear()
    second = followup.run(
        tmp_path,
        tmp_path / "Jeff",
        session_factory=lambda _: pytest.fail("Completed rerun should skip GPU"),
    )
    assert second["backbone_loads"] == 0
    assert second["new_forwards_planned"] == 0


def test_followup_rejects_missing_original_full_dev_before_weights(tmp_path, monkeypatch):
    collection, stages = _fake_original_preflight(tmp_path)
    stages[2]["pending"] = ["global2"]
    monkeypatch.setattr(followup.paper, "preflight", lambda root, model_dir: (collection, stages))
    with pytest.raises(ValueError, match="finish run_case_rag_paper_h200"):
        followup.run(
            tmp_path,
            tmp_path / "Jeff",
            session_factory=lambda _: pytest.fail("No weights before complete baseline"),
        )


def test_score_report_exposes_only_finished_followup_arms(tmp_path):
    from scripts.evaluate_case_rag_paper_dev import expected_output_arms

    assert expected_output_arms("s1", tmp_path) == ("global2",)
    assert expected_output_arms("s3", tmp_path) == ("global2",)
    assert expected_output_arms("s2", tmp_path) == (
        "global2",
        "random2",
        "global2_labelhide",
        "global2_labelonly",
    )
    (tmp_path / "global2_labelhide.jsonl").write_text("complete", encoding="utf-8")
    assert expected_output_arms("s3", tmp_path) == ("global2", "global2_labelhide")
    (tmp_path / "random2.jsonl").write_text("complete", encoding="utf-8")
    (tmp_path / "global2_labelonly.jsonl").write_text("complete", encoding="utf-8")
    assert expected_output_arms("s1", tmp_path) == (
        "global2",
        "random2",
        "global2_labelhide",
        "global2_labelonly",
    )


def test_followup_rejects_tampered_run_identity(tmp_path, monkeypatch):
    collection, stages = _fake_original_preflight(tmp_path)
    monkeypatch.setattr(followup.paper, "preflight", lambda root, model_dir: (collection, stages))
    s3 = followup.paper._scores_path(tmp_path, "s3")
    (s3 / "run_manifest.json").write_text(
        json.dumps({"source": "different-samples"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed source"):
        followup.run(
            tmp_path,
            tmp_path / "Jeff",
            session_factory=lambda _: pytest.fail("Manifest tampering before weights"),
        )
