"""CPU regression tests for the locked one-backbone paper experiment."""

from types import SimpleNamespace

import pytest

from scripts import run_case_rag_paper_h200 as paper
from scripts.prepare_case_rag_paper_ablation import ARMS, transform


def test_label_and_clinical_evidence_ablations_are_independent():
    original = {
        "id": "dev-000000",
        "state": {
            "chief_complaint": "患者主訴",
            "retrieved_cases": [
                {
                    "candidate_syndrome": "證A",
                    "chief_complaint": "歷史A",
                    "description": "醫案A",
                    "detection": "四診A",
                },
                {
                    "candidate_syndrome": "證B",
                    "chief_complaint": "歷史B",
                    "description": "醫案B",
                    "detection": "四診B",
                },
            ],
        },
    }
    hidden = transform(original, ARMS[0])
    label_only = transform(original, ARMS[1])
    assert hidden["state"]["retrieved_cases"][0]["candidate_syndrome"] == "未提供"
    assert hidden["state"]["retrieved_cases"][0]["chief_complaint"] == "歷史A"
    assert label_only["state"]["retrieved_cases"][0]["candidate_syndrome"] == "證A"
    assert label_only["state"]["retrieved_cases"][0]["chief_complaint"] == ""
    assert label_only["state"]["chief_complaint"] == "患者主訴"
    assert original["state"]["retrieved_cases"][0]["candidate_syndrome"] == "證A"


def test_one_full_dev_gpu_session_with_six_arms_and_zero_baseline_forwards(
    tmp_path,
    monkeypatch,
):
    paths = {m: tmp_path / f"checkpoint/{m}" for m in paper.METHODS}
    for path in paths.values():
        path.mkdir(parents=True)
    stage = [
        {
            "method": method,
            "checkpoint": paths[method],
            "identity": {"source": paper.TAG},
            "pending": list(paper.GPU_ARMS[method]),
        }
        for method in paper.METHODS
    ]
    monkeypatch.setattr(
        paper,
        "preflight",
        lambda root, model_dir: ({arm: [1] for a in paper.GPU_ARMS.values() for arm in a}, stage),
    )
    calls = []

    class Session:
        model_weight_loads = 0

        def __init__(self, model_dir):
            calls.append("construct_session")

        def original(self):
            if not self.model_weight_loads:
                self.model_weight_loads = 1
                calls.append("load_31b")
            calls.append("switch_lora")
            return SimpleNamespace(decider=object())

    def fake_restore(decider, checkpoint):
        calls.append("restore_" + checkpoint.name)
        return {"method": checkpoint.name}

    monkeypatch.setattr(paper, "restore_checkpoint", fake_restore)
    import tcmdiscern.case_rag_runtime as runtime

    def fake_forward(rows, decider, output):
        calls.append("forward_" + output.stem)
        return {"source": "new_inference", "new_forwards": 5486}

    monkeypatch.setattr(runtime, "run_resumable", fake_forward)
    monkeypatch.setattr(
        paper,
        "evaluate",
        lambda method, *, root, strict: {"status": "CASE_RAG_PAPER_FULL_DEV_COMPLETED"},
    )
    report = paper.run(tmp_path, tmp_path / "model", session_factory=Session)
    assert report["backbone_loads"] == 1
    assert calls.count("load_31b") == 1
    assert calls.count("switch_lora") == 3
    assert len([name for name in calls if name.startswith("forward_")]) == 6
    assert not any(name == "forward_baseline" for name in calls)


def test_check_never_constructs_large_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        paper,
        "preflight",
        lambda *args: ({"global2": [1]}, [{"method": "s2", "pending": ["global2"]}]),
    )
    report = paper.run(
        tmp_path,
        tmp_path / "model",
        check_only=True,
        session_factory=lambda path: pytest.fail("Must not construct GPU model"),
    )
    assert report["gpu_weights_loaded"] == 0
    assert report["planned_new_forwards"] == 5486


def test_preflight_error_blocks_model_load(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("Checkpoint attention backend differs")

    monkeypatch.setattr(paper, "preflight", fail)
    with pytest.raises(ValueError, match="attention backend"):
        paper.run(
            tmp_path,
            tmp_path / "model",
            session_factory=lambda path: pytest.fail("31B loaded before preflight"),
        )


def test_real_preflight_checks_all_checkpoints_and_kernel_before_gpu(tmp_path, monkeypatch):
    import json

    import tcmdiscern.jeff_scoring as scoring

    calls = []
    published = tmp_path / "experiments"
    published.mkdir()
    raw = tmp_path / "data/jeff"
    raw.mkdir(parents=True)
    checkpoints = {}
    for method in paper.METHODS:
        path = tmp_path / method
        path.mkdir()
        (path / "manifest.json").write_text(
            json.dumps({"run_id": f"{method}-seed20261112-e2"}), encoding="utf-8"
        )
        checkpoints[method] = path
        (raw / f"dev_{method}-seed20261112-e2-epoch2_k16_5486.jsonl").write_text(
            "[]\n", encoding="utf-8"
        )
        (published / f"jeff_{method}-seed20261112-e2-epoch2_k16_dev.json").write_text(
            json.dumps(
                {
                    "rows": 5486,
                    "checkpoint": {
                        "method": method,
                        "source_run_id": f"{method}-seed20261112-e2",
                    },
                    "attention_backend": "axolotl_hybrid_fa2_sdpa",
                    "scores": {"top1_correct_n": 4408},
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        paper,
        "_load_inputs",
        lambda root: {arm: [{"id": "x"}] for arm in ("baseline", *paper.GPU_ARMS["s2"])},
    )
    monkeypatch.setattr(paper, "_verify_frozen_prompts", lambda *args: {"cases": 5486})
    monkeypatch.setattr(
        paper, "choose_checkpoint", lambda root, method, explicit: checkpoints[method]
    )
    monkeypatch.setattr(
        paper.batch,
        "preflight_checkpoint_contract",
        lambda model, checkpoint, method, **kwargs: (
            calls.append((method, kwargs))
            or {"checkpoint_attention_backend": "axolotl_hybrid_fa2_sdpa"}
        ),
    )
    monkeypatch.setattr(paper, "expected_run_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(scoring, "read_model_logits", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        paper,
        "evaluate",
        lambda method, *, root, strict: {"cases": 5486, "methods": {"baseline": {"correct": 4408}}},
    )
    _, selected = paper.preflight(tmp_path, tmp_path / "model")
    assert [item["method"] for item in selected] == ["s2", "s1", "s3"]
    assert [item["pending"] for item in selected] == [
        list(paper.GPU_ARMS[m]) for m in paper.METHODS
    ]
    assert [name for name, _ in calls] == ["s2", "s1", "s3"]
    assert [kwargs["check_kernel"] for _, kwargs in calls] == [True, False, False]
