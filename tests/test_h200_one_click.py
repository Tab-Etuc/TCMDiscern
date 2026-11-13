"""H200 one-click orchestration regression tests; CPU and network-free."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import fetch_tcmsd
from scripts.h200 import main as h200_main
from tcmdiscern.h200.artifacts import publish_lines
from tcmdiscern.h200.plan import (
    ACCUMULATION,
    TRAIN_ROWS,
    UPDATES_PER_EPOCH,
    Experiment,
    registered_exploratory,
    registered_main,
)
from tcmdiscern.h200.workflow import (
    H200Runner,
    Stage,
    canonical_command,
)


def test_frozen_h200_training_budget_and_unique_runs():
    assert TRAIN_ROWS == 43180
    assert ACCUMULATION == 64
    assert UPDATES_PER_EPOCH == 675
    main = registered_main()
    exploratory = registered_exploratory()
    assert len(main) == 9 and all(x.updates == 1350 for x in main)
    assert len(exploratory) == 2 and all(x.updates == 675 for x in exploratory)
    assert len({x.run_id for x in (*main, *exploratory)}) == 11
    assert all("--weighting" in x.train_arguments(model_dir="/models/Jeff") for x in main)
    assert all(x.weighting == "uniform" for x in main)
    with pytest.raises(ValueError, match="Only S3"):
        Experiment("s1", 20261112, 1, "recoverable_oof_x2")


def test_no_payload_overwrite_and_atomic_result(tmp_path):
    dest = tmp_path / "private" / "candidate.jsonl"
    publish_lines(dest, iter(({"case_id": "train-000000"}, {"case_id": "train-000001"})))
    original = dest.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        publish_lines(dest, [{"case_id": "different"}])
    assert dest.read_text(encoding="utf-8") == original
    bad = tmp_path / "private" / "bad.jsonl"
    with pytest.raises(TypeError):
        publish_lines(bad, [{"ok": True}, {"not_json": {1}}])
    assert not bad.exists()
    assert not list(dest.parent.glob(".preparing-*"))


def test_stage_is_idempotent_and_output_checked(tmp_path):
    expected = tmp_path / "done.json"
    calls = []

    def invoke(cmd, _cwd, _log):
        calls.append(list(cmd))
        expected.write_text(json.dumps({"n": 2}), encoding="utf-8")

    runner = H200Runner(root=tmp_path, python="/python", invoke=invoke)
    stage = Stage(
        name="demo",
        command=("-m", "demo"),
        expected=expected,
        verify=lambda: json.loads(expected.read_text())["n"] == 2,
    )
    runner.stage(stage)
    runner.stage(stage)
    assert len(calls) == 1
    assert runner.status()[0]["status"] == "complete"
    expected.unlink()
    with pytest.raises(FileNotFoundError, match="output is missing"):
        runner.stage(stage)


def test_interrupted_stage_resumes_without_clearing_settings(tmp_path):
    output = tmp_path / "result.json"
    calls = []

    def invoke(cmd, _cwd, _log):
        calls.append(list(cmd))
        if len(calls) == 1:
            raise RuntimeError("simulated interruption")
        output.write_text('{"complete":true}', encoding="utf-8")

    runner = H200Runner(root=tmp_path, python="/python", invoke=invoke)
    original = Stage(
        "resume-test",
        ("-m", "run", "--steps", "1350"),
        output,
        verify=lambda: json.loads(output.read_text())["complete"],
    )
    with pytest.raises(RuntimeError, match="interruption"):
        runner.stage(original)
    resumed = Stage(
        "resume-test",
        ("-m", "run", "--steps", "1350", "--resume", "/checkpoint"),
        output,
        verify=lambda: json.loads(output.read_text())["complete"],
    )
    runner.stage(resumed)
    assert len(calls) == 2 and calls[1][-2:] == ["--resume", "/checkpoint"]
    assert canonical_command(resumed.command) == list(original.command)
    assert runner.status()[0]["status"] == "complete"
    with pytest.raises(ValueError, match="different command"):
        runner.stage(Stage("resume-test", ("-m", "run", "--steps", "999"), output))


def test_dry_run_never_creates_directories_or_executes_gpu(tmp_path, capsys):
    def forbidden(*_args):
        raise AssertionError("Dry-run must not call subprocess")

    runner = H200Runner(root=tmp_path, dry_run=True, invoke=forbidden)
    runner.full(include_exploratory=True)
    output = capsys.readouterr().out
    assert "train-s1-seed20261112-e2" in output
    assert "train-s2-seed20261112-e2" in output
    assert "train-s3-seed20261114-e2" in output
    assert "train-s3-seed20261112-e1-hard2" in output
    assert not (tmp_path / "data").exists()


def test_full_cli_requires_explicit_confirmation():
    with pytest.raises(SystemExit) as exc:
        h200_main(["full"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        h200_main(["screen"])
    assert exc.value.code == 2


def test_fresh_clone_download_does_not_open_test(tmp_path, monkeypatch):
    downloaded = []

    def fake_download(url, path):
        downloaded.append((url, path.name))

    def fake_unpack(_archive, _destination):
        pass

    monkeypatch.setattr(fetch_tcmsd, "download", fake_download)
    monkeypatch.setattr(fetch_tcmsd, "unpack_selected", fake_unpack)
    monkeypatch.setattr(sys, "argv", ["fetch_tcmsd.py", "--data-root", str(tmp_path / "data")])
    fetch_tcmsd.main()
    assert all("test.json" not in x[0] for x in downloaded)
    assert len(downloaded) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_tcmsd.py",
            "--data-root",
            str(tmp_path / "data"),
            "--include-test",
        ],
    )
    fetch_tcmsd.main()
    assert any("test.json" in x[0] for x in downloaded)
