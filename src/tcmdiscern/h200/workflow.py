"""Resumable, fail-closed H200 experiment orchestrator.

A stage runs exactly once and gets a durable completion marker. A process
interruption can be resumed from complete optimizer checkpoints; neither
existing model results nor patient data are silently overwritten.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .plan import (
    ACCUMULATION,
    DEV_ROWS,
    SEEDS,
    UPDATES_PER_EPOCH,
    Experiment,
    registered_exploratory,
    registered_main,
)

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "data/h200/state"
LOG_DIR = ROOT / "data/h200/logs"


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def save_json(path: Path, payload: dict) -> None:
    """Atomic stage metadata; does not write patient records or raw logits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".progress-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_optimizer_checkpoint(folder: Path) -> bool:
    """Every resumable boundary must contain both weights and training state."""
    if not folder.is_dir():
        return False
    required = (
        "training_state.json",
        "manifest.json",
        "probe_logits.json",
        "head.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
    )
    return (
        all((folder / name).is_file() for name in required)
        and (folder / "adapter/adapter_model.safetensors").is_file()
    )


def canonical_command(command: Sequence[str]) -> list[str]:
    """A resumed command must keep exactly the original frozen run settings."""
    values = list(command)
    if "--resume" in values:
        index = values.index("--resume")
        if index != len(values) - 2:
            raise ValueError("Resume argument must appear only at the end")
        del values[index:]
    return values


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    expected: Path | None = None
    verify: Callable[[], bool] | None = None


class H200Runner:
    def __init__(
        self,
        *,
        root: Path = ROOT,
        python: str = sys.executable,
        model_dir: Path | None = None,
        dry_run: bool = False,
        invoke: Callable[[Sequence[str], Path, Path], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.python = python
        self.model_dir = (model_dir or self.root / "data/models/Jeff-1.1-Large").resolve()
        self.dry_run = dry_run
        self.state_dir = self.root / "data/h200/state"
        self.log_dir = self.root / "data/h200/logs"
        self.invoke = invoke or self._invoke

    @staticmethod
    def _invoke(command: Sequence[str], cwd: Path, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n===== {now_utc()} =====\n")
            stream.write(" ".join(command) + "\n")
            stream.flush()
            with subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)

    def stage(self, stage: Stage) -> None:
        marker = self.state_dir / f"{stage.name}.json"
        cmd = (self.python, *stage.command)
        identity = canonical_command(stage.command)
        if self.dry_run:
            print(f"[DRY-RUN] {stage.name}: {' '.join(cmd)}")
            return
        completed = False
        if marker.exists():
            state = read_json(marker)
            if canonical_command(state.get("command", [])) != identity:
                raise ValueError(f"{stage.name}: prior stage uses different command/configuration")
            completed = state.get("status") == "complete"
        if completed:
            if stage.expected is not None and not stage.expected.exists():
                raise FileNotFoundError(f"{stage.name}: completed marker but output is missing")
            if stage.verify is not None and not stage.verify():
                raise ValueError(f"{stage.name}: completed marker but output is invalid")
            print(f"[SKIP] {stage.name}: verified completed")
            return
        if stage.expected is not None and stage.expected.exists():
            if stage.verify is None or not stage.verify():
                raise RuntimeError(f"{stage.name}: interrupted output requires manual inspection")
            save_json(
                marker,
                {
                    "status": "complete",
                    "command": identity,
                    "recovered_at": now_utc(),
                    "verified": True,
                },
            )
            print(f"[RECOVER] {stage.name}: verified finished output")
            return
        save_json(marker, {"status": "running", "command": identity, "started_at": now_utc()})
        print(f"[RUN] {stage.name}", flush=True)
        try:
            self.invoke(cmd, self.root, self.log_dir / f"{stage.name}.log")
            if stage.expected is not None and not stage.expected.exists():
                raise FileNotFoundError(
                    f"{stage.name}: successful process created no expected output"
                )
            if stage.verify is not None and not stage.verify():
                raise RuntimeError(f"{stage.name}: generated output failed validation")
        except BaseException as error:
            save_json(
                marker,
                {
                    "status": "interrupted",
                    "command": identity,
                    "failed_at": now_utc(),
                    "error_type": type(error).__name__,
                },
            )
            raise
        save_json(
            marker,
            {
                "status": "complete",
                "command": identity,
                "completed_at": now_utc(),
                "verified": True,
            },
        )

    def prepare(self) -> dict:
        if self.dry_run:
            print(
                "[DRY-RUN] fetch official Train/Dev only; build grouped OOF, BM25 Dev, S1/S2/S3/S4 prompts"
            )
            print("[DRY-RUN] independently revalidate every official Train/Dev and training prompt")
            return {"status": "DRY_RUN"}
        from .artifacts import prepare

        report = prepare(self.root, on_stage=lambda line: print(f"[DATA] {line}", flush=True))
        save_json(
            self.state_dir / "prepare-data.json",
            {
                "status": "complete",
                "command": ["-m", "scripts.h200", "prepare"],
                "completed_at": now_utc(),
                "train": report["train"],
                "dev": report["dev"],
                "verified": True,
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report

    def require_gpu(self) -> None:
        """Verify published checkpoint and all pinned H200 packages before loading 31B."""
        from scripts.preflight_jeff_training import preflight

        report = preflight(self.model_dir)
        if report["blockers"]:
            raise RuntimeError("H200 preflight BLOCKED: " + "; ".join(report["blockers"]))
        print("[GPU] Pinned checkpoint and CUDA environment preflight passed")

    def _valid_smoke_two(self, file: Path | None = None) -> bool:
        file = file or (self.root / "data/jeff/dev_zero_shot_k16_smoke0002.jsonl")
        if not file.is_file():
            return False
        try:
            with file.open(encoding="utf-8") as stream:
                values = [json.loads(line) for line in stream]
            return len(values) == 2 and all(
                row.get("case_id") == f"dev-{i:06d}"
                and isinstance(row.get("logits"), list)
                and len(row["logits"]) == 16
                and all(
                    type(value) in (int, float) and math.isfinite(value) for value in row["logits"]
                )
                for i, row in enumerate(values)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _valid_pilot_training(self) -> bool:
        file = self.root / "data/runs/pilot-s3-16updates/summary.json"
        if not file.is_file():
            return False
        result = read_json(file)
        return (
            result.get("loss_method") == "s3"
            and result.get("global_update") == 16
            and result.get("accumulation") == ACCUMULATION
            and result.get("source_count_verified") == 43180
            and complete_optimizer_checkpoint(
                self.root / "data/runs/pilot-s3-16updates/resume-0000016"
            )
        )

    def _complete_inference(self, report_path: Path, *, expected_kind: str) -> bool:
        if not report_path.is_file():
            return False
        result = read_json(report_path)
        if (
            result.get("kind") != expected_kind
            or result.get("rows") != DEV_ROWS
            or result.get("scores", {}).get("n_cases") != DEV_ROWS
        ):
            return False
        raw_path = result.get("raw_logit_file_ignored")
        if not isinstance(raw_path, str):
            return False
        private_file = (self.root / raw_path).resolve()
        if not private_file.is_relative_to(self.root / "data"):
            return False
        from tcmdiscern.jeff_scoring import read_model_logits

        try:
            read_model_logits(private_file, num_cases=DEV_ROWS)
        except (OSError, ValueError, TypeError):
            return False
        return True

    def _complete_training(self, experiment: Experiment) -> bool:
        file = self.root / "data/runs" / experiment.run_id / "summary.json"
        if not file.is_file():
            return False
        result = read_json(file)
        expected = {
            "status": "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
            "run_id": experiment.run_id,
            "loss_method": experiment.method,
            "seed": experiment.seed,
            "epochs": experiment.epochs,
            "target_updates": experiment.updates,
            "global_update": experiment.updates,
            "accumulation": ACCUMULATION,
            "example_weighting": experiment.weighting,
        }
        if any(result.get(k) != v for k, v in expected.items()):
            return False
        return result.get("source_count_verified") == 43180 and complete_optimizer_checkpoint(
            self.root / f"data/runs/{experiment.run_id}/resume-{experiment.updates:07d}"
        )

    def _resume_command(
        self,
        command: Sequence[str],
        *,
        run_id: str,
        method: str,
        seed: int,
        updates: int,
        weighting: str = "uniform",
    ) -> tuple[str, ...]:
        """Resume a registered run only from a complete matching optimizer checkpoint."""
        args = list(command)
        run_folder = self.root / "data/runs" / run_id
        if (run_folder / "summary.json").exists() or not run_folder.exists():
            return tuple(args)
        checkpoints = sorted(run_folder.glob("resume-*"))
        if not checkpoints:
            raise RuntimeError(f"Interrupted run has no complete resume checkpoint: {run_folder}")
        latest = checkpoints[-1]
        if not complete_optimizer_checkpoint(latest):
            raise RuntimeError(f"Incomplete optimizer checkpoint: {latest}")
        state = read_json(latest / "training_state.json")
        required = {
            "run_id": run_id,
            "loss_method": method,
            "target_updates": updates,
            "seed": seed,
            "example_weighting": weighting,
        }
        if any(state.get(key) != value for key, value in required.items()):
            raise ValueError("Saved checkpoint disagrees with the registered run")
        if state.get("global_update", 0) >= updates:
            raise RuntimeError(
                "Final checkpoint exists but summary is missing; inspect before restart"
            )
        print(f"[RESUME] {run_id} from {latest.name}")
        args.extend(["--resume", str(latest)])
        return tuple(args)

    def _training_command(self, experiment: Experiment) -> tuple[str, ...]:
        return self._resume_command(
            experiment.train_arguments(model_dir=str(self.model_dir)),
            run_id=experiment.run_id,
            method=experiment.method,
            seed=experiment.seed,
            updates=experiment.updates,
            weighting=experiment.weighting,
        )

    def train(self, experiment: Experiment) -> None:
        cmd = self._training_command(experiment)
        self.stage(
            Stage(
                name=f"train-{experiment.run_id}",
                command=cmd,
                expected=self.root / "data/runs" / experiment.run_id / "summary.json",
                verify=lambda: self._complete_training(experiment),
            )
        )

    def evaluate(self, experiment: Experiment, epoch: int) -> None:
        if epoch not in range(1, experiment.epochs + 1):
            raise ValueError("Checkpoint epoch is outside the registered training budget")
        step = UPDATES_PER_EPOCH * epoch
        resume = self.root / f"data/runs/{experiment.run_id}/resume-{step:07d}"
        if not self.dry_run and not resume.is_dir():
            raise FileNotFoundError(f"Missing epoch checkpoint: {resume}")
        eval_id = f"{experiment.run_id}-epoch{epoch}"
        report = self.root / "experiments" / f"jeff_{eval_id}_k16_dev.json"
        cmd = (
            "-m",
            "scripts.run_jeff_zeroshot",
            "--model-dir",
            str(self.model_dir),
            "--checkpoint",
            str(resume),
            "--run-id",
            eval_id,
        )
        self.stage(
            Stage(
                name=f"eval-{eval_id}",
                command=cmd,
                expected=report,
                verify=lambda: self._complete_inference(report, expected_kind="FULL_DEV_FINETUNED"),
            )
        )

    def pilot(self) -> None:
        self.prepare()
        if not self.dry_run:
            self.require_gpu()
        self.stage(
            Stage(
                name="pilot-zeroshot-2",
                command=(
                    "-m",
                    "scripts.run_jeff_zeroshot",
                    "--model-dir",
                    str(self.model_dir),
                    "--limit",
                    "2",
                ),
                expected=self.root / "data/jeff/dev_zero_shot_k16_smoke0002.jsonl",
                verify=lambda: self._valid_smoke_two(),
            )
        )
        self.stage(
            Stage(
                name="pilot-step0-s3",
                command=(
                    "-m",
                    "scripts.train_jeff_lora",
                    "--model-dir",
                    str(self.model_dir),
                    "--method",
                    "s3",
                ),
            )
        )
        pilot_run_id = "pilot-s3-16updates"
        pilot_command = self._resume_command(
            (
                "-m",
                "scripts.run_jeff_finetune",
                "--model-dir",
                str(self.model_dir),
                "--method",
                "s3",
                "--run-id",
                pilot_run_id,
                "--steps",
                "16",
                "--epochs",
                "1",
                "--accumulation",
                str(ACCUMULATION),
                "--seed",
                str(SEEDS[0]),
                "--warmup-steps",
                "0",
                "--checkpoint-every",
                "4",
            ),
            run_id=pilot_run_id,
            method="s3",
            seed=SEEDS[0],
            updates=16,
        )
        self.stage(
            Stage(
                name="pilot-16updates-s3",
                command=pilot_command,
                expected=self.root / "data/runs/pilot-s3-16updates/summary.json",
                verify=self._valid_pilot_training,
            )
        )
        checkpoint = self.root / "data/runs/pilot-s3-16updates/resume-0000016"
        pilot_result = self.root / "data/jeff/dev_pilot-s3-16updates-smoke_k16_0002.jsonl"
        self.stage(
            Stage(
                name="pilot-checkpoint-reload-2",
                command=(
                    "-m",
                    "scripts.run_jeff_zeroshot",
                    "--model-dir",
                    str(self.model_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--run-id",
                    "pilot-s3-16updates-smoke",
                    "--limit",
                    "2",
                ),
                expected=pilot_result,
                verify=lambda: self._valid_smoke_two(pilot_result),
            )
        )

    def screen(self) -> None:
        """Three permanent first-seed runs before committing to multi-seed H200 cost.

        Every screen checkpoint belongs to the final registered two-epoch run.
        'full' safely reuses all completed stages, without wasted pilot epochs.
        """
        self.pilot()
        report = self.root / "experiments/jeff_zero_shot_k16_dev.json"
        self.stage(
            Stage(
                name="zeroshot-k16-full",
                command=("-m", "scripts.run_jeff_zeroshot", "--model-dir", str(self.model_dir)),
                expected=report,
                verify=lambda: self._complete_inference(report, expected_kind="FULL_DEV_ZERO_SHOT"),
            )
        )
        for method in ("s1", "s2", "s3"):
            experiment = Experiment(method, SEEDS[0], epochs=2)
            self.train(experiment)
            for epoch in (1, 2):
                self.evaluate(experiment, epoch)

    def full(self, *, include_exploratory: bool = False) -> None:
        self.screen()
        planned = list(registered_main())
        if include_exploratory:
            planned.extend(registered_exploratory())
        for experiment in planned:
            self.train(experiment)
            for epoch in range(1, experiment.epochs + 1):
                self.evaluate(experiment, epoch)

        selection = self.root / "experiments/jeff_h200_dev_selection.json"
        command = ("-m", "scripts.summarize_h200")
        if include_exploratory:
            command = (*command, "--include-exploratory")
        self.stage(
            Stage(
                name="select-best-dev-model",
                command=command,
                expected=selection,
                verify=lambda: (
                    selection.is_file()
                    and read_json(selection).get("status")
                    == "H200_FULL_DEV_RESULTS_SELECTED_NO_TEST_EVALUATION"
                    and read_json(selection).get("test_evaluated") is False
                ),
            )
        )

    def status(self) -> list[dict]:
        rows = []
        if self.state_dir.exists():
            for path in sorted(self.state_dir.glob("*.json")):
                record = read_json(path)
                status = record.get("status")
                if path.stem == "prepare-data" and not all(
                    (self.root / relative).is_file()
                    for relative in (
                        "data/tcm-sd/train.json",
                        "data/tcm-sd/dev.json",
                        "data/candidates/bm25_train_text_grouped.jsonl",
                        "data/candidates/bm25_dev.jsonl",
                        "data/jeff/train_oof_k16_epoch02.jsonl",
                        "data/jeff/train_oof_multipos_k16_epoch03.jsonl",
                    )
                ):
                    status = "missing_inputs"
                rows.append(
                    {
                        "stage": path.stem,
                        "status": status,
                        "at": record.get("completed_at")
                        or record.get("recovered_at")
                        or record.get("failed_at"),
                    }
                )
        return rows
