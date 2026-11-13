"""Reproducible H200 orchestration for disease and disease-syndrome pair K=16 tasks.

Uses the released 31B Jeff checkpoint and existing LoRA+Head trainer. The
original syndrome-148 tasks remain under run_h200.sh. All results for these
additional two tasks are evaluated with complete original Dev rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_joint16 import prepare
from scripts.select_joint16 import MODES, run_id
from tcmdiscern.h200.plan import (
    ACCUMULATION,
    SEEDS,
    UPDATES_PER_EPOCH,
)
from tcmdiscern.h200.workflow import H200Runner, Stage

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 2


def checkpoint(mode: str, seed: int, epoch: int) -> Path:
    if mode not in MODES or seed not in SEEDS or epoch not in (1, 2):
        raise ValueError("Invalid registered checkpoint")
    return ROOT / "data/runs" / run_id(mode, seed) / f"resume-{epoch * UPDATES_PER_EPOCH:07d}"


def evaluate_args(
    mode: str, model_dir: Path, *, name: str, split: str = "dev",
    selected_checkpoint: Path | None = None, limit: int | None = None,
) -> tuple[str, ...]:
    command = [
        "-m", "scripts.evaluate_joint16",
        "--mode", mode, "--split", split,
        "--model-dir", str(model_dir),
        "--run-id", name,
    ]
    if selected_checkpoint is not None:
        command.extend(["--checkpoint", str(selected_checkpoint)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return tuple(command)


def train_args(
    mode: str, seed: int, model_dir: Path, *, dry_pilot: bool = False,
) -> tuple[str, ...]:
    updates = 1 if dry_pilot else EPOCHS * UPDATES_PER_EPOCH
    epochs = 1 if dry_pilot else EPOCHS
    return (
        "-m", "scripts.run_jeff_finetune",
        "--model-dir", str(model_dir),
        "--study-mode", f"{mode}16",
        "--method", "s1",
        "--run-id", f"joint16-pilot-{mode}" if dry_pilot else run_id(mode, seed),
        "--steps", str(updates),
        "--epochs", str(epochs),
        "--accumulation", "1" if dry_pilot else str(ACCUMULATION),
        "--seed", str(seed),
        "--rank", "32", "--alpha", "64",
        "--lora-lr", "2e-5", "--head-lr", "1e-4",
        "--weight-decay", "0.01", "--clip-norm", "1",
        "--warmup-steps", "0" if dry_pilot else "40",
        "--checkpoint-every", "1" if dry_pilot else "300",
    )


def registered_runs():
    return [(mode, seed) for mode in MODES for seed in SEEDS]


def run_study(
    mode: str, model_dir: Path, *, dry_run: bool = False,
    confirm: bool = False,
) -> None:
    if mode == "plan":
        print(json.dumps({
            "tasks": ["disease", "pair"],
            "classes_from_official_train": {"disease": 403, "pair": 1782},
            "official_rows": {"train": 43180, "dev": 5486, "test": 5486},
            "candidate_k": 16, "optimizer_updates_per_epoch": UPDATES_PER_EPOCH,
            "seeds": list(SEEDS), "epochs_per_run": EPOCHS,
            "main_trainings": [
                {"mode": task, "seed": seed, "run_id": run_id(task, seed)}
                for task, seed in registered_runs()
            ],
            "test_requires_frozen_dev_model_selection": True,
        }, ensure_ascii=False, indent=2))
        return

    runner = H200Runner(root=ROOT, model_dir=model_dir, dry_run=dry_run)
    if mode == "status":
        print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
        return
    if mode == "prepare":
        if dry_run:
            print("[DRY-RUN] 5-fold Train-only K16 disease and pair candidates, full Dev")
            return
        runner.prepare()
        print(json.dumps(prepare(split="dev"), ensure_ascii=False, indent=2))
        return

    if mode in ("pilot", "screen", "full"):
        if mode in ("screen", "full") and not (dry_run or confirm):
            raise ValueError(f"{mode} requires --confirm")
        if not dry_run:
            runner.prepare()
            prepare(split="dev")
            runner.require_gpu()

        for task in MODES:
            path = ROOT / "experiments" / f"joint16_dev_{task}_pilot.json"
            runner.stage(Stage(
                "joint16-pilot-zero-" + task,
                evaluate_args(task, model_dir, name="pilot", limit=2),
                expected=path,
            ))
            name = f"joint16-pilot-{task}"
            runner.stage(Stage(
                "joint16-pilot-train-" + task,
                train_args(task, SEEDS[0], model_dir, dry_pilot=True),
                expected=ROOT / "data/runs" / name / "summary.json",
            ))
        if mode == "pilot":
            return
        for task in MODES:
            runner.stage(Stage(
                f"joint16-zero-{task}",
                evaluate_args(task, model_dir, name="zero"),
                expected=ROOT / "experiments" / f"joint16_dev_{task}_zero.json",
            ))
        methods = [(task, SEEDS[0]) for task in MODES]
        if mode == "full":
            methods = registered_runs()

        for task, seed in methods:
            name = run_id(task, seed)
            command = train_args(task, seed, model_dir)
            if not dry_run:
                command = runner._resume_command(
                    command, run_id=name, method="s1",
                    seed=seed, updates=EPOCHS * UPDATES_PER_EPOCH,
                )
            runner.stage(Stage(
                "joint16-train-" + name,
                command,
                expected=ROOT / "data/runs" / name / "summary.json",
            ))
            for epoch in (1, 2):
                evaluation_name = f"{name}-epoch{epoch}"
                runner.stage(Stage(
                    "joint16-dev-" + evaluation_name,
                    evaluate_args(
                        task, model_dir, name=evaluation_name,
                        selected_checkpoint=checkpoint(task, seed, epoch),
                    ),
                    expected=ROOT / "experiments" / (
                        f"joint16_dev_{task}_{evaluation_name}.json"
                    ),
                ))
        if mode == "full":
            runner.stage(Stage(
                "joint16-select-dev",
                ("-m", "scripts.select_joint16"),
                expected=ROOT / "experiments/joint16_dev_selection.json",
            ))
        return

    if mode == "test":
        if not (dry_run or confirm):
            raise ValueError("Final Test requires --confirm")
        selection_path = ROOT / "experiments/joint16_dev_selection.json"
        if not dry_run:
            if not selection_path.is_file():
                raise FileNotFoundError("Missing frozen full-Dev model selection")
            runner.require_gpu()
            prepare(split="test")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("status") != "JOINT16_DEV_SELECTED_TEST_UNOPENED":
                raise ValueError("Invalid frozen model selection")
        for task in MODES:
            if dry_run:
                selected_epoch = 2
            else:
                selected_epoch = selection["selected"][task]["selected_epoch"]
            name = run_id(task, SEEDS[0]) + f"-epoch{selected_epoch}"
            runner.stage(Stage(
                "joint16-test-" + name,
                evaluate_args(
                    task, model_dir, name=name, split="test",
                    selected_checkpoint=checkpoint(task, SEEDS[0], selected_epoch),
                ),
                expected=ROOT / "experiments" / f"joint16_test_{task}_{name}.json",
            ))
        return
    raise ValueError(f"Unsupported joint16 runner stage: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "status", "prepare", "pilot", "screen", "full", "test"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models/Jeff-1.1-Large")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.mode, args.model_dir, dry_run=args.dry_run, confirm=args.confirm)


if __name__ == "__main__":
    main()
