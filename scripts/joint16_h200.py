"""Reproducible H200 orchestration for disease and disease-syndrome pair K=16 tasks.

Uses the released 31B Jeff checkpoint and existing LoRA+Head trainer. The
original syndrome-148 tasks remain under run_h200.sh. All results for these
additional two tasks are evaluated with complete original Dev rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.prepare_joint16 import prepare
from scripts.select_joint16 import MODES, run_id
from tcmdiscern.h200.plan import (
    ACCUMULATION,
    SEEDS,
    UPDATES_PER_EPOCH,
)
from tcmdiscern.h200.workflow import (
    H200Runner,
    Stage,
    complete_optimizer_checkpoint,
    verified_optimizer_learning_signal,
)
from tcmdiscern.jeff_train_loop import expected_case_exposures

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = 2


def valid_eval_report(
    path: Path,
    *,
    task: str,
    split: str,
    run_id: str,
    limit: int | None,
    root: Path = ROOT,
) -> bool:
    """Require a complete original candidate-aligned 16-logit GPU output."""
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        rows = 5486 if limit is None else limit
        if any(
            (
                result.get("kind") != ("FULL_EVALUATION" if limit is None else "SMOKE"),
                result.get("mode") != task,
                result.get("split") != split,
                result.get("run_id") != run_id,
                result.get("k") != 16,
                result.get("attention_backend") not in {"released_default", "eager"},
                result.get("performance", {}).get("n_predictions") != rows,
            )
        ):
            return False
        if limit is None and (
            result.get("scores", {}).get("n_cases") != 5486
            or any(
                not isinstance(result["scores"].get(key), (float, int))
                or not math.isfinite(result["scores"][key])
                or not 0 <= result["scores"][key] <= 1
                for key in ("top1_accuracy", "macro_f1")
            )
        ):
            return False
        raw = result.get("logits_file")
        if not isinstance(raw, str):
            return False
        private = (root / raw).resolve()
        if not private.is_relative_to(root / "data/joint16") or not private.is_file():
            return False
        n = 0
        with private.open(encoding="utf-8") as stream:
            for n, line in enumerate(stream, start=1):
                record = json.loads(line)
                if (
                    n > rows
                    or record.get("case_id") != f"{split}-{n - 1:06d}"
                    or not isinstance(record.get("logits"), list)
                    or len(record["logits"]) != 16
                    or any(
                        type(v) not in (float, int) or not math.isfinite(v)
                        for v in record["logits"]
                    )
                ):
                    return False
        return n == rows
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def valid_training_report(
    path: Path,
    *,
    task: str,
    run_id: str,
    steps: int,
    accumulation: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = path.parent / f"resume-{steps:07d}"
        return all(
            (
                result.get("status") == "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
                result.get("study_mode") == f"{task}16",
                result.get("loss_method") == "s1",
                result.get("run_id") == run_id,
                result.get("global_update") == steps,
                result.get("target_updates") == steps,
                result.get("accumulation") == accumulation,
                result.get("source_count_verified") == 43180,
                result.get("expected_case_exposures")
                == expected_case_exposures(43180, accumulation, steps),
                result.get("microbatches") == expected_case_exposures(43180, accumulation, steps),
                verified_optimizer_learning_signal(result),
                complete_optimizer_checkpoint(checkpoint),
            )
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def checkpoint(mode: str, seed: int, epoch: int) -> Path:
    if mode not in MODES or seed not in SEEDS or epoch not in (1, 2):
        raise ValueError("Invalid registered checkpoint")
    return ROOT / "data/runs" / run_id(mode, seed) / f"resume-{epoch * UPDATES_PER_EPOCH:07d}"


def evaluate_args(
    mode: str,
    model_dir: Path,
    *,
    name: str,
    split: str = "dev",
    selected_checkpoint: Path | None = None,
    limit: int | None = None,
) -> tuple[str, ...]:
    command = [
        "-m",
        "scripts.evaluate_joint16",
        "--mode",
        mode,
        "--split",
        split,
        "--model-dir",
        str(model_dir),
        "--run-id",
        name,
    ]
    if selected_checkpoint is not None:
        command.extend(["--checkpoint", str(selected_checkpoint)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return tuple(command)


def train_args(
    mode: str,
    seed: int,
    model_dir: Path,
    *,
    dry_pilot: bool = False,
) -> tuple[str, ...]:
    updates = 1 if dry_pilot else EPOCHS * UPDATES_PER_EPOCH
    epochs = 1 if dry_pilot else EPOCHS
    return (
        "-m",
        "scripts.run_jeff_finetune",
        "--model-dir",
        str(model_dir),
        "--study-mode",
        f"{mode}16",
        "--method",
        "s1",
        "--run-id",
        f"joint16-pilot-{mode}" if dry_pilot else run_id(mode, seed),
        "--steps",
        str(updates),
        "--epochs",
        str(epochs),
        "--accumulation",
        "1" if dry_pilot else str(ACCUMULATION),
        "--seed",
        str(seed),
        "--rank",
        "32",
        "--alpha",
        "64",
        "--lora-lr",
        "2e-5",
        "--head-lr",
        "1e-4",
        "--weight-decay",
        "0.01",
        "--clip-norm",
        "1",
        "--warmup-steps",
        "0" if dry_pilot else "40",
        "--checkpoint-every",
        "1" if dry_pilot else "300",
    )


def registered_runs():
    return [(mode, seed) for mode in MODES for seed in SEEDS]


def run_study(
    mode: str,
    model_dir: Path,
    *,
    dry_run: bool = False,
    confirm: bool = False,
) -> None:
    if mode == "plan":
        print(
            json.dumps(
                {
                    "tasks": ["disease", "pair"],
                    "classes_from_official_train": {"disease": 403, "pair": 1782},
                    "official_rows": {"train": 43180, "dev": 5486, "test": 5486},
                    "candidate_k": 16,
                    "optimizer_updates_per_epoch": UPDATES_PER_EPOCH,
                    "seeds": list(SEEDS),
                    "epochs_per_run": EPOCHS,
                    "main_trainings": [
                        {"mode": task, "seed": seed, "run_id": run_id(task, seed)}
                        for task, seed in registered_runs()
                    ],
                    "test_requires_frozen_dev_model_selection": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    runner = H200Runner(root=ROOT, model_dir=model_dir, dry_run=dry_run)
    session = None
    if not dry_run and mode in {"pilot", "screen", "full", "test"}:
        from tcmdiscern.h200.session import JeffGPUSession

        session = JeffGPUSession(model_dir)
    if mode == "status":
        print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
        return
    if mode == "prepare":
        if dry_run:
            print("[DRY-RUN] 5-fold Train-only K16 disease and pair candidates, full Dev")
            return
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
            runner.stage(
                Stage(
                    "joint16-pilot-zero-" + task,
                    evaluate_args(task, model_dir, name="pilot", limit=2),
                    expected=path,
                    verify=lambda task=task, path=path: valid_eval_report(
                        path,
                        task=task,
                        split="dev",
                        run_id="pilot",
                        limit=2,
                    ),
                    perform=(
                        lambda task=task: session.joint_dev(
                            mode=task,
                            split="dev",
                            run_id="pilot",
                            limit=2,
                        )
                    )
                    if session is not None
                    else None,
                )
            )
            name = f"joint16-pilot-{task}"
            runner.stage(
                Stage(
                    "joint16-pilot-train-" + task,
                    train_args(task, SEEDS[0], model_dir, dry_pilot=True),
                    expected=ROOT / "data/runs" / name / "summary.json",
                    verify=lambda task=task, name=name: valid_training_report(
                        ROOT / "data/runs" / name / "summary.json",
                        task=task,
                        run_id=name,
                        steps=1,
                        accumulation=1,
                    ),
                    perform=(
                        lambda task=task: session.train_cli(
                            train_args(task, SEEDS[0], model_dir, dry_pilot=True)
                        )
                    )
                    if session is not None
                    else None,
                )
            )
        if mode == "pilot":
            return
        for task in MODES:
            runner.stage(
                Stage(
                    f"joint16-zero-{task}",
                    evaluate_args(task, model_dir, name="zero"),
                    expected=ROOT / "experiments" / f"joint16_dev_{task}_zero.json",
                    verify=lambda task=task: valid_eval_report(
                        ROOT / "experiments" / f"joint16_dev_{task}_zero.json",
                        task=task,
                        split="dev",
                        run_id="zero",
                        limit=None,
                    ),
                    perform=(
                        lambda task=task: session.joint_dev(
                            mode=task,
                            split="dev",
                            run_id="zero",
                        )
                    )
                    if session is not None
                    else None,
                )
            )
        methods = [(task, SEEDS[0]) for task in MODES]
        if mode == "full":
            methods = registered_runs()

        for task, seed in methods:
            name = run_id(task, seed)
            command = train_args(task, seed, model_dir)
            if not dry_run:
                command = runner._resume_command(
                    command,
                    run_id=name,
                    method="s1",
                    seed=seed,
                    updates=EPOCHS * UPDATES_PER_EPOCH,
                )
            runner.stage(
                Stage(
                    "joint16-train-" + name,
                    command,
                    expected=ROOT / "data/runs" / name / "summary.json",
                    verify=lambda task=task, name=name: valid_training_report(
                        ROOT / "data/runs" / name / "summary.json",
                        task=task,
                        run_id=name,
                        steps=EPOCHS * UPDATES_PER_EPOCH,
                        accumulation=ACCUMULATION,
                    ),
                    perform=(lambda command=command: session.train_cli(command))
                    if session is not None
                    else None,
                )
            )
            for epoch in (1, 2):
                evaluation_name = f"{name}-epoch{epoch}"
                eval_checkpoint = checkpoint(task, seed, epoch)
                runner.stage(
                    Stage(
                        "joint16-dev-" + evaluation_name,
                        evaluate_args(
                            task,
                            model_dir,
                            name=evaluation_name,
                            selected_checkpoint=eval_checkpoint,
                        ),
                        expected=ROOT
                        / "experiments"
                        / (f"joint16_dev_{task}_{evaluation_name}.json"),
                        verify=lambda task=task, name=evaluation_name: valid_eval_report(
                            ROOT / "experiments" / f"joint16_dev_{task}_{name}.json",
                            task=task,
                            split="dev",
                            run_id=name,
                            limit=None,
                        ),
                        perform=(
                            lambda task=task, name=evaluation_name, path=eval_checkpoint: (
                                session.joint_dev(
                                    mode=task,
                                    split="dev",
                                    run_id=name,
                                    checkpoint=path,
                                )
                            )
                        )
                        if session is not None
                        else None,
                    )
                )
        if mode == "full":
            runner.stage(
                Stage(
                    "joint16-select-dev",
                    ("-m", "scripts.select_joint16"),
                    expected=ROOT / "experiments/joint16_dev_selection.json",
                )
            )
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
            eval_checkpoint = checkpoint(task, SEEDS[0], selected_epoch)
            runner.stage(
                Stage(
                    "joint16-test-" + name,
                    evaluate_args(
                        task,
                        model_dir,
                        name=name,
                        split="test",
                        selected_checkpoint=eval_checkpoint,
                    ),
                    expected=ROOT / "experiments" / f"joint16_test_{task}_{name}.json",
                    verify=lambda task=task, name=name: valid_eval_report(
                        ROOT / "experiments" / f"joint16_test_{task}_{name}.json",
                        task=task,
                        split="test",
                        run_id=name,
                        limit=None,
                    ),
                    perform=(
                        lambda task=task, name=name, path=eval_checkpoint: session.joint_dev(
                            mode=task,
                            split="test",
                            run_id=name,
                            checkpoint=path,
                        )
                    )
                    if session is not None
                    else None,
                )
            )
        return
    raise ValueError(f"Unsupported joint16 runner stage: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("plan", "status", "prepare", "pilot", "screen", "full", "test")
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models/Jeff-1.1-Large")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_study(args.mode, args.model_dir, dry_run=args.dry_run, confirm=args.confirm)


if __name__ == "__main__":
    main()
