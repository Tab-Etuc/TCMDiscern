"""One registered H200 study runner. Uses the existing Jeff GPU installation and checkpoint utilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_disease_study import prepare
from tcmdiscern.h200.plan import ACCUMULATION, SEEDS, UPDATES_PER_EPOCH
from tcmdiscern.h200.workflow import H200Runner, Stage

SEEDS_PAPER = SEEDS

ROOT = Path(__file__).resolve().parents[1]
MODES = ("global", "conditioned", "conditioned_hard", "pair")
EPOCHS = 2


def task_mode(method: str) -> str:
    return "conditioned" if method == "conditioned_hard" else method


def run_id(method: str, seed: int) -> str:
    return f"disease-{method}-seed{seed}-e{EPOCHS}"


def train_command(method: str, seed: int, model_dir: Path) -> tuple[str, ...]:
    return (
        "-m", "scripts.run_jeff_finetune",
        "--model-dir", str(model_dir),
        "--study-mode", task_mode(method),
        "--method", "s1", "--run-id", run_id(method, seed),
        "--steps", str(UPDATES_PER_EPOCH * EPOCHS),
        "--epochs", str(EPOCHS),
        "--accumulation", str(ACCUMULATION),
        "--seed", str(seed),
        "--rank", "32", "--alpha", "64",
        "--lora-lr", "2e-5", "--head-lr", "1e-4",
        "--weight-decay", "0.01", "--clip-norm", "1",
        "--warmup-steps", "40",
        "--checkpoint-every", "300",
        "--pairwise-weight", "0.2" if method == "conditioned_hard" else "0",
        "--pairwise-margin", "0.2",
    )


def evaluation_command(
    model_dir: Path, *, mode: str, name: str, split: str = "dev",
    checkpoint: Path | None = None, limit: int | None = None,
) -> tuple[str, ...]:
    cmd = [
        "-m", "scripts.evaluate_disease_study", "--model-dir", str(model_dir),
        "--split", split, "--mode", mode, "--run-id", name,
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    return tuple(cmd)


def expected_report(split: str, mode: str, name: str) -> Path:
    return ROOT / "experiments" / f"disease_study_{split}_{mode}_{name}.json"


def registered_runs() -> list[tuple[str, int]]:
    return [(method, seed) for method in MODES for seed in SEEDS_PAPER]


def final_checkpoint(method: str, seed: int) -> Path:
    return ROOT / "data/runs" / run_id(method, seed) / f"resume-{UPDATES_PER_EPOCH * EPOCHS:07d}"


def run_study(
    mode: str, model_dir: Path, *, confirm: bool, dry_run: bool, selected: str | None,
) -> None:
    if mode == "plan":
        print(json.dumps({
            "study": "disease-conditioned Jeff/BM25",
            "official_splits": {"train": 43180, "dev": 5486, "test": 5486},
            "trainings": [{"method": m, "seed": s, "epochs": EPOCHS}
                          for m, s in registered_runs()],
            "k": 5, "accumulation": ACCUMULATION,
            "updates_per_epoch": UPDATES_PER_EPOCH,
            "test_locked_until_model_selection": True,
        }, ensure_ascii=False, indent=2))
        return

    runner = H200Runner(root=ROOT, model_dir=model_dir, dry_run=dry_run)
    if mode == "status":
        print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
        return
    if mode == "prepare":
        if dry_run:
            print("[DRY-RUN] official Train/Dev, global OOF and disease-study K5")
            return
        runner.prepare()
        print(json.dumps(prepare(split="dev"), ensure_ascii=False, indent=2))
        return
    if mode in ("pilot", "full"):
        if not confirm and mode == "full" and not dry_run:
            raise ValueError("Full H200 training requires --confirm")
        if not dry_run:
            runner.prepare()
            prepare(split="dev")
            runner.require_gpu()
        # A simple real two-case smoke verifies the exact K5 official Jeff inference path.
        runner.stage(Stage(
            "disease-pilot-k5",
            evaluation_command(model_dir, mode="conditioned", name="pilot", limit=2),
            expected=expected_report("dev", "conditioned", "pilot"),
        ))
        pilot_name = "disease-hard-pilot-one-step"
        pilot_train = (
            "-m", "scripts.run_jeff_finetune",
            "--model-dir", str(model_dir), "--study-mode", "conditioned",
            "--method", "s1", "--run-id", pilot_name,
            "--steps", "1", "--epochs", "1", "--accumulation", "1",
            "--warmup-steps", "0", "--checkpoint-every", "1",
            "--pairwise-weight", "0.2", "--pairwise-margin", "0.2",
        )
        runner.stage(Stage(
            "disease-pilot-train-one-step",
            pilot_train,
            expected=ROOT / "data/runs" / pilot_name / "summary.json",
        ))
        if mode == "pilot":
            return
        for task in ("global", "conditioned", "pair"):
            runner.stage(Stage(
                f"disease-zero-{task}",
                evaluation_command(model_dir, mode=task, name="zero"),
                expected=expected_report("dev", task, "zero"),
            ))
        for method, seed in registered_runs():
            name = run_id(method, seed)
            full_command = train_command(method, seed, model_dir)
            # Recover from completed optimizer checkpoints according to prior H200 policy.
            if not dry_run:
                full_command = runner._resume_command(
                    full_command, run_id=name, method="s1", seed=seed,
                    updates=UPDATES_PER_EPOCH * EPOCHS,
                )
            runner.stage(Stage(
                "disease-train-" + name, full_command,
                expected=ROOT / "data/runs" / name / "summary.json",
            ))
            runner.stage(Stage(
                "disease-eval-" + name,
                evaluation_command(
                    model_dir, mode=task_mode(method), name=name,
                    checkpoint=final_checkpoint(method, seed),
                ),
                expected=expected_report("dev", task_mode(method), name),
            ))
        selection = ROOT / "experiments/disease_study_dev_selection.json"
        runner.stage(Stage(
            "disease-select-dev",
            ("-m", "scripts.select_disease_study"),
            expected=selection,
        ))
        return
    if mode == "test":
        if not confirm and not dry_run:
            raise ValueError("Test evaluation requires --confirm")
        selection_path = ROOT / "experiments/disease_study_dev_selection.json"
        if dry_run:
            representatives = [selected] if selected else [
                run_id(method, SEEDS_PAPER[0]) for method in MODES
            ]
        else:
            if not selection_path.is_file():
                raise FileNotFoundError("Run the full Dev study before final Test")
            selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection_report.get("status") != "DEV_SELECTED_TEST_SEALED":
                raise ValueError("Invalid Dev model selection")
            representatives = [selected] if selected else [
                selection_report["methods"][method]["representative_checkpoint"]
                for method in MODES
            ]
            runner.require_gpu()
            # Final Test is downloaded and evaluated after the frozen Dev selection.
            prepare(split="test")
        allowed = {run_id(method, seed): (method, seed) for method, seed in registered_runs()}
        for name in representatives:
            if name not in allowed:
                raise ValueError("Test model must be a registered checkpoint")
            method, seed = allowed[name]
            runner.stage(Stage(
                "disease-test-" + name,
                evaluation_command(
                    model_dir, mode=task_mode(method), name=name, split="test",
                    checkpoint=final_checkpoint(method, seed),
                ),
                expected=expected_report("test", task_mode(method), name),
            ))
        return
    raise ValueError(f"Unknown study mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "status", "prepare", "pilot", "full", "test"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models/Jeff-1.1-Large")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--selected-run")
    args = parser.parse_args()
    run_study(
        args.mode, args.model_dir, confirm=args.confirm,
        dry_run=args.dry_run, selected=args.selected_run,
    )


if __name__ == "__main__":
    main()
