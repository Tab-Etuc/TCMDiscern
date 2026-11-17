"""Single prioritized, resumable H200 experiment queue for TCMDiscern.

Run after the pinned H200 environment and the existing first-seed checkpoints
have been prepared. This queue never opens the official Test split.
The 31B Jeff backbone is loaded once for all pending GPU training/evaluation
stages. New run IDs protect all existing models/results.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tcmdiscern.h200.workflow import complete_optimizer_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def default_model_dir() -> Path:
    """Use the pinned weight location without copying a 31B model."""
    provided = os.environ.get("H200_MODEL_DIR")
    if provided:
        return Path(provided).expanduser()
    from tcmdiscern.case_rag_artifacts import artifact_roots

    for base in artifact_roots(ROOT):
        candidate = base / "data/models/Jeff-1.1-Large"
        if candidate.is_dir():
            return candidate
    return ROOT / "data/models/Jeff-1.1-Large"


MODEL = default_model_dir()
STATE = ROOT / "data/h200/priority_experiment_queue.json"
STEPS = 1350
SEED = 20261112


@dataclass(frozen=True)
class Experiment:
    name: str
    method: str = "s1"
    seed: int = SEED
    components: str = "both"
    negatives: str = "hard"
    priority: int = 2

    @property
    def run_id(self) -> str:
        if self.name == "seed":
            return f"{self.method}-seed{self.seed}-e2"
        return f"priority-{self.name}"


EXPERIMENTS = (
    Experiment("s1-head-only", components="head", priority=2),
    Experiment("s1-lora-only", components="lora", priority=2),
    Experiment("s1-random-negative", negatives="random", priority=3),
    Experiment("seed", method="s1", seed=20261113, priority=5),
    Experiment("seed", method="s1", seed=20261114, priority=5),
    Experiment("seed", method="s2", seed=20261113, priority=6),
    Experiment("seed", method="s2", seed=20261114, priority=6),
    Experiment("seed", method="s3", seed=20261113, priority=6),
    Experiment("seed", method="s3", seed=20261114, priority=6),
)


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE)


def valid_report(run_id: str, epoch: int = 2) -> bool:
    """Accept cached results only when metadata and every saved logit row agree."""
    from tcmdiscern.jeff_config import JEFF_MODEL_REVISION
    from tcmdiscern.jeff_scoring import read_model_logits

    report = ROOT / "experiments" / f"jeff_{run_id}-epoch{epoch}_k16_dev.json"
    expected_logits = ROOT / "data/jeff" / f"dev_{run_id}-epoch{epoch}_k16_5486.jsonl"
    if not report.is_file():
        # Raw logits without a completed public report indicate an interrupted
        # evaluation; never overwrite this data or silently mark it complete.
        if expected_logits.exists():
            raise RuntimeError(f"Unpaired partial Dev output; inspect before retry: {expected_logits}")
        return False
    payload = json.loads(report.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    saved = payload.get("checkpoint", {})
    expected_relative = str(expected_logits.relative_to(ROOT))
    if (
        payload.get("kind") != "FULL_DEV_FINETUNED"
        or payload.get("rows") != 5486
        or payload.get("model_revision") != JEFF_MODEL_REVISION
        or payload.get("trained_lora") is not True
        or payload.get("reasoning") is not False
        or payload.get("test_used") is not False
        or payload.get("raw_logit_file_ignored") != expected_relative
        or scores.get("n_cases") != 5486
        or scores.get("label_classes") != 148
        or not isinstance(saved, dict)
        or saved.get("source_run_id") != run_id
    ):
        raise RuntimeError(f"Existing Dev report has mismatched provenance: {report}")
    if not expected_logits.is_file():
        raise RuntimeError(f"Evaluation report lacks raw logits: {expected_logits}")
    read_model_logits(expected_logits, num_cases=5486)
    return True


def baseline_microbatch() -> int:
    """Match the already trained S1 first seed when it is available."""
    from tcmdiscern.case_rag_artifacts import artifact_roots
    from tcmdiscern.jeff_axolotl_attention import requested_attention_backend
    from tcmdiscern.jeff_config import JEFF_MODEL_REVISION

    candidates = [
        path / "data/runs/s1-seed20261112-e2"
        for path in artifact_roots(ROOT)
    ]
    baseline = next(
        (
            path for path in candidates
            if (path / "summary.json").is_file()
            or (path / "resume-0001350/training_state.json").is_file()
        ),
        None,
    )
    if baseline is None:
        return 1  # Dry-run on developer machine, without H200 checkpoints.
    summary = baseline / "summary.json"
    state = baseline / "resume-0001350/training_state.json"
    if summary.is_file():
        original = json.loads(summary.read_text(encoding="utf-8"))
    else:
        original = json.loads(state.read_text(encoding="utf-8"))
    if (
        original.get("loss_method") != "s1"
        or original.get("seed") != SEED
        or original.get("target_updates") != STEPS
        or original.get("accumulation") != 64
        or original.get("source") != "train_oof_multipos_k16_epoch03.jsonl"
        or original.get("rank") != 32
        or original.get("alpha") != 64
        or original.get("warmup_steps") != 40
        or original.get("epochs") != 2
        or original.get("model_revision") != JEFF_MODEL_REVISION
        or original.get("lora_lr") != 2e-5
        or original.get("head_lr") != 1e-4
        or original.get("weight_decay") != 0.01
        or original.get("clip_norm") != 1.0
        or original.get("checkpoint_every") != 300
        or original.get("attention_backend", "released_default") != requested_attention_backend()
    ):
        raise RuntimeError("Existing S1 seed1 training contract differs from priority experiments")
    batch = original.get("micro_batch_size", 1)
    if type(batch) is not int or not 1 <= batch <= 8:
        raise ValueError("Invalid original S1 micro-batch size")
    return batch


def train_args(task: Experiment, model_dir: Path) -> list[str]:
    from scripts.run_jeff_finetune import build_parser
    from tcmdiscern.jeff_train_loop import fixed_run_settings

    run_folder = ROOT / "data/runs" / task.run_id
    micro_batch = baseline_microbatch()
    argv = [
        "-m", "scripts.run_jeff_finetune",
        "--model-dir", str(model_dir),
        "--method", task.method,
        "--run-id", task.run_id,
        "--steps", str(STEPS),
        "--epochs", "2",
        "--accumulation", "64",
        "--micro-batch-size", str(micro_batch),
        "--seed", str(task.seed),
        "--rank", "32",
        "--alpha", "64",
        "--lora-lr", "2e-5",
        "--head-lr", "1e-4",
        "--weight-decay", "0.01",
        "--clip-norm", "1.0",
        "--warmup-steps", "40",
        "--checkpoint-every", "300",
        "--trainable-components", task.components,
        "--negative-sampling", task.negatives,
    ]
    expected = fixed_run_settings(build_parser().parse_args(argv[2:]))

    def verify_metadata(saved: dict, where: Path) -> None:
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(f"Run contract mismatch at {where}: {key}")
        if saved.get("trainable_components", "both") != task.components:
            raise ValueError(f"Trainable components mismatch: {where}")
        if saved.get("negative_sampling", "hard") != task.negatives:
            raise ValueError(f"Negative sampling mismatch: {where}")
        if saved.get("micro_batch_size", 1) != micro_batch:
            raise ValueError(f"Micro-batch mismatch: {where}")

    summary = run_folder / "summary.json"
    if summary.exists():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        verify_metadata(saved, summary)
        if (
            saved.get("status") != "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING"
            or saved.get("global_update") != STEPS
            or saved.get("source_count_verified") != 43180
            or saved.get("expected_case_exposures") != 86360
            or saved.get("microbatches") != 86360
        ):
            raise RuntimeError(f"Existing training summary is incomplete: {summary}")
        checkpoint = run_folder / f"resume-{STEPS:07d}"
        if not complete_optimizer_checkpoint(checkpoint):
            raise RuntimeError(f"Completed training checkpoint invalid: {checkpoint}")
        verify_metadata(
            json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8")),
            checkpoint,
        )
        return []
    if run_folder.exists():
        snapshots = sorted(run_folder.glob("resume-*"))
        if not snapshots:
            raise RuntimeError(f"Interrupted run lacks recoverable snapshot: {run_folder}")
        latest = snapshots[-1]
        if not complete_optimizer_checkpoint(latest):
            raise RuntimeError(f"Incomplete newest snapshot: {latest}")
        saved = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        verify_metadata(saved, latest)
        if saved["global_update"] >= STEPS:
            raise RuntimeError(f"Training finished but summary missing: {latest}")
        argv.extend(["--resume", str(latest)])
    return argv


def verified_smoke(task: Experiment, smoke_id: str) -> bool:
    """Check that the cheap one-update GPU gate actually trained its component."""
    smoke_dir = ROOT / "data/runs" / smoke_id
    summary = smoke_dir / "summary.json"
    if not summary.exists():
        return False
    result = json.loads(summary.read_text(encoding="utf-8"))
    gradients = result.get("first_gradient", {})
    update = result.get("first_optimizer_update_verified", {})
    checkpoint = smoke_dir / "resume-0000001"
    if not (
        result.get("status") == "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING"
        and result.get("run_id") == smoke_id
        and result.get("loss_method") == task.method
        and result.get("trainable_components", "both") == task.components
        and result.get("negative_sampling", "hard") == task.negatives
        and result.get("global_update") == 1
        and result.get("expected_case_exposures") == 64
        and result.get("microbatches") == 64
        and result.get("source_count_verified") == 43180
        and result.get("micro_batch_size", 1) == baseline_microbatch()
        and gradients.get("all_gradients_finite") is True
        and complete_optimizer_checkpoint(checkpoint)
    ):
        raise RuntimeError(f"Incomplete GPU smoke validation: {smoke_dir}")
    if task.components == "both":
        keys = ("lora_B", "head_delta_weight", "head_delta_bias")
    else:
        keys = ("active_component_max_delta",)
    if not all(
        type(update.get(name)) in (int, float) and update[name] > 0
        for name in keys
    ):
        raise RuntimeError(f"GPU smoke failed to update the requested weights: {smoke_id}")
    return True


def pending_tasks(*, skip_seeds: bool, model_dir: Path) -> list[tuple[Experiment, list[str]]]:
    selected = [x for x in EXPERIMENTS if not (skip_seeds and x.name == "seed")]
    output = []
    for task in selected:
        command = train_args(task, model_dir)
        if command or not valid_report(task.run_id):
            output.append((task, command))
    return output


def run_cpu_stage(
    module: str, output: Path, *, python: str,
    args: tuple[str, ...] = (), always_run: bool = False,
) -> None:
    """Rerun cheap checks when correctness depends on the executing environment."""
    if not always_run and output.is_file() and output.stat().st_size > 0:
        if module == "scripts.prepare_h200_random_negative":
            # The generator verifies all 43,180 rows in the existing file.
            from scripts.prepare_h200_random_negative import build

            existing = build()
            if existing["rows"] != 43180:
                raise RuntimeError("Incomplete random-negative control")
        else:
            try:
                cached = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Corrupt cached CPU report: {output}") from exc
            if module == "scripts.evaluate_case_rag_strong_sgd_fusion_cpu":
                import platform

                import numpy as np
                import scipy
                import sklearn

                runtime = cached.get("numerical_environment", {})
                current = {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__,
                    "scipy": scipy.__version__,
                }
                if not (
                    cached.get("status") == "CASE_RAG_STRONG_SGD_CPU_FUSION_COMPLETED"
                    and cached.get("dev_cases") == 5486
                    and cached.get("all_available_completed_arms_evaluated") is True
                ):
                    raise RuntimeError(f"Incorrect cached Case-RAG output: {output}")
                if runtime != current:
                    # Refit the frozen SGD when the Python/scientific
                    # environment has changed, to keep Base/Global2 paired.
                    always_run = True
        if not always_run:
            print(f"[SKIP] Verified CPU artifact: {output.relative_to(ROOT)}", flush=True)
            return
    command = [python, "-m", module, *args]
    print("[CPU] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Stage succeeded but no output: {output}")


def run_queue(*, model_dir: Path, skip_seeds: bool, dry_run: bool) -> None:
    from scripts.prepare_h200_random_negative import TARGET as RANDOM_TARGET
    from scripts.run_jeff_finetune import build_parser
    from tcmdiscern.jeff_train_loop import validate_run_args

    tasks = pending_tasks(skip_seeds=skip_seeds, model_dir=model_dir)
    planned = {
        "status": "planned", "official_test_used": False,
        "timestamp": datetime.now(UTC).isoformat(),
        "queue": [{"priority": t.priority, "run_id": t.run_id, "train_needed": bool(c),
                    "dev_evaluation_needed": not valid_report(t.run_id)}
                   for t, c in tasks],
    }
    print(json.dumps(planned, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return
    # On the H200 the original S1/S2/S3 scores may live in the adjacent
    # TCMDiscern-main run folder. Reuse the pre-registered artifact lookup,
    # which checks both the current checkout and legacy training directories.
    from tcmdiscern.case_rag_artifacts import frozen_paths

    for method in ("s1", "s2", "s3"):
        frozen_paths(ROOT, method)
    from scripts.run_h200_priority_real_gpu_latency import original_s1_checkpoint

    first_checkpoint = original_s1_checkpoint(ROOT)
    if not complete_optimizer_checkpoint(first_checkpoint):
        raise RuntimeError(f"Original S1 checkpoint is incomplete: {first_checkpoint}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Pinned Jeff model not found: {model_dir}")

    # Always refit the frozen CPU SGD under THIS H200 Python/sklearn runtime.
    # The earlier developer/H200 numerical differences changed six diagnoses.
    run_cpu_stage(
        "scripts.run_h200_priority_cpu_ablation",
        ROOT / "experiments/h200_priority_cpu_ablation.json",
        python=sys.executable,
        always_run=True,
    )
    selected20 = ROOT / "data/h200/priority_budget_20pct_indices.json"
    if not selected20.is_file():
        raise RuntimeError("Current-run selective inference indices are missing")
    run_cpu_stage(
        "scripts.prepare_h200_random_negative", RANDOM_TARGET, python=sys.executable
    )

    # Supplemental historical Case-RAG controls must never prevent the core
    # ablations from using the idle H200 if the historical archive is absent.
    from scripts.prepare_case_rag_paper_ablation import TAG

    base = ROOT / "data/case_rag" / TAG
    needed = [
        base / "global2.jsonl",
        *(base / "scores" / f"paper-{method}-full/global2.jsonl"
          for method in ("s1", "s2", "s3")),
    ]
    if all(path.is_file() for path in needed) and (
        ROOT / "src/tcmdiscern/case_rag.py"
    ).is_file():
        strong_report = ROOT / "experiments/h200_priority_strong_fusion_all_arms.json"
        run_cpu_stage(
            "scripts.evaluate_case_rag_strong_sgd_fusion_cpu",
            strong_report,
            python=sys.executable,
            args=("--strict-global2", "--all-arms", "--output", str(strong_report)),
        )
    else:
        print(
            "[OPTIONAL] Historical Case-RAG archive incomplete; continuing core GPU runs. "
            f"Missing: {[str(p.relative_to(ROOT)) for p in needed if not p.is_file()]}",
            flush=True,
        )

    # Recompute queued tasks only after CPU stages have verified all prerequisites.
    tasks = pending_tasks(skip_seeds=skip_seeds, model_dir=model_dir)
    from scripts.run_h200_priority_real_gpu_latency import (
        completed_report,
        selection_indices,
    )

    # Training may be finished even when the GPU timing stage was interrupted.
    latency_ready = completed_report(
        ROOT / "experiments/h200_priority_real_gpu_latency.json",
        selection_indices(ROOT),
    ) is not None
    if not tasks and latency_ready:
        from scripts.summarize_h200_priority_experiments import collect

        collect()
        save_state({**planned, "status": "completed", "pending": 0})
        print("H200_PRIORITY_EXPERIMENTS_COMPLETED", flush=True)
        return

    # Full model is kept resident across selective latency and all training/evaluation.
    from tcmdiscern.h200.session import JeffGPUSession
    session = JeffGPUSession(model_dir)
    progress = {**planned, "status": "running", "completed": []}
    save_state(progress)
    try:
        from scripts.run_h200_priority_real_gpu_latency import run as run_real_gpu_latency

        run_real_gpu_latency(session)
        for task, command in tasks:
            print(f"[GPU] Priority {task.priority}: {task.run_id}", flush=True)
            if command:
                if task.name != "seed":
                    # One optimizer step with the same components and sample
                    # source catches expensive head-only / LoRA-only failures.
                    smoke_id = f"priority-smoke-{task.name}"
                    smoke_dir = ROOT / "data/runs" / smoke_id
                    # A valid resumed full-training checkpoint already proves
                    # optimizer execution. Do not accidentally pass its --resume
                    # path into the unrelated one-step smoke run.
                    if "--resume" not in command and not verified_smoke(task, smoke_id):
                        if smoke_dir.exists():
                            raise RuntimeError(f"Incomplete smoke experiment: {smoke_dir}")
                        smoke = list(command)
                        for name, value in (
                            ("--run-id", smoke_id),
                            ("--steps", "1"),
                            ("--epochs", "1"),
                            ("--warmup-steps", "0"),
                            ("--checkpoint-every", "1"),
                        ):
                            pos = smoke.index(name) + 1
                            smoke[pos] = value
                        session.train_cli(tuple(smoke))
                        if not verified_smoke(task, smoke_id):
                            raise RuntimeError("GPU smoke missing or invalid after execution")
                args = build_parser().parse_args(command[2:])
                validate_run_args(args)
                session.train_cli(tuple(command))
            checkpoint = ROOT / "data/runs" / task.run_id / f"resume-{STEPS:07d}"
            if not complete_optimizer_checkpoint(checkpoint):
                raise RuntimeError(f"Model checkpoint invalid: {checkpoint}")
            for epoch in (1, 2):
                report_id = f"{task.run_id}-epoch{epoch}"
                if valid_report(task.run_id, epoch=epoch):
                    print(f"[SKIP] Existing Dev evaluation {report_id}", flush=True)
                    continue
                checkpoint_epoch = ROOT / "data/runs" / task.run_id / f"resume-{epoch * 675:07d}"
                if not complete_optimizer_checkpoint(checkpoint_epoch):
                    raise RuntimeError(f"Missing valid epoch checkpoint: {checkpoint_epoch}")
                session.syndrome_dev(
                    root=ROOT, checkpoint=checkpoint_epoch, run_id=report_id
                )
                if not valid_report(task.run_id, epoch=epoch):
                    raise RuntimeError(f"Dev evaluation missing after run: {report_id}")
            progress["completed"].append(task.run_id)
            save_state(progress)
        # Summary is recreated after the full GPU queue; all individual public
        # result files are preserved and never rewritten by the training runner.
        from scripts.summarize_h200_priority_experiments import collect

        collect()
        progress["status"] = "completed"
        progress["pending"] = 0
        progress["model_weight_loads"] = session.model_weight_loads
        save_state(progress)
        print("H200_PRIORITY_EXPERIMENTS_COMPLETED", flush=True)
    except BaseException as exc:
        progress["status"] = "failed"
        progress["error"] = repr(exc)
        save_state(progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--skip-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run_queue(
            model_dir=args.model_dir.expanduser().resolve(),
            skip_seeds=args.skip_seeds,
            dry_run=args.dry_run,
        )
    except BaseException as exc:
        # CPU preflight failures happen before the GPU loop's own failure
        # handler. Persist them too so an old 'completed' marker cannot mislead.
        if not args.dry_run:
            try:
                state = (
                    json.loads(STATE.read_text(encoding="utf-8"))
                    if STATE.is_file() else {}
                )
                state.update(
                    status="failed",
                    official_test_used=False,
                    error=repr(exc),
                    failed_at=datetime.now(UTC).isoformat(),
                )
                save_state(state)
            except (OSError, ValueError, TypeError):
                print("[ERROR] Could not persist failure status", file=sys.stderr)
        raise
