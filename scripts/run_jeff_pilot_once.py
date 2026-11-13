"""One-load H200 Jeff syndrome Pilot.

Run the released 31B once, preserve its real GPU state, and complete smoke,
Step-0, 2 optimizer updates, transactional resume and *fresh adapter* reload
without rereading the 31B from disk for each check.

Original TCM-SD Test is never opened. The two Dev items used for smoke are
not benchmark results. Training is limited to 2 optimizer updates.
"""

from __future__ import annotations

import gc
import json
import math
import tempfile
from pathlib import Path

from scripts.run_jeff_finetune import build_parser
from scripts.run_jeff_zeroshot import iter_dev_rows, run_decider
from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.h200.plan import ACCUMULATION, SEEDS
from tcmdiscern.h200.workflow import (
    complete_optimizer_checkpoint,
    verified_optimizer_learning_signal,
)
from tcmdiscern.jeff_checkpoint import reset_released_backbone, restore_checkpoint
from tcmdiscern.jeff_config import JEFF_MODEL_REVISION
from tcmdiscern.jeff_train_loop import run as train

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "pilot-s3-2updates"
PILOT_UPDATES = 2


def pilot_training_args(model_dir: Path, *, root: Path = ROOT):
    """Exactly the registered CLI hyperparameters, including resume settings."""
    params = [
        "--model-dir",
        str(model_dir),
        "--method",
        "s3",
        "--run-id",
        RUN_ID,
        "--steps",
        str(PILOT_UPDATES),
        "--epochs",
        "1",
        "--accumulation",
        str(ACCUMULATION),
        "--seed",
        str(SEEDS[0]),
        "--warmup-steps",
        "0",
        "--checkpoint-every",
        "1",
    ]
    run_dir = root / "data/runs" / RUN_ID
    if not (run_dir / "summary.json").is_file() and run_dir.is_dir():
        snapshots = sorted(run_dir.glob("resume-*"))
        if not snapshots:
            if any(run_dir.iterdir()):
                raise RuntimeError(
                    "Pilot run folder has an incomplete, uncheckpointed run; "
                    f"inspect it before retry: {run_dir}"
                )
            run_dir.rmdir()  # Empty directory from an interrupted initial attempt.
        else:
            if any(not complete_optimizer_checkpoint(item) for item in snapshots):
                raise RuntimeError("Interrupted Pilot has an incomplete optimizer checkpoint")
            latest = snapshots[-1]
            state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
            required = {
                "run_id": RUN_ID,
                "loss_method": "s3",
                "seed": SEEDS[0],
                "target_updates": PILOT_UPDATES,
                "accumulation": ACCUMULATION,
                "model_revision": JEFF_MODEL_REVISION,
            }
            if any(state.get(field) != value for field, value in required.items()):
                raise RuntimeError("Pilot resume checkpoint settings do not match this study")
            if state["global_update"] >= PILOT_UPDATES:
                raise RuntimeError(
                    "Pilot final checkpoint exists without summary; inspect before retry"
                )
            params.extend(["--resume", str(latest)])
    return build_parser().parse_args(params)


def verify_completed_training(root: Path) -> Path:
    """Require the actual 2-step optimizer artifact, not a partial JSON."""
    run_dir = root / "data/runs" / RUN_ID
    summary = run_dir / "summary.json"
    last = run_dir / f"resume-{PILOT_UPDATES:07d}"
    if not summary.is_file() or not complete_optimizer_checkpoint(last):
        raise RuntimeError("Pilot completed training has missing optimizer/checkpoint assets")
    result = json.loads(summary.read_text(encoding="utf-8"))
    if (
        result.get("status") != "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING"
        or result.get("run_id") != RUN_ID
        or result.get("loss_method") != "s3"
        or result.get("global_update") != PILOT_UPDATES
        or result.get("accumulation") != ACCUMULATION
        or result.get("source_count_verified") != 43180
        or result.get("expected_case_exposures") != 128
        or result.get("microbatches") != 128
        or not verified_optimizer_learning_signal(result)
    ):
        raise RuntimeError("Pilot completed training summary is inconsistent")
    return last


def read_two_dev_rows(root: Path) -> list[dict]:
    """Full Dev source/candidate integrity validation before selecting two rows."""
    dev = load_records(root / "data/tcm-sd/dev.json")
    candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    return list(
        iter_dev_rows(root / "data/jeff/dev_k16.jsonl", candidates, limit=2, original_cases=dev)
    )


def valid_two_logits(path: Path) -> bool:
    if not path.is_file():
        return False
    import math

    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return len(rows) == 2 and all(
        row.get("case_id") == f"dev-{i:06d}"
        and isinstance(row.get("logits"), list)
        and len(row["logits"]) == 16
        and all(type(logit) in (int, float) and math.isfinite(logit) for logit in row["logits"])
        for i, row in enumerate(rows)
    )


def score_two_if_missing(rows: list[dict], decider, output: Path) -> None:
    if output.exists():
        if not valid_two_logits(output):
            raise RuntimeError(f"Existing Pilot logits are invalid: {output}")
        # Old partial Pilots may have used a different attention backend.
        # Re-evaluate two items in a private temporary file and prove logit
        # parity before reusing the prior artifact for this experiment.
        with tempfile.TemporaryDirectory(prefix=".pilot-parity-", dir=output.parent) as folder:
            comparison = Path(folder) / output.name
            run_decider(rows, decider, comparison)
            original = [
                json.loads(line)["logits"]
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            current = [
                json.loads(line)["logits"]
                for line in comparison.read_text(encoding="utf-8").splitlines()
            ]
            if (
                len(original) != 2
                or len(current) != 2
                or any(
                    not math.isclose(a, b, abs_tol=1e-3, rel_tol=1e-4)
                    for older, newer in zip(original, current, strict=True)
                    for a, b in zip(older, newer, strict=True)
                )
            ):
                raise RuntimeError(
                    f"Existing Pilot logits disagree with the active causal attention backend: {output}. "
                    "Preserve the file for inspection; do not combine incompatible runs."
                )
        print(f"[PILOT] Existing smoke logits match active backend: {output.name}", flush=True)
        return
    cases, tokens, seconds = run_decider(rows, decider, output)
    if cases != 2 or tokens < 2 or not valid_two_logits(output):
        raise RuntimeError("Two-case official Jeff smoke outputs are invalid")
    print(f"[PILOT] Two Dev decisions PASS ({seconds:.2f}s, {tokens} tokens)", flush=True)


def run_one_load(model_dir: Path, *, root: Path = ROOT) -> dict:
    """Exactly one JeffModel(...) load for the entire syndrome Pilot."""
    from scripts.train_jeff_lora import load_original_checkpoint

    model_dir = model_dir.expanduser().resolve(strict=True)
    if (root / "data/h200/pilot_one_load.json").exists():
        raise FileExistsError("Integrated Pilot report already exists; inspect before replay")
    # Validate Train side and checkpoint state before the costly 31B weight read.
    train_args = pilot_training_args(model_dir, root=root)
    rows = read_two_dev_rows(root)
    summary = root / "data/runs" / RUN_ID / "summary.json"
    original_smoke = root / "data/jeff/dev_zero_shot_k16_smoke0002.jsonl"
    trained_smoke = root / "data/jeff/dev_pilot-s3-2updates-smoke_k16_0002.jsonl"
    if trained_smoke.exists() and not valid_two_logits(trained_smoke):
        raise RuntimeError("Existing trained Pilot smoke is invalid; manual inspection required")

    print("[PILOT] Loading released Jeff-1.1-Large exactly once", flush=True)
    jeff = load_original_checkpoint(model_dir)
    attention_backend = getattr(jeff.decider, "attention_backend", "unrecorded")
    score_two_if_missing(rows, jeff.decider, original_smoke)
    if not summary.exists():
        print("[PILOT] Training syndrome S3 LoRA (2 updates, accumulation=64)", flush=True)
        result = train(train_args, released=jeff)
        if result.get("global_update") != PILOT_UPDATES:
            raise RuntimeError("The integrated Pilot did not finish 2 updates")
    checkpoint = verify_completed_training(root)

    # Original PEFT adapter is removed, without merging LoRA updates into the
    # released backbone. The saved adapter and head are read freshly from disk.
    if not hasattr(jeff.decider.base, "peft_config"):
        # This is the resume-after-training branch: base still holds release.
        restore_checkpoint(jeff.decider, checkpoint)
    else:
        reset_released_backbone(jeff.decider, model_dir)
        gc.collect()
        restore_checkpoint(jeff.decider, checkpoint)
    print("[PILOT] Fresh adapter and Head restored; 16-logit probe PASS", flush=True)
    score_two_if_missing(rows, jeff.decider, trained_smoke)

    report = {
        "status": "JEFF_SYNDROME_PILOT_ONE_LOAD_PASS",
        "model_revision": JEFF_MODEL_REVISION,
        "model_weight_loads": 1,
        "attention_backend": attention_backend,
        "official_train": 43180,
        "dev_smoke_rows": 2,
        "optimizer_steps": PILOT_UPDATES,
        "gradient_accumulation": ACCUMULATION,
        "checkpoint": str(checkpoint.relative_to(root)),
        "fresh_adapter_reload": True,
        "test_opened": False,
    }
    from tcmdiscern.h200.workflow import save_json

    save_json(root / "data/h200/pilot_one_load.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="One-load 31B syndrome LoRA Pilot")
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    run_one_load(args.model_dir)


if __name__ == "__main__":
    main()
