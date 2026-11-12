"""Reproducible Jeff K=16 training with explicit H200 safeguards.

Full official 43,180 Train is required, but only one optimizer step is the
CLI default. No training is started when imported or when inspecting this file.
The released merged model stays frozen; only new LoRA and original Head delta
are optimized. Each epoch reshuffles A..P together with every target.

Resume snapshots are written at completed optimizer boundaries, including
PEFT adapter, Head, optimizer, scheduler, torch RNG and cursor. All outputs
remain under git-ignored data/runs/. Dev/Test are never opened here.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

from .jeff_config import JEFF_MAX_INPUT_TOKENS, JEFF_MODEL_REVISION
from .jeff_gpu_core import (
    decision_logits,
    gradient_audit,
    parameter_audit,
    supervised_loss,
    trainable_parameters,
)
from .jeff_resume import restore_resume, save_resume
from .jeff_schedule import (
    epoch_batches,
    load_frozen_train,
    total_optimizer_steps,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = {
    "s1": "train_oof_k16_epoch02.jsonl",
    "s2": "train_oof_multipos_k16_epoch03.jsonl",
    "s3": "train_oof_multipos_k16_epoch03.jsonl",
    "s4": "train_oof_multipos_k16_epoch03.jsonl",
}


def load_train(method: str) -> list[dict]:
    """Load original Train only, and check all 43,180 rows before any GPU use."""
    if method not in TRAIN_FILE:
        raise ValueError("Unknown Jeff Loss")
    return load_frozen_train(
        ROOT / "data/jeff" / TRAIN_FILE[method],
        multi=method != "s1",
        expected_count=43180,
    )


def fixed_run_settings(args: Any) -> dict:
    """Only these exactly matching settings may be resumed together."""
    return {
        "run_id": args.run_id,
        "loss_method": args.method,
        "model_revision": JEFF_MODEL_REVISION,
        "source": TRAIN_FILE[args.method],
        "official_train_case_count": 43180,
        "seed": args.seed,
        "rank": args.rank,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "target_updates": args.steps,
        "accumulation": args.accumulation,
        "lora_lr": args.lora_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "clip_norm": args.clip_norm,
        "warmup_steps": args.warmup_steps,
    }


def validate_run_args(args: Any) -> Path:
    """Check inputs before loading a 31B checkpoint or allocating GPU memory."""
    if args.method not in TRAIN_FILE:
        raise ValueError("Unknown fine-tuning method")
    if not isinstance(args.run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("Use an ASCII alphanumeric run_id with optional dash/underscore")
    for key in ("steps", "epochs", "accumulation", "rank", "alpha"):
        value = getattr(args, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "warmup_steps", "checkpoint_every"):
        value = getattr(args, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if args.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    for key in ("lora_lr", "head_lr", "clip_norm"):
        v = getattr(args, key)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if args.warmup_steps >= args.steps:
        raise ValueError("warmup_steps must be less than planned optimizer steps")
    possible = total_optimizer_steps(43180, args.accumulation, args.epochs)
    if args.steps > possible:
        raise ValueError(
            f"Requested {args.steps} steps but only {possible} complete updates "
            "exist in the selected number of epochs"
        )
    run_dir = ROOT / "data/runs" / args.run_id
    if args.resume is None:
        if run_dir.exists():
            raise FileExistsError("Run folder exists; specify --resume or a new run ID")
    else:
        saved = args.resume.expanduser().resolve(strict=True)
        if not run_dir.is_dir() or saved.parent != run_dir.resolve():
            raise ValueError("Resume must refer to a checkpoint inside the same run_id folder")
        if not saved.name.startswith("resume-"):
            raise ValueError("Invalid resume checkpoint name")
    return run_dir


def optimizer_schedule(optimizer, *, warmup_steps: int):
    """Deterministic linear warmup; constant thereafter. State is checkpointed."""
    import torch

    def multiplier(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _new_progress(settings: dict) -> dict:
    return {
        **settings,
        "global_update": 0,
        "cursor_epoch": 0,
        "cursor_offset": 0,
        "microbatches": 0,
        "input_tokens": 0,
        "sum_microbatch_loss": 0.0,
    }


def iter_update_groups(
    records: list[dict],
    *,
    epoch: int,
    seed: int,
    accumulation: int,
    start_offset: int,
):
    """Public testable schedule: no partial gradient accumulation is lost."""
    return epoch_batches(
        records,
        epoch=epoch,
        seed=seed,
        accumulation=accumulation,
        start_offset=start_offset,
    )


def run(args: Any) -> dict:
    """Execute explicitly on H200; requires the pinned release on local disk."""
    run_dir = validate_run_args(args)

    import torch
    from jeff.prompting import encode_prompt

    from scripts.train_jeff_lora import (
        inject_lora_and_assert_step_zero,
        load_original_checkpoint,
    )

    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.get_device_properties(0).total_memory < 80 * (2**30)
    ):
        raise RuntimeError("Jeff-1.1-Large training requires an H200-class CUDA BF16 GPU")

    rows = load_train(args.method)
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    began = time.perf_counter()

    released = load_original_checkpoint(model_dir)
    decider = released.decider
    ids0 = encode_prompt(decider.tokenizer, rows[0]["row"], JEFF_MAX_INPUT_TOKENS)
    adapter, _, initial_delta, modules, n_trainable = inject_lora_and_assert_step_zero(
        decider, ids0, rank=args.rank, alpha=args.alpha
    )
    report_parameters = parameter_audit(adapter, decider.head)
    lora_params, head_params = trainable_parameters(adapter, decider.head)
    decider.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    decider.model.enable_input_require_grads()
    decider.model.config.use_cache = False

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optimizer_schedule(optimizer, warmup_steps=args.warmup_steps)
    settings = fixed_run_settings(args)
    progress = _new_progress(settings)
    if args.resume is not None:
        progress = restore_resume(
            args.resume.expanduser().resolve(strict=True),
            expected=settings,
            adapter=adapter,
            head=decider.head,
            backbone=decider.forward,
            tokenizer=decider.tokenizer,
            device=decider.device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if progress["global_update"] >= args.steps:
            raise ValueError("Selected checkpoint already reached the planned update budget")
        if progress["cursor_epoch"] >= args.epochs:
            raise ValueError("Selected checkpoint is already past the final epoch")

    # Resume validation is finished. Only now may we create a new run directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    gradients = None
    most_recent_checkpoint = None
    initial_epoch = progress["cursor_epoch"]
    initial_offset = progress["cursor_offset"]
    for epoch in range(initial_epoch, args.epochs):
        offset = initial_offset if epoch == initial_epoch else 0
        for next_offset, batch in iter_update_groups(
            rows,
            epoch=epoch,
            seed=args.seed,
            accumulation=args.accumulation,
            start_offset=offset,
        ):
            # The last update in an epoch can be shorter than accumulation.
            # Divide by actual size to avoid silently underweighting those rows.
            divisor = len(batch)
            optimizer.zero_grad(set_to_none=True)
            decider.model.train()
            decider.head.train()
            for item in batch:
                tokens = encode_prompt(decider.tokenizer, item["row"], JEFF_MAX_INPUT_TOKENS)
                logits = decision_logits(
                    decider.forward, decider.head, tokens, device=decider.device
                )
                loss = supervised_loss(logits, item, args.method)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Jeff Head returned a non-finite supervised Loss")
                (loss / divisor).backward()
                progress["microbatches"] += 1
                progress["input_tokens"] += len(tokens)
                progress["sum_microbatch_loss"] += float(loss.detach().item())

            if gradients is None:
                gradients = gradient_audit(adapter, decider.head)
            torch.nn.utils.clip_grad_norm_(
                lora_params + head_params,
                max_norm=args.clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            progress["global_update"] += 1
            if next_offset == len(rows):
                progress["cursor_epoch"] = epoch + 1
                progress["cursor_offset"] = 0
            else:
                progress["cursor_epoch"] = epoch
                progress["cursor_offset"] = next_offset

            if (
                progress["global_update"] % args.checkpoint_every == 0
                or progress["global_update"] == args.steps
            ):
                # save_resume() includes optimizer, scheduler, both trainable
                # modules and the *next* complete update boundary.
                most_recent_checkpoint = save_resume(
                    run_dir,
                    state=progress,
                    model_dir=model_dir,
                    adapter=adapter,
                    head=decider.head,
                    backbone=decider.forward,
                    tokenizer=decider.tokenizer,
                    device=decider.device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            if progress["global_update"] >= args.steps:
                break
        if progress["global_update"] >= args.steps:
            break

    if progress["global_update"] != args.steps or most_recent_checkpoint is None:
        raise RuntimeError("Requested optimizer budget was not reached")
    report = {
        "status": "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
        **progress,
        "source_count_verified": len(rows),
        "model_pretrained_head_reinitialized": False,
        "original_model_revision": JEFF_MODEL_REVISION,
        "step0_max_logit_abs_difference": initial_delta,
        "lora_target_modules": modules,
        "lora_trainable_parameters": n_trainable,
        "trainable_parameters": report_parameters,
        "first_gradient": gradients,
        "mean_training_loss": (progress["sum_microbatch_loss"] / progress["microbatches"]),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.perf_counter() - began,
        "final_checkpoint": str(most_recent_checkpoint.relative_to(ROOT)),
        "optimizer_and_scheduler_saved": True,
        "train_gold_outside_prompts": True,
        "dev_test_used": False,
    }
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report
