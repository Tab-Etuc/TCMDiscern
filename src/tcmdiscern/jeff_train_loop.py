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
    capture_first_update,
    decision_logits,
    gradient_audit,
    parameter_audit,
    supervised_loss,
    trainable_parameters,
    verify_first_update,
)
from .jeff_resume import restore_resume, save_resume
from .jeff_schedule import (
    checkpoint_due,
    epoch_batches,
    load_frozen_train,
    total_optimizer_steps,
)
from .jeff_weighting import WEIGHTING_CHOICES, load_train_oof_weights, normalized_batch_weights

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = {
    # The S1 single-gold CE control uses exactly the same sixteen option
    # identities as S2/S3/S4. Otherwise 820/43,180 training candidate sets
    # differ, confounding a paper comparison of supervision objectives.
    "s1": "train_oof_multipos_k16_epoch03.jsonl",
    "s2": "train_oof_multipos_k16_epoch03.jsonl",
    "s3": "train_oof_multipos_k16_epoch03.jsonl",
    "s4": "train_oof_multipos_k16_epoch03.jsonl",
}
STUDY_FILE = {
    "global": "train_global_k5.jsonl",
    "conditioned": "train_conditioned_k5.jsonl",
    "pair": "train_pair_k5.jsonl",
    "disease16": "train_disease_k16.jsonl",
    "pair16": "train_pair_k16.jsonl",
}


def load_train(method: str, *, study_mode: str | None = None) -> list[dict]:
    """Validate all 43,180 official Train rows before allocating GPU memory."""
    if method not in TRAIN_FILE:
        raise ValueError("Unknown Jeff Loss")
    if study_mode is not None:
        if study_mode not in STUDY_FILE or method != "s1":
            raise ValueError("Disease study requires a registered mode and single-label CE")
        candidate_width = 16 if study_mode in {"disease16", "pair16"} else 5
        folder = "joint16" if candidate_width == 16 else "disease_study"
        return load_frozen_train(
            ROOT / "data" / folder / STUDY_FILE[study_mode],
            multi=False,
            expected_count=43180,
            expected_k=candidate_width,
            study_mode=study_mode,
        )
    # Always validate the same multi-positive Train candidate source.
    # The S1 ablation removes only external multi-label supervision; both
    # methods receive identical model-visible question and candidate options.
    shared = load_frozen_train(
        ROOT / "data/jeff" / TRAIN_FILE[method],
        multi=True,
        expected_count=43180,
    )
    if method == "s1":
        return [
            {
                "row": item["row"],
                "target_index": item["target_index"],
                "target_letter": item["target_letter"],
            }
            for item in shared
        ]
    return shared


def fixed_run_settings(args: Any) -> dict:
    """Only these exactly matching settings may be resumed together."""
    study_mode = getattr(args, "study_mode", None)
    source = STUDY_FILE[study_mode] if study_mode is not None else TRAIN_FILE[args.method]
    settings = {
        "run_id": args.run_id,
        "loss_method": args.method,
        "model_revision": JEFF_MODEL_REVISION,
        "source": source,
        "example_weighting": getattr(args, "weighting", "uniform"),
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
        "checkpoint_every": getattr(args, "checkpoint_every", 100),
    }
    if study_mode is not None:
        settings.update(
            {
                "study_mode": study_mode,
                "study_k": 16 if study_mode in {"disease16", "pair16"} else 5,
                "pairwise_weight": getattr(args, "pairwise_weight", 0.0),
                "pairwise_margin": getattr(args, "pairwise_margin", 0.2),
            }
        )
    return settings


def validate_run_args(args: Any) -> Path:
    """Check inputs before loading a 31B checkpoint or allocating GPU memory."""
    if args.method not in TRAIN_FILE:
        raise ValueError("Unknown fine-tuning method")
    study_mode = getattr(args, "study_mode", None)
    pairwise_weight = getattr(args, "pairwise_weight", 0.0)
    pairwise_margin = getattr(args, "pairwise_margin", 0.2)
    if study_mode is not None and (study_mode not in STUDY_FILE or args.method != "s1"):
        raise ValueError("Invalid disease-study training task")
    if not math.isfinite(pairwise_weight) or pairwise_weight < 0:
        raise ValueError("pairwise_weight must be nonnegative and finite")
    if not math.isfinite(pairwise_margin) or pairwise_margin < 0:
        raise ValueError("pairwise_margin must be nonnegative and finite")
    if pairwise_weight and study_mode != "conditioned":
        raise ValueError("Pairwise hard negatives require disease-conditioned training")
    if getattr(args, "weighting", "uniform") not in WEIGHTING_CHOICES:
        raise ValueError("Unknown example-weighting method")
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


def expected_case_exposures(rows: int, accumulation: int, updates: int) -> int:
    """Exact number of Train rows consumed for an optimizer-step budget."""
    if rows <= 0 or accumulation <= 0 or updates <= 0:
        raise ValueError("Training exposure accounting requires positive integers")
    per_epoch = math.ceil(rows / accumulation)
    complete_epochs, remaining_updates = divmod(updates, per_epoch)
    return complete_epochs * rows + min(rows, remaining_updates * accumulation)


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


def run(args: Any, *, released: Any | None = None) -> dict:
    """Train a registered run, optionally reusing one already-validated Jeff model.

    A preloaded released model permits the integrated H200 Pilot to reuse its
    single expensive 31B read, while keeping identical gradient/checkpoint
    logic and exact CLI hyperparameters.
    """
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

    study_mode = getattr(args, "study_mode", None)
    rows = load_train(args.method, study_mode=study_mode)
    weighting_scheme = getattr(args, "weighting", "uniform")
    oof_weights = None
    weighting_report = {"policy": "uniform", "original_rows_removed": 0}
    if weighting_scheme != "uniform":
        oof_weights, weighting_report = load_train_oof_weights(
            rows, source=ROOT / "data/candidates/bm25_train_text_grouped.jsonl"
        )
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    began = time.perf_counter()

    if released is None:
        released = load_original_checkpoint(model_dir)
    else:
        # Refuse accidental training on a previous Pilot's adapter or altered
        # head. The caller must restore the original release first.
        if hasattr(released.decider.base, "peft_config"):
            raise ValueError("Preloaded Jeff model already has an active LoRA adapter")
        if released.decider.base is not released.decider.model.model:
            raise ValueError("Preloaded Jeff decoder does not match its original backbone")
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
    first_parameter_update = None
    most_recent_checkpoint = None
    session_started = time.perf_counter()
    updates_before_session = progress["global_update"]
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
            scales = normalized_batch_weights(
                batch, scheme=weighting_scheme, oof_weights=oof_weights
            )
            optimizer.zero_grad(set_to_none=True)
            decider.model.train()
            decider.head.train()
            for item, scale in zip(batch, scales, strict=True):
                tokens = encode_prompt(decider.tokenizer, item["row"], JEFF_MAX_INPUT_TOKENS)
                logits = decision_logits(
                    decider.forward, decider.head, tokens, device=decider.device
                )
                if study_mode is None:
                    loss = supervised_loss(logits, item, args.method)
                elif study_mode in {"disease16", "pair16"}:
                    loss = supervised_loss(logits, item, "s1")
                else:
                    from .disease_study import study_torch_loss

                    loss = study_torch_loss(
                        logits[:5],
                        item,
                        pairwise_weight=getattr(args, "pairwise_weight", 0.0),
                        margin=getattr(args, "pairwise_margin", 0.2),
                    )
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Jeff Head returned a non-finite supervised Loss")
                (loss * scale).backward()
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
            # Verify an actual LoRA-B and Head weight change after step one.
            # A finite loss/gradient alone is insufficient evidence of training.
            update_probe = (
                capture_first_update(adapter, decider.head)
                if first_parameter_update is None
                else None
            )
            optimizer.step()
            if update_probe is not None:
                first_parameter_update = verify_first_update(update_probe)
                print(
                    f"[TRAIN] First optimizer step changed LoRA and Head: {first_parameter_update}",
                    flush=True,
                )
            scheduler.step()
            progress["global_update"] += 1
            update = progress["global_update"]
            if update == 1 or update % 10 == 0 or update == args.steps:
                elapsed = time.perf_counter() - session_started
                completed_here = max(1, update - updates_before_session)
                remaining = int((args.steps - update) * elapsed / completed_here)
                print(
                    f"[TRAIN] {args.run_id}: {update}/{args.steps} updates "
                    f"({update / args.steps:.1%}), "
                    f"last_loss={float(loss.detach().item()):.5f}, "
                    f"elapsed={elapsed / 60:.1f}m, "
                    f"estimated_remaining={remaining / 60:.1f}m",
                    flush=True,
                )
            if next_offset == len(rows):
                progress["cursor_epoch"] = epoch + 1
                progress["cursor_offset"] = 0
            else:
                progress["cursor_epoch"] = epoch
                progress["cursor_offset"] = next_offset

            if checkpoint_due(
                update=progress["global_update"],
                cadence=args.checkpoint_every,
                at_epoch_end=next_offset == len(rows),
                target_updates=args.steps,
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
    expected_exposures = expected_case_exposures(len(rows), args.accumulation, args.steps)
    if progress["microbatches"] != expected_exposures:
        raise RuntimeError(
            f"Training exposed {progress['microbatches']} cases, expected "
            f"{expected_exposures}; refuse an incomplete or duplicated epoch"
        )
    if args.steps == total_optimizer_steps(len(rows), args.accumulation, args.epochs) and (
        progress["cursor_epoch"] != args.epochs or progress["cursor_offset"] != 0
    ):
        raise RuntimeError("Full epochs are incomplete despite reaching optimizer budget")
    report = {
        "status": "JEFF_LORA_CHECKPOINT_SAVED_DEV_EVALUATION_PENDING",
        **progress,
        "source_count_verified": len(rows),
        "expected_case_exposures": expected_exposures,
        "study_mode": study_mode,
        "model_pretrained_head_reinitialized": False,
        "original_model_revision": JEFF_MODEL_REVISION,
        "step0_max_logit_abs_difference": initial_delta,
        "lora_target_modules": modules,
        "lora_trainable_parameters": n_trainable,
        "trainable_parameters": report_parameters,
        "first_gradient": gradients,
        "first_optimizer_update_verified": first_parameter_update,
        "mean_training_loss": (progress["sum_microbatch_loss"] / progress["microbatches"]),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "attention_backend": getattr(decider, "attention_backend", "unrecorded"),
        "elapsed_seconds": time.perf_counter() - began,
        "final_checkpoint": str(most_recent_checkpoint.relative_to(ROOT)),
        "optimizer_and_scheduler_saved": True,
        "train_gold_outside_prompts": True,
        "weighting_audit": weighting_report,
        "dev_test_used": False,
    }
    with (run_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report
