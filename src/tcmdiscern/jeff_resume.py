"""Transactional, adapter-only resume checkpoints for Jeff LoRA training.

The existing jeff_checkpoint.save_checkpoint writes inference-ready PEFT+Head
weights. This layer adds optimizer/scheduler/training cursor state in a
temporary directory and atomically publishes only completed update boundaries.

All artifacts remain under ignored data/runs/, never include patient text.
Only load checkpoints created by this project; torch.load uses weights_only.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .jeff_checkpoint import checkpoint_probe, save_checkpoint
from .jeff_config import JEFF_MAX_INPUT_TOKENS
from .jeff_gpu_core import decision_logits


def save_resume(
    run_dir: Path,
    *,
    state: dict,
    model_dir: Path,
    adapter,
    head,
    backbone,
    tokenizer,
    device,
    optimizer,
    scheduler,
) -> Path:
    import torch

    update = state["global_update"]
    if not isinstance(update, int) or update < 1:
        raise ValueError("Checkpoint requires a completed optimizer update")
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"resume-{update:07d}"
    if target.exists():
        raise FileExistsError(f"Frozen resume checkpoint exists: {target}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=run_dir) as dirname:
        work = Path(dirname)
        stage = work / "model"
        save_checkpoint(stage, model_dir, adapter, head, backbone, tokenizer, device, state)
        torch.save(optimizer.state_dict(), stage / "optimizer.pt")
        torch.save(scheduler.state_dict(), stage / "scheduler.pt")
        torch.save(
            {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            stage / "rng.pt",
        )
        (stage / "training_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(stage, target)
    return target


def verify_resume_settings(state: dict, expected: dict) -> None:
    """CPU-verifiable guard against changes to optimizer or micro-batch protocol."""
    if not isinstance(state, dict) or not isinstance(expected, dict):
        raise TypeError("Expected JSON checkpoint state and frozen settings")
    for key, required in expected.items():
        if state.get(key) != required:
            raise ValueError(f"Resume checkpoint does not match fixed {key}")
    if state.get("micro_batch_size", 1) != expected.get("micro_batch_size", 1):
        raise ValueError("Resume checkpoint micro_batch_size differs from requested run")


def restore_resume(
    directory: Path,
    *,
    expected: dict,
    adapter,
    head,
    backbone,
    tokenizer,
    device,
    optimizer,
    scheduler,
) -> dict:
    import numpy as np
    import torch
    from jeff.prompting import encode_prompt
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    if not directory.is_dir() or not (directory / "training_state.json").is_file():
        raise FileNotFoundError("Missing training-resume metadata")
    state = json.loads((directory / "training_state.json").read_text(encoding="utf-8"))
    verify_resume_settings(state, expected)
    if not isinstance(state.get("global_update"), int) or state["global_update"] < 1:
        raise ValueError("Invalid resume optimizer step")
    for key in ("cursor_epoch", "cursor_offset"):
        if not isinstance(state.get(key), int) or state[key] < 0:
            raise ValueError(f"Invalid resume training cursor {key}")
    adapter_weights = directory / "adapter/adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError("Missing saved adapter")
    response = set_peft_model_state_dict(
        adapter, load_file(str(adapter_weights)), adapter_name="default"
    )
    if getattr(response, "unexpected_keys", []):
        raise ValueError("Unexpected PEFT keys when reloading adapter")
    head.load_state_dict(load_file(str(directory / "head.safetensors")), strict=True)
    optimizer.load_state_dict(
        torch.load(directory / "optimizer.pt", weights_only=True, map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(directory / "scheduler.pt", weights_only=True, map_location="cpu")
    )
    rng = torch.load(directory / "rng.pt", weights_only=True, map_location="cpu")
    if not isinstance(rng.get("torch_cpu"), torch.Tensor):
        raise TypeError("Saved Torch CPU RNG state is invalid")
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available():
        if len(rng.get("torch_cuda", [])) != torch.cuda.device_count():
            raise ValueError("GPU count changed; cannot guarantee RNG-consistent resume")
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    # Saved synthetic prompt is fully independent of training/Dev/Test patients.
    probe = checkpoint_probe()
    ids = encode_prompt(tokenizer, probe, JEFF_MAX_INPUT_TOKENS)
    adapter.eval()
    head.eval()
    with torch.inference_mode():
        logits = decision_logits(backbone, head, ids, device=device)
    previous = json.loads((directory / "probe_logits.json").read_text(encoding="utf-8"))
    expected_logits = np.asarray(previous["raw_logits"], dtype=np.float32)
    measured = logits.float().cpu().numpy()
    if measured.shape != (16,) or not np.allclose(expected_logits, measured, atol=1e-3, rtol=1e-4):
        raise RuntimeError("Resume checkpoint altered original saved decision logits")
    return state
