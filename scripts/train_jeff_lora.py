"""H200-only Jeff-1.1-Large released-Head + new-LoRA Step-0 safety gate.

This file does NOT start a full experiment. It loads the pinned released
Jeff 1.1 merged weights and trained Head with Jeff's official v1.3.0 runner,
then inserts a new zero-initialized LoRA and verifies that the raw Head logits
are unchanged. With --train-step it performs ONE gradient/optimizer update on
a Train-only K16 case, then saves and reloads LoRA+Head, requiring logit parity.

Do not run on the local RTX 4060; 31B bf16 weights need H200-class VRAM.
Install the official Jeff v1.3.0 stack and PEFT on H200 (see docs).

Examples:
  python -m scripts.train_jeff_lora --model-dir /models/Jeff-1.1-Large --method s3
  python -m scripts.train_jeff_lora --model-dir /models/Jeff-1.1-Large --method s3 --train-step
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.preflight_jeff_training import inspect_checkpoint
from tcmdiscern.jeff_config import JEFF_MAX_INPUT_TOKENS, JEFF_UPSTREAM_CODE_TAG
from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_losses import torch_loss

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
METHODS = ("s1", "s2", "s3", "s4")


def select_lora_target_modules(
    model, targets: tuple[str, ...] = LORA_TARGETS
) -> tuple[list[str], dict[str, int]]:
    """Select only the Gemma 4 TEXT decoder modules by exact qualified name.

    Gemma 4 31B has two independent weight-sharing rules:
    (1) global attention with attention_k_eq_v has no v_proj,
    (2) final shared-KV decoder layers have neither k_proj nor v_proj.
    The vision tower, which also has q/k/v/MLP module names, must not receive
    LoRA for these text-only diagnosis tasks.
    """
    language_model = getattr(model, "language_model", model)
    layer_prefix = "language_model.layers" if language_model is not model else "layers"
    layers = getattr(language_model, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Jeff backbone does not expose Gemma 4 text decoder layers")
    registered = dict(model.named_modules())
    names: list[str] = []
    counts = {name: 0 for name in targets}
    selected = set(targets)
    for index, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attn is None or mlp is None:
            raise RuntimeError(f"Gemma 4 decoder layer {index} is incomplete")
        for section, components, owner in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"), attn),
            ("mlp", ("gate_proj", "up_proj", "down_proj"), mlp),
        ):
            for component in components:
                if component not in selected:
                    continue
                qualified = f"{layer_prefix}.{index}.{section}.{component}"
                if qualified not in registered:
                    shared_kv = bool(getattr(attn, "is_kv_shared_layer", False))
                    if (
                        component in {"k_proj", "v_proj"}
                        and shared_kv
                        and getattr(attn, component, None) is None
                    ):
                        continue  # Last decoder layers reuse previous K/V states.
                    if (
                        component == "v_proj"
                        and bool(getattr(attn, "use_alternative_attention", False))
                        and getattr(attn, "v_proj", None) is None
                    ):
                        continue  # Global attention uses K == V.
                    raise RuntimeError(f"Missing required Gemma 4 text LoRA module: {qualified}")
                if registered[qualified] is not getattr(owner, component, None):
                    raise RuntimeError(f"Incorrect Gemma 4 LoRA target module: {qualified}")
                names.append(qualified)
                counts[component] += 1
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"No text decoder LoRA projections found: {counts}")
    for name, count in counts.items():
        if name not in {"k_proj", "v_proj"} and count != len(layers):
            raise RuntimeError(f"Incomplete Gemma 4 text LoRA coverage: {counts}")
    if counts["k_proj"] > len(layers) or counts["v_proj"] > counts["k_proj"]:
        raise RuntimeError(f"Invalid Gemma 4 shared-KV LoRA counts: {counts}")
    return names, counts


def validate_lora_targets(model, targets: tuple[str, ...] = LORA_TARGETS) -> dict[str, int]:
    """Validate the actual text decoder and conditional K==V weight tying."""
    _, counts = select_lora_target_modules(model, targets)
    return counts


def load_original_checkpoint(model_dir: Path):
    """Never download or initialize missing merged backbone / Head weights."""
    import torch
    from jeff import __version__ as upstream_version
    from jeff.model import JeffModel

    if upstream_version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise RuntimeError("Expected Jeff source exactly v1.3.0")
    if not torch.cuda.is_available():
        raise RuntimeError("Full Jeff-1.1-Large training requires a CUDA H200-class GPU")
    if torch.cuda.get_device_properties(0).total_memory < 80 * (2**30):
        raise RuntimeError("Expected a GPU with at least 80 GiB VRAM")
    checkpoint = inspect_checkpoint(model_dir)
    if not checkpoint["weights_present"] or not checkpoint["decision_head_present"]:
        raise FileNotFoundError("Missing original pinned Jeff merged backbone or head.safetensors")
    if checkpoint["release_metadata_matches"] is not True:
        raise ValueError("Released Jeff metadata does not match the pinned model")
    model = JeffModel(
        model_dir,
        device="cuda",
        reasoning=False,
        long_input="refuse",
        cudnn_attention=False,
        compile=False,
    )
    if model.config.get("bidirectional") is not None:
        raise ValueError("Jeff-1.1-Large must use its released causal attention")
    if model.decider.max_tokens != JEFF_MAX_INPUT_TOKENS:
        raise ValueError("Unexpected released Jeff context length")
    from tcmdiscern.jeff_attention import audit_released_jeff_attention

    audit_released_jeff_attention(model.decider)
    return model


def pick_train_example(root: Path, method: str) -> dict:
    """Read only the original Train-side saved questions, never Dev/Test labels."""
    if method not in METHODS:
        raise ValueError(f"Unknown loss method: {method}")
    file = (
        root / "data/jeff/train_oof_k16_epoch02.jsonl"
        if method == "s1"
        else root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
    )
    with file.open(encoding="utf-8") as stream:
        for line in stream:
            example = json.loads(line)
            # Exercise the actual multi-positive Set/Masked/Soft loss branch.
            if method == "s1" or len(example["positive_indices"]) > 1:
                return example
    raise ValueError("No valid training example found")


def original_logits(decider, prompt_ids: list[int]):
    """Exactly the released Jeff evaluation formula, without temperature changes."""
    import torch

    with torch.inference_mode():
        hidden = decider.hidden(prompt_ids)
        return decider.head(hidden.to(torch.bfloat16).float())[0, :16].float().clone()


def inject_lora_and_assert_step_zero(decider, prompt_ids, *, rank: int, alpha: int):
    from peft import LoraConfig, get_peft_model

    if rank < 1 or alpha < 1:
        raise ValueError("LoRA rank/alpha must be positive")
    baseline = original_logits(decider, prompt_ids)
    target_module_names, target_counts = select_lora_target_modules(decider.base)
    # Original 31B merged backbone stays frozen; do not inject the old
    # release adapter/ again (it is already merged in the HF weights).
    for parameter in decider.model.parameters():
        parameter.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_module_names,
        lora_dropout=0.0,
        bias="none",
    )
    adapter = get_peft_model(decider.base, config)
    decider.model.model = adapter
    decider.base = adapter
    decider.forward = adapter
    decider.model.eval()
    decider.head.eval()
    current = original_logits(decider, prompt_ids)
    delta = (baseline - current).abs()
    maximum = float(delta.max().item())
    if not math.isfinite(maximum) or maximum > 1e-3:
        raise RuntimeError(f"Zero-step LoRA altered official Jeff Head logits: max diff={maximum}")
    if int(baseline.argmax().item()) != int(current.argmax().item()):
        raise RuntimeError("Zero-step model changed the original decision")
    lora = [
        (name, parameter)
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    if not lora or any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in adapter.named_parameters()
    ):
        raise RuntimeError("Frozen backbone unexpectedly has other trainable parameters")
    trainable = sum(p.numel() for _, p in lora)
    if not trainable:
        raise RuntimeError("LoRA has no trainable adapter parameters")
    return adapter, baseline, maximum, target_counts, trainable


def perform_one_training_step(decider, adapter, example: dict, *, method: str, seed: int):
    import torch

    # Single example, but re-randomize all answer indices as in training.
    item = reshuffle_training_example(example, epoch=10, seed=seed)
    from jeff.prompting import encode_prompt

    ids = encode_prompt(decider.tokenizer, item["row"], JEFF_MAX_INPUT_TOKENS)
    if len(ids) > JEFF_MAX_INPUT_TOKENS or not ids:
        raise RuntimeError("Invalid Jeff train prompt length")

    decider.model.train()
    decider.head.train()
    # Non-reentrant checkpointing keeps gradients through the frozen backbone
    # to LoRA matrices. Fail rather than silently training with incorrect grads.
    decider.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    lora_parameters = [p for p in adapter.parameters() if p.requires_grad]
    head_parameters = list(decider.head.delta.parameters())
    if not all(p.requires_grad for p in head_parameters):
        raise RuntimeError("Original released Decision Head delta is not trainable")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": 2e-5},
            {"params": head_parameters, "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    input_ids = torch.tensor([ids], device=decider.device)
    # Same final post-norm hidden state and bf16->fp32 cast as official Decider.
    hidden = decider.forward(input_ids=input_ids, use_cache=False, return_dict=True)
    hidden = hidden.last_hidden_state[:, -1]
    logits = decider.head(hidden.to(torch.bfloat16).float())[0, :16]
    positives = item.get("positive_indices", [item["target_index"]])
    loss = torch_loss(
        logits,
        positive_indices=positives,
        gold_index=item["target_index"],
        method=method,
    )
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite Jeff Loss before backward")
    loss.backward()
    lora_gradient = sum(
        float(p.grad.detach().abs().sum().item()) for p in lora_parameters if p.grad is not None
    )
    head_gradient = sum(
        float(p.grad.detach().abs().sum().item()) for p in head_parameters if p.grad is not None
    )
    if not math.isfinite(lora_gradient) or lora_gradient <= 0:
        raise RuntimeError("No valid gradient reached trainable LoRA parameters")
    if not math.isfinite(head_gradient) or head_gradient <= 0:
        raise RuntimeError("No valid gradient reached the released Decision Head")
    torch.nn.utils.clip_grad_norm_(
        [*lora_parameters, *head_parameters], max_norm=1.0, error_if_nonfinite=True
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    decider.model.eval()
    decider.head.eval()
    return {
        "method": method,
        "loss": float(loss.item()),
        "train_prompt_tokens": len(ids),
        "positive_options": len(positives),
        "lora_gradient_l1": lora_gradient,
        "head_gradient_l1": head_gradient,
        "optimizer_steps": 1,
    }


def save_verify_reload(decider, adapter, example: dict, checkpoint: Path):
    """Save original LoRA+Head, alter them, reload, and assert logits recover."""
    import torch
    from jeff.prompting import encode_prompt
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file, save_file

    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint exists: {checkpoint}")
    checkpoint.mkdir(parents=True, exist_ok=False)
    ids = encode_prompt(decider.tokenizer, example["row"], JEFF_MAX_INPUT_TOKENS)
    before = original_logits(decider, ids)
    adapter.save_pretrained(checkpoint / "adapter", safe_serialization=True)
    head_file = checkpoint / "head.safetensors"
    save_file(
        {key: val.detach().cpu().contiguous() for key, val in decider.head.state_dict().items()},
        str(head_file),
    )
    weights = checkpoint / "adapter/adapter_model.safetensors"
    if not weights.exists():
        raise RuntimeError("PEFT adapter checkpoint did not write safetensors")
    # Perturb an adapter parameter and Head bias, then restore both from disk.
    with torch.no_grad():
        modified = next(
            p for name, p in adapter.named_parameters() if p.requires_grad and "lora_B" in name
        )
        modified.add_(0.125)
        decider.head.delta.bias.add_(0.5)
    changed = original_logits(decider, ids)
    if torch.allclose(changed, before, atol=1e-4, rtol=0):
        raise RuntimeError("Checkpoint corruption probe did not change Head logits")
    status = set_peft_model_state_dict(adapter, load_file(str(weights)), adapter_name="default")
    if getattr(status, "unexpected_keys", []):
        raise RuntimeError(f"Unexpected PEFT weights when reloading: {status}")
    decider.head.load_state_dict(load_file(str(head_file)), strict=True)
    recovered = original_logits(decider, ids)
    maximum = float((recovered - before).abs().max().item())
    if maximum > 1e-3:
        raise RuntimeError(f"LoRA+Head reload did not preserve logits: max diff={maximum}")
    return {
        "reload_max_logit_difference": maximum,
        "adapter_file_present": True,
        "head_file_present": True,
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="H200 Jeff released-Head LoRA Step-0/single-update gate"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="s3")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Actually perform ONE optimizer update and save/reload a checkpoint",
    )
    args = parser.parse_args()
    import torch
    from jeff.prompting import encode_prompt

    torch.manual_seed(args.seed)
    released = load_original_checkpoint(args.model_dir)
    decider = released.decider
    source = pick_train_example(root, args.method)
    original_item = reshuffle_training_example(source, epoch=0, seed=args.seed)
    ids = encode_prompt(decider.tokenizer, original_item["row"], JEFF_MAX_INPUT_TOKENS)
    adapter, baseline, parity, modules, count = inject_lora_and_assert_step_zero(
        decider, ids, rank=args.rank, alpha=args.alpha
    )
    report = {
        "status": "JEFF_LORA_STEP_ZERO_PARITY_PASS"
        if not args.train_step
        else "JEFF_LORA_ONE_OPTIMIZER_STEP_VERIFIED",
        "method": args.method,
        "train_only": True,
        "test_used": False,
        "checkpoint_model": "original merged Jeff 1.1 Large + original released Head",
        "new_adapter": f"LoRA rank={args.rank}, alpha={args.alpha}",
        "lora_target_module_counts": modules,
        "trainable_lora_parameters": count,
        "parity_max_abs_logit_difference": parity,
        "original_choice_position": int(baseline.argmax().item()),
        "model_training": False,
    }
    if args.train_step:
        report["one_step"] = perform_one_training_step(
            decider, adapter, source, method=args.method, seed=args.seed
        )
        checkpoint = root / "data/checkpoints" / f"jeff_{args.method}_step1_seed{args.seed}"
        report["reload_check"] = save_verify_reload(decider, adapter, source, checkpoint)
        report["model_training"] = "ONE_UPDATE_ONLY_NO_FULL_TRAINING"
        with (checkpoint / "smoke_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
