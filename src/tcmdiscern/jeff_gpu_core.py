"""Jeff-1.1-Large K=16 decision-head runtime primitives.

Heavy imports are optional and lazy. This module deliberately runs the decoder
stack, not the vocabulary language-model head. Official Jeff Decider reads
post-final-norm last_hidden_state and evaluates the published DecisionHead
on h.to(bfloat16).float(). For throughput experiments, use the upstream
Jeff Decider.hidden_batch implementation: it right-pads and gathers each
example's last valid token instead of indexing the padding tail.

Only the H200 entrypoints should import the official jeff package/PEFT.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from .jeff_config import MAX_DIRECT_JEFF_K
from .jeff_losses import torch_loss


def decision_logits(backbone: Any, head: Any, token_ids: Sequence[int], *, device: Any):
    """Compute raw 16-option logits from the official decoder and Head."""
    import torch

    if not token_ids:
        raise ValueError("Empty Jeff prompt token list")
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    out = backbone(
        input_ids=ids,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
    )
    hidden = out.last_hidden_state
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != len(token_ids):
        raise ValueError("Official backbone did not return [1, sequence, hidden]")
    logits = head(hidden[:, -1].to(torch.bfloat16).float())
    if logits.shape != (1, MAX_DIRECT_JEFF_K):
        raise ValueError("Decision Head must return exactly 16 slot logits")
    return logits[0]


def decision_logits_batch(decider: Any, token_batches: Sequence[Sequence[int]]):
    """Official Jeff v1.3 batch path, with the same 16-slot Head as single scoring.

    Uses Decider.hidden_batch, which right-pads, masks the padding and gathers
    each example's *actual* final token; the single-item branch preserves the
    original exact unpadded decision forward for numerical comparisons.
    """
    import torch

    if not token_batches or any(not tokens for tokens in token_batches):
        raise ValueError("Jeff batch must have nonempty token sequences")
    if len(token_batches) == 1:
        return decision_logits(
            decider.forward, decider.head, token_batches[0], device=decider.device
        ).unsqueeze(0)
    hidden = decider.hidden_batch([list(tokens) for tokens in token_batches])
    if hidden.ndim != 2 or hidden.shape[0] != len(token_batches):
        raise ValueError("Upstream Jeff hidden_batch did not return [batch, hidden]")
    logits = decider.head(hidden.to(torch.bfloat16).float())
    if logits.shape != (len(token_batches), MAX_DIRECT_JEFF_K):
        raise ValueError("Batched Jeff Head must return [batch, 16] logits")
    return logits


def verify_official_batched_logits(
    decider: Any,
    token_sequences: Sequence[Sequence[int]],
    *,
    atol: float = 0.1,
    rtol: float = 0.05,
) -> dict:
    """Block accelerated training if Jeff's padded path changes the decisions.

    Use two real prompts with different lengths, compare all 16 slots to the
    published single-example forward and preserve the actual parity evidence.
    The upstream Jeff authors explicitly support this hidden_batch path.
    """
    import torch

    if len(token_sequences) < 2:
        raise ValueError("Batch parity requires at least two examples")
    was_training = bool(decider.model.training)
    decider.model.eval()
    decider.head.eval()
    try:
        with torch.inference_mode():
            selected = token_sequences[:2]
            reference = torch.stack(
                [
                    decision_logits(decider.forward, decider.head, tokens, device=decider.device)
                    for tokens in selected
                ]
            )
            batched = decision_logits_batch(decider, selected)
            difference = (batched.float() - reference.float()).abs()
            if (
                not torch.isfinite(reference).all().item()
                or not torch.isfinite(batched).all().item()
            ):
                raise RuntimeError("Batched Jeff decisions contain non-finite logits")
            max_abs = float(difference.max().item())
            if not torch.allclose(batched, reference, atol=atol, rtol=rtol):
                raise RuntimeError(
                    "Padded Jeff hidden_batch changed 16-slot decision logits "
                    f"beyond tolerance (max_abs={max_abs:.6g}); "
                    "retain micro_batch_size=1 until investigated"
                )
            return {
                "two_real_examples": True,
                "input_lengths": [len(tokens) for tokens in selected],
                "max_abs_logit_difference": max_abs,
                "all_16_logits_within_tolerance": True,
                "atol": atol,
                "rtol": rtol,
            }
    finally:
        if was_training:
            decider.model.train()


def parameter_audit(adapter_model: Any, head: Any) -> dict:
    """Fail closed if backbone/vocab head or wrong decision-head weights are trainable."""
    adapter_trainable = [
        (name, parameter)
        for name, parameter in adapter_model.named_parameters()
        if parameter.requires_grad
    ]
    if not adapter_trainable:
        raise ValueError("LoRA adapter has no trainable parameters")
    if any("lora_" not in name.lower() for name, _ in adapter_trainable):
        raise ValueError("Non-LoRA backbone weights were accidentally made trainable")
    head_trainable = [
        (name, parameter) for name, parameter in head.named_parameters() if parameter.requires_grad
    ]
    if {name for name, _ in head_trainable} != {"delta.weight", "delta.bias"}:
        raise ValueError("Only the released Head delta weight/bias should be optimized")
    for name in ("base_rows", "mu", "sigma"):
        if name not in dict(head.named_buffers()):
            raise ValueError(f"Missing original Jeff Head buffer: {name}")
    if not all(parameter.dtype.is_floating_point for _, parameter in head_trainable):
        raise ValueError("Jeff Head correction must be floating-point")
    return {
        "trainable_adapter_parameters": sum(p.numel() for _, p in adapter_trainable),
        "trainable_head_parameters": sum(p.numel() for _, p in head_trainable),
        "trainable_adapter_tensors": len(adapter_trainable),
        "trainable_head_tensors": [name for name, _ in head_trainable],
        "frozen_backbone": True,
        "published_head_buffers_preserved": True,
    }


def trainable_parameters(adapter_model: Any, head: Any) -> tuple[list[Any], list[Any]]:
    report = parameter_audit(adapter_model, head)
    if report["trainable_adapter_parameters"] <= 0:
        raise ValueError("LoRA parameter count must be positive")
    return (
        [p for p in adapter_model.parameters() if p.requires_grad],
        [p for p in head.parameters() if p.requires_grad],
    )


def validate_step0(
    official_logits: Sequence[float],
    injected_logits: Any,
    *,
    atol: float = 0.001,
    rtol: float = 0.0001,
) -> dict:
    """New LoRA starts at zero and must preserve every released Head logit."""
    import torch

    expected = torch.as_tensor(
        list(official_logits), dtype=torch.float32, device=injected_logits.device
    )
    actual = injected_logits.detach().float()
    if actual.shape != (16,) or not torch.isfinite(actual).all():
        raise ValueError("Injected model did not return 16 finite decision logits")
    diff = (actual - expected).abs()
    largest = float(diff.max().item())
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(
            "Jeff Step-0 logits differ after fresh LoRA injection "
            f"(max_abs={largest:.6g}); abort fine-tuning"
        )
    if int(actual.argmax().item()) != int(expected.argmax().item()):
        raise ValueError("Jeff initial top-1 decision changed before training")
    return {
        "comparison": "official Jeff Decider versus fresh LoRA+original Head",
        "all_16_logits_match": True,
        "top1_unchanged": True,
        "max_abs_logit_difference": largest,
        "atol": atol,
        "rtol": rtol,
    }


def supervised_loss(logits: Any, example: dict, method: str):
    """Use *outside-prompt* gold indices, never language-generation labels."""
    gold_index = example["target_index"]
    positives = [gold_index] if method == "s1" else example["positive_indices"]
    return torch_loss(
        logits,
        positive_indices=positives,
        gold_index=gold_index,
        method=method,
    )


def gradient_audit(adapter_model: Any, head: Any) -> dict:
    """First LoRA A gradient may be zero because B initializes to zero."""
    import torch

    adapter = [
        (name, p)
        for name, p in adapter_model.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    correction = [(name, p) for name, p in head.named_parameters() if p.requires_grad]
    if not adapter or not correction:
        raise ValueError("No LoRA or Jeff Head gradients were produced")
    if any(
        not torch.isfinite(p.grad).all().item()
        for _, p in adapter + [(name, p) for name, p in correction if p.grad is not None]
    ):
        raise ValueError("Non-finite adapter or Decision Head gradient")
    has_adapter = any(p.grad.detach().abs().max().item() > 0 for _, p in adapter)
    has_head = all(
        p.grad is not None and p.grad.detach().abs().max().item() > 0 for _, p in correction
    )
    if not has_adapter or not has_head:
        raise ValueError("Adapter or released Decision Head received zero gradients")
    return {
        "adapter_gradients_present": True,
        "head_delta_gradients_present": True,
        "adapter_tensors_with_grad": len(adapter),
        "all_gradients_finite": True,
    }


def capture_first_update(adapter_model: Any, head: Any) -> dict:
    """Capture real tensors *after backward* and before the first optimizer step.

    A valid nonzero gradient alone does not prove that an Optimizer actually
    updated the LoRA matrices or the released Decision Head. Save one active
    LoRA-B matrix and the complete two Head correction tensors on device.
    """
    import torch

    lora_b = [
        (name, parameter)
        for name, parameter in adapter_model.named_parameters()
        if parameter.requires_grad
        and "lora_B" in name
        and parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all().item())
        and bool(torch.count_nonzero(parameter.grad).item())
    ]
    if not lora_b:
        raise RuntimeError("No active LoRA-B gradient before the first optimizer update")
    head_params = dict(head.named_parameters())
    names = ("delta.weight", "delta.bias")
    if any(
        name not in head_params
        or head_params[name].grad is None
        or not bool(torch.isfinite(head_params[name].grad).all().item())
        or not bool(torch.count_nonzero(head_params[name].grad).item())
        for name in names
    ):
        raise RuntimeError("Decision Head correction has no valid first-step gradient")
    key, active = lora_b[0]
    return {
        "lora_B": (key, active, active.detach().clone()),
        "head_delta_weight": (
            "delta.weight",
            head_params["delta.weight"],
            head_params["delta.weight"].detach().clone(),
        ),
        "head_delta_bias": (
            "delta.bias",
            head_params["delta.bias"],
            head_params["delta.bias"].detach().clone(),
        ),
    }


def validate_update_magnitudes(updates: dict[str, float]) -> dict[str, float]:
    """Machine-checkable evidence of a real optimizer update."""
    required = ("lora_B", "head_delta_weight", "head_delta_bias")
    if set(updates) != set(required) or any(
        type(updates[name]) not in (float, int)
        or not math.isfinite(updates[name])
        or updates[name] <= 0
        for name in required
    ):
        raise RuntimeError("Optimizer failed to change both LoRA and Head parameters")
    return updates


def verify_first_update(snapshot: dict) -> dict[str, float]:
    """Fail the run immediately if any required trainable tensor stayed fixed."""
    import torch

    updates: dict[str, float] = {}
    for kind in ("lora_B", "head_delta_weight", "head_delta_bias"):
        name, parameter, before = snapshot[kind]
        after = parameter.detach()
        if after.shape != before.shape or not torch.isfinite(after).all().item():
            raise RuntimeError(f"First optimizer update damaged {kind} ({name})")
        difference = float((after.float() - before.float()).abs().max().item())
        if not math.isfinite(difference) or difference <= 0:
            raise RuntimeError(f"First optimizer step did not update {kind} ({name})")
        updates[kind] = difference
    return validate_update_magnitudes(updates)


def count_trainable_bytes(params: Iterable[Any]) -> int:
    return sum(p.numel() * p.element_size() for p in params)
