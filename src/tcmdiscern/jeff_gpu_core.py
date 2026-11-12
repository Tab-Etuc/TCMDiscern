"""Jeff-1.1-Large K=16 decision-head runtime primitives.

Heavy imports are optional and lazy. This module deliberately runs the decoder
stack, not the vocabulary language-model head. Official Jeff Decider reads
post-final-norm last_hidden_state[:, -1] then evaluates the published
DecisionHead on h.to(bfloat16).float(). Single-example forward avoids padding
and incorrect last-token indexing.

Only the H200 entrypoints should import the official jeff package/PEFT.
"""

from __future__ import annotations

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


def count_trainable_bytes(params: Iterable[Any]) -> int:
    return sum(p.numel() * p.element_size() for p in params)
