"""H200 preflight for unexpected noncausal Gemma 4 attention behavior.

Compare hidden states for identical prefix tokens with and without a suffix.
A properly causal decoder cannot let later tokens alter earlier states. The
check uses a synthetic fixed prompt, never patient data. This directly guards
the attention-backend failure mode described in newer decision-model work.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compare_causal_prefix(
    prefix_states: np.ndarray,
    extended_states: np.ndarray,
    *,
    atol: float = 0.1,
    rtol: float = 0.05,
) -> dict:
    """Validate all prefix hidden positions; allow BF16/kernel numeric noise."""
    first = np.asarray(prefix_states, dtype=np.float32)
    longer = np.asarray(extended_states, dtype=np.float32)
    if first.ndim != 3 or longer.ndim != 3 or first.shape[0] != 1 or longer.shape[0] != 1:
        raise ValueError("Expected [1, tokens, hidden] causal-decoder tensors")
    if first.shape[2] != longer.shape[2] or not 0 < first.shape[1] < longer.shape[1]:
        raise ValueError("Appended sequence must preserve hidden width and original prefix")
    earlier = longer[:, : first.shape[1], :]
    if not np.isfinite(first).all() or not np.isfinite(earlier).all():
        raise RuntimeError("Non-finite hidden states in Jeff causal attention smoke")
    difference = np.abs(first - earlier)
    allowed = atol + rtol * np.abs(first)
    violations = int((difference > allowed).sum())
    result = {
        "short_sequence_tokens": int(first.shape[1]),
        "extended_sequence_tokens": int(longer.shape[1]),
        "max_hidden_absolute_difference": float(difference.max()),
        "mean_hidden_absolute_difference": float(difference.mean()),
        "mismatched_features": violations,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
    }
    if violations:
        raise RuntimeError(
            "Gemma 4 attention backend changed earlier hidden states after appending tokens: "
            f"{violations} mismatched features (max_abs={result['max_hidden_absolute_difference']:.5g}); "
            "verify truly causal attention or switch to verified eager backend before H200 training"
        )
    return result


def audit_released_jeff_attention(decider) -> dict:
    """Use one fixed non-patient Jeff/SemIf probe and the official tokenizer."""
    from jeff.prompting import encode_prompt

    from .jeff_checkpoint import checkpoint_probe
    from .jeff_config import JEFF_MAX_INPUT_TOKENS

    ids = encode_prompt(decider.tokenizer, checkpoint_probe(), JEFF_MAX_INPUT_TOKENS)
    return audit_official_causal_attention(decider, ids)


def audit_official_causal_attention(decider, prompt_ids: Sequence[int]) -> dict:
    """Run two true Jeff decoder forwards before LoRA injection or scoring."""
    import torch

    if getattr(decider, "bidirectional", None):
        raise ValueError("This H200 causal test only applies to Jeff-1.1-Large")
    if len(prompt_ids) < 40:
        raise ValueError("Synthetic prompt needs >=40 tokens for prefix audit")
    n = max(32, len(prompt_ids) // 2)
    if n >= len(prompt_ids):
        raise ValueError("Synthetic causal test has no appended suffix")
    dev = decider.device
    ids_short = torch.tensor([list(prompt_ids[:n])], dtype=torch.long, device=dev)
    ids_extended = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    with torch.inference_mode():
        prefix = decider.forward(
            input_ids=ids_short, attention_mask=None, use_cache=False, return_dict=True
        ).last_hidden_state
        extended = decider.forward(
            input_ids=ids_extended, attention_mask=None, use_cache=False, return_dict=True
        ).last_hidden_state
    result = compare_causal_prefix(
        prefix.detach().float().cpu().numpy(), extended.detach().float().cpu().numpy()
    )
    result["synthetic_probe_only"] = True
    return result
