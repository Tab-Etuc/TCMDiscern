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
    from .jeff_release_attention import runtime_attention_contract

    contract = runtime_attention_contract(decider)
    if contract == "bidirectional":
        # The released decision model intentionally lets later options affect
        # earlier hidden states. Prefix equality is the WRONG invariant there.
        report = audit_bidirectional_determinism(decider, ids)
        report["attention_backend"] = "released_default"
    else:
        report = audit_with_safe_backend(decider, ids)
    report["attention_contract"] = contract
    decider.attention_backend = report["attention_backend"]
    return report


def audit_bidirectional_determinism(decider, prompt_ids: Sequence[int]) -> dict:
    """Check a released bidirectional decision twice with the exact Jeff mask.

    A bidirectional model is expected to change earlier states after a suffix
    is appended, so never use a causal-prefix test on that configuration.
    """
    import torch

    if len(prompt_ids) < 40:
        raise ValueError("Bidirectional audit requires a realistic synthetic prompt")
    a = list(prompt_ids)
    b = a.copy()
    b[-1] = 1 if b[-1] != 1 else 2
    with torch.inference_mode():
        first = decider.hidden(a).float()
        second = decider.hidden(a).float()
        changed = decider.hidden(b).float()
    if (
        not torch.isfinite(first).all().item()
        or not torch.isfinite(second).all().item()
        or not torch.isfinite(changed).all().item()
        or not torch.allclose(first, second, rtol=1e-4, atol=1e-3)
    ):
        raise RuntimeError("Released bidirectional Jeff is not deterministic or finite")
    if torch.allclose(first, changed, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Released Jeff did not react to a changed prompt token")
    return {
        "attention_contract": "bidirectional",
        "same_prompt_repeatable": True,
        "different_prompt_response": True,
        "synthetic_probe_only": True,
    }


def audit_with_safe_backend(decider, prompt_ids: Sequence[int]) -> dict:
    """Use the released attention if causal; otherwise verify eager in-place."""
    try:
        report = audit_official_causal_attention(decider, prompt_ids)
        report["attention_backend"] = "released_default"
        return report
    except RuntimeError as original_error:
        # With torch 2.11 / CUDA 12.8, the default SDPA kernel on a GH200 may
        # violate text-prefix causality. Fail closed unless the actual released
        # backbone can switch to eager and independently passes the same audit.
        if "attention backend changed earlier hidden states" not in str(original_error):
            raise
        model = decider.model
        set_backend = getattr(model, "set_attn_implementation", None)
        if not callable(set_backend):
            raise TypeError(
                "Default Gemma 4 attention is not causal, and the official model "
                "does not expose set_attn_implementation('eager')"
            ) from original_error
        print(
            "[ATTENTION] Default backend violates causal-prefix check; "
            "switching the loaded Jeff model to eager and rechecking",
            flush=True,
        )
        set_backend("eager")
        try:
            report = audit_official_causal_attention(decider, prompt_ids)
        except Exception as fallback_error:
            raise RuntimeError(
                "Neither default nor eager attention passed causal-prefix safety; "
                "LoRA and evaluation remain blocked"
            ) from fallback_error
        report["attention_backend"] = "eager"
        report["default_backend_causal_pass"] = False
        print("[ATTENTION] Eager backend causal-prefix PASS", flush=True)
        return report


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
