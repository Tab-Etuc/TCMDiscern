"""H200 preflight for noncausal Gemma 4 attention behavior.

Compare prefix states from two equal-length prompts that differ only in
suffix tokens. This avoids false alarms from BF16 GEMM kernel changes when
short and long sequences use different matrix shapes. The probe is synthetic,
contains no patient data, and runs before 31B training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def perturb_suffix_tokens(prompt_ids: Sequence[int]) -> tuple[list[int], int]:
    """Replace only the suffix while keeping the *sequence shape* unchanged.

    Shape changes can select different BF16 GEMM kernels and magnify rounding
    differences across 60 layers, even with a correct causal mask. A fixed
    input shape isolates the causal question from that numerical confound.
    """
    if len(prompt_ids) < 40:
        raise ValueError("Fixed-length causal probe needs at least 40 tokens")
    original = list(prompt_ids)
    split = max(32, len(original) // 2)
    changed = original.copy()
    changed[split:] = original[split + 1 :] + original[split : split + 1]
    if changed == original:
        alternative = next((token for token in original[:split] if token != original[-1]), None)
        if alternative is None:
            raise ValueError("Synthetic prompt cannot create a distinct suffix")
        changed[-1] = alternative
    if changed[:split] != original[:split]:
        raise RuntimeError("Causal audit altered prefix tokens")
    return changed, split


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
            "Gemma 4 attention backend changed earlier hidden states after suffix perturbation: "
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
        config = getattr(decider.model, "config", None)
        get_text = getattr(config, "get_text_config", None)
        text_config = get_text() if callable(get_text) else config
        active_backend = getattr(text_config, "_attn_implementation", None)
        if hasattr(decider, "_tcmdiscern_hybrid_attention"):
            report["attention_backend"] = "axolotl_hybrid_fa2_sdpa"
        else:
            report["attention_backend"] = (
                "eager" if active_backend == "eager" else "released_default"
            )
        return report
    except RuntimeError as original_error:
        if hasattr(decider, "_tcmdiscern_hybrid_attention"):
            raise RuntimeError(
                "Axolotl Gemma 4 hybrid FA2/SDPA failed the causal-prefix audit; "
                "check the prebuilt kernel and attention masks"
            ) from original_error
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
            # One 31B load takes 42 minutes on GH200. Preserve useful
            # evidence from the same resident weights before stopping.
            try:
                diagnostic = first_divergent_decoder_layer(decider, prompt_ids)
            except (
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
                OSError,
                ImportError,
            ) as debug_error:
                diagnostic = {"diagnostic_error": str(debug_error)}
            raise RuntimeError(
                "Neither default nor eager attention passed causal-prefix safety; "
                f"decoder_layer_diagnostic={diagnostic}. "
                "LoRA and evaluation remain blocked"
            ) from fallback_error
        report["attention_backend"] = "eager"
        report["default_backend_causal_pass"] = False
        print("[ATTENTION] Eager backend causal-prefix PASS", flush=True)
        return report


def first_divergent_decoder_layer(decider, prompt_ids: Sequence[int]) -> dict:
    """Diagnose the earliest layer leaking suffix tokens in an already loaded 31B.

    This runs only after both attention backends fail. It reuses resident model
    weights and retains CPU copies of only the short prefix at each layer.
    """
    import torch

    backbone = decider.base
    text = getattr(backbone, "language_model", backbone)
    layers = getattr(text, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Cannot identify Gemma 4 text decoder layers")
    n = max(32, len(prompt_ids) // 2)
    records: dict[str, dict[int, torch.Tensor]] = {}
    changed_ids, split = perturb_suffix_tokens(prompt_ids)
    n = split
    for label, ids in (
        ("short", list(prompt_ids)),
        ("extended", changed_ids),
    ):
        records[label] = {}
        handles = []

        def collect(index: int, phase: str):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    records[phase][index] = tensor[:, :n, :].detach().float().cpu()

            return hook

        try:
            for index, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(collect(index, label)))
            with torch.inference_mode():
                decider.forward(
                    input_ids=torch.tensor([ids], device=decider.device, dtype=torch.long),
                    attention_mask=None,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

    if len(records["short"]) != len(layers) or len(records["extended"]) != len(layers):
        raise RuntimeError("Decoder did not execute every expected layer")
    for index in range(len(layers)):
        shorter = records["short"][index]
        extended = records["extended"][index]
        if shorter.shape != extended.shape:
            raise RuntimeError(f"Decoder layer {index} changed prefix tensor shape")
        diff = (shorter - extended).abs()
        mask = diff > 0.1 + 0.05 * shorter.abs()
        if bool(mask.any()):
            return {
                "first_divergent_layer": index,
                "layer_type": text.config.layer_types[index],
                "mismatched_features": int(mask.sum().item()),
                "max_abs": float(diff.max().item()),
                "total_decoder_layers": len(layers),
            }
    return {"first_divergent_layer": None, "total_decoder_layers": len(layers)}


def audit_official_causal_attention(decider, prompt_ids: Sequence[int]) -> dict:
    """Check fixed-length suffix perturbations in true Jeff decoder forwards."""
    import torch

    if getattr(decider, "bidirectional", None):
        raise ValueError("This H200 causal test only applies to Jeff-1.1-Large")
    if len(prompt_ids) < 40:
        raise ValueError("Synthetic prompt needs >=40 tokens for prefix audit")
    changed_ids, n = perturb_suffix_tokens(prompt_ids)
    dev = decider.device
    # Both forwards have the same length, GEMM shapes and attention-mask shape.
    # Only tokens strictly after n change.
    ids_original = torch.tensor([list(prompt_ids)], dtype=torch.long, device=dev)
    ids_changed = torch.tensor([changed_ids], dtype=torch.long, device=dev)
    with torch.inference_mode():
        original = decider.forward(
            input_ids=ids_original, attention_mask=None, use_cache=False, return_dict=True
        ).last_hidden_state
        changed = decider.forward(
            input_ids=ids_changed, attention_mask=None, use_cache=False, return_dict=True
        ).last_hidden_state
    result = compare_causal_prefix(
        original[:, :n, :].detach().float().cpu().numpy(),
        changed.detach().float().cpu().numpy(),
    )
    result["synthetic_probe_only"] = True
    result["audit_type"] = "fixed_length_suffix_perturbation"
    result["input_sequence_tokens"] = len(prompt_ids)
    result["mutated_suffix_tokens"] = sum(
        a != b for a, b in zip(prompt_ids[n:], changed_ids[n:], strict=True)
    )
    return result
