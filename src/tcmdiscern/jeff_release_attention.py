"""Inspect the released Jeff/Gemma 4 attention contract without loading weights.

Both metadata files are tiny. This is the first step of a cheap preflight:
native Gemma 4 bidirectionality or Jeff 1.1 bidir hooks invalidate a causal
prefix-invariance assertion. An unexpected attention mode fails immediately.
"""

from __future__ import annotations

import json
from pathlib import Path


def inspect_release_attention(folder: Path) -> dict:
    folder = Path(folder)
    with (folder / "jeff_config.json").open(encoding="utf-8") as stream:
        jeff_config = json.load(stream)
    with (folder / "config.json").open(encoding="utf-8") as stream:
        model_config = json.load(stream)
    if not isinstance(model_config, dict) or not isinstance(jeff_config, dict):
        raise TypeError("Released Jeff configuration must contain JSON objects")
    if model_config.get("model_type") not in {"gemma4", "gemma4_text"}:
        raise ValueError("Expected the released Gemma 4 checkpoint")
    text_config = model_config.get("text_config", model_config)
    if not isinstance(text_config, dict):
        raise TypeError("Expected Gemma 4 text configuration")
    release_mode = jeff_config.get("bidirectional")
    native_mode = text_config.get("use_bidirectional_attention")
    if release_mode not in (None, "full", "all"):
        raise ValueError(f"Unknown released Jeff attention mode: {release_mode!r}")
    if native_mode not in (None, "vision", "all"):
        raise ValueError(f"Unknown native Gemma 4 attention mode: {native_mode!r}")
    if release_mode and native_mode == "all":
        raise ValueError("Jeff bidirectional hooks and native Gemma 4 all-mode conflict")
    expected = "bidirectional" if release_mode or native_mode == "all" else "causal"
    return {
        "model_type": model_config["model_type"],
        "release_jeff_bidirectional": release_mode,
        "gemma4_native_attention": native_mode,
        "expected_attention": expected,
        "attention_k_eq_v": bool(text_config.get("attention_k_eq_v", False)),
        "num_kv_shared_layers": int(text_config.get("num_kv_shared_layers", 0)),
        "has_per_layer_embeddings": bool(text_config.get("hidden_size_per_layer_input", 0)),
        "base_model": jeff_config.get("base_model"),
    }


def runtime_attention_contract(decider) -> str:
    """Resolve actual transformer settings, including native Gemma4TextConfig."""
    native = getattr(getattr(decider.base, "config", None), "use_bidirectional_attention", None)
    jeff = getattr(decider, "bidirectional", None)
    if native not in (None, "vision", "all") or jeff not in (None, "full", "all"):
        raise RuntimeError("Unexpected loaded Jeff attention configuration")
    if native == "all" and jeff:
        raise RuntimeError("Conflicting native and patched bidirectional attention")
    return "bidirectional" if native == "all" or jeff else "causal"
