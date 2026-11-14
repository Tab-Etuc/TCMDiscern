"""Axolotl's Gemma-4 hybrid attention for released Jeff 1.1 Large.

Modified from Axolotl (Apache License 2.0; license copy in
licenses/axolotl-APACHE-2.0.txt):
- src/axolotl/loaders/patch_manager.py::_apply_gemma_hybrid_attention
- src/axolotl/monkeypatch/gemma4_hybrid_mask.py
  https://github.com/axolotl-ai-cloud/axolotl/tree/f2158ff1922b2bcf701854a17e47cc91ba26e7e9

TCMDiscern only trains text-only SemIf inputs, with no sample packing, so
the generic Axolotl packed-global attention router is unnecessary.
Use flash_attention_2 on 50 sliding layers and SDPA on the 10 global
head_dim=512 layers; force full-mask construction into the SDPA layout.

The released Jeff merged weights, tokenizer, 16-slot Head, LoRA targets and
S1-S4 supervision are unchanged. Do not import the full Axolotl training
framework into the pinned Jeff environment.
"""

from __future__ import annotations

import copy
import os
from typing import Any

HYBRID_BACKEND = "axolotl_hybrid_fa2_sdpa"
DEFAULT_BACKEND = "released_default"
VALID_BACKENDS = frozenset({HYBRID_BACKEND, DEFAULT_BACKEND, "eager"})
_TARGET = "transformers.models.gemma4.modeling_gemma4"


def requested_attention_backend() -> str:
    setting = os.environ.get("TCMDISCERN_GEMMA4_ATTENTION", "hybrid")
    if setting not in {"hybrid", "sdpa"}:
        raise ValueError("TCMDISCERN_GEMMA4_ATTENTION must be hybrid or sdpa")
    return HYBRID_BACKEND if setting == "hybrid" else DEFAULT_BACKEND


def preflight_hybrid_kernel() -> dict[str, str]:
    """Resolve the pre-built Hugging Face FA2 kernel *before* mapping 64GB.

    transformers 5.17 uses the kernels-community/flash-attn2 Hub artifact
    when the optional flash_attn binary package is absent. The kernels wheel
    is tiny; neither Axolotl nor flash-attn need compiling on GH200 aarch64.
    """
    import transformers
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    if transformers.__version__ != "5.17.0":
        raise RuntimeError("Axolotl Gemma 4 patch is pinned to transformers 5.17.0")
    functions, _ = lazy_import_flash_attention("flash_attention_2")
    if functions[0] is None or functions[1] is None:
        raise RuntimeError("FlashAttention 2 forward/varlen kernels are unavailable")
    return {
        "attention_backend": HYBRID_BACKEND,
        "flash_attention": "flash_attention_2",
        "kernel": "transformers 5.17.0 / flash-attn2 (installed or Hub)",
        "global_attention": "sdpa",
    }


def _patch_gemma4_mask() -> None:
    """Port of Axolotl's global-mask-only patch; no packed examples in Jeff."""
    import importlib

    module = importlib.import_module(_TARGET)
    current = module.create_causal_mask
    if getattr(current, "_tcmdiscern_axolotl_mask", False):
        return

    def hybrid_create_causal_mask(config, *args, **kwargs):
        # Gemma 4 vision/overlay masks and sliding masks keep original
        # dispatch. Only full_attention needs a 4D SDPA-compatible mask.
        if kwargs.get("and_mask_function") is not None:
            return current(config, *args, **kwargs)
        sdpa_config = copy.copy(config)
        sdpa_config._attn_implementation = "sdpa"
        return current(sdpa_config, *args, **kwargs)

    hybrid_create_causal_mask._tcmdiscern_axolotl_mask = True
    module.create_causal_mask = hybrid_create_causal_mask


def _patch_sdpa_large_head_gqa() -> None:
    """Axolotl guard: head_dim 512 GQA would force SDPA math attention."""
    from transformers.integrations import sdpa_attention as sdpa_module

    previous = sdpa_module.use_gqa_in_sdpa
    if getattr(previous, "_tcmdiscern_axolotl_gqa", False):
        return

    def guarded(attention_mask, key, *args, **kwargs):
        if key.shape[-1] > 256:
            return False
        return previous(attention_mask, key, *args, **kwargs)

    guarded._tcmdiscern_axolotl_gqa = True
    sdpa_module.use_gqa_in_sdpa = guarded


def enable_hybrid_attention(decider: Any) -> dict[str, Any]:
    """Select Axolotl's mixed FA2/SDPA strategy on a loaded original Jeff.

    Called once, before released attention audit and before LoRA injection.
    Updates attention functions/config only; it never touches parameter data.
    """
    if requested_attention_backend() != HYBRID_BACKEND:
        return {"attention_backend": DEFAULT_BACKEND, "enabled": False}

    model = decider.model
    text = getattr(model, "model", None)
    if text is None or not hasattr(text, "layers"):
        raise RuntimeError("Expected the Jeff 1.1 Gemma4TextModel decoder")
    layers = text.layers
    layer_types = list(text.config.layer_types)
    if len(layers) != 60 or len(layer_types) != len(layers):
        raise RuntimeError("Expected the 60-layer released Jeff 1.1 Large")
    if layer_types.count("sliding_attention") != 50 or layer_types.count("full_attention") != 10:
        raise RuntimeError("The released Jeff Gemma 4 attention layout changed")
    if getattr(text.config, "global_head_dim", None) != 512:
        raise RuntimeError("Expected Gemma 4 global attention head_dim=512")

    # set_attn_implementation registers/validates the HF prebuilt FA2 kernel.
    model.set_attn_implementation("flash_attention_2")
    _patch_gemma4_mask()
    _patch_sdpa_large_head_gqa()
    global_layers = 0
    for index, layer in enumerate(layers):
        if layer_types[index] != "full_attention":
            if layer.self_attn.config._attn_implementation != "flash_attention_2":
                raise RuntimeError("Gemma 4 sliding attention did not enable FA2")
            continue
        attn = layer.self_attn
        config = copy.copy(attn.config)
        config._attn_implementation = "sdpa"
        attn.config = config
        global_layers += 1

    report = {
        "attention_backend": HYBRID_BACKEND,
        "enabled": True,
        "sliding_flash2_layers": len(layers) - global_layers,
        "global_sdpa_layers": global_layers,
        "global_head_dim": 512,
        "sample_packing": False,
        "axolotl_patch_revision": "f2158ff1922b2bcf701854a17e47cc91ba26e7e9",
    }
    decider._tcmdiscern_hybrid_attention = report
    return report
