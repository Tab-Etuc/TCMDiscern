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
import json
import os
from pathlib import Path
from typing import Any

HYBRID_BACKEND = "axolotl_hybrid_fa2_sdpa"
DEFAULT_BACKEND = "released_default"
VALID_BACKENDS = frozenset({HYBRID_BACKEND, DEFAULT_BACKEND, "eager"})
# Hub v3 on CUDA 12.8 has a documented GQA backward failure. Pin the v2
# CXX11 kernel compatible with the GH200 torch 2.11.0+cu128 runtime.
# https://github.com/huggingface/kernels-community/issues/1085
FLASH2_KERNEL = "kernels-community/flash-attn2@v2"
_TARGET = "transformers.models.gemma4.modeling_gemma4"


def requested_attention_backend() -> str:
    setting = os.environ.get("TCMDISCERN_GEMMA4_ATTENTION", "hybrid")
    if setting not in {"hybrid", "sdpa"}:
        raise ValueError("TCMDISCERN_GEMMA4_ATTENTION must be hybrid or sdpa")
    return HYBRID_BACKEND if setting == "hybrid" else DEFAULT_BACKEND


def inspect_hybrid_layer_geometry(config: Any) -> dict[str, Any]:
    """Inspect the same resolved per-layer config as Gemma4TextAttention.

    Transformers 5.17 consumes global_head_dim in __post_init__, storing the
    result in per_layer_config. Reading config.global_head_dim on a loaded
    model is invalid and previously caused a failure after 42 minutes.
    """
    layer_types = list(config.layer_types)
    count = config.num_hidden_layers
    if len(layer_types) != count or count != 60:
        raise ValueError("Expected all 60 released Gemma 4 decoder layers")
    if layer_types.count("sliding_attention") != 50 or layer_types.count("full_attention") != 10:
        raise ValueError("Expected 50 sliding and 10 full Gemma 4 attention layers")
    sliding_dims: set[int] = set()
    full_dims: set[int] = set()
    for index, kind in enumerate(layer_types):
        try:
            layer_head_dim = config.per_layer_config[index].head_dim
        except (IndexError, KeyError, AttributeError, TypeError) as exc:
            raise ValueError(f"Gemma 4 per-layer head dimension missing at {index}") from exc
        if type(layer_head_dim) is not int or layer_head_dim <= 0:
            raise ValueError(f"Invalid Gemma 4 head dimension in layer {index}")
        if kind == "sliding_attention":
            if layer_head_dim > 256:
                raise ValueError(
                    f"Sliding layer {index} head_dim={layer_head_dim} exceeds FA2 support"
                )
            sliding_dims.add(layer_head_dim)
        else:
            full_dims.add(layer_head_dim)
    return {
        "num_hidden_layers": count,
        "sliding_flash2_layers": 50,
        "global_sdpa_layers": 10,
        "sliding_head_dims": sorted(sliding_dims),
        "global_head_dims": sorted(full_dims),
    }


def inspect_release_hybrid_layout(model_dir: Path) -> dict[str, Any]:
    """Resolve the released local JSON through official HF config before loading weights."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    with (Path(model_dir) / "config.json").open(encoding="utf-8") as stream:
        model_config = json.load(stream)
    if not isinstance(model_config, dict):
        raise TypeError("Released Gemma 4 config.json must contain an object")
    text_config = model_config.get("text_config", model_config)
    if not isinstance(text_config, dict):
        raise TypeError("Released Gemma 4 text config must contain an object")
    resolved = Gemma4TextConfig.from_dict(text_config)
    return inspect_hybrid_layer_geometry(resolved)


def preflight_hybrid_adapter(model_dir: Path) -> dict[str, Any]:
    """Apply the real Hybrid patch to a tiny 60-layer Gemma 4 before 31B IO.

    The tiny model carries the actual released sliding/full-layer ordering
    but only 64 hidden channels. This catches Python integration errors in
    set_attn_implementation, per-layer config and mask patching without
    mapping any large released weight tensors.
    """
    from types import SimpleNamespace

    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from scripts.smoke_tiny_jeff import miniature_text_config
    from tcmdiscern.jeff_release_attention import inspect_release_attention

    release = inspect_release_attention(model_dir)
    tiny = Gemma4ForCausalLM(miniature_text_config(release, deep=True))
    try:
        report = enable_hybrid_attention(SimpleNamespace(model=tiny))
    finally:
        del tiny
    return report


def preflight_hybrid_kernel(model_dir: Path | None = None) -> dict[str, Any]:
    """Resolve the pre-built Hugging Face FA2 kernel *before* mapping 64GB.

    transformers 5.17 uses the kernels-community/flash-attn2 Hub artifact
    when the optional flash_attn binary package is absent. The kernels wheel
    is tiny; neither Axolotl nor flash-attn need compiling on GH200 aarch64.
    """
    import transformers
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

    if transformers.__version__ != "5.17.0":
        raise RuntimeError("Axolotl Gemma 4 patch is pinned to transformers 5.17.0")
    geometry = inspect_release_hybrid_layout(model_dir) if model_dir is not None else None
    functions, _ = lazy_import_flash_attention(FLASH2_KERNEL)
    if functions[0] is None or functions[1] is None:
        raise RuntimeError("FlashAttention 2 forward/varlen kernels are unavailable")
    report: dict[str, Any] = {
        "attention_backend": HYBRID_BACKEND,
        "flash_attention": FLASH2_KERNEL,
        "kernel": "transformers 5.17.0 / Hub flash-attn2 v2",
        "global_attention": "sdpa",
    }
    if geometry is not None:
        report["release_geometry"] = geometry
        report["tiny_adapter_preflight"] = preflight_hybrid_adapter(Path(model_dir))
    return report


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
    geometry = inspect_hybrid_layer_geometry(text.config)
    if len(layers) != geometry["num_hidden_layers"]:
        raise RuntimeError("Loaded Jeff decoder depth differs from release config")
    for index, layer in enumerate(layers):
        expected_head_dim = text.config.per_layer_config[index].head_dim
        if layer.self_attn.head_dim != expected_head_dim:
            raise RuntimeError(f"Loaded Gemma 4 layer {index} head dimension differs from config")

    # Use the explicit, pinned Hub v2, never the default v3 that can
    # succeed in forward but fail GQA LoRA backward on CUDA 12.8.
    model.set_attn_implementation(FLASH2_KERNEL)
    _patch_gemma4_mask()
    _patch_sdpa_large_head_gqa()
    global_layers = 0
    for index, layer in enumerate(layers):
        if layer_types[index] != "full_attention":
            if layer.self_attn.config._attn_implementation != FLASH2_KERNEL:
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
        "sliding_head_dims": geometry["sliding_head_dims"],
        "global_head_dims": geometry["global_head_dims"],
        "flash_kernel": FLASH2_KERNEL,
        "sample_packing": False,
        "axolotl_patch_revision": "f2158ff1922b2bcf701854a17e47cc91ba26e7e9",
    }
    decider._tcmdiscern_hybrid_attention = report
    return report
