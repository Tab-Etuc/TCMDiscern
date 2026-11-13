"""CPU-only checks for GH200 causal fallback and Gemma 4 31B text LoRA targets."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.train_jeff_lora import (
    LORA_TARGETS,
    select_lora_target_modules,
    validate_lora_targets,
)
from tcmdiscern import jeff_attention


def test_attention_keeps_verified_default_backend(monkeypatch):
    decider = SimpleNamespace(model=SimpleNamespace())
    monkeypatch.setattr(
        jeff_attention,
        "audit_official_causal_attention",
        lambda decider, ids: {"mismatched_features": 0},
    )
    result = jeff_attention.audit_with_safe_backend(decider, list(range(50)))
    assert result["attention_backend"] == "released_default"


def test_attention_falls_back_only_after_failing_prefix_audit(monkeypatch):
    class Model:
        backend = "sdpa"

        def set_attn_implementation(self, backend):
            self.backend = backend

    decider = SimpleNamespace(model=Model())
    observed = []

    def audit(decider, ids):
        observed.append(decider.model.backend)
        if decider.model.backend == "sdpa":
            raise RuntimeError("Gemma 4 attention backend changed earlier hidden states")
        return {"mismatched_features": 0}

    monkeypatch.setattr(jeff_attention, "audit_official_causal_attention", audit)
    result = jeff_attention.audit_with_safe_backend(decider, list(range(50)))
    assert observed == ["sdpa", "eager"]
    assert decider.model.backend == "eager"
    assert result["attention_backend"] == "eager"
    assert result["default_backend_causal_pass"] is False


def test_attention_eager_must_also_pass(monkeypatch):
    decider = SimpleNamespace(
        model=SimpleNamespace(set_attn_implementation=lambda _: None)
    )

    def audit(decider, ids):
        raise RuntimeError("Gemma 4 attention backend changed earlier hidden states")

    monkeypatch.setattr(jeff_attention, "audit_official_causal_attention", audit)
    with pytest.raises(RuntimeError, match="Neither default nor eager attention"):
        jeff_attention.audit_with_safe_backend(decider, list(range(50)))


def test_unsupported_backend_errors_are_not_swallowed(monkeypatch):
    decider = SimpleNamespace(model=SimpleNamespace(set_attn_implementation=lambda _: None))

    def audit(decider, ids):
        raise RuntimeError("GPU out of memory")

    monkeypatch.setattr(jeff_attention, "audit_official_causal_attention", audit)
    with pytest.raises(RuntimeError, match="GPU out of memory"):
        jeff_attention.audit_with_safe_backend(decider, list(range(50)))


class TextBackbone:
    def __init__(self, *, omit_k=False):
        self.language_model = SimpleNamespace(layers=[])
        self.omit_k = omit_k
        for global_layer in (True, False):
            attn = SimpleNamespace(
                use_alternative_attention=global_layer,
                q_proj=object(),
                k_proj=object(),
                v_proj=None if global_layer else object(),
                o_proj=object(),
            )
            mlp = SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object())
            self.language_model.layers.append(SimpleNamespace(self_attn=attn, mlp=mlp))

    def named_modules(self):
        # The multimodal Gemma4Model also contains vision attention projections.
        yield "vision_tower.encoder.layers.0.self_attn.q_proj", object()
        for i, layer in enumerate(self.language_model.layers):
            for section, obj, keys in (
                ("self_attn", layer.self_attn, LORA_TARGETS[:4]),
                ("mlp", layer.mlp, LORA_TARGETS[4:]),
            ):
                for key in keys:
                    if self.omit_k and i == 1 and key == "k_proj":
                        continue
                    value = getattr(obj, key)
                    if value is not None:
                        yield f"language_model.layers.{i}.{section}.{key}", value


def test_text_only_targets_match_gemma4_global_k_eq_v():
    model = TextBackbone()
    modules, counts = select_lora_target_modules(model)
    assert counts == {
        "q_proj": 2, "k_proj": 2, "v_proj": 1, "o_proj": 2,
        "gate_proj": 2, "up_proj": 2, "down_proj": 2,
    }
    assert len(modules) == 13
    assert all(name.startswith("language_model.layers.") for name in modules)
    assert validate_lora_targets(model) == counts


def test_missing_required_text_decoder_projection_is_rejected():
    with pytest.raises(RuntimeError, match="Missing required Gemma 4 text LoRA module"):
        select_lora_target_modules(TextBackbone(omit_k=True))
