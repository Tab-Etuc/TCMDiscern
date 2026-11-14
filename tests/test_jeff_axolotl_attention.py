"""Regression checks for Axolotl Gemma 4 configuration handling.

The official Transformers 5.17 config consumes global_head_dim and stores
per-layer head_dim values in per_layer_config; the original runtime attribute
does not exist after config construction.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tcmdiscern import jeff_axolotl_attention as hybrid


def release_config(*, full_dim: int = 384, sliding_dim: int = 256):
    kinds = [
        "full_attention" if (index + 1) % 6 == 0 else "sliding_attention" for index in range(60)
    ]
    return SimpleNamespace(
        num_hidden_layers=60,
        layer_types=kinds,
        per_layer_config={
            index: SimpleNamespace(head_dim=full_dim if kind == "full_attention" else sliding_dim)
            for index, kind in enumerate(kinds)
        },
    )


@pytest.mark.parametrize("full_dim", [128, 256, 384, 512])
def test_resolved_gemma4_per_layer_dimensions_without_global_head_dim(full_dim):
    config = release_config(full_dim=full_dim)
    assert not hasattr(config, "global_head_dim")
    result = hybrid.inspect_hybrid_layer_geometry(config)
    assert result["num_hidden_layers"] == 60
    assert result["sliding_flash2_layers"] == 50
    assert result["global_sdpa_layers"] == 10
    assert result["sliding_head_dims"] == [256]
    assert result["global_head_dims"] == [full_dim]


def test_oversized_sliding_layer_rejected_before_weight_loading():
    with pytest.raises(ValueError, match="exceeds FA2 support"):
        hybrid.inspect_hybrid_layer_geometry(release_config(sliding_dim=384))


def test_missing_per_layer_head_dimension_reports_layer_before_load():
    config = release_config()
    del config.per_layer_config[17]
    with pytest.raises(ValueError, match="at 17"):
        hybrid.inspect_hybrid_layer_geometry(config)


def test_preflight_uses_official_resolved_config_not_consumed_constructor_arg(
    tmp_path, monkeypatch
):
    config = release_config(full_dim=384)
    path = tmp_path / "Jeff-1.1-Large"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "num_hidden_layers": 60,
                "layer_types": config.layer_types,
                "global_head_dim": 384,
            }
        ),
        encoding="utf-8",
    )

    class FakeGemma4TextConfig:
        @classmethod
        def from_dict(cls, values):
            assert values["global_head_dim"] == 384
            return config

    modules = [
        "transformers",
        "transformers.models",
        "transformers.models.gemma4",
        "transformers.models.gemma4.configuration_gemma4",
    ]
    for name in modules:
        fake = ModuleType(name)
        fake.__path__ = []
        if name == modules[-1]:
            fake.Gemma4TextConfig = FakeGemma4TextConfig
        monkeypatch.setitem(sys.modules, name, fake)

    result = hybrid.inspect_release_hybrid_layout(path)
    assert result["global_head_dims"] == [384]
    assert result["sliding_flash2_layers"] == 50


def test_hybrid_selection_follows_actual_layer_dimensions(monkeypatch):
    monkeypatch.setenv("TCMDISCERN_GEMMA4_ATTENTION", "hybrid")
    monkeypatch.setattr(hybrid, "_patch_gemma4_mask", lambda: None)
    monkeypatch.setattr(hybrid, "_patch_sdpa_large_head_gqa", lambda: None)
    config = release_config(full_dim=384)
    config._attn_implementation = "sdpa"
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                config=config,
                head_dim=config.per_layer_config[index].head_dim,
            )
        )
        for index in range(60)
    ]
    model = SimpleNamespace(model=SimpleNamespace(config=config, layers=layers))

    def set_backend(name):
        config._attn_implementation = name

    model.set_attn_implementation = set_backend
    decider = SimpleNamespace(model=model)
    result = hybrid.enable_hybrid_attention(decider)
    assert result["global_head_dims"] == [384]
    assert result["global_sdpa_layers"] == 10
    assert result["sliding_flash2_layers"] == 50
    assert result["flash_kernel"] == hybrid.FLASH2_KERNEL
    assert all(
        layer.self_attn.config._attn_implementation
        == ("sdpa" if (i + 1) % 6 == 0 else hybrid.FLASH2_KERNEL)
        for i, layer in enumerate(layers)
    )
