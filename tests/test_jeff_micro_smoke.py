"""Fast release-attention and miniature-Pilot checks without CUDA or 31B weights."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.smoke_tiny_jeff import miniature_token_ids, read_training_probe
from tcmdiscern.jeff_release_attention import (
    inspect_release_attention,
    runtime_attention_contract,
)


def release_folder(tmp_path, *, native=None, jeff=None):
    folder = tmp_path / "released"
    folder.mkdir()
    (folder / "jeff_config.json").write_text(
        json.dumps({"base_model": "google/gemma-4-31B-it", "bidirectional": jeff}),
        encoding="utf-8",
    )
    (folder / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "use_bidirectional_attention": native,
                    "attention_k_eq_v": True,
                    "num_kv_shared_layers": 3,
                    "hidden_size_per_layer_input": 64,
                },
            }
        ),
        encoding="utf-8",
    )
    return folder


@pytest.mark.parametrize(
    ("native", "jeff", "expected"),
    [
        (None, None, "causal"),
        ("vision", None, "causal"),
        ("all", None, "bidirectional"),
        (None, "full", "bidirectional"),
        (None, "all", "bidirectional"),
    ],
)
def test_metadata_discovers_attention_without_loading_weights(tmp_path, native, jeff, expected):
    report = inspect_release_attention(release_folder(tmp_path, native=native, jeff=jeff))
    assert report["expected_attention"] == expected
    assert report["attention_k_eq_v"] is True
    assert report["num_kv_shared_layers"] == 3
    assert report["has_per_layer_embeddings"] is True


def test_audit_uses_released_bidirectional_contract(monkeypatch):
    from tcmdiscern import jeff_attention

    fake_jeff = ModuleType("jeff")
    fake_prompt = ModuleType("jeff.prompting")
    fake_prompt.encode_prompt = lambda *_: list(range(64))
    fake_jeff.prompting = fake_prompt
    monkeypatch.setitem(sys.modules, "jeff", fake_jeff)
    monkeypatch.setitem(sys.modules, "jeff.prompting", fake_prompt)
    monkeypatch.setattr(
        jeff_attention,
        "audit_bidirectional_determinism",
        lambda *_: {"same_prompt_repeatable": True},
    )
    monkeypatch.setattr(
        jeff_attention,
        "audit_with_safe_backend",
        lambda *_: (_ for _ in ()).throw(AssertionError("causal check must not run")),
    )
    decider = SimpleNamespace(
        base=SimpleNamespace(config=SimpleNamespace(use_bidirectional_attention="all")),
        bidirectional=None,
        tokenizer=SimpleNamespace(),
    )
    report = jeff_attention.audit_released_jeff_attention(decider)
    assert report["attention_contract"] == "bidirectional"
    assert report["attention_backend"] == "released_default"
    assert decider.attention_backend == "released_default"


def test_reject_conflicting_native_and_patched_bidirectional(tmp_path):
    folder = release_folder(tmp_path, native="all", jeff="full")
    with pytest.raises(ValueError, match="conflict"):
        inspect_release_attention(folder)


@pytest.mark.parametrize(
    "native,jeff",
    [
        (None, None),
        ("vision", None),
        ("all", None),
        (None, "full"),
    ],
)
def test_runtime_contract_matches_config(native, jeff):
    decider = SimpleNamespace(
        base=SimpleNamespace(config=SimpleNamespace(use_bidirectional_attention=native)),
        bidirectional=jeff,
    )
    assert runtime_attention_contract(decider) == (
        "bidirectional" if native == "all" or jeff else "causal"
    )


def test_miniature_token_mapping_is_bounded_without_reordered_tokens():
    real = list(range(1_000, 1_120))
    mini = miniature_token_ids(real)
    assert len(mini) == 64
    assert mini[0] == 1 + real[-64] % 511
    assert all(1 <= token < 512 for token in mini)


def test_real_train_probe_only_reads_original_train_labels(tmp_path):
    sample = {
        "row": {
            "id": "train-000001",
            "options": [{"id": f"label{i}", "description": f"label{i}"} for i in range(16)],
        },
        "target_index": 0,
        "target_letter": "A",
        "positive_indices": [0, 1],
        "positive_mask": [i in (0, 1) for i in range(16)],
        "target_distribution": [0.5 if i in (0, 1) else 0.0 for i in range(16)],
    }
    target = tmp_path / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    item = read_training_probe(tmp_path)
    gold = item["row"]["options"][item["target_index"]]["id"]
    assert gold == "label0"
    assert len(item["positive_indices"]) == 2
    assert "label1" in [item["row"]["options"][i]["id"] for i in item["positive_indices"]]
