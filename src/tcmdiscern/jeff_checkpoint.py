"""Save published Jeff head + new PEFT adapter without duplicating the 31B base."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .jeff_config import JEFF_MAX_INPUT_TOKENS
from .jeff_gpu_core import decision_logits


def checkpoint_probe() -> dict:
    """Non-patient synthetic prompt shared by save and reload checks."""
    from .jeff_format import TCM_QUESTION

    return {
        "id": "synthetic-checkpoint-probe",
        "qtype": "choice",
        "state": {
            "chief_complaint": "Synthetic example.",
            "description": "No actual patient record.",
            "detection": "Test data.",
        },
        "question": TCM_QUESTION,
        "options": [{"id": f"synthetic_{i}", "description": f"synthetic_{i}"} for i in range(16)],
    }


def save_checkpoint(
    directory: Path,
    model_directory: Path,
    adapter: Any,
    head: Any,
    backbone: Any,
    tokenizer: Any,
    device: Any,
    manifest: dict,
) -> dict:
    """Use released head_config and all original head buffers; save no optimizer state."""
    import torch
    from jeff.head import load_head
    from jeff.prompting import encode_prompt
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=False)
    adapter.eval()
    head.eval()
    adapter.save_pretrained(directory / "adapter", safe_serialization=True)
    shutil.copy2(model_directory / "head_config.json", directory / "head_config.json")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
        str(directory / "head.safetensors"),
    )
    ids = encode_prompt(tokenizer, checkpoint_probe(), JEFF_MAX_INPUT_TOKENS)
    with torch.inference_mode():
        original = decision_logits(backbone, head, ids, device=device)
        reloaded_head = load_head(directory).to(device).eval()
        restored = decision_logits(backbone, reloaded_head, ids, device=device)
    if not torch.allclose(original, restored, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Saved Head differs after loading the published file schema")
    (directory / "probe_logits.json").write_text(
        json.dumps(
            {"raw_logits": original.detach().float().cpu().tolist()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "adapter_saved": True,
        "head_reloaded_same_logits": True,
        "original_head_config_preserved": True,
        "requires_fresh_adapter_reload": True,
    }


def validate_checkpoint_manifest(manifest: dict) -> None:
    """Same frozen research contract for both resumable saves and fresh Dev reload."""
    from .jeff_config import JEFF_MODEL_REVISION

    if not isinstance(manifest, dict):
        raise TypeError("Checkpoint manifest must be a JSON object")
    if manifest.get("official_train_case_count") != 43180:
        raise ValueError("Checkpoint missing full original Train audit")
    if manifest.get("model_revision") != JEFF_MODEL_REVISION:
        raise ValueError("Saved adapter was trained on a different Jeff revision")
    if manifest.get("loss_method") not in {"s1", "s2", "s3", "s4"}:
        raise ValueError("Checkpoint has no valid loss method")
    expected_source = (
        "train_oof_k16_epoch02.jsonl"
        if manifest["loss_method"] == "s1"
        else "train_oof_multipos_k16_epoch03.jsonl"
    )
    if manifest.get("source") != expected_source:
        raise ValueError("Checkpoint method and frozen Train source disagree")


def reset_released_backbone(decider: Any, original_model_dir: Path) -> dict:
    """Remove this study's LoRA without merging it into the released 31B.

    Reuses the already allocated model weights and restores the published Head
    from its small safetensors. This is the only supported transition between
    independent LoRA runs within one GPU process.
    """
    from jeff.head import load_head
    from peft import PeftModel
    from peft.tuners.tuners_utils import BaseTunerLayer

    adapter = decider.base
    if not isinstance(adapter, PeftModel):
        raise TypeError("Expected the fine-tuned PEFT adapter before reset")
    original = adapter.base_model.unload()
    # PEFT 0.21.1 BaseTuner.__init__ sets self.model.peft_config even
    # after unloading all LoRA layers. Check real modules before removing
    # this stale metadata to keep the next independent injection clean.
    if any(isinstance(module, BaseTunerLayer) for module in original.modules()):
        raise RuntimeError("PEFT unload left real tuner layers in the released backbone")
    if hasattr(original, "peft_config"):
        delattr(original, "peft_config")
    decider.model.model = original
    decider.base = original
    decider.forward = original
    # Trainer freezes backbone permanently; newly initialized adapters will
    # carry all new gradients, and the original fp32 Head is restored here.
    decider.head = load_head(original_model_dir).to(decider.device).eval()
    decider.model.eval()
    return {"released_backbone_reused": True, "fresh_original_head": True}


def restore_checkpoint(decider: Any, checkpoint_dir: Path) -> dict:
    """Fresh-release reload: new LoRA + trained FP32 Head, probe logit parity."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("Checkpoint directory does not exist")
    required = (
        checkpoint_dir / "adapter",
        checkpoint_dir / "head.safetensors",
        checkpoint_dir / "head_config.json",
        checkpoint_dir / "probe_logits.json",
        checkpoint_dir / "manifest.json",
    )
    if any(not item.exists() for item in required):
        raise FileNotFoundError("Incomplete Jeff fine-tuning checkpoint")

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_checkpoint_manifest(manifest)

    import numpy as np
    import torch
    from jeff.head import load_head
    from jeff.prompting import encode_prompt
    from peft import PeftModel

    adapter = PeftModel.from_pretrained(
        decider.base, checkpoint_dir / "adapter", is_trainable=False
    )
    decider.model.model = adapter
    decider.base = adapter
    decider.forward = adapter
    decider.head = load_head(checkpoint_dir).to(decider.device).eval()
    decider.model.eval()
    ids = encode_prompt(decider.tokenizer, checkpoint_probe(), decider.max_tokens)
    with torch.inference_mode():
        actual = decider.first(decider.hidden(ids), 16)[1]
    expected = json.loads((checkpoint_dir / "probe_logits.json").read_text(encoding="utf-8"))[
        "raw_logits"
    ]
    if len(expected) != 16 or not np.allclose(actual, expected, atol=1e-3, rtol=1e-4):
        raise RuntimeError("Fresh Jeff adapter+Head reload changed synthetic logits")
    return {
        "adapter_reloaded": True,
        "head_reloaded": True,
        "probe_16_logits_match": True,
        "max_abs_probe_difference": float(np.abs(np.asarray(actual) - np.asarray(expected)).max()),
        "source_run_id": manifest.get("run_id"),
        "method": manifest.get("loss_method"),
    }
