"""Read-only hardware, version and checkpoint preflight for Jeff-1.1-Large LoRA.

Does not download weights or start training. Use on the target H200 host.
Only prints environment-level information; no patient records are touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from tcmdiscern.jeff_config import (
    JEFF_BASE_MODEL,
    JEFF_HEAD_SOFTCAP,
    JEFF_HIDDEN_DIM,
    JEFF_MAX_INPUT_TOKENS,
    JEFF_MODEL_ID,
    JEFF_MODEL_REVISION,
    JEFF_PROMPT_VERSION,
    JEFF_UPSTREAM_CODE_TAG,
    MAX_DIRECT_JEFF_K,
    PRIMARY_JEFF_K,
)

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
    "torch": "2.13.0",
    "transformers": "5.17.0",
    "safetensors": "0.8.0",
    "huggingface_hub": "1.32.0",
    "accelerate": "1.15.0",
}


def installed_versions() -> dict[str, str | None]:
    found = {}
    for package in PINNED:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    try:
        found["peft"] = metadata.version("peft")
    except metadata.PackageNotFoundError:
        found["peft"] = None
    return found


def inspect_gpu() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=12,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    output = []
    for row in result.stdout.strip().splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 3:
            name, memory, driver = parts
            output.append(
                {
                    "name": name,
                    "memory_mib": int(memory),
                    "driver_version": driver,
                }
            )
    return output


def inspect_checkpoint(folder: Path) -> dict:
    index = folder / "model.safetensors.index.json"
    meta = folder / "jeff_config.json"
    head = folder / "head_config.json"
    detail = {
        "checkpoint_directory_exists": folder.is_dir(),
        "weight_index_present": index.is_file(),
        "decision_head_present": (folder / "head.safetensors").is_file(),
        "head_config_present": head.is_file(),
        "release_config_present": meta.is_file(),
        "tokenizer_present": (folder / "tokenizer.json").is_file(),
        "weights_present": False,
        "missing_weight_shards": None,
        "release_metadata_matches": None,
    }
    if index.exists():
        contents = json.loads(index.read_text(encoding="utf-8"))
        shards = set(contents.get("weight_map", {}).values())
        missing = sorted(name for name in shards if not (folder / name).is_file())
        detail["missing_weight_shards"] = len(missing)
        detail["weights_present"] = bool(shards) and not missing
    if meta.exists() and head.exists():
        cfg = json.loads(meta.read_text(encoding="utf-8"))
        h = json.loads(head.read_text(encoding="utf-8"))
        detail["release_metadata_matches"] = all(
            [
                cfg.get("base_model") == JEFF_BASE_MODEL,
                cfg.get("prompt_version") == JEFF_PROMPT_VERSION,
                cfg.get("max_input_tokens") == JEFF_MAX_INPUT_TOKENS,
                cfg.get("max_options") == MAX_DIRECT_JEFF_K,
                h.get("d_model") == JEFF_HIDDEN_DIM,
                h.get("k_slots") == MAX_DIRECT_JEFF_K,
                h.get("softcap") == JEFF_HEAD_SOFTCAP,
            ]
        )
    return detail


def preflight(folder: Path) -> dict:
    versions = installed_versions()
    devices = inspect_gpu()
    model = inspect_checkpoint(folder)
    disk_free_gib = round(shutil.disk_usage(folder if folder.exists() else ROOT).free / (2**30), 2)
    blockers = []
    warnings = []
    if not devices:
        blockers.append("NVIDIA GPU unavailable")
    elif not any(d["memory_mib"] >= 81920 for d in devices):
        blockers.append("No GPU with >=80GiB VRAM for 31B bf16 inference")
    if versions["torch"] != PINNED["torch"]:
        blockers.append("PyTorch does not match upstream tested version")
    if versions["transformers"] != PINNED["transformers"]:
        blockers.append("Transformers does not match upstream tested version")
    if versions["peft"] is None:
        blockers.append("PEFT not installed for new trainable LoRA")
    if not model["weights_present"]:
        blockers.append("Pinned Jeff-1.1-Large merged model weights are unavailable")
        if disk_free_gib < 70:
            blockers.append("Less than 70 GiB free disk for the ~64 GB model download")
    if not model["decision_head_present"]:
        blockers.append("Released head.safetensors is unavailable")
    if model["release_metadata_matches"] is False:
        blockers.append("Released model/head metadata do not match the research configuration")
    if not (ROOT / "scripts/train_jeff_lora.py").exists():
        blockers.append("Gradient-enabled LoRA + Decision Head trainer not implemented")
    warnings.extend(
        [
            "The 80GiB threshold is an upstream INFERENCE minimum, not a demonstrated LoRA training minimum.",
            "Single-GPU H200 training fit must be measured using forward/backward, checkpointing and optimizer states.",
            "The released model already has merged LoRA; do not apply its stored adapter a second time.",
            "Before training, prove zero-step logits match the untouched Jeff checkpoint.",
            "The official training script has not been published; the new domain trainer requires independent verification.",
        ]
    )
    return {
        "study": "TCMDiscern",
        "model": JEFF_MODEL_ID,
        "model_revision": JEFF_MODEL_REVISION,
        "upstream_code_tag": JEFF_UPSTREAM_CODE_TAG,
        "primary_k": PRIMARY_JEFF_K,
        "status": "BLOCKED" if blockers else "PRELIMINARY_PASS_REQUIRES_GPU_SMOKE",
        "gpus": devices,
        "disk_free_gib": disk_free_gib,
        "installed_versions": versions,
        "upstream_versions": PINNED,
        "checkpoint": model,
        "blockers": blockers,
        "warnings": warnings,
        "training_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Jeff LoRA target host")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "data/models/Jeff-1.1-Large",
    )
    args = parser.parse_args()
    print(json.dumps(preflight(args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
