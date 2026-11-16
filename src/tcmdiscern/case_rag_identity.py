"""Freeze experiment inputs and checkpoint sidecars for resumable Jeff evaluations.

Use deterministic JSON content fingerprints rather than directory names. The
30B released base weights are independently validated by Jeff preflight; only
small fine-tuning provenance files and large-adapter file metadata are read here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _digest_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def prompt_fingerprints(rows: Mapping[str, Sequence[dict]]) -> dict[str, str]:
    """Canonical digests of exactly the model-visible subset (including order)."""
    result = {}
    for condition, samples in rows.items():
        h = hashlib.sha256()
        for case in samples:
            encoded = json.dumps(
                case, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        result[condition] = h.hexdigest()
    return result


def checkpoint_fingerprint(directory: Path) -> dict:
    """Fingerprint small provenance and stat the large trained adapter/head."""
    if not directory.is_dir():
        raise FileNotFoundError("Jeff checkpoint directory is missing")
    required = (
        "manifest.json", "head_config.json", "probe_logits.json",
        "adapter/adapter_config.json",
    )
    payload = {}
    for name in required:
        file = directory / name
        if not file.is_file():
            raise FileNotFoundError(f"Incomplete checkpoint provenance: {name}")
        payload[name] = _digest_bytes(file.read_bytes())
    for name in ("head.safetensors", "adapter/adapter_model.safetensors"):
        file = directory / name
        if not file.is_file():
            raise FileNotFoundError(f"Fine-tuned checkpoint weights missing: {name}")
        stat = file.stat()
        payload[name] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return payload


def expected_run_identity(
    rows: Mapping[str, Sequence[dict]],
    model_dir: Path,
    checkpoint: Path,
    *,
    source: str,
) -> dict:
    if not rows or "baseline" not in rows:
        raise ValueError("A complete frozen baseline is required")
    return {
        "source": source,
        "cases": len(rows["baseline"]),
        "model_dir": str(model_dir.expanduser().resolve()),
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "conditions": list(rows),
        "prompt_sha256": prompt_fingerprints(rows),
        "checkpoint_provenance": checkpoint_fingerprint(checkpoint),
    }




def validate_score_only_inputs(
    saved: dict, rows: Mapping[str, Sequence[dict]], *, source: str,
) -> None:
    """Prevent recomputing model metrics against edited prompts or options."""
    expected = {
        "source": source,
        "cases": len(rows["baseline"]),
        "conditions": list(rows),
        "prompt_sha256": prompt_fingerprints(rows),
    }
    validate_saved_identity(saved, expected)

def validate_saved_identity(saved: dict, expected: dict, *, strict: bool = False) -> None:
    """Refuse stale results before allocating the 31B model on H200."""
    if not isinstance(saved, dict):
        raise TypeError("Invalid saved inference manifest")
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Existing Case-RAG run changed {key}; use a new run-id")
    if strict and set(saved) != set(expected):
        raise ValueError("Unexpected inference manifest schema")
