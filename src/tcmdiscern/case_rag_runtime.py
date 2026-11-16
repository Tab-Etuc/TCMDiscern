"""Crash-safe raw Jeff logits and synchronized per-case latency artifacts.

Never overwrite a completed result. A write interruption may leave one extra
logit row without its timing; resume rewinds that unpaired row and reruns it.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np

from .jeff_config import JEFF_MAX_INPUT_TOKENS


def _validate_row(value: dict, expected_id: str) -> list[float]:
    if not isinstance(value, dict) or set(value) != {"case_id", "logits"}:
        raise ValueError("Saved inference record contains unexpected fields")
    if value["case_id"] != expected_id:
        raise ValueError("Saved inference case IDs differ from prepared prompts")
    logits = value["logits"]
    if not isinstance(logits, list) or len(logits) != 16:
        raise ValueError("Saved inference must contain 16 decision logits")
    if not all(isinstance(s, (float, int)) and math.isfinite(s) for s in logits):
        raise ValueError("Saved inference contains nonfinite or nonnumeric logits")
    return [float(s) for s in logits]


def _validate_timing(value: dict, expected_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "case_id", "input_tokens", "wall_seconds", "peak_cuda_bytes",
    }:
        raise ValueError("Invalid inference timing record structure")
    if value["case_id"] != expected_id:
        raise ValueError("Inference timing case IDs disagree with logits")
    seconds = value["wall_seconds"]
    tokens = value["input_tokens"]
    if seconds is None and tokens is None and value["peak_cuda_bytes"] is None:
        # A legacy partial had valid logits but no per-case timing sidecar.
        return
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid synchronized inference latency")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("Invalid recorded input token count")
    cuda = value["peak_cuda_bytes"]
    if cuda is not None and (not isinstance(cuda, int) or cuda < 0):
        raise ValueError("Invalid peak CUDA memory measurement")


def _verify_file(
    file: Path, rows: Sequence[dict], *, allow_prefix: bool, timings: bool = False,
) -> int:
    count = 0
    with file.open(encoding="utf-8") as source:
        for line in source:
            if count >= len(rows):
                raise ValueError("Inference artifact contains too many cases")
            if not line.endswith("\n"):
                raise ValueError("Truncated inference record; inspect partial file")
            record = json.loads(line)
            if timings:
                _validate_timing(record, rows[count]["id"])
            else:
                _validate_row(record, rows[count]["id"])
            count += 1
    if not allow_prefix and count != len(rows):
        raise ValueError("Completed inference artifact has missing cases")
    return count


def _recover_incomplete_tail(file: Path) -> None:
    """Discard only a final non-newline-terminated, interrupted JSONL record."""
    if not file.exists():
        return
    with file.open("rb+") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            stream.truncate(data.rfind(b"\n") + 1)


def _keep_prefix(file: Path, count: int) -> None:
    """Align two valid append-only prefix files after a write interruption."""
    with file.open("rb+") as stream:
        offset = 0
        for _ in range(count):
            line = stream.readline()
            if not line:
                raise ValueError("Inference timing/score prefixes cannot be aligned")
            offset += len(line)
        stream.truncate(offset)


def _timing_paths(output: Path) -> tuple[Path, Path]:
    complete = output.with_name(output.stem + ".timings.jsonl")
    partial = complete.with_suffix(complete.suffix + ".partial")
    return complete, partial


def _cuda_start():
    """Reset stats after synchronization; timing includes tokenization and model."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch


def _timing_summary(path: Path, *, count: int) -> dict:
    times, tokens, peaks = [], [], []
    total = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            total += 1
            entry = json.loads(line)
            if entry["wall_seconds"] is None:
                continue
            times.append(entry["wall_seconds"] * 1000.0)
            tokens.append(entry["input_tokens"])
            if entry["peak_cuda_bytes"] is not None:
                peaks.append(entry["peak_cuda_bytes"])
    if total != count:
        raise ValueError("Missing Case-RAG latency records")
    if not times:
        return {"latency_status": "unavailable_from_legacy_partial", "unmeasured_cases": count}
    arr = np.asarray(times, dtype=np.float64)
    return {
        "latency_sample_n": len(times),
        "unmeasured_legacy_cases": count - len(times),
        "mean_ms": round(mean(times), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "first_ms": round(times[0], 3),
        "steady_p95_ms": round(float(np.percentile(arr[1:], 95)), 3)
        if len(times) > 1 else None,
        "mean_input_tokens": round(mean(tokens), 2),
        "peak_cuda_allocated_bytes": max(peaks) if peaks else None,
        "cuda_synchronized": bool(peaks),
        "timing_scope": "per-case Jeff decide wall-time, excludes BM25 search and model load",
    }


def run_resumable(
    rows: Sequence[dict],
    decider: Any,
    output: Path,
    *,
    max_tokens: int = JEFF_MAX_INPUT_TOKENS,
    progress_every: int = 20,
) -> dict:
    """Stream one forward per patient and recover only complete paired prefixes."""
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Inference requires nonempty unique case IDs")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    timings_done, timings_partial = _timing_paths(output)
    if output.exists():
        _verify_file(output, rows, allow_prefix=False)
        if timings_partial.exists() and not timings_done.exists():
            _recover_incomplete_tail(timings_partial)
            _verify_file(timings_partial, rows, allow_prefix=False, timings=True)
            os.link(timings_partial, timings_done)
            timings_partial.unlink()
        if timings_done.exists():
            _verify_file(timings_done, rows, allow_prefix=False, timings=True)
            profile = _timing_summary(timings_done, count=len(rows))
        else:
            profile = {"latency_status": "unavailable_from_legacy_completed_logits"}
        return {
            "source": "complete_verified", "cases": len(rows), "new_forwards": 0,
            "timing": profile,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if timings_partial.exists() and not partial.exists():
        raise ValueError("Timing prefix has no corresponding Jeff logits")
    if partial.exists() and not timings_partial.exists():
        # Backward-compatible recovery for older, valid partial scores.
        # Retain existing logits, and explicitly mark their timing as unknown.
        _recover_incomplete_tail(partial)
        historical = _verify_file(partial, rows, allow_prefix=True)
        with timings_partial.open("x", encoding="utf-8") as stream:
            for row in rows[:historical]:
                stream.write(json.dumps({
                    "case_id": row["id"], "input_tokens": None,
                    "wall_seconds": None, "peak_cuda_bytes": None,
                }) + "\n")
    _recover_incomplete_tail(partial)
    _recover_incomplete_tail(timings_partial)
    completed = _verify_file(partial, rows, allow_prefix=True) if partial.exists() else 0
    measured = (
        _verify_file(timings_partial, rows, allow_prefix=True, timings=True)
        if timings_partial.exists() else 0
    )
    if completed != measured:
        resume_at = min(completed, measured)
        _keep_prefix(partial, resume_at)
        _keep_prefix(timings_partial, resume_at)
        completed = resume_at
    started = perf_counter()
    tokens = 0
    new_forwards = 0
    with (
        partial.open("a" if partial.exists() else "x", encoding="utf-8") as stream,
        timings_partial.open("a" if timings_partial.exists() else "x", encoding="utf-8")
        as profile_stream,
    ):
        for row in rows[completed:]:
            cuda = _cuda_start()
            begin = perf_counter()
            probabilities, raw, n_tokens, _ = decider.decide(row)
            if cuda is not None:
                cuda.cuda.synchronize()
            duration = perf_counter() - begin
            peak = int(cuda.cuda.max_memory_allocated()) if cuda is not None else None
            if not isinstance(n_tokens, int) or not 1 <= n_tokens <= max_tokens:
                raise ValueError("Jeff prompt exceeded the frozen context budget")
            if decider.last.get("shortened"):
                raise ValueError("Jeff shortened a RAG prompt")
            if decider.last.get("thought") or decider.last.get("output_tokens", 0):
                raise ValueError("Reasoning/generation mode unexpectedly activated")
            if len(probabilities) != 16:
                raise ValueError("Jeff decision head returned the wrong candidate count")
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            vector = _validate_row({"case_id": row["id"], "logits": list(raw)}, row["id"])
            timing = {
                "case_id": row["id"],
                "input_tokens": n_tokens,
                "wall_seconds": round(duration, 6),
                "peak_cuda_bytes": peak,
            }
            _validate_timing(timing, row["id"])
            # Logits are written first. Crash between these writes is repaired
            # by rewinding that final unmatched logit record on next resume.
            stream.write(json.dumps(
                {"case_id": row["id"], "logits": vector},
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
            stream.flush()
            profile_stream.write(json.dumps(timing, ensure_ascii=False) + "\n")
            profile_stream.flush()
            tokens += n_tokens
            new_forwards += 1
            if new_forwards % progress_every == 0:
                print(f"[case-rag] {completed + new_forwards}/{len(rows)} completed",
                      flush=True)
    _verify_file(partial, rows, allow_prefix=False)
    _verify_file(timings_partial, rows, allow_prefix=False, timings=True)
    os.link(partial, output)
    os.link(timings_partial, timings_done)
    partial.unlink()
    timings_partial.unlink()
    return {
        "source": "resumed_and_completed" if completed else "new_inference",
        "cases": len(rows),
        "resumed_cases": completed,
        "new_forwards": new_forwards,
        "new_input_tokens": tokens,
        "new_forward_seconds": round(perf_counter() - started, 3),
        "timing": _timing_summary(timings_done, count=len(rows)),
    }
