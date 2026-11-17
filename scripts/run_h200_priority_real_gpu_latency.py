"""Measure and resume real selective Jeff inference on the fixed Dev 20% subset.

Clinical records remain in the ignored data tree. Intermediate timing records
store only Dev indices and timings; no gold labels or patient text are exported.
The official Test remains sealed. One resident original S1 Jeff checkpoint is
used, with explicit logit parity against the previously completed 5,486-case run.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np

from tcmdiscern.case_rag_artifacts import frozen_paths
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_scoring import read_model_logits

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/h200/priority_budget_20pct_indices.json"
OUTPUT = ROOT / "experiments/h200_priority_real_gpu_latency.json"
PARTIAL = ROOT / "data/h200/priority_real_gpu_latency_partial.jsonl"
BASELINE = ROOT / "data/runs/s1-seed20261112-e2/resume-0001350"
RUN_ID = "s1-seed20261112-e2"


def selection_indices(root: Path = ROOT) -> list[int]:
    payload = json.loads((root / "data/h200/priority_budget_20pct_indices.json").read_text(encoding="utf-8"))
    indices = payload.get("selected_indices")
    if (
        payload.get("dev_cases") != 5486
        or payload.get("policy") != "lowest SGD top2 margin"
        or not isinstance(indices, list)
        or len(indices) != int(5486 * 0.20)
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or not 0 <= i < 5486 for i in indices)
    ):
        raise ValueError("Invalid or stale fixed Dev selective-inference indices")
    return indices


def completed_report(output: Path, indices: list[int]) -> dict | None:
    if not output.exists():
        return None
    result = json.loads(output.read_text(encoding="utf-8"))
    if (
        result.get("status") != "H200_PRIORITY_GPU_LATENCY_COMPLETED"
        or result.get("official_test_used") is not False
        or result.get("model_run_id") != RUN_ID
        or result.get("total_dev_cases") != 5486
        or result.get("model_calls") != len(indices)
        or result.get("selected_indices") != indices
        or result.get("same_checkpoint_saved_logit_argmax_mismatches") != 0
        or not all(
            isinstance(result.get(metric), (int, float))
            and math.isfinite(result[metric])
            and result[metric] > 0
            for metric in ("p50_seconds_per_call", "p95_seconds_per_call", "mean_seconds_per_call")
        )
    ):
        raise RuntimeError(f"Existing latency report is incomplete or for another policy: {output}")
    return result


def read_partial(path: Path, indices: list[int]) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(records) >= len(indices):
                raise ValueError("Partial latency file exceeds selected case count")
            row = json.loads(line)
            expected = indices[len(records)]
            if (
                row.get("row_index") != expected
                or row.get("model_run_id") != RUN_ID
                or row.get("top1_match") is not True
                or type(row.get("tokens")) is not int
                or not 1 <= row["tokens"] <= 8192
                or not isinstance(row.get("seconds"), (int, float))
                or not math.isfinite(row["seconds"])
                or row["seconds"] <= 0
                or not isinstance(row.get("max_abs_logits_delta"), (float, int))
                or not math.isfinite(row["max_abs_logits_delta"])
            ):
                raise ValueError(f"Partial GPU inference does not match selected row {expected}")
            records.append(row)
    return records


def original_s1_checkpoint(root: Path) -> Path:
    """Resolve the original H200 checkpoint from this checkout or legacy root."""
    from tcmdiscern.case_rag_artifacts import artifact_roots

    candidates = [
        base / "data/runs/s1-seed20261112-e2/resume-0001350"
        for base in artifact_roots(root)
    ]
    checkpoint = next((path for path in candidates if path.is_dir()), None)
    if checkpoint is None:
        raise FileNotFoundError(
            f"Original S1 final checkpoint is required. Checked: {candidates}"
        )
    return checkpoint


def run(session, *, root: Path = ROOT, output: Path = OUTPUT) -> dict:
    indices = selection_indices(root)
    cached = completed_report(output, indices)
    if cached is not None:
        print("[SKIP] Verified complete real-GPU selective latency report", flush=True)
        return cached

    checkpoint = original_s1_checkpoint(root)
    partial = root / "data/h200/priority_real_gpu_latency_partial.jsonl"
    rows = [json.loads(line) for line in (root / "data/jeff/dev_k16.jsonl").open(encoding="utf-8")]
    if len(rows) != 5486:
        raise ValueError("Incomplete original 5,486 Dev prompts")
    previous = read_partial(partial, indices)
    _, saved_score_path = frozen_paths(root, "s1")
    saved_logits = read_model_logits(saved_score_path, num_cases=5486)

    began = time.perf_counter()
    new_calls = 0
    if len(previous) < len(indices):
        released = session.original()
        restore_checkpoint(released.decider, checkpoint)
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("a", encoding="utf-8") as stream:
            for position in range(len(previous), len(indices)):
                index = indices[position]
                row = rows[index]
                if row["id"] != f"dev-{index:06d}" or len(row["options"]) != 16:
                    raise ValueError(f"Dev prompt ID or options corrupted: {index}")
                before = time.perf_counter()
                _, logits, n_tokens, _ = released.decider.decide(row)
                seconds = time.perf_counter() - before
                if released.decider.last.get("shortened") or released.decider.last.get("thought"):
                    raise RuntimeError("Jeff inference shortened evidence or enabled thinking")
                if released.decider.last.get("output_tokens", 0):
                    raise RuntimeError("Unexpected Jeff text generation")
                values = np.asarray(logits, dtype=np.float64)
                expected = saved_logits[index]
                if (
                    values.shape != (16,) or not np.isfinite(values).all()
                    or not 1 <= n_tokens <= 8192 or seconds <= 0
                ):
                    raise RuntimeError("Invalid model output or timing")
                same = bool(values.argmax() == expected.argmax())
                if not same:
                    raise RuntimeError(
                        f"Frozen S1 checkpoint prediction mismatch at Dev case {index}"
                    )
                record = {
                    "row_index": index,
                    "model_run_id": RUN_ID,
                    "seconds": seconds,
                    "tokens": int(n_tokens),
                    "max_abs_logits_delta": float(np.max(np.abs(values - expected))),
                    "top1_match": same,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                previous.append(record)
                new_calls += 1
                if new_calls % 64 == 0 or position == len(indices) - 1:
                    stream.flush()
                    os.fsync(stream.fileno())
                if new_calls % 200 == 0:
                    print(
                        f"[GPU] Selective S1: {len(previous)}/{len(indices)} "
                        f"({len(previous) / len(indices):.1%})",
                        flush=True,
                    )

    if len(previous) != len(indices):
        raise RuntimeError("Selective GPU evaluation was not fully completed")
    times = [r["seconds"] for r in previous]
    tokens = sum(r["tokens"] for r in previous)
    report = {
        "status": "H200_PRIORITY_GPU_LATENCY_COMPLETED",
        "scope": "official Dev, warm GPU, fixed 20% low-SGD-margin gate",
        "model_run_id": RUN_ID,
        "model_calls": len(indices),
        "selected_indices": indices,
        "total_dev_cases": len(rows),
        "actual_call_fraction": len(indices) / len(rows),
        "elapsed_current_session_seconds": time.perf_counter() - began,
        "reused_partial_calls": len(previous) - new_calls,
        "executed_current_session_calls": new_calls,
        "mean_seconds_per_call": statistics.mean(times),
        "p50_seconds_per_call": float(np.percentile(times, 50)),
        "p95_seconds_per_call": float(np.percentile(times, 95)),
        "max_seconds_per_call": max(times),
        "input_tokens_selected": tokens,
        "same_checkpoint_saved_logit_argmax_mismatches": 0,
        "max_abs_logit_difference": max(r["max_abs_logits_delta"] for r in previous),
        "preprocessing_and_sgd_fit_included": False,
        "official_test_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".preparing")
    if tmp.exists():
        raise FileExistsError(f"Incomplete final latency report: {tmp}")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(
        f"H200_PRIORITY_GPU_LATENCY_COMPLETED calls={len(indices)} "
        f"p50={report['p50_seconds_per_call']:.4f}s "
        f"p95={report['p95_seconds_per_call']:.4f}s",
        flush=True,
    )
    return report
