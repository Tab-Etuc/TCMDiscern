"""H200 Jeff K=1..35 evaluation with a full read-only CPU wiring test.

The --dry-run path evaluates every official Dev case using synthetic BM25-rank-1
scores. It never loads Jeff and must never be reported as Jeff accuracy.
For real model runs, K>16 uses the official grouped Decider.decide_wide path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from scripts.preflight_jeff_training import inspect_checkpoint
from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import Prediction, read_predictions
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_config import JEFF_MAX_INPUT_TOKENS, JEFF_UPSTREAM_CODE_TAG
from tcmdiscern.jeff_format import decision_row
from tcmdiscern.jeff_scoring import score_k

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DEV = 5486
MAX_K = 35
DEFAULT_KS = (1, 5, 16, 17, 35)


def validate_dev(cases: list[dict], predictions: list[Prediction], *, expected: int = OFFICIAL_DEV):
    """Verify complete fixed BM25 candidate order against original Dev rows."""
    if len(cases) != expected or len(predictions) != expected:
        raise ValueError("Original Dev and retrieval results must have the full expected row count")
    for i, (case, pred) in enumerate(zip(cases, predictions, strict=True)):
        case_id = f"dev-{i:06d}"
        if pred.case_id != case_id:
            raise ValueError(f"Dev case ID mismatch at {i}")
        if pred.gold_labels != (str(case["norm_syndrome"]),):
            raise ValueError(f"Dev gold mismatch at {i}")
        if len(pred.candidate_labels) < MAX_K or len(set(pred.candidate_labels[:MAX_K])) != MAX_K:
            raise ValueError(f"Dev BM25 candidate order or uniqueness mismatch at {i}")
    return sorted({str(case["norm_syndrome"]) for case in cases})


def iter_k_rows(cases: list[dict], predictions: list[Prediction], *, k: int, limit: int):
    """Generate model-visible input from whitelisted clinical fields only."""
    if not 2 <= k <= MAX_K:
        raise ValueError("Jeff requires K=2..35; K=1 is BM25 without model inference")
    if not 1 <= limit <= len(cases) or len(cases) != len(predictions):
        raise ValueError("Invalid evaluation count")
    for i in range(limit):
        yield decision_row(
            cases[i],
            case_id=f"dev-{i:06d}",
            candidates=predictions[i].candidate_labels[:k],
            allow_grouped=k > 16,
        )


def decider_scores(rows, decider, *, k: int, destination: Path):
    """Write original candidate-order scores with atomic no-overwrite semantics."""
    if destination.exists():
        raise FileExistsError(f"Completed output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".jeff-grid-", dir=destination.parent)
    vectors: list[list[float]] = []
    tokens = 0
    started = time.perf_counter()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                probabilities, logits, n_tokens, _ = decider.decide(row)
                if len(probabilities) != k or len(logits) != k:
                    raise ValueError("Jeff returned the wrong number of candidate scores")
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value) for value in logits
                ):
                    raise ValueError("Jeff returned nonfinite candidate scores")
                if not 0 < n_tokens <= (math.ceil(k / 16) + (k > 16)) * JEFF_MAX_INPUT_TOKENS:
                    raise ValueError("Jeff reported an invalid processed token count")
                if decider.last.get("shortened") or decider.last.get("thought"):
                    raise ValueError("Jeff inference unexpectedly shortened or reasoned")
                if decider.last.get("output_tokens", 0):
                    raise ValueError("Jeff generated text despite reasoning=False")
                values = [float(value) for value in logits]
                vectors.append(values)
                tokens += n_tokens
                stream.write(
                    json.dumps({"case_id": row["id"], "scores": values}, ensure_ascii=False) + "\n"
                )
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return vectors, tokens, time.perf_counter() - started


def evaluate(
    *,
    ks: list[int],
    limit: int,
    model_dir: Path | None,
    checkpoint: Path | None,
    run_id: str,
    dry_run: bool,
    root: Path = ROOT,
):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("run_id must contain only ASCII letters, digits, dash or underscore")
    if (
        not ks
        or len(ks) != len(set(ks))
        or any(type(k) is not int or not 1 <= k <= MAX_K for k in ks)
    ):
        raise ValueError("K selection must contain unique integers in 1..35")
    if not 1 <= limit <= OFFICIAL_DEV:
        raise ValueError("limit is outside the official Dev range")
    if not dry_run and model_dir is None and any(k > 1 for k in ks):
        raise ValueError("Real Jeff evaluation requires --model-dir")

    cases = load_records(root / "data/tcm-sd/dev.json")
    predictions = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    classes = validate_dev(cases, predictions)
    if len(classes) != 148:
        raise ValueError("Expected all 148 official Dev syndrome labels")

    # Refuse collisions before spending H200 time loading 31B weights.
    output_base = root / "data/jeff" / f"grid_{run_id}"
    planned = {}
    for k in ks:
        if dry_run or k == 1:
            continue
        suffix = "full" if limit == OFFICIAL_DEV else f"smoke{limit:04d}"
        output = output_base / f"k{k:02d}_{suffix}.jsonl"
        public = root / "experiments" / f"jeff_grid_{run_id}_k{k:02d}.json"
        if output.exists() or (limit == OFFICIAL_DEV and public.exists()):
            raise FileExistsError(f"K={k} already has results; specify only unfinished --ks")
        planned[k] = output, public

    decider = None
    if not dry_run and planned:
        from jeff import __version__ as installed_jeff
        from jeff.model import JeffModel

        if installed_jeff != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
            raise RuntimeError("Expected pinned official Jeff v1.3.0")
        assert model_dir is not None
        model_folder = model_dir.expanduser().resolve(strict=True)
        inspection = inspect_checkpoint(model_folder)
        if not inspection["weights_present"] or not inspection["decision_head_present"]:
            raise FileNotFoundError("Pinned Jeff original model or Head is missing")
        if inspection["release_metadata_matches"] is not True:
            raise ValueError("Jeff model release metadata disagrees with the research protocol")
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_properties(
            0
        ).total_memory < 80 * (2**30):
            raise RuntimeError("Jeff-1.1-Large requires an H200-class GPU")
        model = JeffModel(
            model_folder,
            device="cuda",
            reasoning=False,
            long_input="refuse",
            cudnn_attention=False,
            compile=False,
        )
        if (
            model.config.get("bidirectional") is not None
            or model.decider.max_tokens != JEFF_MAX_INPUT_TOKENS
        ):
            raise ValueError(
                "Jeff attention/context configuration does not match the pinned protocol"
            )
        from tcmdiscern.jeff_attention import audit_released_jeff_attention

        audit_released_jeff_attention(model.decider)
        if checkpoint is not None:
            restore_checkpoint(model.decider, checkpoint.expanduser().resolve(strict=True))
        decider = model.decider

    reports = []
    for k in ks:
        if k == 1:
            if limit != OFFICIAL_DEV and not dry_run:
                reports.append({"k": 1, "status": "BM25_ONLY_NO_PARTIAL_METRICS"})
                continue
            vectors = np.zeros((OFFICIAL_DEV, 1), dtype=float)
            result = score_k(predictions, vectors, k=1, all_classes=classes)
            result.update(status="BM25_RANK1_BASELINE", inference_seconds=0, model_forward_calls=0)
        elif dry_run:
            vectors = np.zeros((OFFICIAL_DEV, k), dtype=float)
            vectors[:, 0] = 1.0
            result = score_k(predictions, vectors, k=k, all_classes=classes)
            result.update(status="SYNTHETIC_BM25_RANK1_DRY_RUN_NOT_JEFF", model_forward_calls=0)
            if result["top1_correct_n"] != result["bm25_top1_correct_n"]:
                raise RuntimeError("Candidate scoring does not reproduce BM25 rank-1")
        else:
            assert decider is not None
            output, public = planned[k]
            vectors, tokens, elapsed = decider_scores(
                iter_k_rows(cases, predictions, k=k, limit=limit),
                decider,
                k=k,
                destination=output,
            )
            result = {
                "k": k,
                "status": "SMOKE_NOT_A_BENCHMARK",
                "cases": limit,
                "tokens": tokens,
                "inference_seconds": elapsed,
                "score_type": "raw_head_logits"
                if k <= 16
                else "official_grouped_log_probabilities",
                "model_forward_calls_per_case": 1 if k <= 16 else math.ceil(k / 16) + 1,
                "source": "original BM25 ranks, no gold inserted into model prompts",
                "checkpoint": str(checkpoint)
                if checkpoint is not None
                else "released_jeff_zero_shot",
            }
            if limit == OFFICIAL_DEV:
                result.update(score_k(predictions, vectors, k=k, all_classes=classes))
                result["status"] = "FULL_OFFICIAL_DEV"
                with public.open("x", encoding="utf-8") as stream:
                    json.dump(result, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
        reports.append(result)
    return {
        "run_id": run_id,
        "dry_run": dry_run,
        "official_dev_rows": OFFICIAL_DEV,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight or evaluate Jeff K=1..35 on official Dev"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-id", default="zeroshot")
    parser.add_argument(
        "--limit", type=int, default=2, help="Default 2 cases only; 5486 for full Dev"
    )
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument(
        "--all-ks", action="store_true", help="Evaluate every K=1..35 (expensive on H200)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Full-Dev synthetic rank-1 CPU validation without Jeff",
    )
    args = parser.parse_args()
    if args.checkpoint is not None and args.run_id == "zeroshot":
        parser.error("A checkpoint evaluation requires an explicit unique --run-id")
    ks = list(range(1, MAX_K + 1)) if args.all_ks else args.ks
    result = evaluate(
        ks=ks,
        limit=OFFICIAL_DEV if args.dry_run else args.limit,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
