"""Jeff v1.3.0 zero-shot full-Dev inference on H200, using the released runner.

Inputs:
- local, pinned Jeff-1.1-Large model directory (weights and original Head);
- original, unlabeled Jeff/SemIf Dev K=16 examples;
- original Dev BM25 candidates (for option-order integrity checks only).

No Dev gold is inserted into prompts. Raw logits stay under git-ignored data/.
A full 5,486-case run produces a public aggregate-only report in experiments/.
Partial --limit runs are explicitly called SMOKE, never a benchmark result.

H200:
  python -m scripts.run_jeff_zeroshot --model-dir /models/Jeff-1.1-Large --limit 2
  python -m scripts.run_jeff_zeroshot --model-dir /models/Jeff-1.1-Large
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from scripts.preflight_jeff_training import inspect_checkpoint
from tcmdiscern.datasets import load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_config import (
    JEFF_MAX_INPUT_TOKENS,
    JEFF_MODEL_ID,
    JEFF_MODEL_REVISION,
    JEFF_UPSTREAM_CODE_TAG,
    PRIMARY_JEFF_K,
)
from tcmdiscern.jeff_format import TCM_QUESTION, patient_state
from tcmdiscern.jeff_scoring import read_model_logits, score_k16

OFFICIAL_DEV = 5486


def iter_dev_rows(
    jsonl: Path,
    candidates: list,
    *,
    limit: int,
    original_cases: list[dict] | None = None,
):
    """Yield model-visible decisions with matched BM25 options and original clinical evidence."""
    if len(candidates) != OFFICIAL_DEV:
        raise ValueError("Requires 5,486 original BM25 Dev candidate rankings")
    if not 1 <= limit <= OFFICIAL_DEV:
        raise ValueError("limit must be within the original Dev size")
    if original_cases is not None and len(original_cases) != OFFICIAL_DEV:
        raise ValueError("Expected full original official Dev for clinical evidence validation")
    seen = 0
    with jsonl.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            seen += 1
            if i >= OFFICIAL_DEV:
                raise ValueError("Unexpected extra Dev decision rows")
            row = json.loads(line)
            expected_id = f"dev-{i:06d}"
            if row["id"] != expected_id or candidates[i].case_id != expected_id:
                raise ValueError(f"Incorrect Dev ID/order at position {i}")
            labels = [candidate["id"] for candidate in row["options"]]
            if len(labels) != PRIMARY_JEFF_K:
                raise ValueError(f"Unexpected Dev K at row {i}")
            if labels != list(candidates[i].candidate_labels[:PRIMARY_JEFF_K]):
                raise ValueError(f"BM25 candidates are out of order at Dev row {i}")
            if set(row) != {"id", "state", "question", "options", "qtype"}:
                raise ValueError("Dev row has extra fields, possibly supervision labels")
            if row["question"] != TCM_QUESTION or row["qtype"] != "choice":
                raise ValueError("Dev row does not match the official TCM Jeff choice prompt")
            if any(option["description"] != option["id"] for option in row["options"]):
                raise ValueError("The Dev option descriptions differ from frozen BM25 labels")
            if original_cases is not None:
                if row["state"] != patient_state(original_cases[i]):
                    raise ValueError(f"Dev clinical evidence differs from official source at {i}")
                if candidates[i].gold_labels != (str(original_cases[i]["norm_syndrome"]),):
                    raise ValueError(f"Dev BM25 gold labels differ from official source at {i}")
            if i < limit:
                yield row
        if seen != OFFICIAL_DEV:
            raise ValueError("Incomplete original Dev prompt source")


def run_decider(rows, decider, output: Path) -> tuple[int, int, float]:
    """Write each decision's *raw* logits atomically, fail before overwrite.

    Takes a Jeff v1.3 Decider-compatible object so CPU unit tests can verify
    the entire mapping without downloading 31B model weights.
    """
    if output.exists():
        raise FileExistsError(f"Output exists: {output}. Do not overwrite a prior run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".jeff-zeroshot-", dir=output.parent)
    count = 0
    total_tokens = 0
    start = time.perf_counter()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                probabilities, logits, n_tokens, _ = decider.decide(row)
                if len(logits) != PRIMARY_JEFF_K or len(probabilities) != PRIMARY_JEFF_K:
                    raise ValueError("Jeff returned wrong number of decision slots")
                if n_tokens < 1 or n_tokens > JEFF_MAX_INPUT_TOKENS:
                    raise ValueError("Jeff inference prompt was shortened or has wrong length")
                if decider.last.get("shortened"):
                    raise ValueError("A Dev prompt was shortened; the frozen protocol forbids it")
                if decider.last.get("thought") or decider.last.get("output_tokens", 0):
                    raise ValueError("Reasoning was accidentally enabled for the baseline")
                if not all(isinstance(v, (int, float)) for v in logits):
                    raise ValueError("Invalid raw logits")
                import math

                if not all(math.isfinite(v) for v in logits):
                    raise ValueError("Non-finite raw logits")
                stream.write(
                    json.dumps({"case_id": row["id"], "logits": logits}, ensure_ascii=False) + "\n"
                )
                count += 1
                total_tokens += n_tokens
        os.link(temp_name, output)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return count, total_tokens, time.perf_counter() - start


def run(
    model_folder: Path,
    *,
    limit: int,
    root: Path,
    checkpoint: Path | None = None,
    run_id: str | None = None,
    released=None,
) -> dict:
    from jeff import __version__ as jeff_version
    from jeff.model import JeffModel

    if jeff_version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise RuntimeError("Install exactly Jeff v1.3.0, not an unpinned upstream revision")
    inspection = inspect_checkpoint(model_folder)
    if not inspection["weights_present"] or not inspection["decision_head_present"]:
        raise FileNotFoundError("Pinned Jeff original weights/head not available locally")
    if inspection["release_metadata_matches"] is not True:
        raise ValueError("Released Jeff model/head metadata differs from pinned protocol")

    dev_candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    original_dev = load_records(root / "data/tcm-sd/dev.json")
    # Check ALL original Dev prompts before loading the large model. Gold labels
    # are used for source integrity here, never inserted into model-visible rows.
    rows = list(
        iter_dev_rows(
            root / "data/jeff/dev_k16.jsonl",
            dev_candidates,
            limit=limit,
            original_cases=original_dev,
        )
    )
    # Official Jeff loads its *merged* model weights and original released Head.
    # Its Decider uses post-final-norm last-token hidden state and native head logits.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This 31B released Jeff baseline requires H200-class GPU")
    jeff = (
        released
        if released is not None
        else JeffModel(
            model_folder,
            device="cuda",
            reasoning=False,
            long_input="refuse",
            cudnn_attention=False,
            compile=False,
        )
    )
    if jeff.config.get("bidirectional") is not None:
        raise ValueError("The pinned Jeff-1.1-Large must use causal attention")
    if jeff.decider.max_tokens != JEFF_MAX_INPUT_TOKENS:
        raise ValueError("Unexpected Jeff model input token limit")
    from tcmdiscern.jeff_attention import audit_released_jeff_attention

    # The resident session performed the same audit at model load. Preserve its
    # verified backend and avoid costly repeated causal-prefix forwards.
    if released is None or not hasattr(jeff.decider, "attention_backend"):
        attention_report = audit_released_jeff_attention(jeff.decider)
    else:
        attention_report = {"attention_backend": jeff.decider.attention_backend}
    if checkpoint is None and hasattr(jeff.decider.base, "peft_config"):
        raise ValueError("Zero-shot requires the released adapter-free backbone")

    reload_info = None
    if checkpoint is not None:
        import re

        if not run_id or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", run_id):
            raise ValueError("Checkpoint evaluation requires safe --run-id")
        reload_info = restore_checkpoint(jeff.decider, checkpoint.expanduser().resolve())
        file_name = f"dev_{run_id}_k16_{limit:04d}.jsonl"
    else:
        file_name = (
            f"dev_zero_shot_k16_smoke{limit:04d}.jsonl"
            if limit < OFFICIAL_DEV
            else "dev_zero_shot_k16_full.jsonl"
        )
    output = root / "data/jeff" / file_name
    count, total_tokens, seconds = run_decider(rows, jeff.decider, output)
    if count != limit:
        raise ValueError("The number of inference results differs from the selected limit")
    report = {
        "study": "TCMDiscern",
        "model": JEFF_MODEL_ID,
        "model_revision": JEFF_MODEL_REVISION,
        "official_upstream_version": JEFF_UPSTREAM_CODE_TAG,
        "source": "original official TCM-SD Dev, original BM25 Top-16, no gold injection",
        "kind": (
            "SMOKE_NOT_A_BENCHMARK"
            if limit < OFFICIAL_DEV
            else "FULL_DEV_FINETUNED"
            if checkpoint is not None
            else "FULL_DEV_ZERO_SHOT"
        ),
        "rows": count,
        "tokens": total_tokens,
        "wall_seconds": seconds,
        "mean_wall_seconds_per_decision": seconds / count,
        "raw_logit_file_ignored": str(output.relative_to(root)),
        "long_input_policy": "refuse",
        "reasoning": False,
        "attention_backend": attention_report["attention_backend"],
        "trained_lora": checkpoint is not None,
        "checkpoint": (
            reload_info
            if checkpoint is not None
            else "original released Jeff Head and merged model weights"
        ),
        "test_used": False,
    }
    if limit == OFFICIAL_DEV:
        official_dev = original_dev
        if len(official_dev) != OFFICIAL_DEV:
            raise ValueError("Original official Dev count has changed")
        all_labels = sorted(str(row["norm_syndrome"]) for row in official_dev)
        if len(set(all_labels)) != 148:
            raise ValueError("Expected original 148 label classes")
        logits = read_model_logits(output, num_cases=OFFICIAL_DEV)
        report["scores"] = score_k16(dev_candidates, logits, all_classes=sorted(set(all_labels)))
        report_path = (
            root / "experiments" / f"jeff_{run_id}_k16_dev.json"
            if checkpoint is not None
            else root / "experiments/jeff_zero_shot_k16_dev.json"
        )
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Official Jeff-1.1-Large one-pass Dev baseline")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=OFFICIAL_DEV)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-id", help="Required when loading a trained adapter and Head")
    args = parser.parse_args()
    if args.checkpoint is None and args.run_id is not None:
        parser.error("--run-id only applies to --checkpoint evaluation")
    result = run(
        args.model_dir,
        limit=args.limit,
        root=root,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
