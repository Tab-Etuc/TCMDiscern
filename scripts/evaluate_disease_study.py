"""Run released Jeff-1.1-Large or a trained adapter on TCM-SD disease-study Top-5.

Gold labels are in a separate file and never added to model prompts.
Store per-row logits in data/ and only publish aggregate score reports.
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
from tcmdiscern.disease_study import MODES, K, gold_for, score_predictions
from tcmdiscern.jeff_checkpoint import restore_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def score_decider(
    decider, prompts: list[dict], labels: list[dict],
    *, output: Path,
) -> tuple[list[str], dict]:
    if len(prompts) != len(labels):
        raise ValueError("Prompt/label row counts disagree")
    if output.exists():
        raise FileExistsError(f"Results already exist: {output}")
    predictions = []
    total_tokens = 0
    began = time.perf_counter()
    fd, temporary = tempfile.mkstemp(prefix=".jeff-k5-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for i, (prompt, metadata) in enumerate(zip(prompts, labels, strict=True)):
                if prompt["id"] != metadata["case_id"]:
                    raise ValueError(f"Prompt/metadata row ID mismatch at {i}")
                options = [opt["id"] for opt in prompt["options"]]
                if len(options) != K or options != metadata["candidates"]:
                    raise ValueError(f"Prompt choices differ from metadata at {i}")
                _probabilities, logits, tokens, _ = decider.decide(prompt)
                if len(logits) != K:
                    raise RuntimeError(f"Jeff returned {len(logits)} slots for five-option study")
                import math
                if not all(math.isfinite(float(value)) for value in logits):
                    raise ValueError("Non-finite Jeff logits")
                winning = max(range(K), key=lambda index: float(logits[index]))
                predicted = options[winning]
                predictions.append(predicted)
                total_tokens += tokens
                stream.write(json.dumps(
                    {"case_id": prompt["id"], "prediction": predicted,
                     "logits": [float(v) for v in logits]}, ensure_ascii=False
                ) + "\n")
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return predictions, {
        "elapsed_seconds": time.perf_counter() - began,
        "total_tokens": total_tokens,
        "n_predictions": len(predictions),
    }


def run(*, model_dir: Path, split: str, mode: str, run_id: str,
        checkpoint: Path | None = None, limit: int | None = None) -> dict:
    from jeff import __version__ as version
    from jeff.model import JeffModel

    from tcmdiscern.jeff_config import JEFF_UPSTREAM_CODE_TAG

    if version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise RuntimeError("Official Jeff version differs from the pinned release")
    if mode not in MODES or split not in ("dev", "test"):
        raise ValueError("Unsupported disease-study task/split")
    inspection = inspect_checkpoint(model_dir)
    if (
        inspection["release_metadata_matches"] is not True
        or not inspection["weights_present"]
        or not inspection["decision_head_present"]
    ):
        raise ValueError("Pinned Jeff release model/head unavailable")

    folder = ROOT / "data/disease_study"
    paths = (folder / f"{split}_{mode}_prompts.jsonl",
             folder / f"{split}_{mode}_labels.jsonl")
    prompts = [
        json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()
    ]
    metadata = [
        json.loads(line) for line in paths[1].read_text(encoding="utf-8").splitlines()
    ]
    cases = load_records(ROOT / f"data/tcm-sd/{split}.json")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    if len(cases) != 5486 or len(metadata) != len(cases) or len(prompts) != len(cases):
        raise ValueError("Entire original official evaluation partition required")
    for i, (case, row) in enumerate(zip(cases, metadata, strict=True)):
        if row["case_id"] != f"{split}-{i:06d}" or row["gold"] != gold_for(case, mode):
            raise ValueError("Evaluation metadata disagree with official source")
    if limit is not None:
        if not 1 <= limit <= len(cases):
            raise ValueError("Smoke limit outside evaluation partition")
        prompts, metadata, cases = prompts[:limit], metadata[:limit], cases[:limit]

    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H200 CUDA BF16 is required")
    model = JeffModel(
        model_dir, device="cuda", reasoning=False, long_input="refuse",
        cudnn_attention=False, compile=False,
    )
    if checkpoint is not None:
        restore_checkpoint(model.decider, checkpoint.expanduser().resolve())
    output = folder / f"{split}_{mode}_{run_id}_logits.jsonl"
    predictions, performance = score_decider(model.decider, prompts, metadata, output=output)
    result = {
        "status": "SMOKE_NOT_BENCHMARK" if limit is not None else "COMPLETE",
        "mode": mode, "split": split, "run_id": run_id, "k": K,
        "checkpoint": str(checkpoint) if checkpoint else "original_released",
        "logits_file": str(output.relative_to(ROOT)),
        "performance": performance,
    }
    if limit is None:
        result["scores"] = score_predictions(
            cases, metadata, predictions, mode=mode,
            train_labels=[gold_for(row, mode) for row in train],
        )
    report = ROOT / "experiments" / f"disease_study_{split}_{mode}_{run_id}.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(
        model_dir=args.model_dir, split=args.split, mode=args.mode,
        run_id=args.run_id, checkpoint=args.checkpoint, limit=args.limit,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
