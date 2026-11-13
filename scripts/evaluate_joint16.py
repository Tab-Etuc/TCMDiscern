"""H200 Jeff-1.1-Large K16 disease / disease-syndrome pair evaluation.

Full official 5,486-row evaluation with original Train-only BM25 candidates,
checkpointed raw 16-slot logits and all observed labels (including Dev-unseen)
in Macro-F1. Inference never receives the gold label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from statistics import median

from sklearn.metrics import accuracy_score, f1_score

from scripts.preflight_jeff_training import inspect_checkpoint
from tcmdiscern.datasets import load_records
from tcmdiscern.disease_study import pair_parts
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_format import patient_state
from tcmdiscern.joint_study16 import MODES, QUESTION, K, gold_label

ROOT = Path(__file__).resolve().parents[1]
FULL_COUNT = 5486


def score(
    *,
    train: list[dict],
    eval_cases: list[dict],
    metadata: list[dict],
    predicted: list[str],
    mode: str,
) -> dict:
    if not (len(eval_cases) == len(metadata) == len(predicted) == FULL_COUNT):
        raise ValueError("Full official 5,486-case scoring only")
    train_label_set = {gold_label(case, mode) for case in train}
    classes = sorted(train_label_set | {gold_label(case, mode) for case in eval_cases})
    truths = [gold_label(case, mode) for case in eval_cases]
    bm25_top1 = [entry["candidates"][0] for entry in metadata]
    if any(
        entry["gold"] != truth or prediction not in entry["candidates"]
        for entry, truth, prediction in zip(metadata, truths, predicted, strict=True)
    ):
        raise ValueError("Predictions must match original labels and sixteen candidates")
    covered = [label in entry["candidates"] for label, entry in zip(truths, metadata, strict=True)]
    corrected = [p == g and b != g for p, b, g in zip(predicted, bm25_top1, truths, strict=True)]
    broken = [p != g and b == g for p, b, g in zip(predicted, bm25_top1, truths, strict=True)]
    result = {
        "n_cases": FULL_COUNT,
        "all_label_classes": len(classes),
        "top1_accuracy": float(accuracy_score(truths, predicted)),
        "macro_f1": float(
            f1_score(truths, predicted, average="macro", labels=classes, zero_division=0)
        ),
        "bm25_top1": float(accuracy_score(truths, bm25_top1)),
        "oracle16": sum(covered) / FULL_COUNT,
        "rescued": sum(corrected),
        "broken": sum(broken),
        "net_correct": sum(corrected) - sum(broken),
        "gold_absent_from_train": sum(truth not in train_label_set for truth in truths),
        "predictions_within_candidate_set": True,
    }
    if mode == "pair":
        actual = [pair_parts(t) for t in truths]
        outputs = [pair_parts(t) for t in predicted]
        result["disease_top1"] = float(
            accuracy_score([p[0] for p in actual], [p[0] for p in outputs])
        )
        result["syndrome_top1"] = float(
            accuracy_score([p[1] for p in actual], [p[1] for p in outputs])
        )
    return result


def evaluate_decider(
    decider, prompts: list[dict], metadata: list[dict], *, dest: Path
) -> tuple[list[str], dict]:
    if dest.exists():
        raise FileExistsError(f"Existing logit run: {dest}")
    if len(prompts) != len(metadata):
        raise ValueError("Prompt/metadata count mismatch")
    count = 0
    tokens = 0
    durations = []
    predictions = []
    start = time.perf_counter()
    fd, temporary = tempfile.mkstemp(prefix=".jeff-joint16-", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for i, (row, entry) in enumerate(zip(prompts, metadata, strict=True)):
                if row["id"] != entry["case_id"] or len(row["options"]) != K:
                    raise ValueError(f"Prompt and label metadata misaligned at {i}")
                options = [opt["id"] for opt in row["options"]]
                if options != entry["candidates"]:
                    raise ValueError(f"BM25 candidate order changed at {i}")
                _, logits, count_tokens, wall = decider.decide(row)
                if (
                    len(logits) != K
                    or any(not math.isfinite(float(v)) for v in logits)
                    or count_tokens < 1
                ):
                    raise ValueError("Invalid native Jeff 16-slot output")
                predicted = options[max(range(K), key=lambda j: logits[j])]
                predictions.append(predicted)
                count += 1
                tokens += count_tokens
                durations.append(wall)
                stream.write(
                    json.dumps(
                        {"case_id": row["id"], "logits": [float(v) for v in logits]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.link(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    durations.sort()
    return predictions, {
        "n_predictions": count,
        "total_tokens": tokens,
        "elapsed_seconds": time.perf_counter() - start,
        "median_inference_seconds": median(durations),
        "p95_inference_seconds": durations[
            min(len(durations) - 1, math.ceil(0.95 * len(durations)) - 1)
        ],
    }


def evaluate(
    *,
    mode: str,
    split: str,
    model_dir: Path,
    run_id: str,
    checkpoint: Path | None,
    limit: int | None = None,
    released=None,
) -> dict:
    from jeff import __version__ as jeff_version
    from jeff.model import JeffModel

    from tcmdiscern.jeff_config import JEFF_UPSTREAM_CODE_TAG

    if mode not in MODES or split not in ("dev", "test"):
        raise ValueError("Invalid official evaluation mode")
    if split == "test" and not (ROOT / "experiments/joint16_dev_selection.json").is_file():
        raise ValueError("Final Test requires frozen Dev model selection")
    if jeff_version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise RuntimeError("Jeff upstream runtime version does not match pinned version")
    inspection = inspect_checkpoint(model_dir)
    if (
        inspection.get("release_metadata_matches") is not True
        or not inspection.get("weights_present")
        or not inspection.get("decision_head_present")
    ):
        raise ValueError("Pinned released Jeff backbone and Decision Head unavailable")

    directory = ROOT / "data/joint16"
    prompts = [
        json.loads(line)
        for line in (directory / f"{split}_{mode}_prompts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    labels = [
        json.loads(line)
        for line in (directory / f"{split}_{mode}_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    official = load_records(ROOT / f"data/tcm-sd/{split}.json")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    if len(prompts) != FULL_COUNT or len(labels) != FULL_COUNT or len(official) != FULL_COUNT:
        raise ValueError("Incomplete official evaluation partition")
    for i, (case, prompt, entry) in enumerate(zip(official, prompts, labels, strict=True)):
        if (
            prompt["id"] != f"{split}-{i:06d}"
            or prompt["state"] != patient_state(case)
            or prompt["question"] != QUESTION[mode]
            or prompt.get("qtype") != "choice"
            or set(prompt) != {"id", "state", "question", "options", "qtype"}
            or entry["gold"] != gold_label(case, mode)
            or entry["case_id"] != prompt["id"]
        ):
            raise ValueError(f"Raw source/prompt/label integrity failure at row {i}")
    if limit is not None:
        if not 1 <= limit <= FULL_COUNT:
            raise ValueError("Invalid limited GPU pilot size")
        prompts = prompts[:limit]
        labels = labels[:limit]

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("H200 CUDA BF16 is required")
    model = (
        released
        if released is not None
        else JeffModel(
            model_dir,
            device="cuda",
            reasoning=False,
            long_input="refuse",
            cudnn_attention=False,
            compile=False,
        )
    )
    from tcmdiscern.jeff_attention import audit_released_jeff_attention

    # A resident GPU session already validated this exact backbone/backend.
    if released is None or not hasattr(model.decider, "attention_backend"):
        attention_report = audit_released_jeff_attention(model.decider)
    else:
        attention_report = {"attention_backend": model.decider.attention_backend}
    if checkpoint is None and hasattr(model.decider.base, "peft_config"):
        raise ValueError("Released baseline must not contain an active training adapter")
    if checkpoint is not None:
        restore_checkpoint(model.decider, checkpoint.expanduser().resolve(strict=True))
    logit_file = directory / f"{split}_{mode}_{run_id}_logits.jsonl"
    predictions, perf = evaluate_decider(model.decider, prompts, labels, dest=logit_file)
    result = {
        "kind": "SMOKE" if limit is not None else "FULL_EVALUATION",
        "mode": mode,
        "split": split,
        "run_id": run_id,
        "checkpoint": str(checkpoint) if checkpoint else "released",
        "k": K,
        "performance": perf,
        "attention_backend": attention_report["attention_backend"],
        "logits_file": str(logit_file.relative_to(ROOT)),
        "test_used_for_selection": False,
    }
    if limit is None:
        result["scores"] = score(
            train=train,
            eval_cases=official,
            metadata=labels,
            predicted=predictions,
            mode=mode,
        )
    report_file = ROOT / "experiments" / f"joint16_{split}_{mode}_{run_id}.json"
    with report_file.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                mode=args.mode,
                split=args.split,
                model_dir=args.model_dir,
                run_id=args.run_id,
                checkpoint=args.checkpoint,
                limit=args.limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
