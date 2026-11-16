"""Evaluate frozen Jeff S2 on five equal-evidence-budget Case-RAG conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from tcmdiscern.case_rag import CONDITIONS
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_format import semif_messages

ROOT = Path(__file__).resolve().parents[1]


def load_prompts(directory: Path) -> dict[str, list[dict]]:
    result = {}
    original_ids = None
    original_options = None
    for condition in CONDITIONS:
        with (directory / f"{condition}.jsonl").open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        ids = [row["id"] for row in rows]
        options = [[opt["id"] for opt in row["options"]] for row in rows]
        if len(rows) == 0 or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicated query ids")
        if any(len(c) != 16 or len(set(c)) != 16 for c in options):
            raise ValueError("Invalid Jeff candidates")
        if original_ids is not None and (ids != original_ids or options != original_options):
            raise ValueError("Different source rows or option order across conditions")
        for row in rows:
            if set(row) != {"id", "state", "question", "options", "qtype"}:
                raise ValueError("Unexpected model-visible prompt fields")
            if "norm_syndrome" in row["state"]:
                raise ValueError("Query gold label leaked")
            semif_messages(row)
        original_ids = ids
        original_options = options
        result[condition] = rows
    return result


def read_scores(path: Path, original: list[dict]) -> list[str]:
    predictions = []
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) != len(original):
        raise ValueError("Jeff output length does not match the input")
    for record, example in zip(rows, original, strict=True):
        if set(record) != {"case_id", "logits"} or record["case_id"] != example["id"]:
            raise ValueError("Jeff output case IDs changed")
        logits = np.asarray(record["logits"], dtype=np.float64)
        if logits.shape != (16,) or not np.isfinite(logits).all():
            raise ValueError("Invalid 16-slot Jeff logits")
        predictions.append(example["options"][int(logits.argmax())]["id"])
    return predictions


def metric(
    gold: list[str], predicted: list[str], base: list[str], vocabulary: list[str],
    train_texts: set[str], dev: list[dict], source: list[dict],
) -> dict:
    ids = [int(row["id"].split("-")[1]) for row in source]
    unseen = [j for j, i in enumerate(ids) if clinical_text(dev[i]) not in train_texts]
    correct = sum(a == b for a, b in zip(gold, predicted, strict=True))
    changed = sum(a != b for a, b in zip(predicted, base, strict=True))
    rescued = sum(p == y and b != y for p, b, y in zip(predicted, base, gold, strict=True))
    broken = sum(p != y and b == y for p, b, y in zip(predicted, base, gold, strict=True))
    disagreements = rescued + broken
    return {
        "top1_percent": round(100 * correct / len(gold), 4),
        "macro_f1_148_percent": round(100 * float(
            f1_score(gold, predicted, labels=vocabulary, average="macro", zero_division=0)
        ), 4),
        "no_train_exact_overlap_n": len(unseen),
        "no_train_exact_overlap_top1_percent": round(
            100 * sum(predicted[j] == gold[j] for j in unseen) / len(unseen), 4
        ) if unseen else None,
        "changed_vs_baseline": changed,
        "rescued_vs_baseline": rescued,
        "broken_vs_baseline": broken,
        "mcnemar_p_vs_baseline": float(
            binomtest(min(rescued, broken), disagreements).pvalue
        ) if disagreements else 1.0,
    }


def evaluate(
    source_dir: Path, model_dir: Path | None, checkpoint: Path | None,
    *, dry_run: bool,
) -> dict:
    rows = load_prompts(source_dir)
    count = len(rows["baseline"])
    if dry_run:
        return {
            "ready": True,
            "rows_per_condition": count,
            "conditions": list(CONDITIONS),
            "all_have_original_candidate_order": True,
            "real_model_executed": False,
        }
    if model_dir is None or checkpoint is None:
        raise ValueError("Provide --model-dir and --checkpoint for real S2 evaluation")
    import torch
    from jeff import __version__
    from jeff.model import JeffModel

    from scripts.preflight_jeff_training import inspect_checkpoint
    from scripts.run_jeff_zeroshot import run_decider
    from tcmdiscern.jeff_attention import audit_released_jeff_attention
    from tcmdiscern.jeff_checkpoint import restore_checkpoint

    if __version__ != "1.3.0" or not torch.cuda.is_available():
        raise RuntimeError("Pinned Jeff 1.3.0 and a CUDA GPU are required")
    inspection = inspect_checkpoint(model_dir)
    if not inspection["weights_present"] or not inspection["decision_head_present"]:
        raise FileNotFoundError("Released Jeff model/head weights are unavailable")
    model = JeffModel(
        model_dir, device="cuda", reasoning=False, long_input="refuse",
        cudnn_attention=False, compile=False,
    )
    audit_released_jeff_attention(model.decider)
    restored = restore_checkpoint(model.decider, checkpoint)
    if restored["method"] != "s2":
        raise ValueError("This pilot requires the original S2 checkpoint")
    output = source_dir / "jeff_s2_inference"
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "checkpoint_run_id": restored["source_run_id"],
        "method": restored["method"],
        "model_dir": str(model_dir.resolve()),
        "checkpoint_dir": str(checkpoint.resolve()),
        "query_ids": [row["id"] for row in rows["baseline"]],
        "conditions": list(CONDITIONS),
    }
    identity_file = output / "run_manifest.json"
    if identity_file.exists():
        previous = json.loads(identity_file.read_text(encoding="utf-8"))
        if previous != provenance:
            raise ValueError("Incompatible checkpoint or query set for resumed logits")
    else:
        if any(output.glob("*.jsonl")):
            raise ValueError("Found model outputs without a run identity manifest")
        with identity_file.open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2)
    costs = {}
    for condition in CONDITIONS:
        start = perf_counter()
        path = output / f"{condition}.jsonl"
        if path.exists():
            read_scores(path, rows[condition])  # Verify completed atomic result before resuming.
            costs[condition] = {"resumed": True, "mean_seconds_per_case": None}
            print(f"[case-rag] {condition}: reused verified logits", flush=True)
            continue
        n, n_tokens, seconds = run_decider(rows[condition], model.decider, path)
        if n != count:
            raise AssertionError("Incomplete Jeff inference")
        costs[condition] = {
            "resumed": False,
            "seconds": round(seconds, 2),
            "mean_seconds_per_case": round(seconds / count, 4),
            "mean_input_tokens": round(n_tokens / count, 1),
            "total_seconds_including_runner": round(perf_counter()-start, 2),
        }
        print(f"[case-rag] {condition}: {n} real Jeff decisions", flush=True)
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    vocabulary = sorted({str(row["norm_syndrome"]) for row in train})
    gold = [str(dev[int(row["id"].split("-")[1])]["norm_syndrome"])
            for row in rows["baseline"]]
    base = read_scores(output / "baseline.jsonl", rows["baseline"])
    memory_texts = {clinical_text(row) for row in train}
    result = {
        "status": "REAL_JEFF_S2_EXPLORATORY_DEV_PILOT",
        "split": "original Dev; subset chosen without query gold",
        "source_samples": count,
        "checkpoint": restored["source_run_id"],
        "test_used": False,
        "conditions": {
            condition: metric(
                gold, read_scores(output / f"{condition}.jsonl", rows[condition]),
                base, vocabulary, memory_texts, dev, rows[condition],
            )
            for condition in CONDITIONS
        },
        "runtime": costs,
    }
    (output / "aggregate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/case_rag/pilotclean_0300_20261116")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.source, args.model_dir, args.checkpoint, dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

