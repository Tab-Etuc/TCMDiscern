"""Run several zero-retraining Case-RAG suites with one Jeff model load.

Prompt-identical conditions across suites share their already scored raw
16-slot logits (hard links after strict equality and output validation).
Checkpoint identity, candidate order, patient cases and source labels remain
validated; every other model forward is independently resumable.

H200:
    python -m scripts.run_case_rag_batch_jeff --preflight
    python -m scripts.run_case_rag_batch_jeff --model-dir MODEL_DIR \
      --checkpoint S2_CHECKPOINT --expected-method s2 \
      --run-id case-rag-joint-s2
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter

from scripts.preflight_jeff_training import inspect_checkpoint
from scripts.run_case_rag_jeff import (
    ROOT,
    check_reference_baseline,
    load_inputs,
    score_outputs,
)
from tcmdiscern.case_rag_identity import expected_run_identity, validate_saved_identity
from tcmdiscern.case_rag_runtime import run_resumable
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_config import JEFF_UPSTREAM_CODE_TAG

DEFAULT_TAGS = (
    "budget2_0300_20261116",
    "excerpt2_0300_20261116",
    "cf2factorial_0300_20261116",
)


def load_suites(tags: tuple[str, ...], limit: int | None = None) -> dict[str, dict]:
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("Expected nonempty, unique suite names")
    suites = {}
    first_ids = None
    for tag in tags:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", tag):
            raise ValueError("Unsafe Case-RAG suite name")
        folder = ROOT / "data/case_rag" / tag
        collection = load_inputs(folder, limit)
        ids = [row["id"] for row in collection["baseline"]]
        if first_ids is None:
            first_ids = ids
        elif ids != first_ids:
            raise ValueError("Case-RAG suites have different sampled patients/order")
        suites[tag] = collection
    return suites


def plan_identical_inferences(suites: dict[str, dict]) -> list[dict]:
    """Always compare complete model-visible prompts, not case IDs alone."""
    seen = []
    plan = []
    for tag, collection in suites.items():
        for arm, rows in collection.items():
            match = next(
                ((old_tag, old_arm) for old_tag, old_arm, old_rows in seen
                 if old_rows == rows), None,
            )
            plan.append({
                "tag": tag,
                "condition": arm,
                "reuse": (
                    {"tag": match[0], "condition": match[1]} if match else None
                ),
                "rows": len(rows),
            })
            if match is None:
                seen.append((tag, arm, rows))
    return plan


def load_checkpoint(model_dir: Path, checkpoint: Path, expected_method: str | None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and the original Jeff-31B weights are required")
    from jeff import __version__ as jeff_version
    from jeff.model import JeffModel

    if jeff_version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise ValueError("Jeff version differs from the pinned release")
    verified = inspect_checkpoint(model_dir)
    if not verified["weights_present"] or not verified["decision_head_present"]:
        raise FileNotFoundError("Released Jeff model or Head missing")
    if verified["release_metadata_matches"] is not True:
        raise ValueError("Released Jeff metadata mismatch")
    jeff = JeffModel(
        model_dir, device="cuda", reasoning=False, long_input="refuse",
        cudnn_attention=False, compile=False,
    )
    if jeff.config.get("bidirectional") is not None:
        raise ValueError("Unexpected released backbone architecture")
    restored = restore_checkpoint(jeff.decider, checkpoint)
    if expected_method is not None and restored.get("method") != expected_method:
        raise ValueError("The checkpoint does not match the expected training method")
    return jeff.decider, restored


def _expected_suite_identity(
    folder: Path, collection: dict,
    model_dir: Path, checkpoint: Path,
) -> dict:
    return expected_run_identity(
        collection, model_dir, checkpoint, source=folder.name,
    )


def _preflight_existing_scores(
    suites: dict[str, dict], *,
    model_dir: Path, checkpoint: Path, run_id: str,
    expected_method: str | None,
) -> None:
    """Refuse mixed prompts/checkpoints before Jeff's large backbone loads."""
    if expected_method is not None:
        metadata = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("loss_method") != expected_method:
            raise ValueError("Fine-tuning checkpoint method differs from expected")
    for tag, collection in suites.items():
        folder = ROOT / "data/case_rag" / tag
        expected = _expected_suite_identity(folder, collection, model_dir, checkpoint)
        target = folder / "scores" / run_id
        manifest = target / "run_manifest.json"
        if manifest.exists():
            validate_saved_identity(
                json.loads(manifest.read_text(encoding="utf-8")), expected,
            )
        elif target.exists() and any(target.glob("*.jsonl*")):
            raise ValueError(
                "Unidentified existing Case-RAG batch results cannot be resumed; "
                "use another run-id"
            )


def _verify_run_identity(
    folder: Path, target: Path, collection: dict,
    model_dir: Path, checkpoint: Path, restored: dict,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    identity = {
        **_expected_suite_identity(folder, collection, model_dir, checkpoint),
        "checkpoint_method": restored["method"],
        "checkpoint_run_id": restored["source_run_id"],
    }
    manifest = target / "run_manifest.json"
    if manifest.exists():
        validate_saved_identity(
            json.loads(manifest.read_text(encoding="utf-8")), identity,
            strict=True,
        )
    else:
        if any(target.glob("*.jsonl*")):
            raise ValueError("Existing scores lack a matching inference manifest")
        manifest.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _timing_path(path: Path) -> Path:
    return path.with_name(path.stem + ".timings.jsonl")


def execute(
    suites: dict[str, dict], plan: list[dict], *,
    model_dir: Path, checkpoint: Path, expected_method: str | None,
    run_id: str, limit: int | None = None,
    require_baseline_parity: bool = False,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("Unsafe run ID")
    started = perf_counter()
    _preflight_existing_scores(
        suites, model_dir=model_dir, checkpoint=checkpoint,
        run_id=run_id, expected_method=expected_method,
    )
    decider, restored = load_checkpoint(model_dir, checkpoint, expected_method)
    if require_baseline_parity and restored["method"] != "s2":
        raise ValueError("The frozen baseline comparison applies only to S2")
    frozen = ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl"
    if require_baseline_parity and not frozen.is_file():
        raise FileNotFoundError("S2 original checkpoint logit reference missing")
    first_scored = {}
    completed = {}
    for entry in plan:
        tag, arm = entry["tag"], entry["condition"]
        collection = suites[tag]
        folder = ROOT / "data/case_rag" / tag
        destination = folder / "scores" / run_id
        _verify_run_identity(folder, destination, collection, model_dir, checkpoint, restored)
        out = destination / f"{arm}.jsonl"
        if entry["reuse"] is not None:
            prev = entry["reuse"]
            earlier = first_scored[(prev["tag"], prev["condition"])]
            if suites[prev["tag"]][prev["condition"]] != collection[arm]:
                raise ValueError("Shared logits do not have byte-identical model inputs")
            source_timing = _timing_path(earlier)
            if not source_timing.is_file():
                raise ValueError("Cannot share logits without measured source timing")
            if out.is_file():
                # A crash may occur after linking scores but before timing.
                if not _timing_path(out).exists():
                    os.link(source_timing, _timing_path(out))
                event = run_resumable(collection[arm], decider, out)
            elif out.with_suffix(".jsonl.partial").exists():
                event = run_resumable(collection[arm], decider, out)
            else:
                os.link(earlier, out)
                os.link(source_timing, _timing_path(out))
                event = run_resumable(collection[arm], decider, out)
                event["source"] = "identical_prompt_shared_scores_and_timing"
        else:
            event = run_resumable(collection[arm], decider, out)
        first_scored[(tag, arm)] = out
        completed.setdefault(tag, {})[arm] = event
        if arm == "baseline" and require_baseline_parity:
            comparison = check_reference_baseline(collection[arm], out, frozen)
            if comparison["decision_disagreements_vs_saved_s2"]:
                raise ValueError("The original S2 checkpoint baseline changed")
        print(f"[case-rag-batch] {tag}/{arm}: {event['source']}", flush=True)
        if arm == list(collection)[-1]:
            report = score_outputs(folder, destination, limit=limit)
            report["inference"] = {
                "one_model_load_for_all_suites": True,
                "checkpoint_run_id": restored["source_run_id"],
                "checkpoint_method": restored["method"],
                "arms": completed[tag],
            }
            public = ROOT / "experiments" / f"case_rag_batch_{tag}_{run_id}.json"
            if public.exists():
                saved = json.loads(public.read_text(encoding="utf-8"))
                if saved.get("status") != report["status"] or saved.get("samples") != report.get("samples"):
                    raise ValueError("Existing published summary is from another study")
            else:
                with public.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {
        "status": "REAL_JEFF_BATCH_CASE_RAG_EVALUATION",
        "suite_count": len(suites),
        "arm_count": len(plan),
        "deduplicated_arm_count": sum(item["reuse"] is not None for item in plan),
        "total_new_forwards": sum(
            value.get("new_forwards", 0)
            for group in completed.values() for value in group.values()
        ),
        "single_jeff_checkpoint_load": True,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-method", choices=("s1", "s2", "s3", "s4"))
    parser.add_argument("--require-baseline-parity", action="store_true")
    parser.add_argument("--run-id", default="case-rag-batch")
    args = parser.parse_args()
    suites = load_suites(tuple(args.tags), limit=args.limit)
    plan = plan_identical_inferences(suites)
    if args.preflight:
        print(json.dumps({
            "status": "BATCH_READY_FOR_H200",
            "suites": list(suites),
            "total_arms": len(plan),
            "unique_model_input_sets": sum(item["reuse"] is None for item in plan),
            "deduplicated_arms": sum(item["reuse"] is not None for item in plan),
            "planned_unique_forwards": sum(
                item["rows"] for item in plan if item["reuse"] is None
            ),
            "arm_reuse_plan": plan,
            "test_used": False,
        }, ensure_ascii=False, indent=2))
        return
    if args.model_dir is None or args.checkpoint is None:
        parser.error("--model-dir and --checkpoint are required for real inference")
    result = execute(
        suites, plan, model_dir=args.model_dir, checkpoint=args.checkpoint,
        expected_method=args.expected_method, run_id=args.run_id,
        limit=args.limit, require_baseline_parity=args.require_baseline_parity,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
