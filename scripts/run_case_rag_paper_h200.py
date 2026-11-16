"""Run the locked full-Dev Case-RAG paper experiments in one resident Jeff GPU session.

Order: S2 4 arms, then S1 Global2, then S3 Global2.
Prior full-Dev baseline logits are always reused. No LoRA training or Test reads.

  bash run_case_rag_paper_h200.sh --prepare-only
  bash run_case_rag_paper_h200.sh --check
  bash run_case_rag_paper_h200.sh
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import run_case_rag_batch_jeff as batch
from scripts.evaluate_case_rag_paper_dev import evaluate
from scripts.prepare_case_rag_paper_ablation import ARMS, ROOT, TAG, prepare, transform
from scripts.run_case_rag_h200 import choose_checkpoint
from scripts.run_case_rag_jeff import load_inputs, read_condition_logits
from tcmdiscern.case_rag_identity import expected_run_identity, validate_saved_identity
from tcmdiscern.jeff_checkpoint import restore_checkpoint
from tcmdiscern.jeff_config import JEFF_MAX_INPUT_TOKENS

METHODS = ("s2", "s1", "s3")
GPU_ARMS = {
    "s2": ("global2", "random2", "global2_labelhide", "global2_labelonly"),
    "s1": ("global2",),
    "s3": ("global2",),
}
MODEL_PATH = Path(os.environ.get("H200_MODEL_DIR", str(ROOT / "data/models/Jeff-1.1-Large")))


def _prepare_legacy(root: Path) -> None:
    from scripts.prepare_case_rag_pilot import run

    folder = root / "data/case_rag" / TAG
    if not (folder / "manifest.json").is_file():
        if folder.exists():
            raise ValueError(f"Partial full-Dev preparation: {folder}")
        run(5486, 20261116, "paper", evidence_budget=2)
    report = prepare(root)
    if report["status"] != "PAPER_5486_ABLATIONS_VERIFIED":
        raise ValueError("Paper full-Dev ablations are incomplete")


def _load_inputs(root: Path) -> dict:

    folder = root / "data/case_rag" / TAG
    collection = load_inputs(folder)
    for arm in ARMS:
        with (folder / f"{arm}.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != 5486:
            raise ValueError(f"Paper {arm} has incomplete prompts")
        for original, altered in zip(collection["global2"], rows, strict=True):
            if altered != transform(original, arm):
                raise ValueError(f"Frozen {arm} changed retrieved case evidence")
        collection[arm] = rows
    if len(collection["baseline"]) != 5486:
        raise ValueError("Full Dev baseline is missing patients")
    return collection


def _scores_path(root: Path, method: str) -> Path:
    return root / "data/case_rag" / TAG / "scores" / f"paper-{method}-full"


def _verify_frozen_prompts(root: Path, model_dir: Path, collection: dict) -> dict:
    """Audit every actual Jeff input ID before the expensive 31B weight load.

    The HF tokenizer can return BatchEncoding (two keys); the shared counter
    explicitly reads input_ids and fails on any implausibly small token count.
    """
    from statistics import mean

    from tcmdiscern.case_rag_tokenizer import (
        count_jeff_prompt_tokens,
        load_jeff_token_counter,
    )

    folder = root / "data/case_rag" / TAG
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    ablation = json.loads((folder / "paper_ablations.json").read_text(encoding="utf-8"))
    if manifest.get("samples") != 5486 or ablation.get("samples") != 5486:
        raise ValueError("Frozen full-Dev preparation is incomplete")
    if manifest.get("test_used") is not False or ablation.get("test_used") is not False:
        raise ValueError("Unexpected official Test usage in input preparation")

    # Frozen original decision inputs must match the official Jeff Dev file,
    # including patient bytes, candidate sequence and all 5486 case IDs.
    original_path = root / "data/jeff/dev_k16.jsonl"
    with original_path.open(encoding="utf-8") as stream:
        for i, (row, line) in enumerate(zip(collection["baseline"], stream, strict=True)):
            if row["id"] != f"dev-{i:06d}" or row != json.loads(line):
                raise ValueError(f"Original Jeff Dev prompt drift at case {i}")

    tokenizer, source = load_jeff_token_counter(root, model_dir=model_dir)
    checked = {}
    for arm in ("baseline", "global2", "random2", *ARMS):
        expected = ablation["input_tokens"][arm] if arm in ARMS else manifest["input_tokens"][arm]
        lengths = []
        for i, row in enumerate(collection[arm]):
            n = count_jeff_prompt_tokens(tokenizer, row)
            if n < 32 or n > JEFF_MAX_INPUT_TOKENS:
                raise ValueError(
                    f"Jeff {arm} case {i} has {n} tokens; "
                    f"limit is {JEFF_MAX_INPUT_TOKENS}. GPU inference blocked."
                )
            lengths.append(n)
        observed = {
            "minimum": min(lengths),
            "mean": round(mean(lengths), 2),
            "maximum": max(lengths),
        }
        if (
            observed["minimum"] != expected["minimum"]
            or observed["maximum"] != expected["maximum"]
            or abs(observed["mean"] - expected["mean"]) > 0.12
        ):
            raise ValueError(
                f"Jeff {arm} tokenizer or prompt changed since preparation: "
                f"expected {expected}, observed {observed}"
            )
        checked[arm] = observed
        print(
            f"[FULL DEV TOKEN PREFLIGHT] {arm}: "
            f"{len(lengths)} real token sequences, max={observed['maximum']}, "
            f"limit={JEFF_MAX_INPUT_TOKENS}",
            flush=True,
        )
    return {"tokenizer": str(source), "arms": checked, "cases": 5486}


def _validated_saved_prefix(path: Path, cases: list[dict], *, timing: bool) -> int:
    """Verify resumable JSONL without mutating files or starting Jeff."""
    from tcmdiscern.case_rag_runtime import _validate_row, _validate_timing

    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as stream:
        for fragment in stream:
            if not fragment.endswith(b"\n"):
                # A final interrupted write will be trimmed on resume.
                break
            if count >= len(cases):
                raise ValueError(f"Too many saved rows in {path}")
            record = json.loads(fragment)
            case_id = cases[count]["id"]
            if timing:
                _validate_timing(record, case_id)
            else:
                _validate_row(record, case_id)
            count += 1
    return count


def _saved_progress(folder: Path, arm: str, cases: list[dict]) -> int:
    """Catch corrupt output and mismatched timing before the 31B load."""
    from tcmdiscern.case_rag_runtime import _verify_file

    result = folder / f"{arm}.jsonl"
    timings = folder / f"{arm}.timings.jsonl"
    partial = result.with_suffix(".jsonl.partial")
    timing_partial = timings.with_suffix(".jsonl.partial")

    if result.is_file():
        read_condition_logits(result, cases)
        if timings.is_file():
            _verify_file(timings, cases, allow_prefix=False, timings=True)
        elif timing_partial.is_file() and (
            _validated_saved_prefix(timing_partial, cases, timing=True) != len(cases)
        ):
            raise ValueError(f"Incomplete timing sidecar for completed logits: {timing_partial}")
        return len(cases)
    if timings.is_file():
        raise ValueError(f"Orphan completed timing records: {timings}")
    if timing_partial.exists() and not partial.exists():
        raise ValueError(f"Orphan partial timing records: {timing_partial}")
    scored = _validated_saved_prefix(partial, cases, timing=False)
    measured = _validated_saved_prefix(timing_partial, cases, timing=True)
    # Older valid partial logits can be retained with unknown timing markers.
    return min(scored, measured) if timing_partial.exists() else scored


def preflight(root: Path = ROOT, model_dir: Path = MODEL_PATH) -> tuple[dict, list[dict]]:
    """Validate all sources, checkpoints, old logits, and resume state before GPU."""
    from tcmdiscern.jeff_scoring import read_model_logits

    collection = _load_inputs(root)
    _verify_frozen_prompts(root, model_dir, collection)
    selected = []
    for index, method in enumerate(METHODS):
        checkpoint = choose_checkpoint(root, method, None)
        contract = batch.preflight_checkpoint_contract(
            model_dir,
            checkpoint,
            method,
            require_baseline_parity=(method == "s2"),
            check_kernel=(index == 0),
        )
        study = json.loads(
            (root / f"experiments/jeff_{method}-seed20261112-e2-epoch2_k16_dev.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        if (
            study["rows"] != 5486
            or study["checkpoint"]["method"] != method
            or study["checkpoint"]["source_run_id"] != metadata["run_id"]
            or study["attention_backend"] != contract["checkpoint_attention_backend"]
        ):
            raise ValueError(f"{method}: trained checkpoint and original full Dev differ")
        reference = root / f"data/jeff/dev_{method}-seed20261112-e2-epoch2_k16_5486.jsonl"
        read_model_logits(reference, num_cases=5486)
        directory = _scores_path(root, method)
        expected = expected_run_identity(
            collection,
            model_dir,
            checkpoint,
            source=TAG,
        )
        stamp = directory / "run_manifest.json"
        if stamp.exists():
            validate_saved_identity(
                json.loads(stamp.read_text(encoding="utf-8")), expected, strict=True
            )
        elif directory.exists() and list(directory.glob("*.jsonl*")):
            raise ValueError(f"{method} incomplete output identity, inspect {directory}")
        pending = []
        remaining = {}
        for arm in GPU_ARMS[method]:
            done = _saved_progress(directory, arm, collection[arm])
            if done < len(collection[arm]):
                pending.append(arm)
                remaining[arm] = len(collection[arm]) - done
        selected.append(
            {
                "method": method,
                "checkpoint": checkpoint,
                "identity": expected,
                "pending": pending,
                "pending_remaining": remaining,
            }
        )
        print(f"[FULL DEV PREFLIGHT] {method} PASS; remaining forwards: {remaining}", flush=True)
    return collection, selected


def run(
    root: Path = ROOT,
    model_dir: Path = MODEL_PATH,
    *,
    check_only: bool = False,
    session_factory=None,
) -> dict:
    from tcmdiscern.case_rag_runtime import run_resumable

    inputs, stages = preflight(root, model_dir)
    if check_only:
        return {
            "status": "CASE_RAG_PAPER_ALL_PREFLIGHTS_PASSED",
            "cases": 5486,
            "methods": [item["method"] for item in stages],
            "planned_new_forwards": sum(
                sum(item.get("pending_remaining", {}).values())
                if "pending_remaining" in item
                else len(item["pending"]) * 5486
                for item in stages
            ),
            "gpu_weights_loaded": 0,
            "test_used": False,
        }
    session = None
    completed = []
    for stage in stages:
        method = stage["method"]
        directory = _scores_path(root, method)
        if stage["pending"]:
            if session_factory is None:
                from tcmdiscern.h200.session import JeffGPUSession

                session_factory = JeffGPUSession
            if session is None:
                session = session_factory(model_dir)
            released = session.original()
            decider = released.decider
            restored = restore_checkpoint(decider, stage["checkpoint"])
            if restored.get("method") != method:
                raise ValueError(f"{method} LoRA+Head mismatch after restore")
            directory.mkdir(parents=True, exist_ok=True)
            identity = directory / "run_manifest.json"
            if identity.exists():
                validate_saved_identity(
                    json.loads(identity.read_text(encoding="utf-8")),
                    stage["identity"],
                    strict=True,
                )
            else:
                identity.write_text(
                    json.dumps(stage["identity"], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[CASE-RAG PAPER] {method} checkpoint ready; "
                f"base loads={session.model_weight_loads}",
                flush=True,
            )
            for arm in GPU_ARMS[method]:
                destination = directory / f"{arm}.jsonl"
                event = run_resumable(inputs[arm], decider, destination)
                print(
                    f"[CASE-RAG PAPER] {method}/{arm}: "
                    f"{event['source']}, new forwards={event['new_forwards']}",
                    flush=True,
                )
        report = evaluate(method, root=root, strict=True)
        completed.append(
            {
                "method": method,
                "status": report["status"],
                "report": f"experiments/case_rag_paper_full_dev_{method}_20261117.json",
            }
        )
        print(f"[CASE-RAG PAPER] {method} full Dev results scored", flush=True)
    loads = session.model_weight_loads if session is not None else 0
    if loads not in (0, 1):
        raise RuntimeError(f"Unexpected repeated 31B base-model load: {loads}")
    return {
        "status": "CASE_RAG_PAPER_FULL_DEV_COMPLETE",
        "backbone_loads": loads,
        "methods": completed,
        "official_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.check and args.prepare_only:
        parser.error("--check and --prepare-only cannot be combined")
    if args.prepare_only:
        _prepare_legacy(ROOT)
        result = {"status": "CASE_RAG_PAPER_FULL_DEV_INPUTS_PREPARED", "cases": 5486}
    else:
        result = run(ROOT, args.model_dir.expanduser().resolve(), check_only=args.check)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
