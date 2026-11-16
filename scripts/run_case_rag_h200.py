"""One-command, frozen Case-RAG H200 launch after original Jeff preparation.

H200:
    bash run_case_rag_h200.sh s2 --check
    bash run_case_rag_h200.sh s2 --limit 5
    bash run_case_rag_h200.sh s2
    bash run_case_rag_h200.sh s3 --all

Runs official Train/Dev-only inputs; keeps Test sealed and never trains Jeff.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from random import Random

from scripts.run_case_rag_batch_jeff import (
    load_suites,
    plan_identical_inferences,
)
from tcmdiscern.case_rag_tokenizer import locate_jeff_tokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261116
SAMPLES = 300
PREFIX = "budget2_0300_20261116"
EXCERPT = "excerpt2_0300_20261116"
COUNTERFACTUAL = "cf2factorial_0300_20261116"
REQUIRED_INPUTS = (
    "data/tcm-sd/train.json",
    "data/tcm-sd/dev.json",
    "data/candidates/bm25_dev.jsonl",
    "data/jeff/dev_k16.jsonl",
)
REQUIRED_CHECKPOINT = (
    "manifest.json",
    "head_config.json",
    "head.safetensors",
    "probe_logits.json",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)


def checkpoints(root: Path, method: str) -> list[dict]:
    """Discover runnable Jeff training boundaries without opening patient data."""
    candidates = []
    folder = root / "data/runs"
    if not folder.is_dir():
        return []
    for file in folder.rglob("manifest.json"):
        path = file.parent
        if not all((path / name).is_file() for name in REQUIRED_CHECKPOINT):
            continue
        try:
            meta = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("loss_method") != method:
            continue
        candidates.append({
            "path": str(path.resolve()),
            "run_id": str(meta.get("run_id") or ""),
            "global_update": int(meta.get("global_update") or 0),
            "cursor_epoch": int(meta.get("cursor_epoch") or 0),
        })
    return sorted(
        candidates,
        key=lambda row: (
            -row["cursor_epoch"], -row["global_update"], row["path"],
        ),
    )


def choose_checkpoint(
    root: Path, method: str, explicit: Path | None,
) -> Path:
    if explicit is not None:
        directory = explicit.expanduser().resolve()
        if not all((directory / name).is_file() for name in REQUIRED_CHECKPOINT):
            raise FileNotFoundError(
                f"Incomplete {method.upper()} Jeff checkpoint: {directory}"
            )
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("loss_method") != method:
            raise ValueError("Checkpoint method differs from the requested experiment")
        return directory
    matches = checkpoints(root, method)
    if not matches:
        raise FileNotFoundError(
            f"No complete Jeff {method.upper()} checkpoint under data/runs/. "
            "Use --checkpoint /absolute/path/to/checkpoint."
        )
    if len(matches) > 1:
        selected = matches[0]
        second = matches[1]
        if (
            selected["cursor_epoch"], selected["global_update"]
        ) == (
            second["cursor_epoch"], second["global_update"]
        ):
            raise ValueError(
                f"Multiple equally progressed {method.upper()} checkpoints. "
                f"Select explicitly with --checkpoint. Candidates: "
                + ", ".join(row["path"] for row in matches[:12])
            )
        print(
            f"[H200 Case-RAG] Highest-progress {method.upper()} checkpoint: "
            f"{selected['path']} (epoch={selected['cursor_epoch']}, "
            f"update={selected['global_update']})",
            flush=True,
        )
    return Path(matches[0]["path"])


def required_data_status(root: Path) -> dict[str, bool]:
    return {name: (root / name).is_file() for name in REQUIRED_INPUTS}


def _frozen_input(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("samples") != SAMPLES
        or manifest.get("seed") != SEED
        or manifest.get("all_within_8192") is not True
    ):
        raise ValueError(f"Unexpected frozen Case-RAG manifest: {folder.name}")
    return manifest


def prepare_inputs(root: Path, *, include_counterfactual: bool) -> dict:
    missing = [name for name, exists in required_data_status(root).items() if not exists]
    if missing:
        raise FileNotFoundError(
            "Official Train/Dev artifacts missing: " + ", ".join(missing)
            + ". On H200 run 'bash run_h200.sh prepare'."
        )
    locate_jeff_tokenizer(root)
    from scripts.prepare_case_rag_counterfactual import prepare as counterfactual
    from scripts.prepare_case_rag_excerpt import prepare as excerpt
    from scripts.prepare_case_rag_pilot import run as pilot
    from tcmdiscern.case_rag_counterfactual import ARMS as CF_ARMS
    from tcmdiscern.case_rag_excerpt import EXCERPT_ARMS

    base = root / "data/case_rag" / PREFIX
    secondary = root / "data/case_rag" / EXCERPT
    if not base.exists():
        print("[CASE-RAG] Building frozen two-case BM25 supports", flush=True)
        pilot(SAMPLES, SEED, "budget2", evidence_budget=2)
    _frozen_input(base)
    if not secondary.exists():
        print("[CASE-RAG] Building extractive/full-case controls", flush=True)
        excerpt(PREFIX, EXCERPT)
    _frozen_input(secondary)
    names = [PREFIX, EXCERPT]
    if include_counterfactual:
        cf = root / "data/case_rag" / COUNTERFACTUAL
        if not cf.exists():
            print("[CASE-RAG] Building diagnostic label/order stress tests", flush=True)
            counterfactual(PREFIX, COUNTERFACTUAL)
        _frozen_input(cf)
        names.append(COUNTERFACTUAL)
    suites = load_suites(tuple(names))
    expected_ids = [f"dev-{i:06d}" for i in sorted(Random(SEED).sample(range(5486), SAMPLES))]
    for name, rows in suites.items():
        if [row["id"] for row in rows["baseline"]] != expected_ids:
            raise ValueError(f"{name}: Dev subset was changed")
        if rows["baseline"] != suites[PREFIX]["baseline"]:
            raise ValueError(f"{name}: original patient/option inputs differ")
        if name != PREFIX and rows["paired2"] != suites[PREFIX]["paired2"]:
            raise ValueError(f"{name}: source cases differ")
        manifest = _frozen_input(root / "data/case_rag" / name)
        if name == EXCERPT and tuple(manifest["conditions"]) != EXCERPT_ARMS:
            raise ValueError("Incorrect excerpt controls")
        if name == COUNTERFACTUAL and tuple(manifest["conditions"]) != CF_ARMS:
            raise ValueError("Incorrect counterfactual controls")
    # The read-only source audit rederives each excerpt from the exact Train
    # positions. On H200, prevent a wrong reconstruction from reaching Jeff.
    from scripts.audit_case_rag_excerpt_provenance import verify
    verify(PREFIX, EXCERPT)
    plan = plan_identical_inferences(suites)
    return {
        "tags": names,
        "total_arms": len(plan),
        "unique_arms": sum(row["reuse"] is None for row in plan),
        "unique_forwards_for_300": sum(row["rows"] for row in plan if row["reuse"] is None),
        "inputs_checked": True,
        "suites": suites,
        "plan": plan,
    }


def check(root: Path, method: str, model_dir: Path, *, include_counterfactual: bool) -> dict:
    tags = [PREFIX, EXCERPT]
    if include_counterfactual:
        tags.append(COUNTERFACTUAL)
    generated = {
        tag: (root / "data/case_rag" / tag / "manifest.json").is_file()
        for tag in tags
    }
    try:
        tokenizer_dir = str(locate_jeff_tokenizer(root, model_dir=model_dir))
    except FileNotFoundError:
        tokenizer_dir = None
    from scripts.preflight_jeff_training import inspect_checkpoint
    model_status = inspect_checkpoint(model_dir)
    return {
        "status": "CASE_RAG_READINESS_CHECK",
        "method": method,
        "official_artifacts": required_data_status(root),
        "model_directory": str(model_dir),
        "model_weights_present": model_status["weights_present"],
        "model_decision_head_present": model_status["decision_head_present"],
        "released_model_metadata_matches": model_status["release_metadata_matches"],
        "tokenizer_directory": tokenizer_dir,
        "frozen_suites_prepared": generated,
        "available_checkpoints": checkpoints(root, method)[:12],
        "test_opened": False,
        "next_command": f"bash run_case_rag_h200.sh {method} --limit 5",
    }


def run(
    *,
    root: Path,
    method: str,
    checkpoint: Path | None,
    model_dir: Path,
    mode: str,
    limit: int | None,
    include_counterfactual: bool,
    run_id: str | None,
    require_baseline_parity: bool,
) -> dict:
    if mode == "check":
        return check(root, method, model_dir, include_counterfactual=include_counterfactual)
    import scripts.run_case_rag_batch_jeff as batch

    # The batch runner is the only owner of paths and scientific scoring.
    # CLI orchestration may be unit-tested in a separate temporary ROOT.
    if root.resolve() != batch.ROOT.resolve():
        raise ValueError("Runner root differs from project workspace")
    planned = prepare_inputs(root, include_counterfactual=include_counterfactual)
    summary = {
        k: v for k, v in planned.items() if k not in {"suites", "plan"}
    }
    if mode == "prepare":
        return {"status": "CASE_RAG_INPUTS_VERIFIED", **summary, "test_opened": False}
    chosen = choose_checkpoint(root, method, checkpoint)
    from scripts.preflight_jeff_training import inspect_checkpoint
    inspected = inspect_checkpoint(model_dir)
    if not (
        inspected["weights_present"]
        and inspected["decision_head_present"]
        and inspected["release_metadata_matches"] is True
        and inspected["tokenizer_present"]
    ):
        raise FileNotFoundError(
            "Pinned released 31B Jeff model is incomplete. On H200 use "
            "'bash run_h200.sh pilot' first."
        )
    if limit is not None and not 1 <= limit <= SAMPLES:
        raise ValueError(f"--limit must be in 1..{SAMPLES}")
    suffix = f"-n{limit}" if limit is not None and limit != SAMPLES else ""
    label = "all" if include_counterfactual else "core"
    name = run_id or f"case-rag-{method}-{label}{suffix}"
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ValueError("Unsafe run-id")
    if limit is not None:
        suites = batch.load_suites(tuple(planned["tags"]), limit=limit)
        plan = batch.plan_identical_inferences(suites)
    else:
        suites, plan = planned["suites"], planned["plan"]
    report = batch.execute(
        suites,
        plan,
        model_dir=model_dir,
        checkpoint=chosen,
        expected_method=method,
        run_id=name,
        limit=limit,
        require_baseline_parity=require_baseline_parity,
    )
    return {
        **report,
        "checkpoint": str(chosen),
        "run_id": name,
        "tags": planned["tags"],
        "private_logits": [
            f"data/case_rag/{tag}/scores/{name}/" for tag in planned["tags"]
        ],
        "public_reports": [
            f"experiments/case_rag_batch_{tag}_{name}.json"
            for tag in planned["tags"]
        ],
        "gold_used_for_model_input": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("s1", "s2", "s3", "s4"), nargs="?", default="s2")
    parser.add_argument("--check", action="store_true", help="inspect files without GPU or writes")
    parser.add_argument("--prepare-only", action="store_true", help="generate/verify inputs without GPU")
    parser.add_argument("--all", action="store_true",
                        help="include seven counterfactual stress controls")
    parser.add_argument("--checkpoint", type=Path, help="explicit trained checkpoint")
    parser.add_argument("--model-dir", type=Path,
                        default=Path(os.environ.get(
                            "H200_MODEL_DIR", str(ROOT / "data/models/Jeff-1.1-Large"),
                        )))
    parser.add_argument("--limit", type=int, help="distinct smoke run ID when <300")
    parser.add_argument("--run-id", help="resume-safe distinct run ID")
    parser.add_argument("--require-baseline-parity", action="store_true",
                        help="require exact saved S2 epoch2 prediction match")
    args = parser.parse_args(argv)
    if args.check and args.prepare_only:
        parser.error("--check and --prepare-only are mutually exclusive")
    if args.require_baseline_parity and args.method != "s2":
        parser.error("Frozen baseline parity applies only to the S2 checkpoint")
    mode = "check" if args.check else "prepare" if args.prepare_only else "run"
    result = run(
        root=ROOT, method=args.method, checkpoint=args.checkpoint,
        model_dir=args.model_dir.expanduser().resolve(),
        mode=mode, limit=args.limit,
        include_counterfactual=args.all, run_id=args.run_id,
        require_baseline_parity=args.require_baseline_parity,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
