"""Run missing S1/S3 full-Dev source-attribution controls with one Jeff 31B load.

The first full-Dev pass is already finished: S2 has all four arms and S1/S3
each have Global2. This follow-up computes only their missing Random2,
Label-Hide and Label-Only arms. Seed 2/3 training is intentionally out of scope.

H200:
    bash run_case_rag_followup_h200.sh --check
    bash run_case_rag_followup_h200.sh

Resume by executing the same command. 0 new forwards => 0 model loads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts import run_case_rag_paper_h200 as paper
from scripts.evaluate_case_rag_paper_dev import evaluate
from tcmdiscern.case_rag_identity import validate_saved_identity
from tcmdiscern.jeff_checkpoint import restore_checkpoint

METHODS = ("s3", "s1")  # highest Top-1 first; S2 already has all these controls
CONTROLS = ("global2_labelhide", "global2_labelonly", "random2")


def preflight(
    root: Path = paper.ROOT,
    model_dir: Path = paper.MODEL_PATH,
) -> tuple[dict, list[dict]]:
    """Reject incomplete original GPU results, stale identities and broken partials."""
    inputs, original_stages = paper.preflight(root, model_dir)
    stages_by_method = {stage["method"]: stage for stage in original_stages}
    selected = []
    for method in METHODS:
        stage = stages_by_method[method]
        if stage["pending"]:
            raise ValueError(
                f"{method} original Global2 experiment is not complete; "
                f"finish run_case_rag_paper_h200.sh first: {stage['pending']}"
            )
        directory = paper._scores_path(root, method)
        identity_file = directory / "run_manifest.json"
        if not identity_file.is_file():
            raise ValueError(f"{method}: frozen original run identity is missing")
        validate_saved_identity(
            json.loads(identity_file.read_text(encoding="utf-8")),
            stage["identity"],
            strict=True,
        )
        remaining = {}
        for arm in CONTROLS:
            done = paper._saved_progress(directory, arm, inputs[arm])
            if done < len(inputs[arm]):
                remaining[arm] = len(inputs[arm]) - done
        selected.append(
            {
                "method": method,
                "checkpoint": stage["checkpoint"],
                "directory": directory,
                "remaining": remaining,
            }
        )
        print(
            f"[FOLLOWUP PREFLIGHT] {method.upper()} PASS; remaining={remaining}",
            flush=True,
        )
    return inputs, selected


def run(
    root: Path = paper.ROOT,
    model_dir: Path = paper.MODEL_PATH,
    *,
    check_only: bool = False,
    session_factory=None,
) -> dict:
    from tcmdiscern.case_rag_runtime import run_resumable

    inputs, stages = preflight(root, model_dir)
    planned = sum(sum(stage["remaining"].values()) for stage in stages)
    if check_only:
        return {
            "status": "CASE_RAG_FOLLOWUP_PREFLIGHT_PASSED",
            "cases": 5486,
            "methods": list(METHODS),
            "arms": list(CONTROLS),
            "planned_new_forwards": planned,
            "new_backbone_loads": 0,
            "official_test_used": False,
        }

    session = None
    results = []
    for stage in stages:
        method = stage["method"]
        if stage["remaining"]:
            if session is None:
                if session_factory is None:
                    from tcmdiscern.h200.session import JeffGPUSession

                    session_factory = JeffGPUSession
                session = session_factory(model_dir)
            decider = session.original().decider
            restored = restore_checkpoint(decider, stage["checkpoint"])
            if restored.get("method") != method:
                raise ValueError(f"Restored LoRA/Head differs from {method}")
            if session.model_weight_loads != 1:
                raise RuntimeError("Unexpected repeated released Jeff 31B load")
            for arm in CONTROLS:
                if arm not in stage["remaining"]:
                    continue
                path = stage["directory"] / f"{arm}.jsonl"
                outcome = run_resumable(inputs[arm], decider, path)
                print(
                    f"[FOLLOWUP] {method}/{arm}: {outcome['source']} "
                    f"new forwards={outcome['new_forwards']}",
                    flush=True,
                )
        for arm in CONTROLS:
            # Independently revalidate all completed outputs, including existing arms.
            path = stage["directory"] / f"{arm}.jsonl"
            paper.read_condition_logits(path, inputs[arm])
        report = evaluate(method, root=root, strict=True)
        if not all(arm in report["methods"] for arm in CONTROLS):
            raise RuntimeError(f"{method}: follow-up score report lacks an attribution control")
        results.append(
            {
                "method": method,
                "report": f"experiments/case_rag_paper_full_dev_{method}_20261117.json",
                "controls": {
                    name: {
                        key: report["methods"][name][key]
                        for key in (
                            "top1_percent",
                            "macro_f1_148_percent",
                            "tail_37_macro_f1_percent",
                        )
                    }
                    for name in CONTROLS
                },
            }
        )
    loads = session.model_weight_loads if session is not None else 0
    if loads not in (0, 1):
        raise RuntimeError("More than one released Jeff weight load")
    return {
        "status": "CASE_RAG_FOLLOWUP_COMPLETED",
        "backbone_loads": loads,
        "new_forwards_planned": planned,
        "methods": results,
        "official_test_used": False,
        "seed_2_3_deferred": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("H200_MODEL_DIR", str(paper.MODEL_PATH))),
    )
    args = parser.parse_args()
    report = run(
        paper.ROOT,
        args.model_dir.expanduser().resolve(),
        check_only=args.check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
