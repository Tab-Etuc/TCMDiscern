"""Paired, checkpoint-matched evaluation of Case-RAG excerpting styles.

Compares original frozen-prefix supports against query-focused *same* Train
cases, labels and options. Model logits must exist from real H200 inference.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from scripts.run_case_rag_jeff import load_inputs, read_condition_logits
from tcmdiscern.case_rag import transitive_group_ids
from tcmdiscern.datasets import clinical_text, load_records

ROOT = Path(__file__).resolve().parents[1]


def predictions(
    folder: Path,
    output: Path,
) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    rows = load_inputs(folder)
    result = {}
    for condition, input_rows in rows.items():
        logits = read_condition_logits(output / f"{condition}.jsonl", input_rows)
        result[condition] = [
            row["options"][int(np.argmax(scores))]["id"]
            for row, scores in zip(input_rows, logits, strict=True)
        ]
    return result, rows


def compare(budget: int, *, reference_run: str, focused_run: str) -> dict:
    if budget not in (2, 4):
        raise ValueError("Only 2- and 4-case matched comparisons are supported")
    for value in (reference_run, focused_run):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
            raise ValueError("Unsafe run name")
    reference_tag = "budget2_0300_20261116" if budget == 2 else "pilotclean_0300_20261116"
    focus_tag = f"focused{budget}_0300_20261116"
    original = ROOT / "data/case_rag" / reference_tag
    focused = ROOT / "data/case_rag" / focus_tag
    original_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    focused_manifest = json.loads((focused / "manifest.json").read_text(encoding="utf-8"))
    if focused_manifest.get("source_tag") != reference_tag:
        raise ValueError("Focused prompts were not derived from the frozen original")
    if focused_manifest["seed"] != original_manifest["seed"]:
        raise ValueError("Frozen sample seed differs")
    if (original / "provenance.jsonl").read_bytes() != (
        focused / "provenance.jsonl"
    ).read_bytes():
        raise ValueError("Training support identities differ between styles")
    ref_out = original / "scores" / reference_run
    focus_out = focused / "scores" / focused_run
    ref_identity = json.loads((ref_out / "run_manifest.json").read_text(encoding="utf-8"))
    focus_identity = json.loads((focus_out / "run_manifest.json").read_text(encoding="utf-8"))
    matched_keys = (
        "model_dir", "checkpoint", "checkpoint_method",
        "checkpoint_run_id", "cases", "conditions",
    )
    if any(ref_identity[k] != focus_identity[k] for k in matched_keys):
        raise ValueError("Different Jeff checkpoints or sample definitions were compared")
    ref_preds, ref_rows = predictions(original, ref_out)
    focus_preds, focus_rows = predictions(focused, focus_out)
    for condition in ref_rows:
        for left, right in zip(ref_rows[condition], focus_rows[condition], strict=True):
            if (
                left["id"] != right["id"]
                or left["options"] != right["options"]
                or left["state"]["chief_complaint"] != right["state"]["chief_complaint"]
                or left["state"]["description"] != right["state"]["description"]
                or left["state"]["detection"] != right["state"]["detection"]
            ):
                raise ValueError("Patient or candidate differs between excerpt styles")
    if ref_preds["baseline"] != focus_preds["baseline"]:
        raise ValueError("Identical no-RAG baseline changed across style runs")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    ids = [int(row["id"].removeprefix("dev-")) for row in ref_rows["baseline"]]
    gold = [str(dev[i]["norm_syndrome"]) for i in ids]
    classes = sorted({str(row["norm_syndrome"]) for row in train})
    if len(classes) != 148:
        raise ValueError("Expected 148 original syndrome labels")
    source_text = {clinical_text(row) for row in train}
    unseen = np.array([clinical_text(dev[i]) not in source_text for i in ids], dtype=bool)
    group = transitive_group_ids(dev)
    group_ids = [group[i] for i in ids]
    partitions = [
        np.asarray([j for j, v in enumerate(group_ids) if v == g], dtype=int)
        for g in sorted(set(group_ids))
    ]
    rng = np.random.default_rng(20261116)
    report = {
        "status": "REAL_JEFF_PREFIX_VS_FOCUSED_EVIDENCE_ABLATION",
        "budget": budget,
        "case_count": len(gold),
        "checkpoint_method": ref_identity["checkpoint_method"],
        "saved_source_cases_equal": True,
        "baseline_predictions_equal": True,
        "conditions": {},
        "test_used": False,
    }
    for condition in ref_rows:
        before = np.array([p == y for p, y in zip(ref_preds[condition], gold, strict=True)])
        after = np.array([p == y for p, y in zip(focus_preds[condition], gold, strict=True)])
        rescue = int((after & ~before).sum())
        broke = int((before & ~after).sum())
        differences = []
        for _ in range(600):
            indices = np.concatenate([partitions[i] for i in rng.integers(0, len(partitions), size=len(partitions))])
            differences.append(100 * float(after[indices].mean() - before[indices].mean()))
        report["conditions"][condition] = {
            "prefix_top1_percent": round(100 * float(before.mean()), 4),
            "focused_top1_percent": round(100 * float(after.mean()), 4),
            "delta_top1_pp": round(100 * float(after.mean() - before.mean()), 4),
            "delta_macro_f1_148_pp": round(100 * float(
                f1_score(gold, focus_preds[condition], labels=classes, average="macro", zero_division=0)
                - f1_score(gold, ref_preds[condition], labels=classes, average="macro", zero_division=0)
            ), 4),
            "delta_top1_no_train_exact_overlap_pp": (
                round(100 * float(after[unseen].mean() - before[unseen].mean()), 4)
                if unseen.any() else None
            ),
            "rescued": rescue,
            "broken": broke,
            "mcnemar_p": float(binomtest(min(rescue, broke), rescue + broke).pvalue) if rescue + broke else 1.0,
            "delta_top1_pp_group_bootstrap_95ci": [
                round(float(v), 4) for v in np.percentile(differences, [2.5, 97.5])
            ],
            "prefix_mean_tokens": original_manifest["input_tokens"][condition]["mean"],
            "focused_mean_tokens": focused_manifest["input_tokens"][condition]["mean"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-budget", type=int, choices=(2, 4), required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--focused-run", required=True)
    args = parser.parse_args()
    result = compare(
        args.evidence_budget,
        reference_run=args.reference_run,
        focused_run=args.focused_run,
    )
    destination = ROOT / "experiments" / (
        f"case_rag_focused{args.evidence_budget}_vs_prefix_{args.focused_run}.json"
    )
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
