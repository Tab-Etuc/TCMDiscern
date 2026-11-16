"""Evaluate existing Jeff S2 checkpoint on fixed four-case RAG prompts.

Usage on H200 after preparing the exact same 300-case inputs there:
  python -m scripts.run_case_rag_jeff --model-dir /models/Jeff-1.1-Large \
    --checkpoint data/runs/s2-seed20261112-e2/resume-0001350 \
    --run-id case-rag-s2-300

This never trains a model. Every option list is identical to baseline.
Private row-level logits stay in ignored data/; public output is aggregate-only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score

from tcmdiscern.case_rag import CONDITIONS, CONDITIONS_2, transitive_group_ids
from tcmdiscern.case_rag_counterfactual import ARMS as CF_ARMS
from tcmdiscern.case_rag_counterfactual import assert_only_allowed_changes
from tcmdiscern.case_rag_excerpt import EXCERPT_ARMS
from tcmdiscern.case_rag_identity import (
    expected_run_identity,
    validate_saved_identity,
    validate_score_only_inputs,
)
from tcmdiscern.case_rag_pilot_eval import pilot_label_coverage
from tcmdiscern.case_rag_runtime import run_resumable
from tcmdiscern.datasets import clinical_text, load_records

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "pilotclean_0300_20261116"
DEFAULT_RUN_ID = "case-rag-s2-300"


def load_inputs(folder: Path, limit: int | None = None) -> dict[str, list[dict]]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    total = manifest["samples"]
    arms = tuple(manifest["conditions"])
    if arms not in (CONDITIONS, CONDITIONS_2, CF_ARMS, EXCERPT_ARMS) or not manifest["all_within_8192"]:
        raise ValueError("Original prompt preparation manifest failed")
    evidence_budget = 2 if arms in (CONDITIONS_2, CF_ARMS, EXCERPT_ARMS) else 4
    if manifest.get("evidence_budget", evidence_budget) != evidence_budget:
        raise ValueError("Evidence budget contradicts the arm definitions")
    n = total if limit is None else limit
    if not 1 <= n <= total:
        raise ValueError("Invalid prompt limit")
    collection: dict[str, list[dict]] = {}
    for condition in arms:
        with (folder / f"{condition}.jsonl").open(encoding="utf-8") as source:
            all_rows = [json.loads(line) for line in source]
        if len(all_rows) != total:
            raise ValueError("Prompt count differs from manifest")
        collection[condition] = all_rows[:n]
    original = collection["baseline"]
    for index, row in enumerate(original):
        if len(row["options"]) != 16 or len({x["id"] for x in row["options"]}) != 16:
            raise ValueError("Baseline missing 16 distinct labels")
        if not re.fullmatch(r"dev-\d{6}", row["id"]):
            raise ValueError("Unexpected case ID format")
        if arms == CF_ARMS:
            assert_only_allowed_changes({arm: collection[arm][index] for arm in arms})
        if arms == EXCERPT_ARMS:
            source = collection["paired2"][index]["state"]["retrieved_cases"]
            for arm in ("querywindow2", "randomwindow2", "fullcase2"):
                entries = collection[arm][index]["state"]["retrieved_cases"]
                for before, after in zip(source, entries, strict=True):
                    if before["candidate_syndrome"] != after["candidate_syndrome"]:
                        raise ValueError("Excerpt ablation changed the source case label")
                if collection[arm][index]["state"]["retrieval_note"] != (
                    collection["paired2"][index]["state"]["retrieval_note"]
                ):
                    raise ValueError("Excerpt ablation changed the source instruction")
        for condition in arms[1:]:
            rag = collection[condition][index]
            if (
                rag["id"] != row["id"] or rag["options"] != row["options"]
                or rag["question"] != row["question"]
                or rag["qtype"] != row["qtype"]
            ):
                raise ValueError("RAG intervention changed the decision task")
            if len(rag["state"].get("retrieved_cases", [])) != evidence_budget:
                raise ValueError("RAG condition has wrong case count")
            if any(rag["state"][f] != row["state"][f]
                   for f in ("chief_complaint", "description", "detection")):
                raise ValueError("RAG condition altered patient input")
    if len({row["id"] for row in original}) != n:
        raise ValueError("Duplicate patient ID in prepared pilot")
    return collection


def read_condition_logits(path: Path, cases: list[dict]) -> np.ndarray:
    with path.open(encoding="utf-8") as source:
        values = [json.loads(line) for line in source]
    if len(values) != len(cases):
        raise ValueError("Incomplete Jeff condition scoring")
    matrix = []
    for value, row in zip(values, cases, strict=True):
        if set(value) != {"case_id", "logits"} or value["case_id"] != row["id"]:
            raise ValueError("Wrong Jeff logits case ID/order")
        matrix.append(value["logits"])
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (len(cases), 16) or not np.isfinite(result).all():
        raise ValueError("Expected sixteen finite Jeff scores per case")
    return result


def check_reference_baseline(
    rows: list[dict], local_logits: Path, reference_logits: Path, *, total_cases: int = 5486,
) -> dict:
    """Check true checkpoint/prompt parity before spending GPU on RAG arms."""
    from tcmdiscern.jeff_scoring import read_model_logits

    frozen = read_model_logits(reference_logits, num_cases=total_cases)
    received = read_condition_logits(local_logits, rows)
    expected = frozen[np.asarray([int(row["id"].removeprefix("dev-")) for row in rows])]
    disagreements = int(np.count_nonzero(expected.argmax(axis=1) != received.argmax(axis=1)))
    absolute = np.abs(expected - received)
    return {
        "sample_count": len(rows),
        "decision_disagreements_vs_saved_s2": disagreements,
        "max_absolute_logit_difference": round(float(absolute.max()), 6),
        "mean_absolute_logit_difference": round(float(absolute.mean()), 6),
        "reference_file": reference_logits.name,
    }


def accuracy(gold: list[str], predictions: list[str]) -> float:
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Mismatched prediction array")
    return float(np.mean([a == b for a, b in zip(gold, predictions, strict=True)]))


def score_counterfactual_outputs(
    collection: dict[str, list[dict]], output: Path,
) -> dict:
    """Quantify sensitivity to the historical source label, order, and provenance."""
    from scipy.stats import binomtest

    originals = collection["baseline"]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    all_labels = sorted({str(row["norm_syndrome"]) for row in train})
    if len(all_labels) != 148:
        raise ValueError("Missing original 148 syndrome labels")
    indices = [int(row["id"].removeprefix("dev-")) for row in originals]
    gold = [str(dev[i]["norm_syndrome"]) for i in indices]
    exact_train = {clinical_text(row) for row in train}
    unseen = np.asarray(
        [clinical_text(dev[i]) not in exact_train for i in indices], dtype=bool,
    )
    groups = transitive_group_ids(dev)
    group_indices = {}
    for i, original_index in enumerate(indices):
        group_indices.setdefault(groups[original_index], []).append(i)
    grouped = [np.asarray(members, dtype=int) for members in group_indices.values()]
    random = np.random.default_rng(20261116)
    bootstrap = random.integers(0, len(grouped), (600, len(grouped)))
    logits = {
        arm: read_condition_logits(output / f"{arm}.jsonl", collection[arm])
        for arm in CF_ARMS
    }
    positions = {arm: score.argmax(axis=1) for arm, score in logits.items()}
    chosen = {
        arm: [original["options"][int(j)]["id"]
              for original, j in zip(originals, pred, strict=True)]
        for arm, pred in positions.items()
    }
    result = {
        "status": "COUNTERFACTUAL_SOURCE_ATTRIBUTION_ANALYSIS",
        "samples": len(originals),
        "samples_no_train_exact_text": int(unseen.sum()),
        "pilot_label_coverage": pilot_label_coverage(train, dev, indices),
        "scope": "Experimental stress test: deliberately false historical labels",
        "test_used": False,
        "methods": {},
        "source_interventions_vs_paired2": {},
    }
    original_correct = np.asarray([
        label == truth for label, truth in zip(chosen["paired2"], gold, strict=True)
    ])
    original_margin = logits["paired2"][:, 0] - logits["paired2"][:, 1]
    for arm in CF_ARMS:
        prediction = chosen[arm]
        correct = np.asarray([label == truth for label, truth in zip(prediction, gold, strict=True)])
        result["methods"][arm] = {
            "top1_percent": round(100 * float(correct.mean()), 4),
            "top1_no_train_exact_text_percent": (
                round(100 * float(correct[unseen].mean()), 4) if unseen.any() else None
            ),
            "macro_f1_148_percent": round(
                100 * float(f1_score(
                    gold, prediction, labels=all_labels, average="macro", zero_division=0,
                )), 4,
            ),
            "macro_f1_observed_gold_classes_percent": round(
                100 * float(f1_score(
                    gold, prediction, labels=sorted(set(gold)), average="macro",
                    zero_division=0,
                )), 4,
            ),
            "correct": int(correct.sum()),
        }
        if arm == "paired2":
            continue
        new_margin = logits[arm][:, 0] - logits[arm][:, 1]
        changed = positions[arm] != positions["paired2"]
        recovered = int((correct & ~original_correct).sum())
        damaged = int((~correct & original_correct).sum())
        changes = []
        for draw in bootstrap:
            sample = np.concatenate([grouped[int(j)] for j in draw])
            changes.append(100 * float(correct[sample].mean() - original_correct[sample].mean()))
        paired_pred = positions["paired2"]
        new_pred = positions[arm]
        result["source_interventions_vs_paired2"][arm] = {
            "prediction_flip_rate_percent": round(100 * float(changed.mean()), 4),
            "top2_pair_flip_rate_percent": round(
                100 * float(np.mean(
                    ((paired_pred == 0) & (new_pred == 1))
                    | ((paired_pred == 1) & (new_pred == 0))
                )), 4,
            ),
            "mean_absolute_top2_margin_change": round(
                float(np.abs(new_margin - original_margin).mean()), 6,
            ),
            "delta_top1_pp": round(
                100 * float(correct.mean() - original_correct.mean()), 4,
            ),
            "rescued_vs_paired2": recovered,
            "broken_vs_paired2": damaged,
            "mcnemar_p_vs_paired2": (
                float(binomtest(min(recovered, damaged), recovered + damaged).pvalue)
                if recovered + damaged else 1.0
            ),
            "delta_top1_pp_cluster_bootstrap_95ci": [
                round(float(v), 4) for v in np.percentile(changes, [2.5, 97.5])
            ],
        }
    # Full 2x2 intervention: true vs swapped source labels crossed with
    # original vs reversed case order. Both clinical texts remain unchanged.
    margins = {
        arm: values[:, 0] - values[:, 1] for arm, values in logits.items()
    }
    base = margins["paired2"]
    swapped = margins["labelswap2"]
    reversed_order = margins["orderflip2"]
    doubled = margins["doubleflip2"]
    label_effect = 0.5 * ((swapped - base) + (doubled - reversed_order))
    order_effect = 0.5 * ((reversed_order - base) + (doubled - swapped))
    interaction = doubled - reversed_order - swapped + base
    result["label_order_factorial"] = {
        "design": "2x2 source label swap x pair order; fixed patient and source case text",
        "mean_label_swap_effect_on_top2_logit_margin": round(float(label_effect.mean()), 6),
        "mean_abs_label_swap_effect_on_top2_logit_margin": round(
            float(np.abs(label_effect).mean()), 6,
        ),
        "mean_order_effect_on_top2_logit_margin": round(float(order_effect.mean()), 6),
        "mean_abs_order_effect_on_top2_logit_margin": round(
            float(np.abs(order_effect).mean()), 6,
        ),
        "mean_abs_label_order_interaction": round(float(np.abs(interaction).mean()), 6),
    }
    result["interpretation"] = (
        "Labelswap2 and doubleflip2 intentionally falsify historical case labels "
        "for stress testing only. The balanced 2x2 factorial distinguishes source-"
        "label changes from prompt ordering; neither predicts standalone diagnostic "
        "utility. Labelhide2 isolates the presence of support labels. Sourceguard2 "
        "tests whether explicit source attribution reduces inappropriate influence."
    )
    return result


def score_excerpt_outputs(
    collection: dict[str, list[dict]], output: Path,
) -> dict:
    """Evaluate clinical-evidence extraction under controlled support identity."""
    from scipy.stats import binomtest

    original = collection["baseline"]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    labels = sorted({str(row["norm_syndrome"]) for row in train})
    if len(labels) != 148:
        raise ValueError("The original full 148-class reference set is missing")
    case_indices = [int(row["id"].removeprefix("dev-")) for row in original]
    gold = [str(dev[i]["norm_syndrome"]) for i in case_indices]
    exact_train = {clinical_text(row) for row in train}
    unseen = np.asarray([
        clinical_text(dev[i]) not in exact_train for i in case_indices
    ], dtype=bool)
    groups = transitive_group_ids(dev)
    group_members = {}
    for j, dev_index in enumerate(case_indices):
        group_members.setdefault(groups[dev_index], []).append(j)
    group_arr = [np.asarray(items, dtype=int) for items in group_members.values()]
    rng = np.random.default_rng(20261116)
    sampled_groups = rng.integers(0, len(group_arr), size=(600, len(group_arr)))
    predictions = {}
    for mode in EXCERPT_ARMS:
        scores = read_condition_logits(output / f"{mode}.jsonl", collection[mode])
        predictions[mode] = [
            row["options"][int(scores[i].argmax())]["id"]
            for i, row in enumerate(collection[mode])
        ]
    paired = predictions["paired2"]
    original_correct = np.asarray([
        p == y for p, y in zip(paired, gold, strict=True)
    ])
    result = {
        "status": "REAL_JEFF_CLINICAL_EXCERPT_ABLATION",
        "samples": len(original),
        "samples_no_train_exact_text": int(unseen.sum()),
        "pilot_label_coverage": pilot_label_coverage(train, dev, case_indices),
        "protocol": "Same two source patient cases and correct historical labels",
        "methods": {},
        "vs_paired2_fixed_prefix": {},
        "test_used": False,
    }
    for mode, predictions_for_arm in predictions.items():
        current_correct = np.asarray([
            p == y for p, y in zip(predictions_for_arm, gold, strict=True)
        ])
        result["methods"][mode] = {
            "top1_percent": round(float(current_correct.mean() * 100), 4),
            "macro_f1_148_percent": round(float(
                100 * f1_score(
                    gold, predictions_for_arm, labels=labels,
                    average="macro", zero_division=0,
                )), 4,
            ),
            "macro_f1_observed_gold_classes_percent": round(
                float(100 * f1_score(
                    gold, predictions_for_arm, labels=sorted(set(gold)),
                    average="macro", zero_division=0,
                )), 4,
            ),
            "top1_no_train_exact_text_percent": (
                round(float(current_correct[unseen].mean() * 100), 4)
                if unseen.any() else None
            ),
            "correct": int(current_correct.sum()),
        }
        if mode == "paired2":
            continue
        rescued = int((current_correct & ~original_correct).sum())
        damaged = int((~current_correct & original_correct).sum())
        diffs = []
        for draw in sampled_groups:
            sample = np.concatenate([group_arr[int(j)] for j in draw])
            diffs.append(100 * float(
                current_correct[sample].mean() - original_correct[sample].mean()
            ))
        result["vs_paired2_fixed_prefix"][mode] = {
            "delta_top1_pp": round(100 * float(
                current_correct.mean() - original_correct.mean()
            ), 4),
            "delta_macro_f1_148_pp": round(100 * float(
                f1_score(
                    gold, predictions_for_arm, labels=labels, average="macro",
                    zero_division=0,
                ) - f1_score(
                    gold, paired, labels=labels, average="macro", zero_division=0,
                )
            ), 4),
            "rescued": rescued, "broken": damaged,
            "mcnemar_p": float(binomtest(
                min(rescued, damaged), rescued + damaged,
            ).pvalue) if rescued + damaged else 1.0,
            "cluster_bootstrap_delta_top1_pp_95ci": [
                round(float(v), 4) for v in np.percentile(diffs, (2.5, 97.5))
            ],
        }
    result["interpretation"] = (
        "Querywindow2 and randomwindow2 retain verbatim case excerpts at "
        "approximately the same context budget as paired2. Fullcase2 tests "
        "the upper context limit with higher tokens. The lexical pilot is not "
        "a measure of syndrome accuracy; only real logits produce that score."
    )
    return result


def score_outputs(
    folder: Path,
    output: Path,
    *,
    limit: int | None = None,
) -> dict:
    collection = load_inputs(folder, limit)
    if tuple(collection) == CF_ARMS:
        return score_counterfactual_outputs(collection, output)
    if tuple(collection) == EXCERPT_ARMS:
        return score_excerpt_outputs(collection, output)
    baseline = collection["baseline"]
    ids = [row["id"] for row in baseline]
    source_dev = load_records(ROOT / "data/tcm-sd/dev.json")
    source_train = load_records(ROOT / "data/tcm-sd/train.json")
    classes = sorted({str(r["norm_syndrome"]) for r in source_train})
    if len(classes) != 148:
        raise ValueError("Missing original 148 classes")
    indices = [int(cid[4:]) for cid in ids]
    gold = [str(source_dev[i]["norm_syndrome"]) for i in indices]
    exact_train = {clinical_text(r) for r in source_train}
    unseen = np.asarray(
        [clinical_text(source_dev[i]) not in exact_train for i in indices], dtype=bool
    )
    all_groups = transitive_group_ids(source_dev)
    group_ids = [all_groups[i] for i in indices]
    unique_groups = sorted(set(group_ids))
    group_members = {
        g: np.asarray([j for j, h in enumerate(group_ids) if h == g], dtype=int)
        for g in unique_groups
    }
    from scipy.stats import binomtest

    predictions: dict[str, list[str]] = {}
    for mode in collection:
        logits = read_condition_logits(output / f"{mode}.jsonl", collection[mode])
        predictions[mode] = [
            row["options"][int(np.argmax(scores))]["id"]
            for row, scores in zip(collection[mode], logits, strict=True)
        ]
    baseline_pred = predictions["baseline"]
    rng = np.random.default_rng(20261116)
    bootstrap = rng.integers(0, len(unique_groups), size=(600, len(unique_groups)))
    report = {
        "status": "REAL_JEFF_ZERO_RETRAINING_RAG_EVALUATION",
        "sample_count": len(gold),
        "samples_exact_train_text_unseen": int(unseen.sum()),
        "pilot_label_coverage": pilot_label_coverage(source_train, source_dev, indices),
        "full_148_class_macro_f1_sample_limit": (
            "Only 300 sampled Dev records; macro-F1 has high uncertainty "
            "for rarely represented classes"
        ),
        "conditions": {},
        "test_used": False,
    }
    for mode in collection:
        pred = predictions[mode]
        c = np.asarray([a == b for a, b in zip(gold, pred, strict=True)])
        b = np.asarray([a == d for a, d in zip(gold, baseline_pred, strict=True)])
        rescued = int((c & ~b).sum())
        broken = int((~c & b).sum())
        cluster_diffs = []
        for replicate in bootstrap:
            row_ids = np.concatenate([group_members[unique_groups[int(g)]] for g in replicate])
            cluster_diffs.append(float((c[row_ids].mean() - b[row_ids].mean()) * 100))
        report["conditions"][mode] = {
            "top1_percent": round(100 * float(c.mean()), 4),
            "macro_f1_148_percent": round(
                100 * float(f1_score(gold, pred, labels=classes,
                                     average="macro", zero_division=0)), 4
            ),
            "top1_no_train_exact_overlap_percent": (
                round(100 * float(c[unseen].mean()), 4) if unseen.any() else None
            ),
            "macro_f1_observed_gold_classes_percent": round(
                100 * float(f1_score(
                    gold, pred, labels=sorted(set(gold)),
                    average="macro", zero_division=0,
                )), 4,
            ),
            "correct": int(c.sum()),
            "rescued_vs_baseline": rescued,
            "broken_vs_baseline": broken,
            "delta_top1_pp_cluster_bootstrap_95ci": [
                round(float(v), 4) for v in np.percentile(cluster_diffs, [2.5, 97.5])
            ],
            "mcnemar_p_vs_baseline": (
                float(binomtest(min(rescued, broken), rescued+broken).pvalue)
                if rescued+broken else 1.0
            ),
        }
    # Predeclared primary test: paired clinical cases vs equal-budget Top-2
    # independent supports. Secondary arms explain why any gain occurs.
    comparison = {}
    suffix = "2" if "paired2" in collection else "4"
    for comparator in (f"balanced{suffix}", f"global{suffix}", f"random{suffix}"):
        paired = predictions[f"paired{suffix}"]
        other = predictions[comparator]
        c = np.asarray([a == y for a, y in zip(paired, gold, strict=True)])
        b = np.asarray([a == y for a, y in zip(other, gold, strict=True)])
        rescue = int((c & ~b).sum())
        breakage = int((~c & b).sum())
        group_delta = []
        for replicate in bootstrap:
            sampled = np.concatenate(
                [group_members[unique_groups[int(g)]] for g in replicate]
            )
            group_delta.append(100 * float(c[sampled].mean() - b[sampled].mean()))
        comparison[comparator] = {
            "delta_top1_pp": round(100 * float(c.mean() - b.mean()), 4),
            "delta_macro_f1_148_pp": round(100 * float(
                f1_score(gold, paired, labels=classes, average="macro", zero_division=0)
                - f1_score(gold, other, labels=classes, average="macro", zero_division=0)
            ), 4),
            "delta_top1_no_train_exact_overlap_pp": (
                round(100 * float(c[unseen].mean() - b[unseen].mean()), 4)
                if unseen.any() else None
            ),
            "rescued_vs_comparator": rescue,
            "broken_vs_comparator": breakage,
            "mcnemar_p": (
                float(binomtest(min(rescue, breakage), rescue + breakage).pvalue)
                if rescue + breakage else 1.0
            ),
            "delta_top1_pp_group_bootstrap_95ci": [
                round(float(v), 4) for v in np.percentile(group_delta, [2.5, 97.5])
            ],
        }
    report[f"paired{suffix}_vs_controls"] = comparison
    return report


def run_inference(
    folder: Path,
    output: Path,
    *,
    model_dir: Path,
    checkpoint: Path,
    limit: int | None = None,
    expected_method: str | None = None,
    require_baseline_parity: bool = False,
) -> dict:
    import torch

    from scripts.preflight_jeff_training import inspect_checkpoint
    from tcmdiscern.jeff_checkpoint import restore_checkpoint
    from tcmdiscern.jeff_config import JEFF_UPSTREAM_CODE_TAG

    if not torch.cuda.is_available():
        raise RuntimeError("Real Jeff inference needs H200-class CUDA GPU")
    from jeff import __version__ as jeff_version
    from jeff.model import JeffModel

    if jeff_version != JEFF_UPSTREAM_CODE_TAG.removeprefix("v"):
        raise RuntimeError("Installed Jeff version differs from pinned protocol")
    inspection = inspect_checkpoint(model_dir)
    if not inspection["weights_present"] or not inspection["decision_head_present"]:
        raise FileNotFoundError("Full released Jeff weights + decision head missing")
    if inspection["release_metadata_matches"] is not True:
        raise ValueError("Jeff released checkpoint metadata mismatch")
    collection = load_inputs(folder, limit)
    # Verify the *actual* patient/evidence/option bytes and checkpoint
    # provenance before allocating Jeff's large released backbone.
    base_identity = expected_run_identity(
        collection, model_dir, checkpoint, source=folder.name,
    )
    if expected_method is not None:
        metadata = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("loss_method") != expected_method:
            raise ValueError("Checkpoint loss_method differs from --expected-method")
    stamp = output / "run_manifest.json"
    if stamp.exists():
        validate_saved_identity(
            json.loads(stamp.read_text(encoding="utf-8")), base_identity,
        )
    elif output.exists() and any(output.glob("*.jsonl*")):
        raise ValueError(
            "Unidentified existing logits/partials cannot be safely resumed; "
            "use a new run-id"
        )
    if output.exists() and any(output.glob("*.jsonl")):
        # Existing artifacts must be complete; partial files are handled by
        # run_resumable after restoring exactly the same checkpoint.
        for mode in collection:
            file = output / f"{mode}.jsonl"
            if file.exists():
                read_condition_logits(file, collection[mode])
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    jeff = JeffModel(
        model_dir, device="cuda", reasoning=False, long_input="refuse",
        cudnn_attention=False, compile=False,
    )
    if jeff.config.get("bidirectional") is not None:
        raise ValueError("Incorrect Jeff backbone architecture")
    checkpoint_info = restore_checkpoint(jeff.decider, checkpoint)
    if expected_method is not None and checkpoint_info.get("method") != expected_method:
        raise ValueError("Checkpoint method does not match --expected-method")
    # Protect partial runs against silently reusing a different checkpoint.
    identity = {
        **base_identity,
        "checkpoint_method": checkpoint_info["method"],
        "checkpoint_run_id": checkpoint_info["source_run_id"],
    }
    if stamp.exists():
        validate_saved_identity(
            json.loads(stamp.read_text(encoding="utf-8")), identity, strict=True,
        )
    else:
        stamp.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    records = {}
    frozen_baseline_check = None
    if require_baseline_parity:
        if checkpoint_info["method"] != "s2":
            raise ValueError("Frozen baseline parity check requires S2")
        frozen_path = ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl"
        if not frozen_path.exists():
            raise FileNotFoundError("The frozen S2 epoch2 reference logits are unavailable")
    for condition in collection:
        path = output / f"{condition}.jsonl"
        records[condition] = run_resumable(collection[condition], jeff.decider, path)
        print(f"[case-rag] {condition}: {records[condition]['cases']} real Jeff decisions", flush=True)
        if condition == "baseline" and require_baseline_parity:
            frozen_baseline_check = check_reference_baseline(
                collection["baseline"], path, frozen_path,
            )
            if frozen_baseline_check["decision_disagreements_vs_saved_s2"] != 0:
                raise ValueError(
                    "Current checkpoint differs from the frozen S2 baseline; "
                    "review the checkpoint before RAG inference"
                )
    return {
        "checkpoint": checkpoint_info,
        "baseline_reference_check": frozen_baseline_check,
        "elapsed_seconds": round(perf_counter()-started, 2),
        "run_conditions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-method", choices=("s1", "s2", "s3", "s4"))
    parser.add_argument("--require-baseline-parity", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.run_id):
        parser.error("Unsafe run-id")
    folder = ROOT / "data/case_rag" / args.tag
    prompts = load_inputs(folder, args.limit)
    if args.preflight:
        print(json.dumps({
            "status": "READY_FOR_H200",
            "rows_per_condition": len(prompts["baseline"]),
            "conditions": list(prompts),
            "evidence_budget": 2 if "paired2" in prompts else 4,
            "candidate_order": "verified equal",
            "gold_injected": False,
            "supports_checkpoint_methods": ["s1", "s2", "s3", "s4"],
            "counterfactual_source_attribution_stress_test": tuple(prompts) == CF_ARMS,
            "extractive_evidence_content_ablation": tuple(prompts) == EXCERPT_ARMS,
            "test_used": False,
        }, ensure_ascii=False, indent=2))
        return
    output = folder / "scores" / args.run_id
    compute_report = {}
    if args.score_only:
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            parser.error("Score-only evaluation requires the original inference manifest")
        validate_score_only_inputs(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            prompts, source=folder.name,
        )
    if not args.score_only:
        if args.model_dir is None or args.checkpoint is None:
            parser.error("Real scoring requires --model-dir and --checkpoint")
        compute_report = run_inference(
            folder, output, model_dir=args.model_dir, checkpoint=args.checkpoint,
            limit=args.limit, expected_method=args.expected_method,
            require_baseline_parity=args.require_baseline_parity,
        )
    report = score_outputs(folder, output, limit=args.limit)
    report["inference"] = compute_report
    summary = ROOT / "experiments" / f"case_rag_jeff_{args.run_id}.json"
    with summary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
