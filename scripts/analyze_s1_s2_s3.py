"""Compare frozen S1/S2/S3 full-Dev logits without retraining or Test access.

All reported fixed policies were defined before seeing S3 outcomes. The previously
chosen S1 prior/gate was explored on Dev, so even this frozen reuse remains
descriptive and is not independent validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from scripts.analyze_s1_s2_ensemble import (
    N_CLASSES,
    ROOT,
    S1,
    S2,
    choose,
    metrics,
    paired_bootstrap,
    prepare,
    read_jsonl,
)

S3 = ROOT / "data/jeff/dev_s3-seed20261112-e2-epoch2_k16_5486.jsonl"
OUT = ROOT / "experiments/s1_s2_s3_cpu/summary.json"
RUN_ID = "seed20261112-e2"


def load_s3(path: Path, n: int, reference_ids: list[str]) -> np.ndarray:
    rows = read_jsonl(path)
    if len(rows) != n:
        raise ValueError(f"Expected {n} S3 Dev predictions, got {len(rows)}")
    if [row.get("case_id") for row in rows] != reference_ids:
        raise ValueError("S3 case IDs/order differ from the official Dev retrieval")
    if any(not isinstance(row.get("logits"), list) or len(row["logits"]) != 16 for row in rows):
        raise ValueError("S3 logits must be complete, finite, and 16-dimensional")
    try:
        logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("S3 logits contain non-numeric values") from exc
    if logits.shape != (n, 16) or not np.isfinite(logits).all():
        raise ValueError("S3 logits must be complete, finite, and 16-dimensional")
    return logits


def verify_aggregate_reports(
    raw_pred: dict[str, np.ndarray],
    gold: np.ndarray,
    expected_rows: int,
) -> dict:
    """Ensure supplied full-Dev aggregates describe the same underlying predictions."""
    summary = {}
    reference = None
    for method in ("s1", "s2", "s3"):
        report_path = ROOT / f"experiments/jeff_{method}-{RUN_ID}-epoch2_k16_dev.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        score = report["scores"]
        ckpt = report["checkpoint"]
        if (
            report.get("kind") != "FULL_DEV_FINETUNED"
            or report.get("rows") != expected_rows
            or ckpt.get("method") != method
            or ckpt.get("source_run_id") != f"{method}-{RUN_ID}"
            or score.get("n_cases") != expected_rows
            or score.get("label_classes") != N_CLASSES
            or not ckpt.get("probe_16_logits_match")
        ):
            raise ValueError(f"Inconsistent completed Jeff report: {report_path}")
        correct = int(np.sum(raw_pred[method] == gold))
        if (
            correct != score["top1_correct_n"]
            or abs(correct / expected_rows - score["top1_accuracy"]) > 1e-12
        ):
            raise ValueError(f"Prediction/report accuracy mismatch: {method}")
        metadata = (report["model_revision"], report["attention_backend"], report["tokens"])
        if reference is None:
            reference = metadata
        elif reference != metadata:
            raise ValueError("Models differ in version, backend, or input token budget")
        summary[method] = {
            "source": str(report_path.relative_to(ROOT)),
            "model_revision": report["model_revision"],
            "attention_backend": report["attention_backend"],
            "reported_macro_f1": score["macro_f1"],
            "wall_seconds": report["wall_seconds"],
            "mean_wall_seconds_per_decision": report["mean_wall_seconds_per_decision"],
        }
    return summary


def pair_counts(gold: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    first_ok = first == gold
    second_ok = second == gold
    gain = int(np.count_nonzero(~first_ok & second_ok))
    loss = int(np.count_nonzero(first_ok & ~second_ok))
    discordant = gain + loss
    return {
        "first_only_correct": loss,
        "second_only_correct": gain,
        "net_more_correct_second": gain - loss,
        "mcnemar_exact_two_sided_p": (
            float(binomtest(min(gain, loss), n=discordant, p=0.5).pvalue) if discordant else 1.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", type=Path, default=S3)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 200:
        raise ValueError("Need at least 200 cluster bootstrap replicates")

    gold, cand, z1, z2, counts, groups, no_train_overlap = prepare(S1, S2)
    rows = read_jsonl(ROOT / "data/candidates/bm25_dev.jsonl")
    z3 = load_s3(args.s3, len(gold), [r["case_id"] for r in rows])
    n = len(gold)
    ix = np.arange(n)
    log_prior = np.log(counts[cand])
    zs = {"s1": z1, "s2": z2, "s3": z3}

    positions = {"bm25": np.zeros(n, dtype=int)}
    for method, z in zs.items():
        positions[f"{method}_raw"] = z.argmax(axis=1)
        positions[f"{method}_fixed_gate"] = choose(z - 0.6 * log_prior, 10, 3.0)

    # Unweighted, symmetric controls only: no new thresholds fitted to Dev.
    for pair in (("s1", "s2"), ("s1", "s3"), ("s2", "s3")):
        key = "_".join(pair)
        z = np.mean([zs[m] for m in pair], axis=0)
        positions[f"equal_mean_{key}_fixed_gate"] = choose(z - 0.6 * log_prior, 10, 3.0)
    z_all = np.mean(list(zs.values()), axis=0)
    positions["equal_mean_s1_s2_s3_fixed_gate"] = choose(z_all - 0.6 * log_prior, 10, 3.0)

    predictions = {}
    score = {}
    for name, pos in positions.items():
        if pos.shape != (n,) or np.any((pos < 0) | (pos >= 16)):
            raise ValueError(f"Invalid candidate position: {name}")
        pred = cand[ix, pos]
        predictions[name] = pred
        scored = metrics(gold, pred, counts)
        scored["rescue_vs_bm25"] = int(np.sum((cand[:, 0] != gold) & (pred == gold)))
        scored["break_vs_bm25"] = int(np.sum((cand[:, 0] == gold) & (pred != gold)))
        score[name] = {
            "overall": scored,
            "exclude_exact_train_text": metrics(gold, pred, counts, no_train_overlap),
        }

    raw = {method: predictions[f"{method}_raw"] for method in zs}
    source_report = verify_aggregate_reports(raw, gold, n)
    for method in zs:
        observed_macro = score[f"{method}_raw"]["overall"]["macro_f1"]
        if abs(observed_macro - source_report[method]["reported_macro_f1"]) > 1e-10:
            raise ValueError(f"Prediction/report Macro-F1 mismatch: {method}")

    comparisons = {}
    for first, second in (
        ("s1_raw", "s2_raw"),
        ("s2_raw", "s3_raw"),
        ("s1_raw", "s3_raw"),
        ("s1_fixed_gate", "s2_fixed_gate"),
        ("s1_fixed_gate", "s3_fixed_gate"),
        ("s2_fixed_gate", "s3_fixed_gate"),
        ("s1_fixed_gate", "equal_mean_s1_s2_fixed_gate"),
        ("s1_fixed_gate", "equal_mean_s1_s2_s3_fixed_gate"),
    ):
        comparisons[f"{second}_minus_{first}"] = {
            **pair_counts(gold, predictions[first], predictions[second]),
            "group_bootstrap_95_delta_pp": paired_bootstrap(
                gold,
                predictions[first],
                predictions[second],
                counts,
                groups,
                replicates=args.bootstrap_replicates,
            ),
        }

    anchor = predictions["bm25"]
    oracles = {}
    for name, keys in (
        ("bm25_s1_raw", ("bm25", "s1_raw")),
        ("bm25_s1_s2_raw", ("bm25", "s1_raw", "s2_raw")),
        ("bm25_s1_s2_s3_raw", ("bm25", "s1_raw", "s2_raw", "s3_raw")),
        (
            "bm25_s1_s2_fixed_gate",
            ("bm25", "s1_fixed_gate", "s2_fixed_gate"),
        ),
        (
            "bm25_s1_s2_s3_fixed_gate",
            ("bm25", "s1_fixed_gate", "s2_fixed_gate", "s3_fixed_gate"),
        ),
    ):
        correct = np.logical_or.reduce([predictions[x] == gold for x in keys])
        oracles[name] = {"correct": int(correct.sum()), "top1": float(correct.mean())}

    raw_other_correct = (anchor == gold) | (raw["s1"] == gold) | (raw["s2"] == gold)
    gate_other_correct = (
        (anchor == gold)
        | (predictions["s1_fixed_gate"] == gold)
        | (predictions["s2_fixed_gate"] == gold)
    )
    report = {
        "study": "TCMDiscern S1/S2/S3 single-seed retrospective Dev comparison",
        "status": "single_seed_exploratory_no_test_access",
        "seed": 20261112,
        "epoch": 2,
        "dev_cases": n,
        "label_classes": N_CLASSES,
        "data_integrity": {
            "consistent_case_order_and_16_candidates": True,
            "expected_s1_s2_s3_aggregate_scores_matched": True,
            "exact_train_text_overlap_dev_rows": int(n - len(no_train_overlap)),
            "dev_unique_clinical_text_groups": len(set(groups)),
        },
        "training_run_reports": source_report,
        "uncertainty_estimation": {
            "method": "paired full-clinical-text cluster bootstrap, fixed 148-class F1",
            "replicates": args.bootstrap_replicates,
            "random_seed": 20261115,
        },
        "fixed_policy": {
            "candidate_source": "original BM25 Top16, identical order",
            "raw": "Jeff 16-slot argmax",
            "posthoc": "argmax(logits - 0.6*log(Train label count)) over Top10; switch from BM25 Top1 only if margin >=3",
            "prior_and_gate_scalar_settings_preexisting": True,
            "equal_weight_ensemble_is_post_s3_exploration": True,
            "selection_caution": "Prior and gate were originally explored on this Dev for S1",
        },
        "scores": score,
        "pairwise": comparisons,
        "agreements": {
            f"{a}_{b}_raw": float(np.mean(raw[a] == raw[b]))
            for a, b in (("s1", "s2"), ("s1", "s3"), ("s2", "s3"))
        },
        "oracles_using_gold_for_diagnostic_only": oracles,
        "s3_unique_correct_when_bm25_s1_s2_raw_wrong": int(
            np.sum((raw["s3"] == gold) & ~raw_other_correct)
        ),
        "s3_unique_correct_when_bm25_s1_s2_fixed_gate_wrong": int(
            np.sum((predictions["s3_fixed_gate"] == gold) & ~gate_other_correct)
        ),
        "s3_correct_but_s2_wrong_raw": int(np.sum((raw["s3"] == gold) & (raw["s2"] != gold))),
        "s2_correct_but_s3_wrong_raw": int(np.sum((raw["s2"] == gold) & (raw["s3"] != gold))),
        "limitations": [
            "One completed training seed per method; 3-seed preregistered selection is pending.",
            "Official Dev has repeatedly guided previous method choices, including S1 prior/gate.",
            "Gold-aware oracles are diagnostic upper bounds, not deployable predictors.",
            "Cluster bootstrap quantifies sampling variability for fixed predictions, not model-search uncertainty.",
            "Full Test was not opened or used for model selection.",
            "RAG with additional cases still needs a Jeff 31B GPU inference experiment.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in (
        "bm25",
        "s1_raw",
        "s2_raw",
        "s3_raw",
        "s1_fixed_gate",
        "s2_fixed_gate",
        "s3_fixed_gate",
        "equal_mean_s1_s2_fixed_gate",
        "equal_mean_s1_s2_s3_fixed_gate",
    ):
        m = score[name]["overall"]
        print(
            f"{name:40} accuracy={m['top1']:.4%} "
            f"macro={m['macro_f1']:.4%} tail={m['tail_f1']:.4%} "
            f"rescue/break={m['rescue_vs_bm25']}/{m['break_vs_bm25']}"
        )
    print("S3 unique raw rescues:", report["s3_unique_correct_when_bm25_s1_s2_raw_wrong"])
    print(
        "S3 unique fixed-gate rescues:",
        report["s3_unique_correct_when_bm25_s1_s2_fixed_gate_wrong"],
    )
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
