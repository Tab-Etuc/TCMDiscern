"""Audit the frozen 2026-11-16 H200 Case-RAG report archive without GPU.

Reads gzipped JSON reports directly (no archive extraction and no Jeff model load).
Usage:
    python -m scripts.analyze_case_rag_archive case_rag_reports.tar.gz
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path

RUN_IDS = {
    "S1": "case-rag-s1-all",
    "S2": "case-rag-s2-core",
    "S3": "case-rag-s3-all",
}
SUITES = {
    "budget": "budget2_0300_20261116",
    "excerpt": "excerpt2_0300_20261116",
    "counterfactual": "cf2factorial_0300_20261116",
}
ARMS = ("baseline", "global2", "balanced2", "paired2", "random2")
CHECKS = ("labelswap2", "labelhide2", "orderflip2", "doubleflip2", "sourceguard2")


def exact_mcnemar_p(rescued: int, broken: int) -> float:
    n = rescued + broken
    if not n:
        return 1.0
    k = min(rescued, broken)
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / (2**n))


def read_reports(archive: Path) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    expected = {
        f"experiments/case_rag_batch_{tag}_{run_id}.json": (method, suite)
        for method, run_id in RUN_IDS.items()
        for suite, tag in SUITES.items()
    }
    found = {}
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            matches = [
                name for name in expected if member.name == name or member.name.endswith("/" + name)
            ]
            if not matches:
                continue
            if len(matches) > 1:
                raise ValueError(f"Ambiguous archive member {member.name}")
            filename = matches[0]
            if filename in found:
                raise ValueError(f"Duplicate report in archive: {filename}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable report {filename}")
            found[filename] = json.load(stream)
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError("Archive missing reports: " + ", ".join(missing))
    return {
        method: {
            suite: found[f"experiments/case_rag_batch_{tag}_{run_id}.json"]
            for suite, tag in SUITES.items()
        }
        for method, run_id in RUN_IDS.items()
    }


def audit(reports: dict) -> dict:
    summary = {
        "status": "CASE_RAG_REPORTS_AUDITED",
        "report_count": 9,
        "sample_per_model": 300,
        "test_used": False,
        "models": {},
        "warnings": [
            "300 Dev cases per model share the same sampled patients; treat model runs as paired, not 900 independent cases",
            "148-class Macro-F1 from only 300 Dev cases has poor rare-class coverage",
            "S1/S2/S3 and retrieval arms are exploratory comparisons; report unadjusted p-values",
            "Exact Train/Dev text non-overlap excludes verbatim duplicates only, not semantic near-duplicates",
            "False historical labels in counterfactual arms assess source-label sensitivity, not clinical reasoning correctness",
        ],
    }
    for method, suites in reports.items():
        budget, excerpt, cf = (suites[name] for name in SUITES)
        if any(value.get("test_used") is not False for value in suites.values()):
            raise ValueError(f"{method}: report opened official Test")
        if set(budget["conditions"]) != set(ARMS):
            raise ValueError(f"{method}: unexpected budget arms")
        if int(budget.get("sample_count", -1)) != 300:
            raise ValueError(f"{method}: expected 300 Dev cases")
        if int(excerpt.get("samples", -1)) != 300 or int(cf.get("samples", -1)) != 300:
            raise ValueError(f"{method}: sample count mismatch")
        baseline = budget["conditions"]["baseline"]
        for suite in (excerpt, cf):
            for arm in ("baseline", "paired2"):
                other = suite["methods"][arm]
                ref = budget["conditions"][arm]
                for key in ("correct", "top1_percent", "macro_f1_148_percent"):
                    if other[key] != ref[key]:
                        raise ValueError(f"{method}: {arm} disagreement across suites: {key}")
        model = {
            "arms": {},
            "retrieval_paired_vs_global2": budget["paired2_vs_controls"]["global2"],
            "excerpt_vs_paired2": excerpt["vs_paired2_fixed_prefix"],
            "counterfactual": {
                "arms": {},
                "source_label_order_factorial": cf["label_order_factorial"],
            },
        }
        for arm in ARMS:
            entry = budget["conditions"][arm]
            rescued = int(entry["rescued_vs_baseline"])
            broken = int(entry["broken_vs_baseline"])
            if entry["correct"] - baseline["correct"] != rescued - broken:
                raise ValueError(f"{method}/{arm}: net rescues inconsistent")
            model["arms"][arm] = {
                "correct": entry["correct"],
                "top1_percent": entry["top1_percent"],
                "delta_top1_pp": round(entry["top1_percent"] - baseline["top1_percent"], 4),
                "macro_f1_148_percent": entry["macro_f1_148_percent"],
                "no_exact_train_text_top1_percent": entry.get(
                    "top1_no_train_exact_overlap_percent"
                ),
                "rescued_vs_baseline": rescued,
                "broken_vs_baseline": broken,
                "mcnemar_exact_p_vs_baseline": exact_mcnemar_p(rescued, broken),
                "cluster_bootstrap_95ci_delta_pp": entry.get(
                    "delta_top1_pp_cluster_bootstrap_95ci"
                ),
            }
        for arm in CHECKS:
            entry = cf["methods"][arm]
            contrast = cf["source_interventions_vs_paired2"][arm]
            model["counterfactual"]["arms"][arm] = {
                "top1_percent": entry["top1_percent"],
                "delta_top1_pp_vs_paired2": contrast["delta_top1_pp"],
                "prediction_flip_rate_percent": contrast["prediction_flip_rate_percent"],
                "mcnemar_p_vs_paired2": contrast["mcnemar_p_vs_paired2"],
            }
        model["pilot_label_coverage"] = budget.get("pilot_label_coverage")
        model["inference"] = {name: suite.get("inference", {}) for name, suite in suites.items()}
        model["no_exact_overlap_n"] = budget["samples_exact_train_text_unseen"]
        model["global2_no_exact_overlap_delta_pp"] = round(
            model["arms"]["global2"]["no_exact_train_text_top1_percent"]
            - model["arms"]["baseline"]["no_exact_train_text_top1_percent"],
            4,
        )
        summary["models"][method] = model
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=Path("case_rag_reports.tar.gz"))
    parser.add_argument("--save", type=Path, help="Optional destination for audited summary JSON")
    args = parser.parse_args()
    report = audit(read_reports(args.archive))
    print("AUDIT: 9/9 reports consistent; official Test untouched")
    print("METHOD  BASELINE  GLOBAL2  DELTA(pp)  MACRO_F1  UNIQUE_TEXT(pp)   RESCUE/BREAK   P")
    for method, item in report["models"].items():
        b, g = item["arms"]["baseline"], item["arms"]["global2"]
        print(
            f"{method:<5}   {b['top1_percent']:6.2f}%   {g['top1_percent']:6.2f}%"
            f"   {g['delta_top1_pp']:+6.2f}     {g['macro_f1_148_percent']:6.2f}%"
            f"       {item['global2_no_exact_overlap_delta_pp']:+6.2f}"
            f"       {g['rescued_vs_baseline']:2}/{g['broken_vs_baseline']:2}"
            f"       {g['mcnemar_exact_p_vs_baseline']:.5f}"
        )
        print(
            f"  paired2-global2={item['retrieval_paired_vs_global2']['delta_top1_pp']:+.2f}pp;"
            f" doubleflip2-paired2={item['counterfactual']['arms']['doubleflip2']['delta_top1_pp_vs_paired2']:+.2f}pp;"
            f" no_exact_overlap_n={item['no_exact_overlap_n']}"
        )
        print("  global2_cluster_bootstrap_95ci=", g["cluster_bootstrap_95ci_delta_pp"])
        print(
            "  pilot_label_coverage=", json.dumps(item["pilot_label_coverage"], ensure_ascii=False)
        )
        timing = item["inference"]["budget"].get("arms", {}).get("global2", {}).get("timing", {})
        if timing:
            print("  global2_inference_timing=", json.dumps(timing, ensure_ascii=False))
        for arm in ("baseline", "paired2"):
            metrics = item["inference"]["budget"].get("arms", {}).get(arm, {}).get("timing", {})
            if metrics:
                print(
                    f"  {arm}_latency_p50_ms=",
                    metrics.get("p50_ms"),
                    "mean_ms=",
                    metrics.get("mean_ms"),
                )
        fullcase = (
            item["inference"]["excerpt"].get("arms", {}).get("fullcase2", {}).get("timing", {})
        )
        if fullcase:
            print(
                "  fullcase2_latency_p50_ms=",
                fullcase.get("p50_ms"),
                "mean_ms=",
                fullcase.get("mean_ms"),
            )
    ranked = sorted(
        (
            (method, report["models"][method]["arms"]["global2"]["mcnemar_exact_p_vs_baseline"])
            for method in RUN_IDS
        ),
        key=lambda value: value[1],
    )
    correction = 0.0
    adjusted = {}
    for index, (method, pvalue) in enumerate(ranked):
        correction = max(correction, min(1.0, (len(ranked) - index) * pvalue))
        adjusted[method] = correction
    report["holm_3_model_global2_vs_baseline"] = adjusted
    print("Holm corrected p for three global2-vs-baseline tests:", adjusted)
    print("CAUTIONS:")
    for warning in report["warnings"]:
        print(" - " + warning)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("SAVED:", args.save)


if __name__ == "__main__":
    main()
