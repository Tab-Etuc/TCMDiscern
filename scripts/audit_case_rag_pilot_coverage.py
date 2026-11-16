"""Audit frozen 300-case Pilot representation before interpreting Jeff scores.

Only aggregate metrics leave private TCM-SD Train/Dev and case provenance.
Never access the official Test split or construct gold-conditioned prompts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tcmdiscern.case_rag_pilot_eval import pilot_label_coverage
from tcmdiscern.datasets import load_records

ROOT = Path(__file__).resolve().parents[1]


def run(tag: str = "budget2_0300_20261116") -> dict:
    source = ROOT / "data/case_rag" / tag / "provenance.jsonl"
    provenance = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    ids = [int(item["case_id"].removeprefix("dev-")) for item in provenance]
    if len(ids) != 300:
        raise ValueError("Frozen pilot size changed")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    reference = pilot_label_coverage(train, dev, ids)
    frequencies = Counter(str(row["norm_syndrome"]) for row in train)
    selected_labels = Counter(str(dev[i]["norm_syndrome"]) for i in ids)
    # These are fixed reference Train-frequency partitions; Dev gold is used
    # only for reporting how the random Pilot happened to sample them.
    strata = {
        "tail_train_support_le_23": lambda size: size <= 23,
        "mid_train_support_24_to_79": lambda size: 24 <= size <= 79,
        "head_train_support_ge_80": lambda size: size >= 80,
    }
    stratification = {}
    for name, predicate in strata.items():
        members = {label for label, frequency in frequencies.items()
                   if predicate(frequency)}
        represented = members & selected_labels.keys()
        stratification[name] = {
            "total_reference_classes": len(members),
            "observed_in_pilot_classes": len(represented),
            "pilot_case_count": sum(selected_labels[label] for label in represented),
        }
    result = {
        "status": "CPU_PILOT_LABEL_COVERAGE_AUDIT",
        "pilot_tag": tag,
        "protocol": "fixed random 300-case Dev sample; not prevalence-stratified",
        "label_coverage": reference,
        "train_frequency_strata": stratification,
        "recommendation": (
            "Pilot: paired rescue/break, Top-1, CPU/Jeff feasibility. "
            "Locked full 5486 Dev: primary full-148 macro-F1 and head/mid/tail."
        ),
        "test_used": False,
    }
    target = ROOT / "experiments/case_rag_pilot_coverage_2026-11-16.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
