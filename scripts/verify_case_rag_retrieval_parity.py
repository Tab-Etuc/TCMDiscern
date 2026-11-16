"""Verify optimized Case-RAG retrieval exactly matches the frozen pilot.

Reads private Train/Dev and frozen provenance; publishes aggregate parity/timing.
Does not train or use hidden Test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from scripts.prepare_case_rag_pilot import load_candidates
from tcmdiscern.case_rag import CONDITIONS, CaseRAGIndex
from tcmdiscern.datasets import load_records

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "pilotclean_0300_20261116"


def main(tag: str = FROZEN) -> dict:
    directory = ROOT / "data/case_rag" / tag
    provenance = [
        json.loads(line)
        for line in (directory / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    candidate_sets = load_candidates(ROOT / "data/candidates/bm25_dev.jsonl")
    index = CaseRAGIndex(train)
    elapsed_ms = []
    mismatch = 0
    details = []
    for record in provenance:
        case_id = record["case_id"]
        dev_index = int(case_id.removeprefix("dev-"))
        query = dev[dev_index]
        labels = candidate_sets[dev_index]
        started = perf_counter()
        pool = index._supports(query, labels)
        elapsed_ms.append((perf_counter() - started) * 1000)
        for condition in CONDITIONS:
            _, selected = index.make_row(
                query, labels, case_id=case_id,
                condition=condition, seed=20261116 + dev_index,
                precomputed_items=pool,
            )
            reference = record["conditions"][condition]
            for field in ("source_train_positions", "support_labels", "support_groups"):
                if selected[field] != reference[field]:
                    mismatch += 1
                    if len(details) < 6:
                        details.append({"case_id": case_id, "condition": condition, "field": field})
    result = {
        "frozen_tag": tag,
        "cases": len(provenance),
        "conditions": len(CONDITIONS),
        "compared_source_selections": len(provenance) * len(CONDITIONS),
        "source_mismatches": mismatch,
        "mismatch_examples": details,
        "retrieval_mean_ms": round(mean(elapsed_ms), 3),
        "retrieval_p50_ms": round(float(np.percentile(elapsed_ms, 50)), 3),
        "retrieval_p95_ms": round(float(np.percentile(elapsed_ms, 95)), 3),
        "scope": "CPU reference-only; no Jeff inference; official Test untouched",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if mismatch:
        raise AssertionError("Optimized retrieval changed frozen model evidence")
    path = ROOT / "experiments/case_rag_retrieval_parity_2026-11-16.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=FROZEN)
    arguments = parser.parse_args()
    main(arguments.tag)
