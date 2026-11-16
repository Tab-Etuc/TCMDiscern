"""Time the complete local BM25 label -> clinical case retrieval -> pairing path.

CPU-only profile on the frozen 300 official Dev queries. Publishes aggregate
timings and source parity, never patient text or labels of individual cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from scripts.prepare_case_rag_pilot import load_candidates
from tcmdiscern.case_rag import CaseRAGIndex
from tcmdiscern.datasets import clinical_text, load_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/case_rag_cpu_latency_2026-11-16.json"


def summarize_ms(samples: list[float]) -> dict:
    return {
        "mean_ms": round(mean(samples), 3),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
    }


def main() -> dict:
    tag = "pilotclean_0300_20261116"
    source = ROOT / "data/case_rag" / tag / "provenance.jsonl"
    query_indices = [
        int(json.loads(line)["case_id"].removeprefix("dev-"))
        for line in source.read_text(encoding="utf-8").splitlines()
    ]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    saved_candidates = load_candidates(ROOT / "data/candidates/bm25_dev.jsonl")
    started = perf_counter()
    index = CaseRAGIndex(train)
    index_seconds = perf_counter() - started
    times = {"bm25_rank": [], "support_retrieval": [], "paired_selection": [], "shared_rank_and_support": []}
    candidate_mismatch = 0
    for i in query_indices:
        query = dev[i]
        candidates = saved_candidates[i]
        start = perf_counter()
        ranked = index.model.rank([clinical_text(query)], max_k=16)
        times["bm25_rank"].append((perf_counter() - start) * 1000)
        candidate_mismatch += int(ranked[0] != candidates)
        start = perf_counter()
        pool = index._supports(query, candidates)
        times["support_retrieval"].append((perf_counter() - start) * 1000)
        start = perf_counter()
        index.choose(
            query, candidates, condition="paired4", seed=20261116 + i,
            precomputed_items=pool,
        )
        times["paired_selection"].append((perf_counter() - start) * 1000)
        start = perf_counter()
        fused_labels, fused_pool = index.rank_and_supports(query, max_k=16)
        times["shared_rank_and_support"].append((perf_counter() - start) * 1000)
        if fused_labels != candidates or fused_pool != pool:
            raise AssertionError("Fused ranking or support cases changed the original protocol")
    overhead = [
        a + b for a, b in zip(
            times["support_retrieval"], times["paired_selection"], strict=True
        )
    ]
    report = {
        "source": "TCM-SD official Train index and frozen 300 sampled Dev inputs",
        "n": len(query_indices),
        "index_initialize_seconds": round(index_seconds, 3),
        "bm25_candidate_parity_failures": candidate_mismatch,
        "stages": {key: summarize_ms(values) for key, values in times.items()},
        "incremental_rag_retrieval_plus_pair": summarize_ms(overhead),
        "shared_rank_and_support_plus_pair": summarize_ms([
            a + b for a, b in zip(
                times["shared_rank_and_support"], times["paired_selection"], strict=True,
            )
        ]),
        "interpretation": "CPU single-request wall time; independent of 31B Jeff GPU forward",
        "test_used": False,
    }
    if candidate_mismatch:
        raise AssertionError("CPU BM25 output differs from frozen Top-16")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
