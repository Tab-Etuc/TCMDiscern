"""Aggregate evidence diversity checks for the frozen 300-case Case-RAG pilot."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from tcmdiscern.datasets import clinical_text, load_records, normalized_input
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def jaccard(a: str, b: str) -> float:
    x, y = set(chinese_bigrams(a)), set(chinese_bigrams(b))
    return len(x & y) / max(1, len(x | y))


def main() -> None:
    source = ROOT / "data/case_rag/pilotclean_0300_20261116"
    rows = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    traces = [json.loads(line) for line in
              (source / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(traces) != 300:
        raise ValueError("Expected original 300 unsupervised sampled Dev queries")
    changed = 0
    j_pair = []
    j_balanced = []
    supported = Counter()
    means = {name: [] for name in ("global4", "balanced4", "paired4", "random4")}
    for record in traces:
        variants = record["conditions"]
        balanced = variants["balanced4"]
        paired = variants["paired4"]
        if balanced["source_train_positions"] != paired["source_train_positions"]:
            changed += 1
        for name, values in means.items():
            entry = variants[name]
            positions = entry["source_train_positions"]
            groups = entry["support_groups"]
            if len(positions) != 4 or len(set(groups)) != 4:
                raise AssertionError("Missing or repeated support groups")
            query = dev[int(record["case_id"].split("-")[1])]
            query_user = str(query.get("user_id") or "").strip()
            for position in positions:
                other = rows[position]
                if normalized_input(other, full=True) == normalized_input(query, full=True):
                    raise AssertionError("Exact query copied into retrieved evidence")
                if query_user and query_user == str(other.get("user_id") or "").strip():
                    raise AssertionError("Same-patient case copied into retrieved evidence")
            values.append(mean(entry["support_scores"]))
            supported[name] += len(set(entry["support_labels"]))
        for name, destination in (("balanced4", j_balanced), ("paired4", j_pair)):
            ids = variants[name]["source_train_positions"]
            destination.extend([
                jaccard(clinical_text(rows[ids[i]]), clinical_text(rows[ids[i+1]]))
                for i in (0, 2)
            ])
    report = {
        "scope": "Original official Train -> 300 preselected official Dev inputs",
        "condition_samples": len(traces),
        "paired_differs_from_independent_balanced_cases": changed,
        "paired_differs_percent": round(100 * changed / len(traces), 2),
        "support_label_count_mean": {
            name: round(supported[name] / len(traces), 3) for name in means
        },
        "support_bm25_score_mean": {
            name: round(mean(values), 3) for name, values in means.items()
        },
        "across_label_patient_similarity": {
            "independent_top2_balanced_mean_jaccard": round(mean(j_balanced), 4),
            "matched_top2_paired_mean_jaccard": round(mean(j_pair), 4),
        },
        "metric_status": "retrieval and prompt diagnostics only; no new model scores",
        "test_used": False,
    }
    destination = ROOT / "experiments/case_rag_input_pilot_2026-11-16.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
