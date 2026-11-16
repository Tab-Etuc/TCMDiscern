"""Aggregate truncation audit of Train-only Case-RAG clinical support text.

No private case text is exported. This tests whether naive fixed character
prefix truncation removes query-overlapping clinical bigrams.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from tcmdiscern.case_rag import CASE_CHARS
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def audit(tag: str = "budget2_0300_20261116") -> dict:
    source = ROOT / "data/case_rag" / tag
    traces = [
        json.loads(line) for line in
        (source / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    cutoff = CASE_CHARS
    total_cases = 0
    any_truncated = 0
    field_stats = {key: Counter() for key in CLINICAL_FIELDS}
    for record in traces:
        i = int(record["case_id"].removeprefix("dev-"))
        patient_terms = set(chinese_bigrams(clinical_text(dev[i])))
        for arm in ("balanced" + str(2 if "budget2" in tag else 4),
                    "paired" + str(2 if "budget2" in tag else 4)):
            for position in record["conditions"][arm]["source_train_positions"]:
                total_cases += 1
                was_truncated = False
                for field in CLINICAL_FIELDS:
                    text = re.sub(r"\s+", " ", str(train[position].get(field) or "")).strip()
                    head, tail = text[:cutoff[field]], text[cutoff[field]:]
                    if tail:
                        was_truncated = True
                        field_stats[field]["truncated_cases"] += 1
                        field_stats[field]["removed_characters"] += len(tail)
                        field_stats[field]["all_characters"] += len(text)
                        removed_unique = (set(chinese_bigrams(tail)) & patient_terms) - (
                            set(chinese_bigrams(head)) & patient_terms
                        )
                        field_stats[field]["lost_unique_query_bigrams"] += len(removed_unique)
                        field_stats[field]["lost_overlap_cases"] += bool(removed_unique)
                    else:
                        field_stats[field]["all_characters"] += len(text)
                any_truncated += int(was_truncated)
    report = {
        "source_tag": tag,
        "n_queries": len(traces),
        "n_support_exposures_balanced_plus_paired": total_cases,
        "supports_with_any_field_truncated": any_truncated,
        "supports_with_any_field_truncated_percent": round(
            100 * any_truncated / total_cases, 4,
        ),
        "fields": {
            key: {
                "prefix_char_limit": cutoff[key],
                "truncated_percent_of_supports": round(
                    100 * value["truncated_cases"] / total_cases, 4,
                ),
                "lost_query_overlap_percent_of_supports": round(
                    100 * value["lost_overlap_cases"] / total_cases, 4,
                ),
                "lost_unique_query_bigram_total": value["lost_unique_query_bigrams"],
                "mean_removed_chars_per_support": round(
                    value["removed_characters"] / total_cases, 3,
                ),
                "mean_raw_field_chars": round(value["all_characters"] / total_cases, 3),
            } for key, value in field_stats.items()
        },
        "note": (
            "Overlap is a lexical bigram diagnostic, not a verified clinical "
            "finding or downstream model gain."
        ),
        "test_used": False,
    }
    destination = ROOT / "experiments/case_rag_clinical_truncation_2026-11-16.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    audit()
