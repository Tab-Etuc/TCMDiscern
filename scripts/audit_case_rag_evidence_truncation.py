"""Audit information lost by fixed-prefix source case truncation (CPU only).

Reads official Train and the frozen Case-RAG provenance. Report contains only
aggregate statistics, never raw patient text or query gold annotations.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tcmdiscern.case_rag import CASE_CHARS, _view
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def overlap(tokens: set[str], excerpt: str) -> int:
    return len(tokens & set(chinese_bigrams(excerpt)))


def summary() -> dict:
    data = ROOT / "data/case_rag/pilotclean_0300_20261116/provenance.jsonl"
    records = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    by_field: dict[str, Counter] = {field: Counter() for field in CLINICAL_FIELDS}
    result = Counter()
    query_lost = 0
    query_full_overlap = 0
    query_excerpt_overlap = 0
    all_positions = set()
    for record in records:
        query = dev[int(record["case_id"].split("-")[1])]
        q_terms = set(chinese_bigrams(clinical_text(query)))
        for condition in ("global4", "balanced4", "paired4", "random4"):
            sources = record["conditions"][condition]["source_train_positions"]
            for position in sources:
                all_positions.add(position)
                row = train[position]
                snippet = _view(row)
                for field in CLINICAL_FIELDS:
                    original = str(row.get(field) or "").strip()
                    limit = CASE_CHARS[field]
                    field_stats = by_field[field]
                    field_stats["cases"] += 1
                    field_stats["truncated"] += len(original) > limit
                    field_stats["original_chars"] += len(original)
                    field_stats["excerpt_chars"] += len(snippet[field])
                query_full_overlap += overlap(q_terms, " ".join(
                    str(row.get(k) or "") for k in CLINICAL_FIELDS
                ))
                query_excerpt_overlap += overlap(q_terms, " ".join(snippet.values()))
                query_lost += (
                    overlap(q_terms, " ".join(str(row.get(k) or "") for k in CLINICAL_FIELDS))
                    > overlap(q_terms, " ".join(snippet.values()))
                )
                result["supports"] += 1
    return {
        "scope": "Frozen 300 Dev × four 4-case arms; official Train support only",
        "total_support_occurrences": result["supports"],
        "unique_source_cases": len(all_positions),
        "per_field": {
            key: {
                "support_occurrences": data["cases"],
                "truncated_percent": round(100 * data["truncated"] / data["cases"], 2),
                "mean_original_chars": round(data["original_chars"] / data["cases"], 1),
                "mean_excerpt_chars": round(data["excerpt_chars"] / data["cases"], 1),
            }
            for key, data in by_field.items()
        },
        "supports_losing_query_lexical_overlap_percent": round(
            100 * query_lost / result["supports"], 2
        ),
        "query_bigrams_matching_original_support_sum": query_full_overlap,
        "query_bigrams_matching_excerpt_support_sum": query_excerpt_overlap,
        "note": "Token overlap is a surrogate for information loss, not clinical diagnostic usefulness",
        "test_used": False,
    }


if __name__ == "__main__":
    result = summary()
    output = ROOT / "experiments/case_rag_truncation_audit_2026-11-16.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
