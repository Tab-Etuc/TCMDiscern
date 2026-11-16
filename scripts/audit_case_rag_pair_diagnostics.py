"""Audit whether matched case pairs expose distinctive query evidence.

A strong match between two historical cases can erase the very differences
that Jeff needs for diagnosis. Counts below are lexical proxies only.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from tcmdiscern.case_rag import _view
from tcmdiscern.case_rag_excerpts import focused_case_view
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def _terms(case: dict) -> set[str]:
    return set(chinese_bigrams(" ".join(str(case.get(k) or "") for k in CLINICAL_FIELDS)))


def compute_pair_features(query: dict, a: dict, b: dict) -> dict:
    q = set(chinese_bigrams(clinical_text(query)))
    aa, bb = _terms(a), _terms(b)
    shared = q & aa & bb
    only_a, only_b = (q & aa) - bb, (q & bb) - aa
    all_q = q & (aa | bb)
    return {
        "shared": len(shared),
        "unique_a": len(only_a),
        "unique_b": len(only_b),
        "both_sides_have_distinct_query_terms": int(bool(only_a) and bool(only_b)),
        "query_coverage": len(all_q),
        "source_jaccard": len(aa & bb) / max(1, len(aa | bb)),
    }


def run() -> dict:
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    report = {}
    for budget in (2, 4):
        folder = ROOT / "data/case_rag" / (
            "budget2_0300_20261116" if budget == 2 else "pilotclean_0300_20261116"
        )
        traces = [json.loads(line) for line in (
            folder / "provenance.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        for style in ("prefix", "focused", "full"):
            for condition in (f"balanced{budget}", f"paired{budget}"):
                summaries = []
                for trace in traces:
                    query = dev[int(trace["case_id"].removeprefix("dev-"))]
                    supports = trace["conditions"][condition]["source_train_positions"]
                    for j in range(0, len(supports), 2):
                        a, b = train[supports[j]], train[supports[j + 1]]
                        if style == "prefix":
                            a, b = _view(a), _view(b)
                        elif style == "focused":
                            a = focused_case_view(a, query)
                            b = focused_case_view(b, query)
                        summaries.append(compute_pair_features(query, a, b))
                report[f"{condition}:{style}"] = {
                    "n_pairs": len(summaries),
                    **{
                        f"mean_{k}": round(mean(value[k] for value in summaries), 4)
                        for k in summaries[0]
                    },
                }
    output = {
        "scope": "Frozen 300 sampled official Dev, official Train supports, lexical only",
        "data": report,
        "interpretation": "Shared and distinct bigram proxies; require Jeff outcome evidence",
        "test_used": False,
    }
    (ROOT / "experiments/case_rag_pair_diagnostic_audit_2026-11-16.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    run()
