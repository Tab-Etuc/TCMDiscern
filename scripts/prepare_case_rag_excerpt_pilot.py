"""Build a frozen-source Case-RAG excerpting ablation without new retrieval.

Every source position, syndrome label, query, candidate order and seed remains
equal to the original pilot. Only the support text excerpt changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from tcmdiscern.case_rag import CONDITIONS, CONDITIONS_2
from tcmdiscern.case_rag_excerpts import focused_case_view, verify_extractive
from tcmdiscern.case_rag_tokenizer import count_jeff_prompt_tokens, load_jeff_token_counter
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def _overlap(query: dict, source: dict) -> int:
    q = set(chinese_bigrams(clinical_text(query)))
    terms = set(chinese_bigrams(" ".join(str(source.get(f) or "") for f in CLINICAL_FIELDS)))
    return len(q & terms)


def main(evidence_budget: int) -> dict:
    if evidence_budget not in (2, 4):
        raise ValueError("Expected a 2- or 4-case evidence budget")
    conditions = CONDITIONS_2 if evidence_budget == 2 else CONDITIONS
    original_tag = "budget2_0300_20261116" if evidence_budget == 2 else "pilotclean_0300_20261116"
    focused_tag = f"focused{evidence_budget}_0300_20261116"
    old = ROOT / "data/case_rag" / original_tag
    target = ROOT / "data/case_rag" / focused_tag
    original_manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    if (
        tuple(original_manifest["conditions"]) != conditions
        or original_manifest["seed"] != 20261116
        or original_manifest["samples"] != 300
    ):
        raise ValueError("Frozen source pilot changed unexpectedly")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    provenance = [
        json.loads(line)
        for line in (old / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(provenance) == 300
    tok, _ = load_jeff_token_counter(ROOT)
    target.mkdir(parents=True, exist_ok=False)
    counts = {arm: [] for arm in conditions}
    overlap_before = 0
    overlap_after = 0
    total_supports = 0
    streams = {arm: (target / f"{arm}.jsonl").open("x", encoding="utf-8") for arm in conditions}
    try:
        for arm in conditions:
            with (old / f"{arm}.jsonl").open(encoding="utf-8") as source:
                for case_i, line in enumerate(source):
                    row = json.loads(line)
                    reference = provenance[case_i]
                    if reference["case_id"] != row["id"]:
                        raise ValueError("Frozen case order changed")
                    query = dev[int(row["id"].removeprefix("dev-"))]
                    if arm != "baseline":
                        positions = reference["conditions"][arm]["source_train_positions"]
                        if len(positions) != evidence_budget:
                            raise ValueError("Incorrect source case budget")
                        for j, pos in enumerate(positions):
                            expected_label = str(train[pos]["norm_syndrome"])
                            if (
                                row["state"]["retrieved_cases"][j]["candidate_syndrome"]
                                != expected_label
                            ):
                                raise ValueError("Source diagnosis changed")
                            new_text = focused_case_view(train[pos], query)
                            verify_extractive(train[pos], new_text)
                            old_text = row["state"]["retrieved_cases"][j]
                            if arm.startswith("paired"):
                                overlap_before += _overlap(query, old_text)
                                overlap_after += _overlap(query, new_text)
                                total_supports += 1
                            row["state"]["retrieved_cases"][j] = {
                                "candidate_syndrome": expected_label,
                                **new_text,
                            }
                    length = count_jeff_prompt_tokens(tok, row)
                    if length > 8192:
                        raise ValueError("Focused case excerpt exceeds Jeff input limit")
                    counts[arm].append(length)
                    streams[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()
    (target / "provenance.jsonl").write_bytes((old / "provenance.jsonl").read_bytes())
    summary = {
        **original_manifest,
        "study": "Case-RAG source-faithful query-focused excerpt ablation",
        "status": "PROMPTS_READY_NO_NEW_JEFF_SCORES",
        "private_directory": str(target.relative_to(ROOT)),
        "source_tag": original_tag,
        "evidence_style": "single-contiguous-query-window",
        "conditions": list(conditions),
        "evidence_budget": evidence_budget,
        "input_tokens": {
            arm: {
                "minimum": min(values),
                "mean": round(mean(values), 1),
                "maximum": max(values),
            }
            for arm, values in counts.items()
        },
        "paired_supports_examined": total_supports,
        "paired_query_overlap_before": overlap_before,
        "paired_query_overlap_after": overlap_after,
        "no_retrieval_change": True,
        "all_within_8192": True,
        "test_used": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-budget", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    main(args.evidence_budget)
