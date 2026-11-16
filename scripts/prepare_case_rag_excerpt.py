"""Prepare verbatim clinical excerpt controls from frozen paired2 source cases."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from statistics import mean

from tcmdiscern.case_rag_excerpt import (
    EXCERPT_ARMS,
    evidence_case,
    validate_excerpts,
)
from tcmdiscern.case_rag_tokenizer import count_jeff_prompt_tokens, load_jeff_token_counter
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.retrieval import chinese_bigrams

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(source_tag: str, tag: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}", tag):
        raise ValueError("Unsafe experiment tag")
    root = ROOT / "data/case_rag" / source_tag
    source_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("evidence_budget") != 2:
        raise ValueError("Requires two-case frozen supports")
    baseline = read_jsonl(root / "baseline.jsonl")
    paired = read_jsonl(root / "paired2.jsonl")
    provenance = read_jsonl(root / "provenance.jsonl")
    if len(baseline) != len(paired) or len(baseline) != len(provenance):
        raise ValueError("Frozen source alignment mismatch")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    encoder, _ = load_jeff_token_counter(ROOT)
    destination = ROOT / "data/case_rag" / tag
    destination.mkdir(parents=True, exist_ok=False)
    files = {
        arm: (destination / f"{arm}.jsonl").open("x", encoding="utf-8") for arm in EXCERPT_ARMS
    }
    measured = {arm: [] for arm in EXCERPT_ARMS}
    coverage = {arm: [] for arm in EXCERPT_ARMS if arm != "baseline"}
    changed = {arm: 0 for arm in coverage}
    try:
        for base, ordinary, trace in zip(baseline, paired, provenance, strict=True):
            if not (base["id"] == ordinary["id"] == trace["case_id"]):
                raise ValueError("Frozen patient ID mismatch")
            i = int(base["id"].removeprefix("dev-"))
            query_text = clinical_text(dev[i])
            query_terms = set(chinese_bigrams(query_text))
            positions = trace["conditions"]["paired2"]["source_train_positions"]
            if len(positions) != 2:
                raise ValueError("Missing two original Train source positions")
            interventions = {"baseline": deepcopy(base), "paired2": deepcopy(ordinary)}
            for name, mode in (
                ("querywindow2", "query"),
                ("randomwindow2", "random"),
                ("fullcase2", "full"),
            ):
                variant = deepcopy(ordinary)
                examples = []
                for j, source_pos in enumerate(positions):
                    source_case = train[source_pos]
                    original = ordinary["state"]["retrieved_cases"][j]
                    label = str(source_case["norm_syndrome"])
                    if original["candidate_syndrome"] != label:
                        raise ValueError("Original support label does not match Train")
                    support = evidence_case(
                        source_case,
                        label,
                        query_text,
                        mode=mode,
                        seed=20261116 + i * 171 + source_pos * 17,
                    )
                    validate_excerpts(original, support, source_case, mode=mode)
                    examples.append(support)
                    changed[name] += int(
                        any(original[field] != support[field] for field in CLINICAL_FIELDS)
                    )
                variant["state"]["retrieved_cases"] = examples
                interventions[name] = variant
            for arm, variant in interventions.items():
                if variant["options"] != base["options"]:
                    raise ValueError("Extractive evidence changed BM25 candidates")
                if any(
                    variant["state"][field] != base["state"][field] for field in CLINICAL_FIELDS
                ):
                    raise ValueError("Extractive evidence changed the patient's observations")
                count = count_jeff_prompt_tokens(encoder, variant)
                if count > 8192:
                    raise ValueError("Full-case evidence exceeds Jeff maximum input")
                measured[arm].append(count)
                if arm in coverage:
                    for support in variant["state"]["retrieved_cases"]:
                        terms = set(
                            chinese_bigrams(" ".join(support[field] for field in CLINICAL_FIELDS))
                        )
                        coverage[arm].append(len(terms & query_terms))
                files[arm].write(json.dumps(variant, ensure_ascii=False) + "\n")
    finally:
        for handle in files.values():
            handle.close()
    report = {
        "study": "Query-grounded verbatim evidence excerpt vs naive prefix under fixed case selection",
        "source_tag": source_tag,
        "excerpt_audit": True,
        "samples": len(baseline),
        "seed": source_manifest["seed"],
        "evidence_budget": 2,
        "conditions": list(EXCERPT_ARMS),
        "candidate_top16_parity": True,
        "baseline_prompt_parity": True,
        "source_positions_and_labels_fixed": True,
        "support_fields_are_verbatim_substrings": True,
        "all_within_8192": True,
        "input_tokens": {
            arm: {"mean": round(mean(values), 2), "minimum": min(values), "maximum": max(values)}
            for arm, values in measured.items()
        },
        "query_bigram_coverage_per_support_mean": {
            arm: round(mean(values), 3) for arm, values in coverage.items()
        },
        "supports_changed_from_prefix": changed,
        "test_used": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "experiments/case_rag_excerpt_cpu_2026-11-16.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tag", default="budget2_0300_20261116")
    parser.add_argument("--tag", default="excerpt2_0300_20261116")
    args = parser.parse_args()
    prepare(args.source_tag, args.tag)
