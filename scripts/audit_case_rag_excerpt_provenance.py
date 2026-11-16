"""Verify the existing two-case excerpt ablation against original Train sources.

The entire audit uses stored source positions and reconstructs every excerpt
from the clinical fields; no hidden diagnosis or official Test is used.
"""
from __future__ import annotations

import json
from pathlib import Path

from tcmdiscern.case_rag_excerpt import (
    EXCERPT_ARMS,
    evidence_case,
    normalize,
    validate_excerpts,
)
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records

ROOT = Path(__file__).resolve().parents[1]


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify(source_tag: str = "budget2_0300_20261116",
           excerpt_tag: str = "excerpt2_0300_20261116") -> dict:
    source = ROOT / "data/case_rag" / source_tag
    target = ROOT / "data/case_rag" / excerpt_tag
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("source_tag") != source_tag
        or tuple(manifest["conditions"]) != EXCERPT_ARMS
        or manifest["evidence_budget"] != 2
    ):
        raise ValueError("Excerpt audit manifest changed")
    prior = {arm: lines(source / f"{arm}.jsonl") for arm in ("baseline", "paired2")}
    current = {arm: lines(target / f"{arm}.jsonl") for arm in EXCERPT_ARMS}
    traces = lines(source / "provenance.jsonl")
    count = len(traces)
    if count != 300 or any(len(value) != count for value in current.values()):
        raise ValueError("Expected the same 300 frozen patients")
    if prior["baseline"] != current["baseline"] or prior["paired2"] != current["paired2"]:
        raise ValueError("Original Jeff or paired supports changed")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    audits = {"querywindow2": 0, "randomwindow2": 0, "fullcase2": 0}
    for idx, detail in enumerate(traces):
        case_id = detail["case_id"]
        if prior["baseline"][idx]["id"] != case_id:
            raise ValueError("Frozen patient order differs")
        dev_index = int(case_id.removeprefix("dev-"))
        query = clinical_text(dev[dev_index])
        positions = detail["conditions"]["paired2"]["source_train_positions"]
        original = current["paired2"][idx]
        for arm, mode in (
            ("querywindow2", "query"),
            ("randomwindow2", "random"),
            ("fullcase2", "full"),
        ):
            row = current[arm][idx]
            original_state = {
                k: v for k, v in original["state"].items() if k != "retrieved_cases"
            }
            state = {
                k: v for k, v in row["state"].items() if k != "retrieved_cases"
            }
            if (state != original_state
                    or any(row[key] != original[key] for key in
                           ("id", "options", "question", "qtype"))):
                raise ValueError(f"{arm}: patient or candidate changed")
            supports = row["state"]["retrieved_cases"]
            for source_position, prior_case, case in zip(
                positions, original["state"]["retrieved_cases"], supports, strict=True,
            ):
                train_case = train[source_position]
                label = str(train_case["norm_syndrome"])
                if label != prior_case["candidate_syndrome"]:
                    raise ValueError("Frozen historical case label changed")
                expected = evidence_case(
                    train_case, label, query,
                    mode=mode,
                    seed=20261116 + dev_index * 171 + source_position * 17,
                )
                if case != expected:
                    raise ValueError(f"{arm}: excerpt differs from deterministic source")
                validate_excerpts(prior_case, case, train_case, mode=mode)
                if mode == "full" and any(
                    case[field] != normalize(train_case.get(field))
                    for field in CLINICAL_FIELDS
                ):
                    raise ValueError("Full-case control omitted clinical evidence")
                audits[arm] += 1
    report = {
        "source_tag": source_tag,
        "excerpt_tag": excerpt_tag,
        "patients": count,
        "source_exposures_per_arm": audits,
        "original_patient_and_candidates_equal": True,
        "all_source_cases_and_labels_verified": True,
        "all_original_clinical_excerpts_verified": True,
        "query_gold_used": False,
        "official_test_used": False,
    }
    output = ROOT / "experiments/case_rag_excerpt_source_audit_2026-11-16.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    verify()
