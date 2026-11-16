"""Build counterfactual retrieval controls from the frozen paired2 inputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tcmdiscern.case_rag_counterfactual import ARMS, assert_only_allowed_changes, variants
from tcmdiscern.case_rag_tokenizer import load_jeff_token_counter
from tcmdiscern.jeff_format import semif_messages

ROOT = Path(__file__).resolve().parents[1]


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare(source_name: str, output_name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,70}", output_name):
        raise ValueError("Unsafe experiment name")
    source = ROOT / "data/case_rag" / source_name
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("evidence_budget") != 2:
        raise ValueError("Requires frozen two-case supports")
    original = records(source / "baseline.jsonl")
    clinical = records(source / "paired2.jsonl")
    if len(original) != manifest["samples"] or len(original) != len(clinical):
        raise ValueError("Source length mismatch")
    codec, _ = load_jeff_token_counter(ROOT)
    lengths = {arm: [] for arm in ARMS}
    output = ROOT / "data/case_rag" / output_name
    output.mkdir(parents=True, exist_ok=False)
    files = {arm: (output / f"{arm}.jsonl").open("x", encoding="utf-8") for arm in ARMS}
    try:
        for left, right in zip(original, clinical, strict=True):
            arms = variants(left, right)
            assert_only_allowed_changes(arms)
            for arm, row in arms.items():
                tokens = len(codec.apply_chat_template(
                    semif_messages(row), tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                ))
                if tokens > 8192:
                    raise ValueError("Counterfactual Jeff input exceeds budget")
                lengths[arm].append(tokens)
                files[arm].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in files.values():
            handle.close()
    report = {
        "status": "COUNTERFACTUAL_INPUTS_ONLY",
        "source_tag": source_name,
        "counterfactual": True,
        "samples": len(original),
        "seed": manifest["seed"],
        "conditions": list(ARMS),
        "evidence_budget": 2,
        "candidate_top16_parity": True,
        "baseline_prompt_parity": True,
        "patient_observations_unchanged": True,
        "support_clinical_text_unchanged": True,
        "all_within_8192": True,
        "input_tokens": {
            arm: {
                "minimum": min(tokens), "mean": round(sum(tokens) / len(tokens), 2),
                "maximum": max(tokens),
            }
            for arm, tokens in lengths.items()
        },
        "test_used": False,
        "stress_test_only": True,
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tag", default="budget2_0300_20261116")
    parser.add_argument("--tag", default="cf2_0300_20261116")
    args = parser.parse_args()
    prepare(args.source_tag, args.tag)
