"""Audit negation-context cuts in frozen extractive Case-RAG prompts.

This is a boundary-risk indicator, not a clinical contradiction classifier.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tcmdiscern.case_rag_excerpt import normalize
from tcmdiscern.datasets import CLINICAL_FIELDS, load_records

ROOT = Path(__file__).resolve().parents[1]
NEGATORS = set("無未不非否沒勿")
PUNCT = set("。，；;！!？?，,：:")


def count_boundaries(source: str, chosen: str) -> dict:
    text = normalize(source)
    excerpt = chosen.strip("…")
    if not excerpt:
        return {"truncated": False, "before": False, "after": False,
                "short_context": False, "ambiguous_position": False}
    where = text.find(excerpt)
    if where < 0:
        raise ValueError("Evidence is not an original contiguous substring")
    end = where + len(excerpt)
    before = where > 0 and text[where - 1] in NEGATORS
    after = end < len(text) and text[end - 1] in NEGATORS
    recent = text[max(0, where - 4):where]
    short_context = where > 0 and any(char in NEGATORS for char in recent) and not any(
        char in PUNCT for char in recent
    )
    return {
        "truncated": len(excerpt) < len(text),
        "before": before,
        "after": after,
        "short_context": short_context,
        "ambiguous_position": text.find(excerpt, where + 1) >= 0,
    }


def audit() -> dict:
    train = load_records(ROOT / "data/tcm-sd/train.json")
    root = ROOT / "data/case_rag"
    tag_to_arm = {
        "budget2_0300_20261116": "paired2",
        "focused2_0300_20261116": "paired2",
        "excerpt2_0300_20261116": "querywindow2",
        "excerpt2_0300_20261116:random": "randomwindow2",
    }
    traces = [
        json.loads(line) for line in
        (root / "budget2_0300_20261116/provenance.jsonl").read_text().splitlines()
    ]
    report = {}
    for tag_arm, arm in tag_to_arm.items():
        tag = tag_arm.split(":")[0]
        prompts = [
            json.loads(line) for line in
            (root / tag / f"{arm}.jsonl").read_text().splitlines()
        ]
        counts = Counter()
        for prompt, trace in zip(prompts, traces, strict=True):
            if prompt["id"] != trace["case_id"]:
                raise ValueError("Excerpted patient order changed")
            positions = trace["conditions"]["paired2"]["source_train_positions"]
            for source_position, support in zip(
                positions, prompt["state"]["retrieved_cases"], strict=True,
            ):
                for field in CLINICAL_FIELDS:
                    measure = count_boundaries(
                        str(train[source_position].get(field) or ""),
                        support[field],
                    )
                    counts["fields"] += 1
                    for key, value in measure.items():
                        counts[key] += bool(value)
        report[tag_arm] = {
            "source_field_exposures": counts["fields"],
            **{
                f"{key}_count": counts[key]
                for key in ("truncated", "before", "after", "short_context", "ambiguous_position")
            },
        }
    result = {
        "scope": "300 Dev / 2 historical Train cases × 3 clinical fields",
        "patterns": report,
        "interpretation": "Boundary proxies include duplicate substring ambiguity and cannot prove a clinical contradiction",
        "test_used": False,
    }
    output = ROOT / "experiments/case_rag_negation_boundary_audit_2026-11-16.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    audit()
