"""Official TCM-SD data loading and aggregate audit helpers.

Raw clinical records remain inside data/ and are never committed.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")
EXPECTED_COUNTS = {"train": 43180, "dev": 5486, "test": 5486}
CLINICAL_FIELDS = ("chief_complaint", "description", "detection")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read the author's JSONL files (with .json suffix), or JSON arrays."""
    with path.open(encoding="utf-8-sig") as stream:
        first = stream.read(1)
        stream.seek(0)
        if first == "[":
            records = json.load(stream)
        else:
            records = [json.loads(line) for line in stream if line.strip()]
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected a JSONL file or JSON array of records: {path}")
    return records


def clinical_text(record: dict[str, Any], *, include_four_examinations: bool = True) -> str:
    """Construct patient-only input; diagnosis fields are deliberately excluded."""
    fields = CLINICAL_FIELDS if include_four_examinations else CLINICAL_FIELDS[:2]
    return "\n".join(f"{field}: {str(record.get(field) or '').strip()}" for field in fields)


def normalized_input(record: dict[str, Any], *, full: bool) -> str:
    return re.sub(r"\s+", "", clinical_text(record, include_four_examinations=full))


def audit_tcmsd(data_dir: Path) -> dict[str, Any]:
    """Produce aggregate statistics; no record text or patient identifiers in results."""
    all_splits = {name: load_records(data_dir / f"{name}.json") for name in SPLITS}
    vocab = [
        line.strip()
        for line in (data_dir / "syndrome_vocab.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    knowledge = load_records(data_dir / "syndrome_knowledge.json")

    counts = {name: len(rows) for name, rows in all_splits.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Official TCM-SD split count mismatch: {counts}")

    all_labels: set[str] = set()
    issues: dict[str, Any] = {}
    for name, rows in all_splits.items():
        missing = Counter()
        empty = Counter()
        labels = Counter()
        ids: list[str] = []
        for row in rows:
            for field in (*CLINICAL_FIELDS, "norm_syndrome", "user_id"):
                if field not in row:
                    missing[field] += 1
                elif row[field] is None or str(row[field]).strip() == "":
                    empty[field] += 1
            gold = str(row.get("norm_syndrome") or "").strip()
            if gold:
                labels[gold] += 1
                all_labels.add(gold)
            if row.get("user_id") is not None:
                ids.append(str(row["user_id"]))
        issues[name] = {
            "schema_keys": sorted({key for row in rows[:10] for key in row}),
            "missing_fields": dict(missing),
            "empty_fields": dict(empty),
            "unique_labels": len(labels),
            "most_common_label_count": labels.most_common(1)[0][1] if labels else 0,
            "least_common_label_count": min(labels.values()) if labels else 0,
            "repeated_user_id_rows": len(ids) - len(set(ids)),
        }
    if len(all_labels) != 148:
        raise ValueError(f"Official TCM-SD label count mismatch: {len(all_labels)}")
    if any(s["missing_fields"] or s["empty_fields"].get("norm_syndrome") for s in issues.values()):
        raise ValueError(f"TCM-SD contains missing required input or gold fields: {issues}")

    patient_ids_in_train = {str(row["user_id"]) for row in all_splits["train"]}
    patient_overlap = {
        split: sum(str(row["user_id"]) in patient_ids_in_train for row in all_splits[split])
        for split in ("dev", "test")
    }
    disease_counts = {
        split: {
            "lcd_id_count": len({str(row.get("lcd_id") or "") for row in rows}),
            "lcd_name_count": len({str(row.get("lcd_name") or "") for row in rows}),
            "gold_label_literal_in_clinical_text": sum(
                str(row["norm_syndrome"]) in clinical_text(row) for row in rows
            ),
        }
        for split, rows in all_splits.items()
    }

    overlap: dict[str, dict[str, int]] = {}
    for full in (False, True):
        mode = "full_four_examinations" if full else "chief_complaint_and_history"
        train_by_text: dict[str, set[str]] = defaultdict(set)
        for row in all_splits["train"]:
            train_by_text[normalized_input(row, full=full)].add(str(row["norm_syndrome"]))
        mode_stat: dict[str, int] = {}
        for split in ("dev", "test"):
            matches = [
                row
                for row in all_splits[split]
                if normalized_input(row, full=full) in train_by_text
            ]
            conflicting = sum(
                str(row["norm_syndrome"]) not in train_by_text[normalized_input(row, full=full)]
                for row in matches
            )
            mode_stat[f"train_{split}_repeated_text_cases"] = len(matches)
            mode_stat[f"train_{split}_conflicting_label_cases"] = conflicting
        overlap[mode] = mode_stat

    return {
        "dataset": "TCM-SD",
        "expected_total_records": 54152,
        "counts": counts,
        "total_records": sum(counts.values()),
        "unique_labels": len(all_labels),
        "syndrome_knowledge_records": len(knowledge),
        "syndrome_vocab_nonblank_lines": len(vocab),
        "splits": issues,
        "cross_split_text_overlap": overlap,
        "train_patient_id_reuse_in_evaluation_rows": patient_overlap,
        "disease_fields_and_literal_label_checks": disease_counts,
        "test_used_for_training": False,
        "notes": [
            "The train/dev/test files are authoritative original publisher splits.",
            "Report original-split and leakage-controlled performance separately.",
            "No test gold or test records may enter retrieval indexes, hard-negative mining, or training.",
            "This audit does not certify absence of semantic near-duplicates.",
        ],
    }
