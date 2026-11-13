"""Full-record, read-only H200 input audit. Only aggregate output; no patient text.

This audit reads every row in the active training and evaluation artifacts and
compares them against the official Train/Dev, full grouped OOF BM25, and the
published Jeff/SemIf schema. Test is checked for schema and count only, never
for performance, model selection or candidate generation.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from scripts.export_jeff_train import load_and_validate_oof
from scripts.verify_jeff_multipos import verify as verify_multi
from scripts.verify_jeff_train import validate_file
from tcmdiscern.datasets import CLINICAL_FIELDS, clinical_text, load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_format import TCM_QUESTION, decision_row
from tcmdiscern.jeff_multi import observed_label_sets

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TRAIN = {
    "s1": "data/jeff/train_oof_k16_epoch02.jsonl",
    "s2_s3_s4": "data/jeff/train_oof_multipos_k16_epoch03.jsonl",
}
TRAIN_COUNT = 43180
DEV_COUNT = 5486


def fail(message: str) -> None:
    raise ValueError("H200 payload BLOCKED: " + message)


def safe_input_scan(records: list[dict], *, split: str, vocab: set[str]) -> dict:
    """Inspect *every* source row, without emitting clinical text or IDs."""
    missing = Counter()
    empties = Counter()
    invalid_types = Counter()
    Unicode = Counter()
    control_locations = []
    unwanted_fields = Counter()
    literal_gold = 0
    lengths = []
    distinct = defaultdict(set)
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            fail(f"{split} row {i}: not a mapping")
        gold = record.get("norm_syndrome")
        if not isinstance(gold, str) or gold not in vocab:
            fail(f"{split} row {i}: gold label outside official vocabulary")
        for field in CLINICAL_FIELDS:
            if field not in record:
                missing[field] += 1
                fail(f"{split} row {i}: missing clinical input {field}")
            value = record[field]
            if value is not None and not isinstance(value, str):
                invalid_types[field] += 1
                fail(f"{split} row {i}: nonstring clinical input {field}")
            if not str(value or "").strip():
                empties[field] += 1
            for char in value or "":
                if char == "\ufffd":
                    Unicode["replacement_character"] += 1
                elif char == "\ufeff":
                    Unicode["bom_character"] += 1
                elif unicodedata.category(char) == "Cc" and char not in "\n\r\t":
                    Unicode["unexpected_controls"] += 1
                    if len(control_locations) < 20:
                        control_locations.append(
                            {"row_zero_based": i, "field": field, "codepoint": f"U+{ord(char):04X}"}
                        )
                elif unicodedata.category(char) == "Cf":
                    Unicode["format_characters"] += 1
        if not any(str(record.get(field) or "").strip() for field in CLINICAL_FIELDS):
            fail(f"{split} row {i}: entirely blank clinical evidence")
        state = clinical_text(record)
        if gold in state:
            literal_gold += 1
        if split != "test":
            distinct[state].add(gold)
        lengths.append(len(state))
        unwanted_fields.update(
            key
            for key in record
            if key
            not in {*CLINICAL_FIELDS, "norm_syndrome", "syndrome", "user_id", "lcd_id", "lcd_name"}
        )
    lengths.sort()
    return {
        "rows_scanned": len(records),
        "schema_all_rows_checked": True,
        "missing_input_fields": dict(missing),
        "empty_input_values": dict(empties),
        "nonstring_values": dict(invalid_types),
        "unicode_warnings": dict(Unicode),
        "unexpected_control_locations_no_raw_text": control_locations,
        "unexpected_schema_fields_present_n": dict(unwanted_fields),
        "gold_literal_in_evidence_n": literal_gold if split != "test" else "SEALED_TEST_NO_REPORT",
        "input_length_chars": {
            "median": lengths[len(lengths) // 2],
            "p99": lengths[int(len(lengths) * 0.99)],
            "max": lengths[-1],
        },
        "unique_evidence_groups": len(distinct) if split != "test" else "SEALED_TEST_NO_REPORT",
        "conflicting_evidence_groups": sum(len(v) > 1 for v in distinct.values())
        if split != "test"
        else "SEALED_TEST_NO_REPORT",
    }


def assert_dev_prompts(dev: list[dict], retrieved, vocab: set[str], *, root: Path) -> dict:
    results = {}
    for k in (5, 16, 17):
        path = root / f"data/jeff/dev_k{k:02d}.jsonl"
        if not path.is_file():
            fail(f"Missing Dev K={k} file")
        n = 0
        with path.open(encoding="utf-8") as stream:
            for i, raw in enumerate(stream):
                if i >= DEV_COUNT:
                    fail(f"Extra Dev prompt K={k}")
                item = json.loads(raw)
                expected = decision_row(
                    dev[i],
                    case_id=f"dev-{i:06d}",
                    candidates=retrieved[i].candidate_labels[:k],
                    allow_grouped=k > 16,
                )
                if item != expected:
                    fail(f"Dev K={k} row {i} differs from original evidence/candidate order")
                if item["question"] != TCM_QUESTION or "target_index" in item:
                    fail(f"Dev K={k} row {i} has wrong question or gold supervision")
                if any(x["id"] not in vocab for x in item["options"]):
                    fail(f"Dev K={k} row {i} has label outside official vocabulary")
                n += 1
        if n != DEV_COUNT:
            fail(f"Dev K={k} incomplete: {n}")
        results[f"dev_k{k:02d}.jsonl"] = {
            "all_rows_matched_official_evidence_and_bm25": True,
            "rows": n,
        }
    return results


def run_audit(root: Path = ROOT, *, include_test_schema: bool = False) -> dict:
    """Revalidate raw inputs; keep Test completely unopened on H200."""
    raw_dir = root / "data/tcm-sd"
    vocab_lines = (raw_dir / "syndrome_vocab.txt").read_text(encoding="utf-8-sig").splitlines()
    vocab = {value.strip() for value in vocab_lines if value.strip()}
    if len(vocab) != 148 or len([x for x in vocab_lines if x.strip()]) != 148:
        fail("Official syndrome vocabulary must contain 148 distinct nonblank labels")
    train = load_records(raw_dir / "train.json")
    dev = load_records(raw_dir / "dev.json")
    if (len(train), len(dev)) != (TRAIN_COUNT, DEV_COUNT):
        fail("Official Train/Dev counts have changed")
    scanned = {
        "train": safe_input_scan(train, split="train", vocab=vocab),
        "dev": safe_input_scan(dev, split="dev", vocab=vocab),
    }
    if include_test_schema:
        test = load_records(raw_dir / "test.json")
        if len(test) != DEV_COUNT:
            fail("Official Test row count has changed")
        scanned["test"] = safe_input_scan(test, split="test", vocab=vocab)
    knowledge_path = raw_dir / "syndrome_knowledge.json"
    knowledge_count = None
    if knowledge_path.exists():
        knowledge_count = len(load_records(knowledge_path))
        if knowledge_count != 1027:
            fail("Syndrome knowledge dataset count differs")
    labels_by_text = observed_label_sets(train)
    multipos_rows = Counter(len(labels_by_text[clinical_text(row)]) for row in train)
    if multipos_rows != {1: 42360, 2: 816, 3: 4}:
        fail(f"Train multi-positive distribution unexpectedly changed {multipos_rows}")

    oof_meta = json.loads(
        (root / "experiments/bm25_train_text_grouped_2026-11-11.json").read_text(encoding="utf-8")
    )
    if (
        oof_meta.get("fold_policy") != "exact_text_grouped"
        or oof_meta["cross_fold_exact_input_text"]["query_rows_with_identical_text_in_another_fold"]
        != 0
    ):
        fail("Train Group-aware OOF metadata not valid")
    oof_rank = load_and_validate_oof(
        train, root / "data/candidates/bm25_train_text_grouped.jsonl", oof_meta
    )
    if any(label not in vocab for ranking in oof_rank for label in ranking):
        fail("Out-of-fold retrieval has labels outside official vocab")

    s1 = validate_file(
        train,
        oof_rank,
        root / "data/jeff/train_oof_k16_epoch02.jsonl",
        expected_k=16,
        exclude_ambiguous_negatives=True,
    )
    multi = verify_multi(train, oof_rank, root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl")
    # Independent exact-row schema check: reject extra fields and any unexpected
    # patient identifiers, diagnosis answers or model-visible supervision.
    for name, relative_path in ACTIVE_TRAIN.items():
        with (root / relative_path).open(encoding="utf-8") as stream:
            for i, raw in enumerate(stream):
                item = json.loads(raw)
                options = [option["id"] for option in item["row"]["options"]]
                expected = decision_row(train[i], case_id=f"train-{i:06d}", candidates=options)
                if item["row"] != expected:
                    fail(f"{name} Train row {i} contains extra fields or mismatched prompt")
    if s1["cases"] != TRAIN_COUNT or multi["total_verified_questions"] != TRAIN_COUNT:
        fail("Train JSONL row counts differ")
    if multi["observed_positive_counts_by_original_row"] != {"1": 42360, "2": 816, "3": 4}:
        fail("Multi-positive Train candidate sets changed")
    retrieved = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if len(retrieved) != DEV_COUNT:
        fail("Original Dev BM25 candidate file incomplete")
    for i, (case, result) in enumerate(zip(dev, retrieved, strict=True)):
        if result.case_id != f"dev-{i:06d}":
            fail(f"Dev candidate order mismatch row {i}")
        if result.gold_labels != (case["norm_syndrome"],):
            fail(f"Dev candidate gold label mismatch row {i}")
        if result.predicted_label is not None:
            fail(f"Dev candidate row {i} contains a prefilled prediction")
        if len(result.candidate_labels) != 50 or len(set(result.candidate_labels)) != 50:
            fail(f"Dev row {i} must have 50 distinct BM25 candidates")
        if any(label not in vocab for label in result.candidate_labels):
            fail(f"Dev row {i} contains a nonofficial syndrome label")
    dev_prompts = assert_dev_prompts(dev, retrieved, vocab, root=root)

    # Source integrity and soft supervision are checked from each row. Old,
    # historical OOF and Train epochs are intentionally NOT part of H200 payload.
    return {
        "status": "FULL_RECORD_SOURCE_AND_PAYLOAD_AUDIT_PASS",
        "patient_text_exported": False,
        "official_data": scanned,
        "syndrome_vocab_labels": len(vocab),
        "knowledge_rows_if_present": knowledge_count,
        "test_opened_for_schema_only": include_test_schema,
        "active_grouped_oof_rows_checked": len(oof_rank),
        "s1_train": s1,
        "s2_s3_s4_train": multi,
        "dev_retrieval_rows_checked": len(retrieved),
        "dev_prompts": dev_prompts,
        "transfer_required_for_repeatable_h200_preflight": [
            "data/tcm-sd/train.json",
            "data/tcm-sd/dev.json",
            "data/tcm-sd/syndrome_vocab.txt",
            "data/candidates/bm25_train_text_grouped.jsonl",
            "data/candidates/bm25_dev.jsonl",
            "data/jeff/train_oof_k16_epoch02.jsonl",
            "data/jeff/train_oof_multipos_k16_epoch03.jsonl",
            "data/jeff/dev_k05.jsonl",
            "data/jeff/dev_k16.jsonl",
            "data/jeff/dev_k17.jsonl",
        ],
        "do_not_transfer_for_training": [
            "data/tcm-sd/test.json",
            "data/secondary/",
            "data/TCM_SD_train_dev.zip",
            "data/candidates/bm25_train_oof.jsonl",
            "data/jeff/train_oof_k16_epoch00.jsonl",
            "data/jeff/train_oof_k16_epoch01.jsonl",
        ],
        "validation_limitation": (
            "Every row structurally and provenance-validated; physician clinical truth "
            "and semantic near-duplicate labels not independently adjudicated"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all official H200 input records")
    parser.add_argument(
        "--check-only", action="store_true", help="Revalidate without writing a frozen report"
    )
    parser.add_argument(
        "--include-test-schema",
        action="store_true",
        help="Local-only format QA; keep disabled on H200 before final evaluation",
    )
    args = parser.parse_args()
    report = run_audit(include_test_schema=args.include_test_schema)
    dest = ROOT / "experiments/h200_payload_full_audit_2026-11-12.json"
    if not args.check_only:
        if dest.exists():
            raise FileExistsError("Do not overwrite an earlier frozen H200 payload audit")
        with dest.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    summary = {
        "status": report["status"],
        "input_records_checked": {k: v["rows_scanned"] for k, v in report["official_data"].items()},
        "active_oof_checked": report["active_grouped_oof_rows_checked"],
        "s1_train_checked": report["s1_train"]["cases"],
        "multi_train_checked": report["s2_s3_s4_train"]["total_verified_questions"],
        "dev_candidate_checked": report["dev_retrieval_rows_checked"],
        "dev_prompts_checked": report["dev_prompts"],
        "unicode_warnings": {k: v["unicode_warnings"] for k, v in report["official_data"].items()},
        "unexpected_control_locations": {
            k: v["unexpected_control_locations_no_raw_text"]
            for k, v in report["official_data"].items()
        },
        "report_file": str(dest.relative_to(ROOT))
        if not args.check_only
        else "CHECK_ONLY_NO_WRITES",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
