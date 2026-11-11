"""Verify all Jeff Train supervision against original TCM-SD and OOF BM25.

Validation never touches Dev/Test. Publishes aggregate-only QA numbers and
keeps raw cases, candidate labels, and target labels under git-ignored data/.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.export_jeff_train import TRAIN_ROWS, load_and_validate_oof
from tcmdiscern.datasets import load_records
from tcmdiscern.jeff_format import LETTERS, TCM_QUESTION, patient_state, semif_messages

EXPECTED_FILES = {
    "k05": "train_oof_k05_epoch00.jsonl",
    "k16": "train_oof_k16_epoch00.jsonl",
    "mixed": "train_oof_mixed_k02_16_epoch00.jsonl",
}


def validate_file(
    cases: list[dict],
    ranked: list[list[str]],
    path: Path,
    *,
    expected_k: int | None,
) -> dict:
    """Validate exact supervision, candidate membership, state and prompt layout."""
    if len(cases) != len(ranked):
        raise ValueError("The original Train and OOF rankings must be aligned")
    length = 0
    by_k: Counter[int] = Counter()
    by_position: Counter[int] = Counter()
    label_literal_in_source = 0
    prompt_chars: list[int] = []
    with path.open(encoding="utf-8") as stream:
        for index, raw in enumerate(stream):
            if index >= len(cases):
                raise ValueError(f"Too many training examples in {path}")
            item = json.loads(raw)
            if set(item) != {"row", "target_index", "target_letter"}:
                raise ValueError(f"{path}: unexpected outer training keys at row {index}")
            row = item["row"]
            case = cases[index]
            gold = str(case["norm_syndrome"])
            if row["id"] != f"train-{index:06d}":
                raise ValueError(f"{path}: training rows not in official order")
            if row["question"] != TCM_QUESTION or row["qtype"] != "choice":
                raise ValueError(f"{path}: not the agreed TCM Jeff choice task")
            state = patient_state(case)
            if row["state"] != state:
                raise ValueError(f"{path}: state differs from original whitelisted fields")
            if set(row["state"]) != {"chief_complaint", "description", "detection"}:
                raise ValueError(f"{path}: private/answer fields in model evidence")

            opts = row["options"]
            k = len(opts)
            if not 2 <= k <= 16 or expected_k is not None and k != expected_k:
                raise ValueError(f"{path}: invalid K={k} at Train row {index}")
            ids = [option["id"] for option in opts]
            if len(set(ids)) != k or any(
                option.get("description") != option["id"] for option in opts
            ):
                raise ValueError(f"{path}: duplicate options or description mismatch")
            target = item["target_index"]
            if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < k:
                raise ValueError(f"{path}: invalid target_index at Train row {index}")
            if ids[target] != gold or item["target_letter"] != LETTERS[target]:
                raise ValueError(f"{path}: gold label is not at the supervised position")
            negatives = [name for name in ranked[index] if name != gold]
            expected = {gold, *negatives[: k - 1]}
            if set(ids) != expected or ids.count(gold) != 1:
                raise ValueError(f"{path}: candidates are not the top BM25 hard negatives")

            messages = semif_messages(row)
            if len(messages) != 2:
                raise ValueError(f"{path}: Jeff message count mismatch")
            payload = json.loads(messages[1]["content"])
            if set(payload) != {"evidence", "criterion", "options"}:
                raise ValueError(f"{path}: Jeff/SemIf payload layout mismatch")
            if payload["evidence"] != state:
                raise ValueError(f"{path}: payload evidence mismatch")
            if [x["letter"] for x in payload["options"]] != list(LETTERS[:k]):
                raise ValueError(f"{path}: Jeff's A-P letter mapping mismatch")

            if gold in " ".join(state.values()):
                label_literal_in_source += 1
            prompt_chars.append(len(messages[1]["content"]))
            by_k[k] += 1
            by_position[target] += 1
            length += 1

    if length != len(cases):
        raise ValueError(f"Expected {len(cases)} rows, got {length} in {path}")
    prompt_chars.sort()
    return {
        "file": f"data/jeff/{path.name}",
        "cases": length,
        "k_distribution": {str(k): by_k[k] for k in sorted(by_k)},
        "gold_position_distribution": {str(k): by_position[k] for k in sorted(by_position)},
        "original_clinical_text_containing_gold_literal": label_literal_in_source,
        "jeff_user_json_char_median": prompt_chars[len(prompt_chars) // 2],
        "jeff_user_json_char_p99": prompt_chars[int(0.99 * (length - 1))],
        "jeff_user_json_char_max": prompt_chars[-1],
        "all_cases_retained": True,
        "train_targets_outside_jeff_prompt": True,
        "gold_excluded_from_hard_negatives": True,
        "prompt_schema": "Jeff direct-options-v1 / SemIf",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = load_records(root / "data/tcm-sd/train.json")
    oof_report = json.loads(
        (root / "experiments/bm25_train_oof_2026-11-11.json").read_text(encoding="utf-8")
    )
    ranks = load_and_validate_oof(cases, root / "data/candidates/bm25_train_oof.jsonl", oof_report)
    if len(cases) != TRAIN_ROWS:
        raise ValueError("Expected full official Train")
    outputs = []
    for name, file in EXPECTED_FILES.items():
        record = validate_file(
            cases,
            ranks,
            root / "data/jeff" / file,
            expected_k={"k05": 5, "k16": 16, "mixed": None}[name],
        )
        outputs.append(record)
        print(
            f"Verified {name}: {record['cases']} cases, {len(record['k_distribution'])} K values",
            flush=True,
        )
    report = {
        "study": "TCMDiscern",
        "date": "2026-11-11",
        "status": "JEFF_TRAIN_DATA_VALIDATED_ALL_129540_ROWS",
        "official_train_rows": len(cases),
        "dataset_deleted_rows": 0,
        "evaluated_training_examples": sum(item["cases"] for item in outputs),
        "test_dev_loaded": False,
        "jeff_model_inference": "NOT_RUN",
        "tokenized_context_length_checked": False,
        "tokenizer_note": (
            "Only prompt character lengths were measured. Actual Gemma 4 token counts "
            "must be measured with the pinned Jeff tokenizer and max length 8192."
        ),
        "all_checks_passed": True,
        "files": outputs,
    }
    dest = root / "experiments/jeff_training_quality_2026-11-11.json"
    with dest.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(f"Saved aggregate QA: {dest}", flush=True)


if __name__ == "__main__":
    main()
