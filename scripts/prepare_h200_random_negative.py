"""Prepare a deterministic Train-only random-negative S1 control.

The original 43,180 Train examples, positive candidate slots, prompts and supervision
are retained. Only the 15/14 other syndrome labels are sampled from the Train
vocabulary. Dev and Test are never opened.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tcmdiscern.jeff_schedule import load_frozen_train

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
TARGET = ROOT / "data/jeff/train_oof_randomneg_k16_epoch03.jsonl"
SEED = 20261117


def build(source: Path = SOURCE, target: Path = TARGET, train_file: Path | None = None) -> dict:
    source_rows = load_frozen_train(source, multi=True, expected_count=43180)
    if train_file is None:
        train_file = ROOT / "data/tcm-sd/train.json"
    from tcmdiscern.datasets import load_records

    train = load_records(train_file)
    if len(train) != 43180:
        raise ValueError("Official Train must contain 43180 rows")
    vocabulary = sorted({str(r["norm_syndrome"]) for r in train})
    if len(vocabulary) != 148:
        raise ValueError("The Train vocabulary must contain 148 syndromes")
    if target.exists():
        existing = load_frozen_train(target, multi=True, expected_count=43180)
        for index, (original, current) in enumerate(
            zip(source_rows, existing, strict=True)
        ):
            if (
                {k: v for k, v in original.items() if k != "row"}
                != {k: v for k, v in current.items() if k != "row"}
                or {k: v for k, v in original["row"].items() if k != "options"}
                != {k: v for k, v in current["row"].items() if k != "options"}
            ):
                raise ValueError(f"Random-negative sample changed noncandidate data: {index}")
            positives = original["positive_indices"]
            protected = {original["row"]["options"][slot]["id"] for slot in positives}
            rng = random.Random(SEED + index * 1_000_003)
            expected_negatives = rng.sample(
                [v for v in vocabulary if v not in protected],
                16 - len(positives),
            )
            expected = [dict(option) for option in original["row"]["options"]]
            for slot, label in zip(
                (j for j in range(16) if j not in positives),
                expected_negatives,
                strict=True,
            ):
                expected[slot] = {"id": label, "description": label}
            if current["row"]["options"] != expected:
                raise ValueError(f"Random-negative sample failed deterministic audit: {index}")
        return {"status": "verified_existing", "rows": len(existing), "path": str(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".preparing")
    if tmp.exists():
        raise FileExistsError(f"Inspect incomplete preparation before retry: {tmp}")
    changed = 0
    try:
        with tmp.open("x", encoding="utf-8") as output:
            for i, item in enumerate(source_rows):
                rng = random.Random(SEED + i * 1_000_003)
                positives = set(item["positive_indices"])
                protected = {item["row"]["options"][slot]["id"] for slot in positives}
                options = [dict(option) for option in item["row"]["options"]]
                negatives = rng.sample([v for v in vocabulary if v not in protected], 16 - len(positives))
                for slot, label in zip((j for j in range(16) if j not in positives), negatives, strict=True):
                    changed += int(options[slot]["id"] != label)
                    options[slot] = {"id": label, "description": label}
                updated = {
                    **item,
                    "row": {**item["row"], "options": options},
                }
                if [options[j]["id"] for j in positives] != [
                    item["row"]["options"][j]["id"] for j in positives
                ]:
                    raise AssertionError("Gold-positive slots changed")
                output.write(json.dumps(updated, ensure_ascii=False) + "\n")
        checked = load_frozen_train(tmp, multi=True, expected_count=43180)
        if any(
            [x["row"]["options"][j]["id"] for j in x["positive_indices"]]
            != [y["row"]["options"][j]["id"] for j in y["positive_indices"]]
            for x, y in zip(checked, source_rows, strict=True)
        ):
            raise AssertionError("Positive slots changed")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {
        "status": "prepared", "rows": len(source_rows), "vocabulary": len(vocabulary),
        "changed_negative_slots": changed, "seed": SEED, "path": str(target),
        "official_test_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only and not TARGET.is_file():
        raise FileNotFoundError(TARGET)
    print(json.dumps(build(), ensure_ascii=False, indent=2))
