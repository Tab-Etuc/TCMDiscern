"""Prove that a clean clone rebuild reproduces every frozen private input row.

Creates an isolated temporary directory under ignored data/reports/, symlinks
official Train/Dev only, and rebuilds all active artifacts from the source.
Compares every JSONL record to the existing audited local source; Test is
never opened. The temporary files are deleted even if verification fails.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from tcmdiscern.h200.artifacts import (
    DEV_CANDIDATES,
    DEV_KS,
    MULTI_TRAIN,
    SINGLE_TRAIN,
    TRAIN_OOF,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = (
    "bm25_train_text_grouped_2026-11-11.json",
    "bm25_k1_35_dev_2026-11-11.json",
    "jeff_train_examples_epoch02_2026-11-11.json",
    "jeff_multipos_k16_epoch03_2026-11-12.json",
)


def assert_equal_jsonl(expected: Path, actual: Path) -> int:
    if not expected.is_file() or not actual.is_file():
        raise FileNotFoundError("Missing source or clean-rebuild clinical artifact")
    count = 0
    with expected.open(encoding="utf-8") as original, actual.open(encoding="utf-8") as rebuilt:
        for a, b in zip(original, rebuilt, strict=True):
            if json.loads(a) != json.loads(b):
                raise ValueError(
                    f"Clean-clone reconstruction differs at {expected.name} row {count}"
                )
            count += 1
    if not count:
        raise ValueError("Zero rows in clean-rebuild artifact")
    return count


def run() -> dict:
    with tempfile.TemporaryDirectory(prefix=".clean-h200-", dir=ROOT / "data/reports") as name:
        temp = Path(name)
        (temp / "experiments").mkdir()
        (temp / "data/tcm-sd").mkdir(parents=True)
        for file in FROZEN:
            shutil.copy2(ROOT / "experiments" / file, temp / "experiments" / file)
        for file in ("train.json", "dev.json", "syndrome_vocab.txt", "syndrome_knowledge.json"):
            (temp / "data/tcm-sd" / file).symlink_to((ROOT / "data/tcm-sd" / file).resolve())
        result = prepare(temp)
        if result["status"] != "H200_TRAIN_DEV_PREPARED":
            raise RuntimeError("Clean-clone preparation failed")
        items = (TRAIN_OOF, DEV_CANDIDATES, SINGLE_TRAIN, MULTI_TRAIN) + tuple(
            f"data/jeff/dev_k{k:02d}.jsonl" for k in DEV_KS
        )
        checked = {}
        for file in items:
            n = assert_equal_jsonl(ROOT / file, temp / file)
            checked[file] = n
            print(f"[EXACT] {file} ({n} original records)", flush=True)
        return {
            "status": "CLEAN_CLONE_ALL_PRIVATE_ARTIFACTS_JSON_EQUAL",
            "artifacts_compared": checked,
            "total_jsonl_records_compared": sum(checked.values()),
            "official_test_opened": False,
            "source_results_overwritten": False,
        }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2), flush=True)
