"""Prepare disease-403 and paired-1782 BM25/Jeff Top-16 experiments on official splits.

Train candidate mining is strictly five-fold out-of-fold on Train. Every Train
row is retained. Dev/Test candidate generation uses the full official Train
index with no gold insertion. Files in ignored data/joint16 contain clinical
text; experiment reports contain only aggregates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.disease_study import syndrome_name
from tcmdiscern.h200.artifacts import publish_lines
from tcmdiscern.jeff_schedule import load_frozen_train
from tcmdiscern.joint_study16 import MODES, K, eval_example, gold_label, train_example
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/joint16"
COUNTS = {"train": 43180, "dev": 5486, "test": 5486}


def read_folds(train: list[dict]) -> list[int]:
    file = ROOT / "data/candidates/bm25_train_text_grouped.jsonl"
    folds: list[int] = []
    with file.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i >= len(train):
                raise ValueError("Extra fold row")
            record = json.loads(line)
            if (
                record.get("case_id") != f"train-{i:06d}"
                or record.get("gold_labels") != [syndrome_name(train[i])]
                or record.get("fold") not in range(5)
            ):
                raise ValueError(f"Corrupt grouped OOF fold at {i}")
            folds.append(record["fold"])
    if len(folds) != len(train) or len(set(folds)) != 5:
        raise ValueError("Incomplete five-fold Train source")
    return folds


def mine_oof(
    train: list[dict], folds: list[int], *, mode: str
) -> list[list[str]]:
    texts = [clinical_text(item) for item in train]
    labels = [gold_label(item, mode) for item in train]
    if len(folds) != len(train):
        raise ValueError("Every Train row must have an OOF fold")
    result: list[list[str] | None] = [None] * len(train)
    for fold in range(5):
        pool_ids = [i for i, assigned in enumerate(folds) if assigned != fold]
        query_ids = [i for i, assigned in enumerate(folds) if assigned == fold]
        bm = CaseBM25([texts[i] for i in pool_ids], [labels[i] for i in pool_ids])
        rankings = bm.rank(
            [texts[i] for i in query_ids],
            max_k=min(50, len(bm.label_names)),
        )
        for i, rank in zip(query_ids, rankings, strict=True):
            result[i] = rank
        print(f"[{mode.upper()}] OOF fold {fold + 1}/5: {len(query_ids)} Train cases", flush=True)
    if any(labels is None or len(labels) < K for labels in result):
        raise ValueError("Incomplete K=16 Train OOF candidates")
    return [labels for labels in result if labels is not None]


def validate_train_file(path: Path, train: list[dict], mode: str) -> None:
    examples = load_frozen_train(
        path, multi=False, expected_count=COUNTS["train"],
        expected_k=K, study_mode=mode,
    )
    for index, (case, item) in enumerate(zip(train, examples, strict=True)):
        label = item["row"]["options"][item["target_index"]]["id"]
        if label != gold_label(case, mode):
            raise ValueError(f"Train gold and frozen K16 item disagree at index {index}")


def publish_eval(
    target: Path, rows: list[dict], cases: list[dict],
    *, mode: str, split: str, metadata: bool,
) -> None:
    if not target.exists():
        publish_lines(target, rows)
    with target.open(encoding="utf-8") as stream:
        count = 0
        for i, line in enumerate(stream):
            if i >= len(cases):
                raise ValueError("Extra evaluation row")
            record = json.loads(line)
            if metadata:
                if (
                    record.get("case_id") != f"{split}-{i:06d}"
                    or record.get("gold") != gold_label(cases[i], mode)
                    or len(record.get("candidates", [])) != K
                ):
                    raise ValueError(f"Invalid {split} metadata at {i}")
            else:
                if (
                    record["id"] != f"{split}-{i:06d}"
                    or len(record["options"]) != K
                    or set(record["state"]) != {
                        "chief_complaint", "description", "detection"
                    }
                    or "target_index" in record
                ):
                    raise ValueError(f"Invalid {split} prompt at {i}")
            count += 1
    if count != len(cases):
        raise ValueError(f"{split} is incomplete")


def prepare(*, split: str = "dev") -> dict:
    if split not in ("dev", "test"):
        raise ValueError("Official Dev/Test only")
    if split == "test" and not (
        ROOT / "experiments/joint16_dev_selection.json"
    ).is_file():
        raise ValueError("Final Test requires a frozen joint16 Dev model selection")

    train = load_records(ROOT / "data/tcm-sd/train.json")
    eval_cases = load_records(ROOT / f"data/tcm-sd/{split}.json")
    if len(train) != COUNTS["train"] or len(eval_cases) != COUNTS[split]:
        raise ValueError("Incorrect official Train/Eval row count")
    DATA.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "split": split, "train_rows": len(train), "eval_rows": len(eval_cases),
        "top_k": K, "modes": {}, "test_used_for_train": False,
    }
    train_text = [clinical_text(case) for case in train]
    eval_text = [clinical_text(case) for case in eval_cases]
    folds = None
    if split == "dev":
        folds = read_folds(train)

    for mode in MODES:
        labels = [gold_label(case, mode) for case in train]
        train_path = DATA / f"train_{mode}_k16.jsonl"
        if split == "dev":
            if not train_path.exists():
                assert folds is not None
                ranked = mine_oof(train, folds, mode=mode)
                outcomes = [
                    train_example(case, index=i, oof_labels=ranked[i], mode=mode)
                    for i, case in enumerate(train)
                ]
                publish_lines(train_path, (entry for entry, _ in outcomes))
                injected = sum(not retrieved for _, retrieved in outcomes)
            else:
                injected = None
            validate_train_file(train_path, train, mode)
        retriever = CaseBM25(train_text, labels)
        rankings = retriever.rank(eval_text, max_k=K)
        examples = [
            eval_example(case, index=i, bm25_labels=rankings[i], mode=mode, split=split)
            for i, case in enumerate(eval_cases)
        ]
        publish_eval(
            DATA / f"{split}_{mode}_prompts.jsonl",
            [prompt for prompt, _ in examples], eval_cases,
            mode=mode, split=split, metadata=False,
        )
        publish_eval(
            DATA / f"{split}_{mode}_labels.jsonl",
            [meta for _, meta in examples], eval_cases,
            mode=mode, split=split, metadata=True,
        )
        gold = [gold_label(case, mode) for case in eval_cases]
        train_label_set = set(labels)
        report["modes"][mode] = {
            "train_label_classes": len(set(labels)),
            "eval_label_classes": len(set(gold)),
            "unseen_eval_gold_rows": sum(label not in train_label_set for label in gold),
            "oracle16_hits": sum(
                label in ranking for label, ranking in zip(gold, rankings, strict=True)
            ),
            "top1_hits": sum(
                label == ranking[0] for label, ranking in zip(gold, rankings, strict=True)
            ),
            "train_gold_injected_only_for_supervised_training": injected,
        }
    dest = ROOT / "experiments" / f"joint16_{split}_prep.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    print(json.dumps(prepare(split=args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
