"""Official TCM-SD disease / disease-syndrome class-space audit and BM25 Dev baselines.

Reports only aggregated Train/Dev results; no Test and no patient-level output.
The original official rows/splits are used as published.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.disease_study import disease_name, pair_key, syndrome_name
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 5, 16, 32, 50)
LABEL_MODES = ("disease", "pair", "syndrome")


def label_of(record: dict, mode: str) -> str:
    if mode == "disease":
        return disease_name(record)
    if mode == "pair":
        return pair_key(disease_name(record), syndrome_name(record))
    if mode == "syndrome":
        return syndrome_name(record)
    raise ValueError("unknown mode")


def summary_counts(train: list[str], dev: list[str]) -> dict:
    tc = Counter(train)
    dc = Counter(dev)
    head = sorted(tc.items(), key=lambda item: (-item[1], item[0]))
    return {
        "train_classes": len(tc),
        "dev_classes": len(dc),
        "train_dev_classes": len(set(tc) | set(dc)),
        "dev_unseen_class_count": len(set(dc) - set(tc)),
        "dev_unseen_rows": sum(1 for value in dev if value not in tc),
        "train_singleton_classes": sum(count == 1 for count in tc.values()),
        "train_support_le_5_classes": sum(count <= 5 for count in tc.values()),
        "train_support_le_10_classes": sum(count <= 10 for count in tc.values()),
        "train_support_le_20_classes": sum(count <= 20 for count in tc.values()),
        "train_top10_class_coverage_rows": sum(count for _, count in head[:10]) / len(train),
        "train_top1_class_coverage_rows": head[0][1] / len(train),
        "dev_class_count_le_10_train_support": sum(
            count for name, count in dc.items() if tc.get(name, 0) <= 10
        ),
    }


def evaluate_rankings(
    train_labels: list[str], dev_labels: list[str], ranked: list[list[str]]
) -> dict:
    if len(dev_labels) != len(ranked):
        raise ValueError("Dev/rankings length mismatch")
    predicted = [row[0] for row in ranked]
    train_label_set = set(train_labels)
    all_classes = sorted(train_label_set | set(dev_labels))
    accuracy = float(accuracy_score(dev_labels, predicted))
    macro = float(
        f1_score(dev_labels, predicted, labels=all_classes, average="macro", zero_division=0)
    )
    oracles = {
        str(k): {
            "hits": sum(gold in ranking[:k] for gold, ranking in zip(dev_labels, ranked, strict=True)),
            "rate": sum(gold in ranking[:k] for gold, ranking in zip(dev_labels, ranked, strict=True))
            / len(dev_labels),
        }
        for k in K_VALUES
    }
    return {
        "accuracy_top1": accuracy,
        "macro_f1": macro,
        "oracle_at_k": oracles,
        "n": len(dev_labels),
        "label_classes": len(all_classes),
        "unseen_dev_gold_absent_from_train_index": sum(
            gold not in train_label_set for gold in dev_labels
        ),
    }


def audit(root: Path, *, include_retrieval: bool) -> dict:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Expected official Train 43,180 / Dev 5,486")
    result: dict = {
        "dataset": "TCM-SD",
        "splits": {"train": len(train), "dev": len(dev)},
        "evaluation_unit": "official original data row",
        "test_used": False,
        "label_spaces": {},
    }
    for mode in LABEL_MODES:
        train_labels = [label_of(item, mode) for item in train]
        dev_labels = [label_of(item, mode) for item in dev]
        item = {"distribution": summary_counts(train_labels, dev_labels)}
        if include_retrieval:
            print(f"[LABEL AUDIT] {mode}: build Train-only BM25, score full Dev", flush=True)
            retriever = CaseBM25([clinical_text(item) for item in train], train_labels)
            if len(retriever.label_names) < max(K_VALUES):
                raise ValueError(f"Insufficient label class count for {mode}")
            ranked = retriever.rank(
                [clinical_text(item) for item in dev], max_k=max(K_VALUES)
            )
            item["bm25"] = evaluate_rankings(train_labels, dev_labels, ranked)
        result["label_spaces"][mode] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="TCM-SD Chinese-disease/paired-label audit")
    parser.add_argument("--retrieval", action="store_true")
    args = parser.parse_args()
    report = audit(ROOT, include_retrieval=args.retrieval)
    dest = ROOT / "experiments/tcmsd_joint_label_space_dev.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
