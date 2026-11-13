"""Frozen TCM-SD BM25 input-feature ablation against the published CCL 2022 setup.

Uses full official Train as the retrieval corpus, full official Dev for score
only. No Dev gold enters indexing/query, and Test is never opened. The CCL 2022
classification baseline used complaint + history (without four-exam detection).
"""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import f1_score

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report_path = ROOT / "experiments/bm25_chief_history_vs_four_diagnoses_2026-11-12.json"
    if report_path.exists():
        raise FileExistsError("Frozen input ablation already exists")
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Full official Train/Dev required")
    labels = [str(row["norm_syndrome"]) for row in train]
    gold = [str(row["norm_syndrome"]) for row in dev]
    classes = sorted(set(labels))
    if len(classes) != 148 or set(gold) != set(classes):
        raise ValueError("Expected the original 148-class official task")
    outputs = {}
    for include_detection in (False, True):
        mode = "chief_history_detection" if include_detection else "chief_history"
        index_texts = [
            clinical_text(row, include_four_examinations=include_detection) for row in train
        ]
        query_texts = [
            clinical_text(row, include_four_examinations=include_detection) for row in dev
        ]
        bm25 = CaseBM25(index_texts, labels, k1=1.5, b=0.75, min_df=2)
        rankings = bm25.rank(query_texts, max_k=35, batch_size=24)
        if len(rankings) != len(dev):
            raise RuntimeError("BM25 failed to return one ranking for every Dev case")
        predictions = [labels[0] for _ in dev]
        predictions[:] = [rank[0] for rank in rankings]
        top1 = sum(a == b for a, b in zip(predictions, gold, strict=True))
        found = {
            str(k): sum(gold[i] in rank[:k] for i, rank in enumerate(rankings))
            for k in (1, 5, 16, 17, 35)
        }
        support = {label: gold.count(label) for label in classes}
        macro = {
            str(k): sum(
                sum(label == gold[i] and label in rank[:k] for i, rank in enumerate(rankings))
                / support[label]
                for label in classes
            )
            / len(classes)
            for k in (1, 5, 16, 17, 35)
        }
        outputs[mode] = {
            "feature_fields": ["chief_complaint", "description"]
            + (["detection"] if include_detection else []),
            "top1_correct": top1,
            "top1_accuracy": top1 / len(dev),
            "macro_f1": f1_score(
                gold, predictions, labels=classes, average="macro", zero_division=0
            ),
            "gold_covered_by_k_n": found,
            "micro_recall": {k: v / len(dev) for k, v in found.items()},
            "macro_recall": macro,
        }
        print(f"{mode}: Top1={top1}/{len(dev)}", flush=True)
    frozen = json.loads(
        (ROOT / "experiments/bm25_k1_35_dev_2026-11-11.json").read_text(encoding="utf-8")
    )["results"][15]
    full = outputs["chief_history_detection"]
    if (
        abs(full["top1_accuracy"] - frozen["bm25_top1_accuracy"]) > 1e-10
        or abs(full["macro_f1"] - frozen["bm25_top1_macro_f1"]) > 1e-10
        or full["gold_covered_by_k_n"]["16"] != frozen["candidate_hit_n"]
    ):
        raise RuntimeError(
            "Recomputed four-field BM25 differs from the frozen official Dev ranking"
        )
    result = {
        "status": "TCMSD_ORIGINAL_INPUT_FEATURE_ABLATION_COMPLETE",
        "dataset": "original TCM-SD Train 43180 and Dev 5486",
        "test_opened": False,
        "query_labels_used_only_for_scoring": True,
        "same_bm25_hyperparameters_and_class_aggregation": True,
        "results": outputs,
        "published_ccl2022_reference": "CCL 2022 Table 4 uses chief complaint + medical history for 148-class classification",
        "restriction": "This is a retrieval feature ablation, not the original paper's model reproduction",
    }
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps({"result_file": str(report_path), **outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
