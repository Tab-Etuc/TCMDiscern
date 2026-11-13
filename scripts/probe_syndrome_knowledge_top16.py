"""CPU-only TCM-SD fixed-Top16 syndrome knowledge probe, exploratory Dev only.

Compares (a) frozen case BM25, (b) syndrome-description BM25 rankings
restricted to the exact original 16, and (c) a predeclared RRF(60) control.
Does not alter any Jeff inputs, gold labels, existing runs or Test files.
Only aggregate metrics are written to a new experiments/ report.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.knowledge_rerank import rerank_fixed_candidates
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/knowledge_top16_cpu_probe_2026-11-12.json"
EXPECTED_TRAIN = 43180
EXPECTED_DEV = 5486
EXPECTED_VOCAB = 148
K = 16


def label_knowledge(vocabulary: set[str], data: list[dict]) -> dict[str, str]:
    """Use only published label-side descriptions, never clinical Dev gold."""
    selected: dict[str, str] = {}
    for row in data:
        name = str(row.get("Name") or "").strip()
        if name not in vocabulary:
            continue
        if name in selected:
            raise ValueError(f"Duplicated knowledge entry for official label {name}")
        definition = str(row.get("Definition") or "").strip()
        typical = str(row.get("Typical_performance") or "").strip()
        if not definition:
            raise ValueError(f"Missing knowledge definition: {name}")
        selected[name] = " ".join(part for part in (name, definition, typical) if part)
    if set(selected) != vocabulary:
        raise ValueError("Knowledge labels differ from official 148-label set")
    return selected


def summarize() -> dict:
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if len(train) != EXPECTED_TRAIN or len(dev) != EXPECTED_DEV:
        raise ValueError("Official complete Train and Dev required")
    train_labels = {str(item["norm_syndrome"]) for item in train}
    vocab = {
        line.strip()
        for line in (ROOT / "data/tcm-sd/syndrome_vocab.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    }
    if len(vocab) != EXPECTED_VOCAB or train_labels != vocab:
        raise ValueError("Official vocabulary drift")
    info = load_records(ROOT / "data/tcm-sd/syndrome_knowledge.json")
    descriptions = label_knowledge(vocab, info)

    # The independent 148-entry knowledge index is built only on label-side
    # knowledge, before any Dev gold enters scoring.
    names = sorted(descriptions)
    knowledge = CaseBM25([descriptions[name] for name in names], names, k1=1.5, b=0.75, min_df=1)
    knowledge_rankings = knowledge.rank(
        [clinical_text(row) for row in dev], max_k=EXPECTED_VOCAB, batch_size=24
    )
    frozen = read_predictions(ROOT / "data/candidates/bm25_dev.jsonl")
    if len(frozen) != EXPECTED_DEV or len(knowledge_rankings) != EXPECTED_DEV:
        raise ValueError("Incomplete Dev inference")

    gold = [str(row["norm_syndrome"]) for row in dev]
    outputs: dict[str, list[str]] = {
        "frozen_case_bm25_top1": [],
        "knowledge_bm25_global_top1": [],
        "knowledge_bm25_within_frozen_top16": [],
        "rrf60_within_frozen_top16": [],
    }
    baseline_gold_rank1 = 0
    recoverable_bm25_wrong = 0
    missed_bm25_top16 = 0
    missed_covered_knowledge_top16 = 0
    knowledge_votes_for_gold = 0
    knowledge_disagrees = 0
    knowledge_disagreement_rescue = 0
    knowledge_disagreement_break = 0
    literal_gold_in_patient_text = 0
    for i, (row, ranking) in enumerate(zip(frozen, knowledge_rankings, strict=True)):
        case_id = f"dev-{i:06d}"
        label = gold[i]
        if row.case_id != case_id or row.gold_labels != (label,):
            raise ValueError(f"Frozen BM25 / official Dev labels disagree at {i}")
        if len(row.candidate_labels) != 50:
            raise ValueError(f"Invalid frozen Top50 candidate count: {case_id}")
        if set(ranking) != vocab or len(ranking) != EXPECTED_VOCAB:
            raise ValueError(f"Knowledge rank omitted official label: {case_id}")
        if label in clinical_text(dev[i]):
            literal_gold_in_patient_text += 1
        combined = rerank_fixed_candidates(row.candidate_labels, ranking, k=K)
        top16 = combined["bm25"]
        if label == top16[0]:
            baseline_gold_rank1 += 1
        elif label in top16:
            recoverable_bm25_wrong += 1
        else:
            missed_bm25_top16 += 1
            if label in ranking[:K]:
                missed_covered_knowledge_top16 += 1
        outputs["frozen_case_bm25_top1"].append(top16[0])
        outputs["knowledge_bm25_global_top1"].append(ranking[0])
        outputs["knowledge_bm25_within_frozen_top16"].append(combined["knowledge_within_k"][0])
        outputs["rrf60_within_frozen_top16"].append(combined["rrf60_within_k"][0])
        knowledge_votes_for_gold += label == combined["knowledge_within_k"][0]
        if top16[0] != combined["knowledge_within_k"][0]:
            knowledge_disagrees += 1
            knowledge_disagreement_rescue += (
                top16[0] != label and combined["knowledge_within_k"][0] == label
            )
            knowledge_disagreement_break += (
                top16[0] == label and combined["knowledge_within_k"][0] != label
            )

    if baseline_gold_rank1 != 4366 or recoverable_bm25_wrong != 1043 or missed_bm25_top16 != 77:
        raise RuntimeError("Baseline BM25 Dev group counts differ from frozen results")
    metrics = {}
    for method, predictions in outputs.items():
        correct = sum(a == b for a, b in zip(gold, predictions, strict=True))
        metrics[method] = {
            "top1_correct": correct,
            "top1": correct / EXPECTED_DEV,
            "macro_f1_148": float(
                f1_score(gold, predictions, labels=names, average="macro", zero_division=0)
            ),
            "unique_predicted_classes": len(set(predictions)),
        }
    train_counts = Counter(str(item["norm_syndrome"]) for item in train)
    long_tail = {name for name in names if train_counts[name] < 20}
    if not long_tail:
        raise ValueError("Empty long-tail support group")
    tail_rows = [i for i, label in enumerate(gold) if label in long_tail]
    for method, predictions in outputs.items():
        metrics[method]["low_support_under20_correct"] = sum(
            predictions[i] == gold[i] for i in tail_rows
        )
        metrics[method]["low_support_under20_dev_n"] = len(tail_rows)
    return {
        "status": "EXPLORATORY_CPU_ONLY_KNOWLEDGE_FIXED_TOP16_DEV_PROBE",
        "official_train_n": EXPECTED_TRAIN,
        "official_dev_n": EXPECTED_DEV,
        "official_classes": EXPECTED_VOCAB,
        "base_retrieval": "frozen case BM25, original full Train index",
        "knowledge_retrieval": (
            "148 exact official syndrome definitions and typical performances; "
            "CaseBM25 k1=1.5,b=0.75,min_df=1, name+Definition+Typical_performance"
        ),
        "knowledge_definitions_covered": len(descriptions),
        "frozen_candidate_width": K,
        "rerank_scope": "the exact frozen BM25 Top16 labels, no candidate replacement",
        "methods": metrics,
        "baseline_breakdown": {
            "bm25_rank1_gold": baseline_gold_rank1,
            "bm25_rank2_to16_gold": recoverable_bm25_wrong,
            "bm25_gold_missing_top16": missed_bm25_top16,
            "missing_gold_recovered_by_independent_knowledge_top16": missed_covered_knowledge_top16,
            "knowledge_top1_within_original_16_hits_gold": knowledge_votes_for_gold,
            "knowledge_vs_bm25_top1_disagreed": knowledge_disagrees,
            "knowledge_reorder_rescued": knowledge_disagreement_rescue,
            "knowledge_reorder_broke": knowledge_disagreement_break,
        },
        "gold_literal_appears_in_clinical_source_n": literal_gold_in_patient_text,
        "selection_note": "Exploratory Dev comparison only; no model or K selected from it",
        "test_opened": False,
        "jeff_model_loaded": False,
        "original_training_artifacts_modified": False,
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Existing exploratory report is immutable")
    result = summarize()
    with REPORT.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
