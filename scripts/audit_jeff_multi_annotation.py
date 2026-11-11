"""Audit multi-annotation evidence and deterministic Jeff K=16 accuracy ceilings.

Reads official Train and Dev plus frozen Dev BM25 ranking; never opens Test.
Writes only aggregate counts and mathematical bounds, no patient-level data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.evaluation import read_predictions
from tcmdiscern.jeff_config import PRIMARY_JEFF_K


def annotation_groups(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        groups[clinical_text(row)].append(str(row["norm_syndrome"]))
    return dict(groups)


def annotation_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = annotation_groups(records)
    histogram = Counter(len(set(values)) for values in groups.values())
    ambiguous = [values for values in groups.values() if len(set(values)) > 1]
    upper = sum(Counter(values).most_common(1)[0][1] for values in groups.values())
    return {
        "rows": len(records),
        "distinct_clinical_inputs": len(groups),
        "groups_by_distinct_observed_gold_count": {str(n): histogram[n] for n in sorted(histogram)},
        "ambiguous_groups": len(ambiguous),
        "rows_in_ambiguous_groups": sum(len(values) for values in ambiguous),
        "max_distinct_observed_labels_per_input": max(histogram),
        "single_answer_deterministic_same_input_upper_bound_n": upper,
        "same_input_different_gold_unavoidable_errors_n": len(records) - upper,
    }


def dev_retrieval_and_label_bounds(
    dev: list[dict[str, Any]], candidates: list[Any], k: int
) -> dict[str, Any]:
    if len(dev) != len(candidates):
        raise ValueError("Dev and frozen BM25 candidate rows are not aligned")
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    bm25_correct = 0
    gold_covered = 0
    for i, (row, prediction) in enumerate(zip(dev, candidates, strict=True)):
        gold = str(row["norm_syndrome"])
        if prediction.case_id != f"dev-{i:06d}" or prediction.gold_labels != (gold,):
            raise ValueError(f"Official Dev labels disagree with BM25 candidates at row {i}")
        options = tuple(prediction.candidate_labels[:k])
        if len(options) != k or len(set(options)) != k:
            raise ValueError(f"Candidate list not complete at Dev row {i}")
        groups[(clinical_text(row), options)].append(gold)
        bm25_correct += options[0] == gold
        gold_covered += gold in options

    deterministic_ceiling = sum(
        Counter(gold for gold in golds if gold in options).most_common(1)[0][1]
        for (_, options), golds in groups.items()
        if any(gold in options for gold in golds)
    )
    return {
        "dev_rows": len(dev),
        "k": k,
        "bm25_top1_correct": bm25_correct,
        "bm25_top1_accuracy": bm25_correct / len(dev),
        "gold_in_top_k_n": gold_covered,
        "gold_in_top_k_rate": gold_covered / len(dev),
        "gold_absent_from_top_k_n": len(dev) - gold_covered,
        "single_answer_deterministic_same_prompt_upper_bound_n": deterministic_ceiling,
        "single_answer_deterministic_same_prompt_upper_bound_rate": deterministic_ceiling
        / len(dev),
        "additional_inevitable_errors_from_conflicting_identical_inputs_n": (
            gold_covered - deterministic_ceiling
        ),
        "bound_assumptions": (
            "Deterministic argmax, prompt depends solely on exactly equal clinical_text "
            "and identical K candidate order; model receives no case identifier or "
            "unobserved clinical fields. This is a test-label consistency bound, "
            "not a measured Jeff result."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Frozen original official Train/Dev case counts required")
    candidates = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    tr = annotation_stats(train)
    dv = annotation_stats(dev)
    bound = dev_retrieval_and_label_bounds(dev, candidates, k=PRIMARY_JEFF_K)
    if tr["ambiguous_groups"] != 252 or tr["rows_in_ambiguous_groups"] != 820:
        raise ValueError("Observed Train ambiguity differs from frozen audit")
    if bound["gold_in_top_k_n"] != 5409 or bound["bm25_top1_correct"] != 4366:
        raise ValueError("Frozen BM25 Dev score changed unexpectedly")

    train_groups = {key: set(values) for key, values in annotation_groups(train).items()}
    seen = Counter()
    for row in dev:
        alternatives = train_groups.get(clinical_text(row))
        if alternatives:
            seen["dev_rows_same_clinical_input_in_train"] += 1
            if len(alternatives) > 1:
                seen["dev_rows_matching_ambiguous_train_input"] += 1
            if str(row["norm_syndrome"]) not in alternatives:
                seen["dev_rows_gold_not_among_observed_train_same_text_labels"] += 1

    report = {
        "study": "TCMDiscern",
        "date": "2026-11-12",
        "status": "MULTI_ANNOTATION_AND_K16_DETERMINISTIC_BOUND_AUDITED",
        "data_source": "TCM-SD official Train/Dev and frozen BM25 Dev",
        "model_predictions_measured": False,
        "test_data_loaded": False,
        "original_rows_removed": 0,
        "train": tr,
        "dev": dv,
        "dev_k16": bound,
        "dev_shared_input_with_train": dict(seen),
        "clinical_status": (
            "Multiple observed labels for identical clinical input; "
            "no physician adjudication of simultaneous validity"
        ),
        "safe_conclusion": (
            "Observed multi-annotation is not automatically validated multi-label truth. "
            "Deterministic single-choice official accuracy can be limited by "
            "inconsistent gold labels for identical model-visible inputs."
        ),
    }
    target = root / "experiments/jeff_multi_annotation_bounds_2026-11-12.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "train": tr,
                "dev": dv,
                "dev_k16": bound,
                "dev_shared_input_with_train": dict(seen),
                "report": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
