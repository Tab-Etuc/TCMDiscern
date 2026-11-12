"""Audit Jeff K16 positive/negative class exposure and duplicate-group weights.

Uses original Train and saved *local* multi-positive S2/S3/S4 rows. Records
only aggregate counts by published syndrome labels, never patient text/IDs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

from tcmdiscern.datasets import clinical_text, load_records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = load_records(root / "data/tcm-sd/train.json")
    if len(train) != 43180:
        raise ValueError("Requires the full original Train, without deduplication")

    support = Counter(str(row["norm_syndrome"]) for row in train)
    group_sizes = Counter(clinical_text(row) for row in train)
    if len(support) != 148:
        raise ValueError("Expected all 148 official labels")
    group_histogram = Counter(group_sizes.values())

    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    multi_rows = 0
    n = 0
    path = root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= len(train):
                raise ValueError("Extra training rows")
            item = json.loads(line)
            row = item["row"]
            if row["id"] != f"train-{index:06d}" or len(row["options"]) != 16:
                raise ValueError("Incorrect training row/candidate alignment")
            observed_indices = set(item["positive_indices"])
            if len(observed_indices) > 1:
                multi_rows += 1
            for position, option in enumerate(row["options"]):
                label = option["id"]
                if label not in support:
                    raise ValueError(f"Unknown label in Jeff training options: {label}")
                if position in observed_indices:
                    positive[label] += 1
                else:
                    negative[label] += 1
            n += 1
    if n != len(train):
        raise ValueError("Missing original Train questions")

    exposure_rows = []
    for label in sorted(support):
        exposure_rows.append(
            {
                "syndrome": label,
                "official_train_support": support[label],
                "positive_option_exposures": positive[label],
                "negative_option_exposures": negative[label],
                "negatives_per_positive_exposure": round(negative[label] / positive[label], 4),
                "negatives_per_official_train_case": round(negative[label] / support[label], 4),
            }
        )
    ratio = sorted(row["negatives_per_positive_exposure"] for row in exposure_rows)
    report = {
        "date": "2026-11-12",
        "status": "LOCAL_JEFF_K16_NEGATIVE_EXPOSURE_AUDITED",
        "data_source": "Original Train + group-aware K16 observed-label training examples",
        "original_train_rows": n,
        "removed_train_rows": 0,
        "official_syndrome_classes": len(support),
        "rows_with_multiple_observed_gold": multi_rows,
        "expected_total_option_exposures": 16 * n,
        "positive_exposures": sum(positive.values()),
        "negative_exposures": sum(negative.values()),
        "negative_to_positive_ratio_median": median(ratio),
        "negative_to_positive_ratio_max": max(ratio),
        "duplicate_groups": {
            "unique_clinical_inputs": len(group_sizes),
            "group_sizes_histogram": {
                str(size): count for size, count in sorted(group_histogram.items())
            },
            "total_train_rows": len(train),
            "all_rows_retained": True,
            "sum_inverse_group_weight": sum(
                count * (1.0 / count) for count in group_sizes.values()
            ),
        },
        "per_syndrome": exposure_rows,
        "notes": [
            "Training sampler is currently unweighted; this report does not change it.",
            "Observed alternative gold labels are counted as positives, never negatives.",
            "Natural frequency of labels and negative option exposure are highly nonuniform.",
            "Compare class/group-balanced sampling in a separate controlled experiment.",
        ],
        "model_training_completed": False,
    }
    if report["positive_exposures"] + report["negative_exposures"] != 16 * n:
        raise ValueError("Exposure counters did not account for every K16 option")
    if report["rows_with_multiple_observed_gold"] != 820:
        raise ValueError("Expected 820 ambiguous-label original rows")
    out = root / "experiments/jeff_multipos_exposure_2026-11-12.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(out),
                "train_cases": n,
                "positive_exposures": report["positive_exposures"],
                "negative_exposures": report["negative_exposures"],
                "negative_to_positive_ratio_median": report["negative_to_positive_ratio_median"],
                "negative_to_positive_ratio_max": report["negative_to_positive_ratio_max"],
                "unique_clinical_inputs": len(group_sizes),
                "multi_label_rows": multi_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
