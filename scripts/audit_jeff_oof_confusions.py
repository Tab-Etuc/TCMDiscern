"""Train-only grouped-OOF diagnostic for Jeff's fixed BM25 Top-16 negatives.

Counts where the BM25 winner is a valid supervised negative versus an
observed alternative annotation. Does not modify candidates, prompts, model,
published baselines, or Test. Output is aggregate confusion counts only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/jeff_oof_confusion_2026-11-12.json"
OOF = ROOT / "data/candidates/bm25_train_text_grouped.jsonl"
MULTIPOS = ROOT / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
TRAIN = 43180


def analyze() -> dict:
    directed: Counter[tuple[str, str]] = Counter()
    hard_directed: Counter[tuple[str, str]] = Counter()
    source_label_support: Counter[str] = Counter()
    hard_top1_label_support: Counter[str] = Counter()
    num_primary_rank1 = num_recoverable = num_unreachable = 0
    alternative_top1_positive = 0
    multi_annotated_cases = 0
    multi_annotated_recoverable = 0
    with OOF.open(encoding="utf-8") as candidates, MULTIPOS.open(encoding="utf-8") as annotations:
        count = 0
        for count, (oof_line, annotated_line) in enumerate(
            zip(candidates, annotations, strict=True), start=1
        ):
            i = count - 1
            oof = json.loads(oof_line)
            item = json.loads(annotated_line)
            expected_id = f"train-{i:06d}"
            if oof["case_id"] != expected_id or item["row"]["id"] != expected_id:
                raise ValueError(f"OOF / annotations not aligned at {i}")
            gold = oof["gold_labels"]
            ranked = oof["candidate_labels"]
            original_gold = item["row"]["options"][item["target_index"]]["id"]
            options = item["row"]["options"]
            indices = item["positive_indices"]
            observed = {options[j]["id"] for j in indices}
            if (
                len(gold) != 1
                or gold[0] != original_gold
                or gold[0] not in observed
                or len(ranked) != 50
                or len(set(ranked)) != 50
                or len(options) != 16
            ):
                raise ValueError(f"Invalid frozen OOF vs multi-annotation: {i}")
            label = gold[0]
            winner = ranked[0]
            source_label_support[label] += 1
            if len(observed) > 1:
                multi_annotated_cases += 1
            if winner == label:
                num_primary_rank1 += 1
                continue
            directed[(label, winner)] += 1
            if label not in ranked[1:16]:
                num_unreachable += 1
                continue
            num_recoverable += 1
            if len(observed) > 1:
                multi_annotated_recoverable += 1
            if winner in observed:
                alternative_top1_positive += 1
            else:
                hard_directed[(label, winner)] += 1
                hard_top1_label_support[label] += 1
    if count != TRAIN or (num_primary_rank1, num_recoverable, num_unreachable) != (
        34656,
        7950,
        574,
    ):
        raise RuntimeError("Frozen Train grouped OOF mismatch")

    counts = list(hard_directed.values())
    rank = sorted(counts, reverse=True)
    top10 = sum(rank[:10])
    top25 = sum(rank[:25])
    top50 = sum(rank[:50])
    total = sum(counts)
    nonempty = sum(value >= 5 for value in counts)
    return {
        "status": "TRAIN_ONLY_GROUPED_OOF_DIFFERENTIAL_NEGATIVE_ANALYSIS",
        "official_train_rows": TRAIN,
        "baseline_bm25_rank1": num_primary_rank1,
        "bm25_rank2_16_recoverable": num_recoverable,
        "bm25_gold_missing_top16": num_unreachable,
        "multi_observed_positive_rows": multi_annotated_cases,
        "recoverable_rows_in_multi_observed_group": multi_annotated_recoverable,
        "recoverable_bm25_winner_is_also_observed_positive": alternative_top1_positive,
        "recoverable_winner_valid_as_observed_negative": total,
        "distinct_directed_all_error_pairs": len(directed),
        "distinct_directed_supervised_hard_pairs": len(hard_directed),
        "directed_hard_pairs_with_at_least5_rows": nonempty,
        "rows_in_top10_confusion_pairs": top10,
        "rows_in_top25_confusion_pairs": top25,
        "rows_in_top50_confusion_pairs": top50,
        "top10_share_of_valid_hard_cases": top10 / total if total else 0,
        "top25_share_of_valid_hard_cases": top25 / total if total else 0,
        "top50_share_of_valid_hard_cases": top50 / total if total else 0,
        "hard_negative_gold_labels_with_at_least_one_case": len(hard_top1_label_support),
        "gold_label_types_in_train": len(source_label_support),
        "top15_directed_confusions": [
            {"gold": a, "bm25_top1": b, "n": v} for (a, b), v in hard_directed.most_common(15)
        ],
        "test_opened": False,
        "dev_gold_opened": False,
        "score_or_candidate_files_modified": False,
        "annotation_note": (
            "Observed-positive alternative is excluded from confirmed wrong-label "
            "training pairs, but other unannotated syndromes remain uncertain."
        ),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Historical OOF confusion report already exists")
    result = analyze()
    with REPORT.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
