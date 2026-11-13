"""Audit BM25 rank prior exposure and official Dev exact-text overlap.

This audit reads Train OOF, S3 candidate prompts and frozen Dev retrieval.
It never opens Test, changes retrieval, or writes raw patient text.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.evaluation import read_predictions

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments/retrieval_rank_shift_2026-11-12.json"


def inspect(root: Path = ROOT) -> dict:
    train = load_records(root / "data/tcm-sd/train.json")
    dev = load_records(root / "data/tcm-sd/dev.json")
    original_oof = root / "data/candidates/bm25_train_text_grouped.jsonl"
    prompts = root / "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
    dev_rankings = read_predictions(root / "data/candidates/bm25_dev.jsonl")
    if (len(train), len(dev), len(dev_rankings)) != (43180, 5486, 5486):
        raise ValueError("Need complete official Train/Dev and frozen Dev candidates")

    train_gold_slot: Counter[int] = Counter()
    original_oof_gold_rank: Counter[int] = Counter()
    with (
        original_oof.open(encoding="utf-8") as oof_stream,
        prompts.open(encoding="utf-8") as prompt_stream,
    ):
        count = 0
        for i, (source_line, prompt_line) in enumerate(zip(oof_stream, prompt_stream, strict=True)):
            if i >= len(train):
                raise ValueError("Extra Train OOF or prompt records")
            source, prompt = json.loads(source_line), json.loads(prompt_line)
            gold = str(train[i]["norm_syndrome"])
            case_id = f"train-{i:06d}"
            if (
                source.get("case_id") != case_id
                or source.get("gold_labels") != [gold]
                or prompt.get("row", {}).get("id") != case_id
            ):
                raise ValueError(f"OOF versus Train prompt alignment mismatch at {i}")
            ranking = source["candidate_labels"]
            options = prompt["row"]["options"]
            slot = prompt["target_index"]
            if (
                len(ranking) != 50
                or len(set(ranking)) != 50
                or len(options) != 16
                or not isinstance(slot, int)
                or not 0 <= slot < 16
                or options[slot]["id"] != gold
            ):
                raise ValueError(f"Invalid OOF or candidate format at {i}")
            train_gold_slot[slot + 1] += 1
            rank = ranking.index(gold) + 1 if gold in ranking else 51
            original_oof_gold_rank[rank] += 1
            count += 1
        if count != len(train):
            raise ValueError("Incomplete Train OOF or prompt dataset")

    known_text: dict[str, set[str]] = defaultdict(set)
    for row in train:
        known_text[clinical_text(row)].add(str(row["norm_syndrome"]))

    dev_gold_slot: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    split_correct: Counter[str] = Counter()
    split_coverage: Counter[str] = Counter()
    inconsistent_repeats = 0
    for i, (row, candidate) in enumerate(zip(dev, dev_rankings, strict=True)):
        case_id = f"dev-{i:06d}"
        gold = str(row["norm_syndrome"])
        if candidate.case_id != case_id or candidate.gold_labels != (gold,):
            raise ValueError(f"Dev labels and candidate ranking mismatch at {i}")
        labels = candidate.candidate_labels[:16]
        if len(labels) != 16 or len(set(labels)) != 16:
            raise ValueError(f"Missing or duplicated Dev Top16 at {i}")
        rank = labels.index(gold) + 1 if gold in labels else 0
        dev_gold_slot[rank] += 1
        source = clinical_text(row)
        group = "overlap" if source in known_text else "no_exact_train_match"
        split_counts[group] += 1
        split_correct[group] += rank == 1
        split_coverage[group] += rank != 0
        if group == "overlap" and gold not in known_text[source]:
            inconsistent_repeats += 1

    if (
        train_gold_slot.total() != 43180
        or original_oof_gold_rank[1] != 34656
        or sum(original_oof_gold_rank[k] for k in range(2, 17)) != 7950
        or sum(original_oof_gold_rank[k] for k in range(17, 52)) != 574
        or dev_gold_slot[1] != 4366
        or sum(dev_gold_slot[k] for k in range(2, 17)) != 1043
        or dev_gold_slot[0] != 77
    ):
        raise ValueError("Frozen rank-position distribution unexpectedly changed")

    return {
        "status": "RETRIEVAL_RANK_AND_OFFICIAL_DEV_OVERLAP_AUDIT_PASS",
        "official_train": len(train),
        "official_dev": len(dev),
        "train_gold_injected_into_each_s3_prompt": True,
        "train_gold_prompt_slot_counts_1based": {str(k): train_gold_slot[k] for k in range(1, 17)},
        "train_gold_prompt_first_slot_rate": train_gold_slot[1] / len(train),
        "original_oof_rank_1": original_oof_gold_rank[1],
        "original_oof_rank_2_to_16": sum(original_oof_gold_rank[k] for k in range(2, 17)),
        "original_oof_rank_over_16": sum(original_oof_gold_rank[k] for k in range(17, 52)),
        "dev_gold_rank_1": dev_gold_slot[1],
        "dev_gold_rank_2_to_16": sum(dev_gold_slot[k] for k in range(2, 17)),
        "dev_gold_absent_top16": dev_gold_slot[0],
        "dev_gold_rank1_rate": dev_gold_slot[1] / len(dev),
        "dev_exact_overlap": {
            group: {
                "rows": split_counts[group],
                "bm25_correct": split_correct[group],
                "bm25_top1": split_correct[group] / split_counts[group],
                "bm25_recall16": split_coverage[group] / split_counts[group],
            }
            for group in ("overlap", "no_exact_train_match")
        },
        "exact_overlap_with_unseen_gold_label_n": inconsistent_repeats,
        "annotation_note": (
            "Train prompts place the original gold in every 16-way choice; "
            "Dev candidates preserve the original BM25 order. Exposing a "
            "synthetic rank >16 for inserted gold would introduce target leakage."
        ),
        "no_patient_text_exported": True,
        "test_opened": False,
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Frozen aggregate report exists: {REPORT}")
    result = inspect()
    with REPORT.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
