"""Build all TCM-SD disease-conditioned Jeff experiment inputs using Train-only BM25.

Usage:
  python -m scripts.prepare_disease_study --split dev
  python -m scripts.prepare_disease_study --split test  # after model selection
All per-case clinical text stays in data/disease_study/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.disease_study import (
    MODES,
    SEED,
    K,
    allowed_syndromes,
    baseline_scores,
    disease_name,
    eval_item,
    gold_for,
    pair_key,
    prioritize_disease,
    syndrome_name,
    train_item,
)
from tcmdiscern.h200.artifacts import publish_lines, require_official_data
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/disease_study"
TRAIN_COUNT, DEV_COUNT = 43180, 5486


def read_global_oof(train: list[dict]) -> tuple[list[int], list[list[str]]]:
    source = ROOT / "data/candidates/bm25_train_text_grouped.jsonl"
    assignments: list[int] = []
    ranked: list[list[str]] = []
    with source.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            obj = json.loads(line)
            if (
                obj["case_id"] != f"train-{i:06d}"
                or obj["gold_labels"] != [syndrome_name(train[i])]
                or not 0 <= obj["fold"] < 5
                or len(obj["candidate_labels"]) < K
            ):
                raise ValueError(f"Invalid Train OOF candidate at row {i}")
            assignments.append(obj["fold"])
            ranked.append(obj["candidate_labels"])
    if len(ranked) != TRAIN_COUNT or len(set(assignments)) != 5:
        raise ValueError("Complete official Train OOF rankings are required")
    return assignments, ranked


def source_global_eval(cases: list[dict], split: str) -> list[list[str]]:
    if split == "dev":
        path = ROOT / "data/candidates/bm25_dev.jsonl"
        source: list[list[str]] = []
        with path.open(encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                if (
                    row["case_id"] != f"dev-{i:06d}"
                    or row["gold_labels"] != [syndrome_name(cases[i])]
                ):
                    raise ValueError("Original Dev and BM25 candidates disagree")
                source.append(row["candidate_labels"])
        if len(source) != DEV_COUNT:
            raise ValueError("Incomplete official Dev candidates")
        return source

    train, _ = require_official_data(ROOT)
    bm = CaseBM25(
        [clinical_text(row) for row in train],
        [syndrome_name(row) for row in train],
    )
    return bm.rank([clinical_text(row) for row in cases], max_k=50)


def rank_pair_candidates(
    train: list[dict], queries: list[dict], *, folds: list[int] | None = None
) -> list[list[str]]:
    train_texts = [clinical_text(row) for row in train]
    train_labels = [pair_key(disease_name(row), syndrome_name(row)) for row in train]
    if folds is None:
        bm = CaseBM25(train_texts, train_labels)
        return bm.rank(
            [clinical_text(row) for row in queries],
            max_k=min(50, len(bm.label_names)),
        )

    if len(folds) != len(train) or len(queries) != len(train):
        raise ValueError("Train OOF pair indexing requires one fold per original row")
    output: list[list[str] | None] = [None] * len(train)
    for fold in range(5):
        source_ids = [i for i, f in enumerate(folds) if f != fold]
        query_ids = [i for i, f in enumerate(folds) if f == fold]
        bm = CaseBM25(
            [train_texts[i] for i in source_ids],
            [train_labels[i] for i in source_ids],
        )
        ranking = bm.rank(
            [train_texts[i] for i in query_ids],
            max_k=min(50, len(bm.label_names)),
        )
        for i, labels in zip(query_ids, ranking, strict=True):
            output[i] = labels
        print(f"  Disease-pair OOF BM25 {fold + 1}/5: {len(query_ids)} Train queries", flush=True)
    if any(row is None for row in output):
        raise RuntimeError("Incomplete disease-pair OOF rankings")
    return [row for row in output if row is not None]


def validate_existing(
    path: Path, *, count: int, mode: str, kind: str, split: str,
) -> None:
    """Only reuse a complete source with the expected identifiers and option width."""
    seen = 0
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            data = json.loads(line)
            row = data["row"] if kind == "train" else data
            case_id = f"{'train' if kind == 'train' else split}-{index:06d}"
            if kind == "labels":
                if data.get("case_id") != case_id or len(data.get("candidates", [])) != K:
                    raise ValueError(f"Corrupt cached metadata: {path}")
            else:
                options = row["options"]
                if row["id"] != case_id or len(options) != K or len({x["id"] for x in options}) != K:
                    raise ValueError(f"Corrupt cached prompts: {path}")
                if kind == "train" and not 0 <= data["target_index"] < K:
                    raise ValueError(f"Invalid cached training target: {path}")
                if mode == "conditioned" and "lcd_name" not in row["state"]:
                    raise ValueError(f"Missing disease context in {path}")
            seen += 1
    if seen != count:
        raise ValueError(f"Expected {count} rows in {path}, found {seen}")


def ensure_source(
    path: Path, *, rows: Iterable[dict], count: int, mode: str, kind: str, split: str,
) -> None:
    if not path.exists():
        publish_lines(path, rows)
    validate_existing(path, count=count, mode=mode, kind=kind, split=split)


def build_training(train: list[dict], folds: list[int], global_rank: list[list[str]]) -> dict:
    all_modes = {
        "global": global_rank,
        "conditioned": [None] * len(train),
        "pair": [None] * len(train),
    }
    per_fold_allow = {}
    for fold in range(5):
        subset = [case for case, assigned in zip(train, folds, strict=True) if assigned != fold]
        per_fold_allow[fold] = allowed_syndromes(subset)

    for i, case in enumerate(train):
        all_modes["conditioned"][i] = prioritize_disease(
            global_rank[i], disease=disease_name(case), allow=per_fold_allow[folds[i]]
        )
    if not (OUT / "train_pair_k5.jsonl").is_file():
        all_modes["pair"] = rank_pair_candidates(train, train, folds=folds)

    built = {}
    for mode in MODES:
        path = OUT / f"train_{mode}_k5.jsonl"
        if path.is_file():
            validate_existing(path, count=TRAIN_COUNT, mode=mode, kind="train", split="train")
        else:
            ranked = all_modes[mode]
            if any(row is None for row in ranked):
                raise RuntimeError(f"No Train OOF data for {mode}")
            ensure_source(
                path,
                rows=(
                    train_item(
                        case, index=i, ranked=ranked[i], mode=mode, seed=SEED,
                        allow=per_fold_allow[folds[i]] if mode == "conditioned" else None,
                    )
                    for i, case in enumerate(train)
                ),
                count=TRAIN_COUNT, mode=mode, kind="train", split="train",
            )
        built[mode] = TRAIN_COUNT
    return built


def build_evaluation(
    train: list[dict], cases: list[dict], *, split: str,
) -> tuple[dict, dict]:
    source_global = source_global_eval(cases, split)
    allowed = allowed_syndromes(train)
    source_cond = [
        prioritize_disease(ranked, disease=disease_name(case), allow=allowed)
        for case, ranked in zip(cases, source_global, strict=True)
    ]
    source_pair = rank_pair_candidates(train, cases)
    rankings = {"global": source_global, "conditioned": source_cond, "pair": source_pair}
    raw_source = {}
    for mode in MODES:
        prompts = OUT / f"{split}_{mode}_prompts.jsonl"
        labels = OUT / f"{split}_{mode}_labels.jsonl"
        candidate_rows = [
            eval_item(case, index=i, ranked=rankings[mode][i], mode=mode, split=split)
            for i, case in enumerate(cases)
        ]
        ensure_source(
            prompts, rows=(row for row, _ in candidate_rows),
            count=len(cases), mode=mode, kind="eval", split=split,
        )
        ensure_source(
            labels, rows=(meta for _, meta in candidate_rows),
            count=len(cases), mode=mode, kind="labels", split=split,
        )
        raw_source[mode] = [meta for _, meta in candidate_rows]
    return baseline_scores(train, cases, raw_source), {
        mode: sum(
            gold_for(case, mode) in ranking[:K]
            for case, ranking in zip(cases, rankings[mode], strict=True)
        )
        for mode in MODES
    }


def prepare(*, split: str) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("Only official Dev or Test split supported")
    train, dev = require_official_data(ROOT)
    if split == "test":
        cases = load_records(ROOT / "data/tcm-sd/test.json")
        if len(cases) != DEV_COUNT:
            raise ValueError("Official Test must contain 5,486 records")
    else:
        cases = dev
    OUT.mkdir(parents=True, exist_ok=True)
    folds, global_rank = read_global_oof(train)
    print("[STUDY] Generating validated five-option Train OOF datasets", flush=True)
    train_sources = build_training(train, folds, global_rank)
    print(f"[STUDY] Generating five-option official {split.upper()} inference data", flush=True)
    baselines, oracle_counts = build_evaluation(train, cases, split=split)
    report = {
        "status": "DISEASE_STUDY_PREPARED",
        "split": split,
        "train_rows": len(train),
        "evaluation_rows": len(cases),
        "k": K,
        "model": "jgeuter/Jeff-1.1-Large",
        "modes": list(MODES),
        "train_sources": train_sources,
        "bm25_baselines": baselines,
        "retrieval_oracle_hits": oracle_counts,
        "benchmark_protocol": "official original Train/Dev/Test, row-level single-label",
        "gold_in_inference_prompts": False,
    }
    output = ROOT / "experiments" / f"disease_study_{split}_baselines.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    result = prepare(split=args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
