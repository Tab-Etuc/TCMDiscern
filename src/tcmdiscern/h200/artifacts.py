"""Clone-safe, resumable preparation of the frozen TCM-SD research artifacts.

Only the official Train and Dev partitions are needed. All generated
record-level data live in ignored data/; published experiments/ are read-only
reference results. Existing artifacts are never silently overwritten.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.jeff_format import decision_row
from tcmdiscern.oof import rank_statistics, rank_train_oof
from tcmdiscern.retrieval import CaseBM25

TRAIN_ROWS = 43180
DEV_ROWS = 5486
FROZEN_OOF = "bm25_train_text_grouped_2026-11-11.json"
FROZEN_DEV = "bm25_k1_35_dev_2026-11-11.json"
TRAIN_OOF = "data/candidates/bm25_train_text_grouped.jsonl"
DEV_CANDIDATES = "data/candidates/bm25_dev.jsonl"
SINGLE_TRAIN = "data/jeff/train_oof_k16_epoch02.jsonl"
MULTI_TRAIN = "data/jeff/train_oof_multipos_k16_epoch03.jsonl"
DEV_KS = (5, 16, 17)


def require_official_data(root: Path) -> tuple[list[dict], list[dict]]:
    """Fail before indexing if a clone is missing the official clinical data."""
    train_file, dev_file = (
        root / "data/tcm-sd/train.json",
        root / "data/tcm-sd/dev.json",
    )
    vocab_file = root / "data/tcm-sd/syndrome_vocab.txt"
    missing = [p for p in (train_file, dev_file, vocab_file) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official TCM-SD Train/Dev files are missing. "
            "Run 'python -m scripts.fetch_tcmsd' (without --include-test). "
            f"Missing paths: {[str(p.relative_to(root)) for p in missing]}"
        )
    train, dev = load_records(train_file), load_records(dev_file)
    if len(train) != TRAIN_ROWS or len(dev) != DEV_ROWS:
        raise ValueError("Official Train/Dev row counts differ from the frozen study")
    vocab_lines = [
        x.strip() for x in vocab_file.read_text(encoding="utf-8-sig").splitlines() if x.strip()
    ]
    vocab = set(vocab_lines)
    if len(vocab_lines) != 148 or len(vocab) != 148:
        raise ValueError("Official 148-class syndrome vocabulary is incomplete")
    for split, rows in (("train", train), ("dev", dev)):
        for i, row in enumerate(rows):
            if row.get("norm_syndrome") not in vocab:
                raise ValueError(f"{split} row {i} contains an invalid official syndrome")
    return train, dev


def publish_lines(destination: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomic, no-overwrite JSONL writer; delete partial output on any failure."""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".preparing-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # Hardlink fails if another process published the same destination.
        os.link(temp_name, destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def publish_generator(destination: Path, generate: Callable[[Path], None]) -> None:
    """Run an existing legacy exporter under an unpublished temporary path."""
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=destination.parent) as dirname:
        temporary = Path(dirname) / destination.name
        generate(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Generator produced no output for {destination.name}")
        os.link(temporary, destination)


def _frozen(root: Path, name: str) -> dict:
    file = root / "experiments" / name
    if not file.is_file():
        raise FileNotFoundError(f"Frozen research snapshot unavailable: {file}")
    return json.loads(file.read_text(encoding="utf-8"))


def _expect_close(actual: float, expected: float, *, name: str) -> None:
    if abs(actual - expected) > 1e-11:
        raise ValueError(f"{name} differs from the pre-registered experiment")


def build_train_oof(root: Path, train: list[dict]) -> None:
    """Regenerate exactly the original 5-fold grouped BM25 candidate source."""
    ref = _frozen(root, FROZEN_OOF)
    if ref.get("fold_policy") != "exact_text_grouped" or ref.get("seed") != 20261111:
        raise ValueError("Frozen OOF configuration does not match the research protocol")
    labels = [str(row["norm_syndrome"]) for row in train]
    texts = [clinical_text(row) for row in train]
    oof = rank_train_oof(
        texts,
        labels,
        folds=5,
        seed=ref["seed"],
        max_k=50,
        batch_size=24,
        fold_policy="exact_text_grouped",
        on_fold_done=lambda fold, indexed, queried, seconds: print(
            f"  OOF {fold + 1}/5: {queried} Train-only queries, {seconds:.1f}s", flush=True
        ),
    )
    actual = rank_statistics(labels, oof, ks=(1, 3, 5, 10, 16, 17, 35, 50))
    if list(oof.fold_query_sizes) != ref["fold_query_rows"]:
        raise ValueError("OOF fold assignments do not match the frozen reference")
    for field in ("bm25_oof_candidate_micro_recall", "bm25_oof_candidate_macro_recall"):
        for k, expected in ref[field].items():
            _expect_close(actual[field][k], expected, name=f"OOF {field}@{k}")

    def iter_rows():
        for i, (gold, fold, candidates) in enumerate(
            zip(labels, oof.fold_by_row, oof.rankings, strict=True)
        ):
            yield {
                "case_id": f"train-{i:06d}",
                "fold": fold,
                "gold_labels": [gold],
                "candidate_labels": list(candidates),
            }

    publish_lines(root / TRAIN_OOF, iter_rows())


def build_dev_candidates(root: Path, train: list[dict], dev: list[dict]) -> None:
    """Train-only index, official untouched Dev, and frozen metric parity."""
    frozen = _frozen(root, FROZEN_DEV)["results"][15]
    retriever = CaseBM25(
        [clinical_text(row) for row in train],
        [str(row["norm_syndrome"]) for row in train],
        k1=1.5,
        b=0.75,
        min_df=2,
    )
    rankings = retriever.rank([clinical_text(row) for row in dev], max_k=50, batch_size=24)
    gold = [str(row["norm_syndrome"]) for row in dev]
    top1 = sum(options[0] == label for options, label in zip(rankings, gold, strict=True))
    found = sum(label in options[:16] for options, label in zip(rankings, gold, strict=True))
    if top1 != 4366 or found != 5409:
        raise ValueError("BM25 Dev correct/covered counts differ from frozen protocol")
    _expect_close(top1 / DEV_ROWS, frozen["bm25_top1_accuracy"], name="BM25 Dev Top-1")

    def iter_rows():
        for i, (label, options) in enumerate(zip(gold, rankings, strict=True)):
            yield {
                "case_id": f"dev-{i:06d}",
                "gold_labels": [label],
                "candidate_labels": options,
            }

    publish_lines(root / DEV_CANDIDATES, iter_rows())


def build_train_prompts(root: Path, train: list[dict]) -> None:
    """Reuse the original audited generators and the original frozen seeds."""
    from scripts.export_jeff_multipos import export as export_multi
    from scripts.export_jeff_train import load_and_validate_oof, write_examples

    reference = _frozen(root, FROZEN_OOF)
    ranks = load_and_validate_oof(train, root / TRAIN_OOF, reference)
    single = root / SINGLE_TRAIN
    if not single.exists():
        frozen = _frozen(root, "jeff_train_examples_epoch02_2026-11-11.json")

        def create_single(path: Path) -> None:
            stats = write_examples(
                train,
                ranks,
                path=path,
                k=16,
                epoch=frozen["shuffling_epoch"],
                seed=frozen["shuffling_seed"],
                exclude_ambiguous_negatives=True,
            )
            if stats["k_counts"] != {"16": TRAIN_ROWS}:
                raise ValueError("S1 source export has wrong candidate count")
            if (
                stats["answer_position_counts_zero_based"]
                != frozen["files"][0]["answer_position_counts_zero_based"]
            ):
                raise ValueError("S1 gold positions differ from frozen generator")

        publish_generator(single, create_single)
    multi = root / MULTI_TRAIN
    if not multi.exists():
        frozen = _frozen(root, "jeff_multipos_k16_epoch03_2026-11-12.json")

        def create_multi(path: Path) -> None:
            stats = export_multi(
                train, ranks, path=path, epoch=frozen["epoch"], seed=frozen["seed"]
            )
            if (
                stats["observed_positive_counts_by_original_row"]
                != frozen["data"]["observed_positive_counts_by_original_row"]
            ):
                raise ValueError("Multi-positive label counts differ from frozen generator")

        publish_generator(multi, create_multi)


def build_dev_prompts(root: Path, dev: list[dict]) -> None:
    from tcmdiscern.evaluation import read_predictions

    retrieved = read_predictions(root / DEV_CANDIDATES)
    if len(retrieved) != DEV_ROWS:
        raise ValueError("Dev BM25 candidates are incomplete")
    for k in DEV_KS:
        destination = root / f"data/jeff/dev_k{k:02d}.jsonl"
        if destination.exists():
            continue

        def iter_rows(*, selected_k: int = k):
            for i, (case, rank) in enumerate(zip(dev, retrieved, strict=True)):
                if rank.case_id != f"dev-{i:06d}" or rank.gold_labels != (
                    str(case["norm_syndrome"]),
                ):
                    raise ValueError("Dev prompts cannot use mismatched BM25 labels")
                yield decision_row(
                    case,
                    case_id=rank.case_id,
                    candidates=rank.candidate_labels[:selected_k],
                    allow_grouped=selected_k > 16,
                )

        publish_lines(destination, iter_rows())


def prepare(root: Path, *, on_stage: Callable[[str], None] = print) -> dict:
    """Resumable CPU data build followed by a fresh complete-record audit."""
    train, dev = require_official_data(root)
    if not (root / TRAIN_OOF).exists():
        on_stage("build Group-aware OOF BM25: 43,180 Train rows")
        build_train_oof(root, train)
    if not (root / DEV_CANDIDATES).exists():
        on_stage("build frozen Dev BM25: 5,486 cases")
        build_dev_candidates(root, train, dev)
    on_stage("verify and build S1 / S2-S4 supervised Train prompts")
    build_train_prompts(root, train)
    on_stage("verify and build K=5,16,17 Dev prompts")
    build_dev_prompts(root, dev)

    # Full independent provenance check is mandatory even when every file
    # already exists, so stale or partially copied clone artifacts cannot pass.
    from scripts.audit_h200_payload import run_audit

    result = run_audit(root)
    if result["status"] != "FULL_RECORD_SOURCE_AND_PAYLOAD_AUDIT_PASS":
        raise RuntimeError("Full-record payload validation did not pass")
    return {
        "status": "H200_TRAIN_DEV_PREPARED",
        "train": TRAIN_ROWS,
        "dev": DEV_ROWS,
        "test_opened": False,
        "oof_rankings": TRAIN_ROWS,
        "supervised_questions_per_method": TRAIN_ROWS,
        "full_audit": result["status"],
    }
