"""CDET: genuine Train-only grouped Fit/Cal/Audit and BM25 retrieval OOF.

Never imports official Dev/Test. No Cal/Audit gold enters any retriever or
patient features. Candidate files intentionally contain no gold labels/text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from sklearn.model_selection import GroupKFold

from tcmdiscern.cdet_protocol import grouped_partition
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]


def dump_candidates(path: Path, indices: list[int], ranked: list[list[str]]) -> None:
    if len(indices) != len(ranked):
        raise ValueError("Row/candidate mismatch")
    with path.open("w", encoding="utf-8") as fh:
        for idx, labels in sorted(zip(indices, ranked, strict=True)):
            fh.write(
                json.dumps(
                    {"case_id": f"train-{idx:06d}", "candidate_labels": labels[:16]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def score_retrieval(indices, rankings, rows) -> dict:
    return {
        "queries": len(indices),
        "bm25_correct_rank1": sum(
            rows[i]["norm_syndrome"] == labels[0]
            for i, labels in zip(indices, rankings, strict=True)
        ),
        "bm25_gold_in_top16": sum(
            rows[i]["norm_syndrome"] in labels[:16]
            for i, labels in zip(indices, rankings, strict=True)
        ),
    }


def prepare(*, limit_records=None, seed=20261115, batch_size=24):
    data = ROOT / "data/tcm-sd/train.json"
    rows = load_records(data)
    if limit_records is not None:
        if limit_records < 500:
            raise ValueError("Smoke case limit must be >=500")
        rows = rows[:limit_records]
    name = "cdet_smoke" if limit_records is not None else "cdet"
    destination = ROOT / "data" / name
    destination.mkdir(parents=True, exist_ok=True)
    start = perf_counter()

    split = grouped_partition(rows, seed=seed)
    groups = split.groups
    fit, cal, audit = list(split.fit), list(split.cal), list(split.audit)
    print(
        f"Split Fit={len(fit)} Cal={len(cal)} Audit={len(audit)}; groups={len(set(groups))}",
        flush=True,
    )
    texts = [clinical_text(row) for row in rows]
    labels = [str(row["norm_syndrome"]) for row in rows]
    positions = {}
    print("Rebuilding BM25 Cal/Audit index using Fit only", flush=True)
    outer = CaseBM25([texts[i] for i in fit], [labels[i] for i in fit])
    for part, indices in [("cal", cal), ("audit", audit)]:
        ranked = outer.rank(
            [texts[i] for i in indices],
            max_k=min(16, len(outer.label_names)),
            batch_size=batch_size,
        )
        dump_candidates(destination / f"{part}_candidates.jsonl", indices, ranked)
        positions[part] = (indices, ranked)
        print(f"Built {part} candidates; {len(indices)} queries", flush=True)
    del outer

    # Inner OOF never indexes the row's patient/text group, nor outer Cal/Audit.
    print("Rebuilding five inner Fit-only BM25 OOF folds", flush=True)
    from numpy import asarray

    fit_array = asarray(fit)
    fold_groups = asarray([groups[i] for i in fit])
    collected = {}
    for fold, (train_pos, val_pos) in enumerate(
        GroupKFold(n_splits=5).split(fit_array, groups=fold_groups)
    ):
        training = fit_array[train_pos].tolist()
        heldout = fit_array[val_pos].tolist()
        if {groups[i] for i in training} & {groups[i] for i in heldout}:
            raise AssertionError("Retrieval index leaks OOF group")
        index = CaseBM25([texts[i] for i in training], [labels[i] for i in training])
        ranked = index.rank(
            [texts[i] for i in heldout],
            max_k=min(16, len(index.label_names)),
            batch_size=batch_size,
        )
        collected.update(zip(heldout, ranked, strict=True))
        print(f"Fit inner OOF fold {fold + 1}/5: {len(heldout)} queries", flush=True)
        del index
    if set(collected) != set(fit):
        raise AssertionError("Fit OOF does not cover the complete fitting partition")
    fit_ranked = [collected[i] for i in fit]
    dump_candidates(destination / "fit_oof_candidates.jsonl", fit, fit_ranked)
    all_parts = {"fit": fit, "cal": cal, "audit": audit}
    (destination / "partition_indices.json").write_text(
        json.dumps(all_parts) + "\n", encoding="utf-8"
    )

    # Audit labels stay sealed until the final one-pass evaluation.
    result = {
        "protocol": "Train-only patient+exact-text transitive grouped Fit/Cal/Audit",
        "seed": seed,
        "smoke_case_limit": limit_records,
        "index_source": "outer Fit only; inner Fit minus one group-disjoint fold",
        "partition_sizes": {key: len(value) for key, value in all_parts.items()},
        "group_counts": {
            key: len({groups[i] for i in indices}) for key, indices in all_parts.items()
        },
        "label_counts": {
            "full_train": len(set(labels)),
            "fit": len({labels[i] for i in fit}),
        },
        "fit_oof_bm25": score_retrieval(fit, fit_ranked, rows),
        "cal_bm25": score_retrieval(*positions["cal"], rows),
        "audit_bm25": "SEALED (no gold evaluated)",
        "seconds": round(perf_counter() - start, 2),
    }
    (destination / "retrieval_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-cases", type=int)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--batch-size", type=int, default=24)
    opts = parser.parse_args()
    prepare(limit_records=opts.smoke_cases, seed=opts.seed, batch_size=opts.batch_size)
