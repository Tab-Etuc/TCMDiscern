"""Train-only probe of BM25 score profiles; ordinary learned pooling controls.

Run from TCMDiscern. It never scores Train-Audit or loads official Dev/Test.
A fold builds features with a group-disjoint Fit index; evaluate fits on all
five OOF caches and scores the previously explored Cal once. No gold injection.
The classifier here is a diagnostic baseline, not a proposed novel algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from tcmdiscern.cdet_protocol import grouped_partition
from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/bm25_profile_probe_2026-11-16"
OUT = ROOT / "experiments/bm25_profile_probe_2026-11-16.json"
KS = (1, 2, 3, 5, 8)


def top_profile(values):
    k = min(8, values.shape[1])
    top = np.sort(np.partition(values, -k, axis=1)[:, -k:], axis=1)[:, ::-1]
    return np.stack([top[:, : min(t, k)].mean(axis=1) for t in KS], axis=-1)


def inputs():
    rows = load_records(ROOT / "data/tcm-sd/train.json")
    parts = json.loads((ROOT / "data/cdet/partition_indices.json").read_text())
    # Reconstruct group identities, and verify the existing frozen partition.
    split = grouped_partition(rows)
    assert tuple(parts["fit"]) == split.fit and tuple(parts["cal"]) == split.cal
    return rows, parts, np.asarray(split.groups)


def features(train_ids, query_ids, rows, groups, expected_path):
    t0 = perf_counter()
    assert not set(groups[train_ids]) & set(groups[query_ids])
    labels = np.asarray(sorted({str(r["norm_syndrome"]) for r in rows}))
    to_id = {c: i for i, c in enumerate(labels)}
    ytrain = [str(rows[i]["norm_syndrome"]) for i in train_ids]
    model = CaseBM25([clinical_text(rows[i]) for i in train_ids], ytrain)
    names = model.label_names
    ids = np.asarray([to_id[n] for n in names])
    ordered = model.ordered_doc_ids
    starts = model.group_starts
    ends = np.r_[starts[1:], len(train_ids)]
    counts = ends - starts
    # One support unit per (patient/exact-text connected component, class).
    pair_to_unit = {}
    units = []
    for idx, label in zip(train_ids, ytrain, strict=True):
        key = (label, int(groups[idx]))
        if key not in pair_to_unit:
            pair_to_unit[key] = len(pair_to_unit)
        units.append(pair_to_unit[key])
    unit_order = np.argsort(units, kind="stable")
    unit_starts = np.flatnonzero(np.r_[True, np.diff(np.asarray(units)[unit_order]) != 0])
    unit_labels = np.asarray([key[0] for key in pair_to_unit])
    group_label_order = np.argsort(unit_labels, kind="stable")
    group_starts = np.flatnonzero(
        np.r_[True, unit_labels[group_label_order][1:] != unit_labels[group_label_order][:-1]]
    )
    group_ends = np.r_[group_starts[1:], len(unit_labels)]
    group_counts = group_ends - group_starts
    assert list(unit_labels[group_label_order][group_starts]) == names

    expected = {}
    for line in expected_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        expected[r["case_id"]] = r["candidate_labels"]
    xx, cc, pp, gg = [], [], [], []
    parity = 0
    for offset in range(0, len(query_ids), 24):
        qids = query_ids[offset : offset + 24]
        q = model.vectorizer.transform([clinical_text(rows[i]) for i in qids])
        q.data[:] = 1
        score = (q @ model.weights_t).toarray()
        docs = score[:, ordered]
        raw = np.stack(
            [top_profile(docs[:, a:b]) for a, b in zip(starts, ends, strict=True)], axis=1
        )
        grouped = np.maximum.reduceat(score[:, unit_order], unit_starts, axis=1)[
            :, group_label_order
        ]
        gp = np.stack(
            [top_profile(grouped[:, a:b]) for a, b in zip(group_starts, group_ends, strict=True)],
            axis=1,
        )
        choice = np.argsort(-raw[:, :, 0], axis=1, kind="stable")[:, :16]
        raw = np.take_along_axis(raw, choice[:, :, None], axis=1)
        gp = np.take_along_axis(gp, choice[:, :, None], axis=1)
        scale = np.maximum(raw[:, 0, 0], 1e-6)[:, None, None]
        ccount = counts[choice]
        gcount = group_counts[choice]
        # Per-class evidence profile, relative competitor margins and density.
        normalized = np.concatenate([raw / scale, gp / scale], axis=2)
        top_two = np.partition(normalized, -2, axis=1)[:, -2:]
        biggest = top_two.max(axis=1, keepdims=True)
        second = top_two.min(axis=1, keepdims=True)
        other = np.where(normalized == biggest, second, biggest)
        contrast = normalized - other
        density = np.stack([np.log1p(ccount), np.log1p(gcount), gcount / ccount], axis=2)
        context = np.broadcast_to(
            np.stack([np.log1p(scale[:, 0, 0]), np.log1p(q.getnnz(axis=1))], axis=1)[:, None, :],
            (len(qids), 16, 2),
        )
        feat = np.concatenate([normalized, contrast, density, context], axis=2)
        global_choice = ids[choice]
        for idx, chosen in zip(qids, global_choice, strict=True):
            if list(labels[chosen]) != expected[f"train-{idx:06d}"][:16]:
                parity += 1
        xx.append(feat.astype(np.float32))
        cc.append(global_choice)
        pp.append(raw.astype(np.float32))
        gg.append(gp.astype(np.float32))
    assert parity == 0, f"Candidate parity failed: {parity}"
    truth = np.asarray([to_id[str(rows[i]["norm_syndrome"])] for i in query_ids])
    return {
        "x": np.concatenate(xx),
        "candidates": np.concatenate(cc),
        "raw": np.concatenate(pp),
        "grouped": np.concatenate(gg),
        "truth": truth,
        "query_ids": np.asarray(query_ids),
        "groups": groups[query_ids],
        "labels": labels,
        "index_rows": np.asarray(len(train_ids)),
        "index_groups": np.asarray(len(set(groups[train_ids]))),
        "seconds": np.asarray(perf_counter() - t0),
    }


def build(stage):
    rows, parts, groups = inputs()
    fit = np.asarray(parts["fit"])
    if stage == "cal":
        training, heldout = fit, np.asarray(parts["cal"])
        source = ROOT / "data/cdet/cal_candidates.jsonl"
    else:
        fold = int(stage)
        train_pos, query_pos = list(GroupKFold(5).split(fit, groups=groups[fit]))[fold]
        training, heldout = fit[train_pos], fit[query_pos]
        source = ROOT / "data/cdet/fit_oof_candidates.jsonl"
    result = features(training, heldout, rows, groups, source)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{stage}.npz", **result)
    print(
        json.dumps(
            {
                "stage": stage,
                "queries": len(heldout),
                "features": result["x"].shape[-1],
                "seconds": float(result["seconds"]),
                "candidate_parity": True,
            }
        ),
        flush=True,
    )


def evaluate():
    t0 = perf_counter()
    folds = [dict(np.load(CACHE / f"{i}.npz", allow_pickle=False)) for i in range(5)]
    cal = dict(np.load(CACHE / "cal.npz", allow_pickle=False))
    assert len(set(np.concatenate([f["query_ids"] for f in folds]))) == 30226
    assert not set(np.concatenate([f["groups"] for f in folds])) & set(cal["groups"])
    assert all(np.array_equal(f["labels"], cal["labels"]) for f in folds)
    x = np.concatenate([f["x"] for f in folds])
    cand = np.concatenate([f["candidates"] for f in folds])
    truth = np.concatenate([f["truth"] for f in folds])
    y = (cand == truth[:, None]).astype(int)
    rows, _parts, _ = inputs()
    support = Counter(str(r["norm_syndrome"]) for r in rows)
    labels = cal["labels"]
    tier = {
        "tail_11": [i for i, c in enumerate(labels) if support[c] < 10],
        "mid_94": [i for i, c in enumerate(labels) if 10 <= support[c] < 100],
        "head_43": [i for i, c in enumerate(labels) if support[c] >= 100],
    }
    gold = cal["truth"]
    preds = {}
    for j, k in enumerate(KS):
        for key, arr in (("raw", cal["raw"]), ("group", cal["grouped"])):
            preds[f"{key}_top{k}"] = cal["candidates"][
                np.arange(len(gold)), arr[:, :, j].argmax(axis=1)
            ]
    # Frozen low-capacity controls; neither searches Cal nor uses class IDs.
    variants = {
        "hgb_raw_profile": list(range(5)) + list(range(10, 15)) + [20, 23, 24],
        "hgb_group_profile": list(range(5, 10)) + list(range(15, 20)) + [21, 23, 24],
        "hgb_joint_profile": list(range(x.shape[-1])),
    }
    learned = {}
    with threadpool_limits(limits=4):
        for name, cols in variants.items():
            start = perf_counter()
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=15,
                learning_rate=0.07,
                min_samples_leaf=40,
                l2_regularization=10.0,
                early_stopping=False,
                random_state=20261116,
            )
            clf.fit(x[:, :, cols].reshape(-1, len(cols)), y.ravel())
            score = clf.predict_proba(cal["x"][:, :, cols].reshape(-1, len(cols)))[:, 1].reshape(
                -1, 16
            )
            preds[name] = cal["candidates"][np.arange(len(gold)), score.argmax(axis=1)]
            learned[name] = {"columns": cols, "seconds": round(perf_counter() - start, 2)}
            print(f"Trained {name}: {learned[name]['seconds']}s", flush=True)
    baseline = preds["raw_top1"]
    top3 = preds["raw_top3"]
    metrics = {}
    for name, pred in preds.items():
        metrics[name] = {
            "top1_percent": float(100 * np.mean(pred == gold)),
            "macro_f1_148_percent": float(
                100
                * f1_score(
                    gold, pred, labels=np.arange(len(labels)), average="macro", zero_division=0
                )
            ),
            **{
                f"{k}_f1_percent": float(
                    100 * f1_score(gold, pred, labels=v, average="macro", zero_division=0)
                )
                for k, v in tier.items()
            },
            "rescued_vs_max": int(np.sum((pred == gold) & (baseline != gold))),
            "broken_vs_max": int(np.sum((pred != gold) & (baseline == gold))),
            "rescued_vs_top3": int(np.sum((pred == gold) & (top3 != gold))),
            "broken_vs_top3": int(np.sum((pred != gold) & (top3 == gold))),
        }
    # Group bootstrap of paired accuracy difference, conditional on fitted models.
    unique, inverse = np.unique(cal["groups"], return_inverse=True)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(20261116)
    delta = np.bincount(
        inverse,
        weights=(preds["hgb_joint_profile"] == gold).astype(int) - (top3 == gold).astype(int),
    )
    boot = []
    for _ in range(2000):
        ix = rng.integers(0, len(unique), size=len(unique))
        boot.append(100 * delta[ix].sum() / sizes[ix].sum())
    result = {
        "scope": "Exploratory ordinary learned-pooling diagnostic, genuine 5-fold group-disjoint Fit OOF -> previously explored Cal",
        "novel_algorithm": False,
        "fit_rows": 30226,
        "cal_rows": 6477,
        "fixed_candidates": 16,
        "audit_dev_test_used": False,
        "candidate_parity_verified_all_folds": True,
        "cal_candidate_oracle_percent": float(
            100 * np.mean(np.any(cal["candidates"] == gold[:, None], axis=1))
        ),
        "group_definition": "transitive shared nonempty user_id or identical whitespace-normalized clinical input",
        "frequency_tiers": "original full official Train: 43 head, 94 mid, 11 tail; fixed across all methods",
        "models": learned,
        "metrics": metrics,
        "joint_vs_top3_cluster_bootstrap_accuracy_pp_95ci": np.quantile(
            boot, [0.025, 0.975]
        ).tolist(),
        "bootstrap_limit": "conditional on fitted models; no correction for research-level multiple exploration",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partition_sha256": hashlib.sha256(
            (ROOT / "data/cdet/partition_indices.json").read_bytes()
        ).hexdigest(),
        "train_sha256": hashlib.sha256((ROOT / "data/tcm-sd/train.json").read_bytes()).hexdigest(),
        "seconds": round(perf_counter() - t0, 2),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        CACHE / "cal_predictions.npz", truth=gold, query_ids=cal["query_ids"], **preds
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["0", "1", "2", "3", "4", "cal", "evaluate"])
    args = parser.parse_args()
    build(args.stage) if args.stage != "evaluate" else evaluate()
