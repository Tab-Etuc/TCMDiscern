"""S1/S2 ensemble exploration on frozen official TCM-SD Dev logits.

Inputs: no Test; fixed S1/S2 16-slot logits and original BM25 Top16.
Fixed policies use no Dev labels. Exploratory rule search uses 5-fold
StratifiedGroupKFold on Dev; these results cannot be claimed independent
validation because Dev was already used to design the policy families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
S1 = ROOT / "data/jeff/dev_s1-seed20261112-e2-epoch2_k16_5486.jsonl"
S2 = ROOT / "data/jeff/dev_s2-seed20261112-e2-epoch2_k16_5486.jsonl"
OUTPUT = ROOT / "experiments/s1_s2_cpu/summary.json"
SEED = 20261115
N_CLASSES = 148


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def clinical_signature(row):
    text = "\n".join(
        str(row.get(k) or "").strip() for k in ("chief_complaint", "description", "detection")
    )
    return hashlib.sha256(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def macro_f1(gold, pred, labels):
    """Fixed-universe macro-F1 without silently excluding missing tail labels."""
    labs = np.asarray(labels, dtype=int)
    if len(labs) == 0:
        return 0.0
    c = int(np.max(labs)) + 1
    c = max(c, N_CLASSES)
    support = np.bincount(gold, minlength=c)
    pred_support = np.bincount(pred, minlength=c)
    tp = np.bincount(gold[gold == pred], minlength=c)
    return float(
        np.divide(
            2 * tp,
            support + pred_support,
            out=np.zeros(c, dtype=float),
            where=(support + pred_support) > 0,
        )[labs].mean()
    )


def metrics(gold, pred, counts, subset=None):
    if subset is not None:
        gold = gold[subset]
        pred = pred[subset]
    cls = np.arange(N_CLASSES)
    heads = cls[counts >= 100]
    mids = cls[(counts >= 10) & (counts < 100)]
    tails = cls[counts < 10]
    return {
        "n": len(gold),
        "correct": int(np.sum(gold == pred)),
        "top1": float(np.mean(gold == pred)),
        "macro_f1": macro_f1(gold, pred, cls),
        "head_f1": macro_f1(gold, pred, heads),
        "mid_f1": macro_f1(gold, pred, mids),
        "tail_f1": macro_f1(gold, pred, tails),
    }


def choose(scores, k, threshold):
    n = len(scores)
    pos = np.argmax(scores[:, :k], axis=1)
    delta = scores[np.arange(n), pos] - scores[:, 0]
    return np.where((pos > 0) & (delta >= threshold), pos, 0)


def prepare(s1_path, s2_path):
    train = read_jsonl(ROOT / "data/tcm-sd/train.json")
    dev = read_jsonl(ROOT / "data/tcm-sd/dev.json")
    retrieval = read_jsonl(ROOT / "data/candidates/bm25_dev.jsonl")
    files = [read_jsonl(s1_path), read_jsonl(s2_path)]
    n = 5486
    if not (
        len(train) == 43180 and len(dev) == len(retrieval) == n and all(len(f) == n for f in files)
    ):
        raise ValueError("Incomplete TCM-SD Train/Dev or Jeff logits")
    names = sorted({row["norm_syndrome"] for row in train})
    label = {s: i for i, s in enumerate(names)}
    if len(names) != N_CLASSES:
        raise ValueError("Need 148 class labels")
    for i in range(n):
        cid = f"dev-{i:06d}"
        if (
            retrieval[i]["case_id"] != cid
            or any(f[i]["case_id"] != cid for f in files)
            or len(retrieval[i]["gold_labels"]) != 1
            or len(retrieval[i]["candidate_labels"]) < 16
        ):
            raise ValueError(f"Source case mismatch at {cid}")
    counts = Counter(row["norm_syndrome"] for row in train)
    cls_cnt = np.array([counts[name] for name in names], dtype=int)
    gold = np.array([label[row["gold_labels"][0]] for row in retrieval], dtype=int)
    cand = np.array(
        [[label[name] for name in row["candidate_labels"][:16]] for row in retrieval], dtype=int
    )
    z1, z2 = (np.array([row["logits"] for row in file], dtype=np.float64) for file in files)
    if (
        z1.shape != (n, 16)
        or z2.shape != (n, 16)
        or not np.isfinite(z1).all()
        or not np.isfinite(z2).all()
    ):
        raise ValueError("Non-finite or incorrectly sized Jeff logits")
    group = np.array([clinical_signature(row) for row in dev])
    train_group = {clinical_signature(row) for row in train}
    no_train_overlap = np.flatnonzero(~np.isin(group, list(train_group)))
    return gold, cand, z1, z2, cls_cnt, group, no_train_overlap


def create_methods(z1, z2, freq):
    n = len(z1)
    ix = np.arange(n)
    ap1 = z1 - 0.6 * freq
    ap2 = z2 - 0.6 * freq
    p1 = choose(ap1, 10, 3.0)
    p2 = choose(ap2, 10, 3.0)
    raw1 = np.argmax(z1, axis=1)
    raw2 = np.argmax(z2, axis=1)
    prior1 = np.argmax(ap1[:, :10], axis=1)
    prior2 = np.argmax(ap2[:, :10], axis=1)
    methods = {
        "bm25": np.zeros(n, dtype=int),
        "s1_raw": raw1,
        "s2_raw": raw2,
        "s1_fixed_prior_gate": p1,
        "s2_fixed_prior_gate": p2,
        "s1_prior_direct": prior1,
        "s2_prior_direct": prior2,
        "both_gate_agree": np.where(p1 == p2, p1, 0),
        "gate_agree_or_s1": np.where((p1 == p2) | (p2 == 0), p1, 0),
        "gate_agree_or_s2": np.where((p1 == p2) | (p1 == 0), p2, 0),
    }
    # S1 and S2 can independently check whether the other's correction
    # is more plausible than the anchored BM25 prediction.
    diff1 = ap1 - ap1[:, [0]]
    diff2 = ap2 - ap2[:, [0]]
    for t in (0.0, 1.0, 2.0, 3.0):
        tag = str(int(t))
        methods["s1_guard_s2_" + tag] = np.where((p1 > 0) & (diff2[ix, p1] >= t), p1, 0)
        methods["s2_guard_s1_" + tag] = np.where((p2 > 0) & (diff1[ix, p2] >= t), p2, 0)
        both = np.where((p1 > 0) & (diff2[ix, p1] >= t), p1, 0)
        s2_only = (p2 > 0) & (diff1[ix, p2] >= t) & (both == 0)
        methods["agree_or_union_" + tag] = np.where(s2_only, p2, both)
    # Prior correction on the average logits is mathematically equivalent to
    # averaging the independently prior-adjusted logits.
    for w in (0.25, 0.5, 0.75):
        for tau in (0.0, 0.3, 0.6, 0.9):
            blended = w * z1 + (1 - w) * z2 - tau * freq
            for k in (5, 10, 16):
                for d in (2.0, 3.0, 4.0, 5.0):
                    name = f"blend_w{w:g}_tau{tau:g}_k{k}_d{d:g}"
                    methods[name] = choose(blended, k, d)
    # Optional conservative consensus: min of improvements over the
    # original BM25 top-1 candidate, after prior correction.
    for k in (5, 10, 16):
        for t in (0.0, 1.0, 2.0, 3.0):
            minimum = np.minimum(diff1, diff2)
            methods[f"min_evidence_k{k}_d{t:g}"] = choose(minimum, k, t)
    for arr in methods.values():
        if arr.shape != (n,) or np.any(arr < 0) or np.any(arr >= 16):
            raise ValueError("Invalid candidate index")
    return methods


def fold_choose(methods, gold, cand, counts, groups):
    n = len(gold)
    gold_counts = counts[gold]
    strata = np.where(gold_counts < 10, 0, np.where(gold_counts < 100, 1, 2))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.arange(n), strata, groups))
    for fit, val in folds:
        if set(groups[fit]) & set(groups[val]):
            raise AssertionError("Fold groups overlap")
    policies = {}
    for policy in ("macro_guard", "joint_guard", "accuracy_guard"):
        pool = np.full(n, -1, dtype=int)
        fold_report = []
        for k, (fit, val) in enumerate(folds, 1):
            base1 = metrics(gold[fit], cand[fit, methods["s1_fixed_prior_gate"][fit]], counts)
            base2 = metrics(gold[fit], cand[fit, methods["s2_fixed_prior_gate"][fit]], counts)
            options = []
            for name, pos in methods.items():
                pred = cand[fit, pos[fit]]
                m = metrics(gold[fit], pred, counts)
                if policy == "macro_guard":
                    if m["top1"] < base1["top1"] - 0.001 or m["macro_f1"] < base1["macro_f1"]:
                        continue
                    score = (m["macro_f1"] - base1["macro_f1"]) + 0.25 * (m["top1"] - base1["top1"])
                elif policy == "joint_guard":
                    if m["top1"] < max(base1["top1"], base2["top1"]) - 0.001:
                        continue
                    if m["macro_f1"] < base1["macro_f1"] - 0.003:
                        continue
                    score = (m["top1"] - base1["top1"]) + 0.5 * (m["macro_f1"] - base1["macro_f1"])
                else:
                    if m["top1"] < max(base1["top1"], base2["top1"]) - 0.001:
                        continue
                    if m["macro_f1"] < base1["macro_f1"] - 0.015:
                        continue
                    score = (m["top1"] - base1["top1"]) + 0.1 * (m["macro_f1"] - base1["macro_f1"])
                options.append((score, m["macro_f1"], m["top1"], name))
            selected = max(options)[3] if options else "s1_fixed_prior_gate"
            pool[val] = methods[selected][val]
            fold_report.append(
                {
                    "fold": k,
                    "fit": len(fit),
                    "validation": len(val),
                    "policy": selected,
                    "n_candidates_meeting_constraints": len(options),
                }
            )
        if (pool < 0).any():
            raise ValueError("Incomplete cross-fit")
        policies[policy] = {"prediction": pool, "selection": fold_report}
    return folds, policies


def paired_bootstrap(gold, first, second, counts, groups, replicates=1500):
    """Cluster-paired fixed prediction resampling; describes Dev uncertainty."""
    _, gids = np.unique(groups, return_inverse=True)
    n_groups = int(gids.max() + 1)
    rng = np.random.default_rng(SEED)
    changes = np.empty((replicates, 2), dtype=np.float64)
    for i in range(replicates):
        selected = rng.integers(0, n_groups, size=n_groups)
        weight = np.bincount(selected, minlength=n_groups)[gids]
        total = float(weight.sum())
        c1 = np.sum(weight * (first == gold)) / total
        c2 = np.sum(weight * (second == gold)) / total
        support = np.bincount(gold, weights=weight, minlength=N_CLASSES)

        def mac(pred, weight=weight, support=support):
            pp = np.bincount(pred, weights=weight, minlength=N_CLASSES)
            yes = pred == gold
            tp = np.bincount(gold[yes], weights=weight[yes], minlength=N_CLASSES)
            return float(
                np.divide(
                    2 * tp, support + pp, out=np.zeros(N_CLASSES), where=(support + pp) > 0
                ).mean()
            )

        changes[i] = [c2 - c1, mac(second) - mac(first)]
    return {
        "top1_delta_pp": (100 * np.quantile(changes[:, 0], [0.025, 0.975])).tolist(),
        "macro_f1_delta_pp": (100 * np.quantile(changes[:, 1], [0.025, 0.975])).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, default=S1)
    parser.add_argument("--s2", type=Path, default=S2)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    gold, cand, z1, z2, counts, group, unseen = prepare(args.s1, args.s2)
    n = len(gold)
    ix = np.arange(n)
    models = create_methods(z1, z2, np.log(counts[cand]))
    results = {}
    for name, pos in models.items():
        pred = cand[ix, pos]
        s = metrics(gold, pred, counts)
        s["rescue_vs_bm25"] = int(np.sum((cand[:, 0] != gold) & (pred == gold)))
        s["break_vs_bm25"] = int(np.sum((cand[:, 0] == gold) & (pred != gold)))
        results[name] = s
    raw1 = cand[ix, models["s1_raw"]]
    raw2 = cand[ix, models["s2_raw"]]
    anchor = cand[:, 0]
    tuned1 = cand[ix, models["s1_fixed_prior_gate"]]
    tuned2 = cand[ix, models["s2_fixed_prior_gate"]]
    prior1 = cand[ix, models["s1_prior_direct"]]
    prior2 = cand[ix, models["s2_prior_direct"]]
    oracle2 = (anchor == gold) | (raw1 == gold)
    oracle3 = oracle2 | (raw2 == gold)
    oracle_all = oracle3 | (prior1 == gold) | (prior2 == gold)
    _fold, policies = fold_choose(models, gold, cand, counts, group)
    for name, value in policies.items():
        pred = cand[ix, value["prediction"]]
        value["metrics"] = metrics(gold, pred, counts)
        value["no_exact_train_text"] = metrics(gold, pred, counts, unseen)
        value["metrics"]["rescue_vs_bm25"] = int(np.sum((anchor != gold) & (pred == gold)))
        value["metrics"]["break_vs_bm25"] = int(np.sum((anchor == gold) & (pred != gold)))
        del value["prediction"]
    rank_macro = sorted(
        results.items(), key=lambda x: (x[1]["macro_f1"], x[1]["top1"]), reverse=True
    )
    rank_acc = sorted(results.items(), key=lambda x: (x[1]["top1"], x[1]["macro_f1"]), reverse=True)
    baseline_names = [
        "bm25",
        "s1_raw",
        "s2_raw",
        "s1_fixed_prior_gate",
        "s2_fixed_prior_gate",
        "both_gate_agree",
        "gate_agree_or_s1",
        "gate_agree_or_s2",
        "s1_guard_s2_0",
        "s2_guard_s1_0",
        "min_evidence_k10_d2",
        "blend_w0.5_tau0.6_k10_d3",
    ]
    for k in range(1, 4):
        top = rank_macro[k][0]
        if top not in baseline_names:
            baseline_names.append(top)
    chosen_names = sorted(
        set(baseline_names + [x[0] for x in rank_macro[:8]] + [x[0] for x in rank_acc[:8]])
    )
    paired = {}
    for name in chosen_names:
        if name == "s1_fixed_prior_gate":
            continue
        pred = cand[ix, models[name]]
        paired[name] = paired_bootstrap(gold, tuned1, pred, counts, group, replicates=1200)
    report = {
        "status": "exploratory S1+S2 CPU fusion; official Test unaccessed",
        "source_paths": {"s1": str(args.s1.resolve()), "s2": str(args.s2.resolve())},
        "data_integrity": {
            "n_dev": n,
            "n_classes": N_CLASSES,
            "train_exact_duplicate_rows_in_dev": int(n - len(unseen)),
            "dev_unique_groups": len(set(group)),
            "class_head_mid_tail": [
                int(np.sum(counts >= 100)),
                int(np.sum((counts >= 10) & (counts < 100))),
                int(np.sum(counts < 10)),
            ],
        },
        "oracle": {
            "bm25_plus_s1_raw": {"n": int(oracle2.sum()), "top1": float(oracle2.mean())},
            "bm25_plus_s1_s2_raw": {"n": int(oracle3.sum()), "top1": float(oracle3.mean())},
            "bm25_plus_s1_s2_raw_plus_two_prior": {
                "n": int(oracle_all.sum()),
                "top1": float(oracle_all.mean()),
            },
            "s2_unique_rescue_over_bm25_and_s1": int(
                np.sum((anchor != gold) & (raw1 != gold) & (raw2 == gold))
            ),
            "s1_unique_rescue_over_bm25_and_s2": int(
                np.sum((anchor != gold) & (raw2 != gold) & (raw1 == gold))
            ),
        },
        "s1_s2_raw_agreement": float(np.mean(raw1 == raw2)),
        "s1_s2_fixed_gate_agreement": float(np.mean(tuned1 == tuned2)),
        "fixed_or_reference": {
            name: {
                "overall": results[name],
                "no_exact_train_text": metrics(gold, cand[ix, models[name]], counts, unseen),
            }
            for name in baseline_names
        },
        "exploratory_full_dev_top_macro": [{"method": name, **s} for name, s in rank_macro[:15]],
        "exploratory_full_dev_top_accuracy": [{"method": name, **s} for name, s in rank_acc[:15]],
        "crossfit_dev_rule_selection": policies,
        "paired_cluster_bootstrap_95_vs_s1_gate": paired,
        "limitations": [
            "Strong baseline S1 prior/gate parameters were already explored on this Dev.",
            "Full-Dev best rule is optimistic and cannot be reported as independent accuracy.",
            "CV selection estimates are on previously inspected Dev; group folds avoid exact-text overlap within selector fitting, not overall study-level exploration bias.",
            "S1/S2 model weights were fitted on official Train; no per-row Train out-of-fold S1/S2 logits are currently available for an independently trained gate.",
            "Macro-F1 is across fixed 148 Train classes, including labels absent from a single fold.",
            "Official Test was not opened.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in baseline_names:
        s = results[name]
        print(
            f"{name:36s} Top1 {s['top1']:.4%} Macro {s['macro_f1']:.4%} "
            f"Tail {s['tail_f1']:.4%} rescue/break {s['rescue_vs_bm25']}/{s['break_vs_bm25']}"
        )
    print("\nTop Macro exploratory")
    for name, s in rank_macro[:10]:
        print(f"{name:36s} {s['top1']:.4%} {s['macro_f1']:.4%} {s['tail_f1']:.4%}")
    print("\nTop Accuracy exploratory")
    for name, s in rank_acc[:10]:
        print(f"{name:36s} {s['top1']:.4%} {s['macro_f1']:.4%} {s['tail_f1']:.4%}")
    print("\nFold CV")
    for name, pol in policies.items():
        m = pol["metrics"]
        print(
            name,
            ":",
            f"{m['top1']:.4%}",
            f"{m['macro_f1']:.4%}",
            [x["policy"] for x in pol["selection"]],
        )
    print("ORACLE", report["oracle"])
    print("saved", args.out)


if __name__ == "__main__":
    main()
