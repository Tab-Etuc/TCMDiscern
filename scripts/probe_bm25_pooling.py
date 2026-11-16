"""Exploratory CPU probe: label-level BM25 pooling and query-null max correction.

Uses official TCM-SD Train Fit for indexing and pre-existing Cal candidates.
No held-out Audit or official Dev/Test, no gold input to scoring.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bm25_pooling_root_probe_2026-11-16.json"


def main() -> None:
    t0 = perf_counter()
    parts = json.loads((ROOT / "data/cdet/partition_indices.json").read_text(encoding="utf-8"))
    fit, cal = parts["fit"], parts["cal"]
    if (len(fit), len(cal)) != (30226, 6477):
        raise ValueError("Expected frozen grouped Fit/Cal")
    rows = load_records(ROOT / "data/tcm-sd/train.json")
    yfit = [str(rows[i]["norm_syndrome"]) for i in fit]
    truth = [str(rows[i]["norm_syndrome"]) for i in cal]
    support = Counter(yfit)
    print("[pooling] building original Fit BM25 index", flush=True)
    bm25 = CaseBM25([clinical_text(rows[i]) for i in fit], yfit)
    names = bm25.label_names
    label_index = {name: i for i, name in enumerate(names)}
    heldout_candidate_indices = []
    heldout_candidate_names = []
    with (ROOT / "data/cdet/cal_candidates.jsonl").open(encoding="utf-8") as inp:
        for i, line in zip(cal, inp, strict=True):
            row = json.loads(line)
            if row["case_id"] != f"train-{i:06d}":
                raise ValueError("Candidate alignment failure")
            choice = row["candidate_labels"][:16]
            heldout_candidate_names.append(choice)
            heldout_candidate_indices.append([label_index[c] for c in choice])
    shortlist = np.asarray(heldout_candidate_indices, dtype=np.int32)
    starts = bm25.group_starts
    ends = np.r_[starts[1:], len(fit)]
    counts = ends - starts
    names_array = np.asarray(names)
    methods = (
        "bm25_max", "mean_top3", "mean_top5", "logmeanexp_t4", "logmeanexp_t12",
        "null_sub_025", "null_sub_050", "null_sub_075", "null_sub_100",
    )
    predictions = {mode: [] for mode in methods}
    candidate_baseline = [labels[0] for labels in heldout_candidate_names]
    for batch_no, start in enumerate(range(0, len(cal), 24)):
        positions = cal[start:start+24]
        cand = shortlist[start:start+24]
        queries = bm25.vectorizer.transform([clinical_text(rows[i]) for i in positions])
        queries.data[:] = 1.0
        score = (queries @ bm25.weights_t).toarray()
        ordered = score[:, bm25.ordered_doc_ids]
        maxes = np.maximum.reduceat(ordered, starts, axis=1)
        top3 = np.empty_like(maxes)
        top5 = np.empty_like(maxes)
        lse4 = np.empty_like(maxes)
        lse12 = np.empty_like(maxes)
        for j, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
            subset = ordered[:, lo:hi]
            k3 = min(3, subset.shape[1])
            k5 = min(5, subset.shape[1])
            top3[:, j] = np.partition(subset, -k3, axis=1)[:, -k3:].mean(axis=1)
            top5[:, j] = np.partition(subset, -k5, axis=1)[:, -k5:].mean(axis=1)
            for t, output in ((4, lse4), (12, lse12)):
                m = maxes[:, j]
                output[:, j] = m + t * np.log(
                    np.exp((subset-m[:, None])/t).mean(axis=1)
                )
        # Query-conditioned background quantiles: analytical max-size correction
        background = np.sort(score, axis=1)
        idx = np.minimum(
            len(fit)-1,
            np.maximum(0, np.ceil((1 - 1/counts) * (len(fit)-1)).astype(int))
        )
        null_expected = background[:, idx]
        score_modes = {
            "bm25_max": maxes,
            "mean_top3": top3,
            "mean_top5": top5,
            "logmeanexp_t4": lse4,
            "logmeanexp_t12": lse12,
        }
        for strength in (25,50,75,100):
            score_modes[f"null_sub_{strength:03d}"] = maxes - (
                strength/100.0 * null_expected
            )
        for method, whole in score_modes.items():
            subset = np.take_along_axis(whole, cand, axis=1)
            best = subset.argmax(axis=1)
            predictions[method].extend([
                str(names_array[cand[i, int(k)]]) for i,k in enumerate(best)
            ])
        if batch_no % 70 == 0:
            print(f"[pooling] completed {min(start+24,len(cal))}/{len(cal)}", flush=True)
    if predictions["bm25_max"] != candidate_baseline:
        bad=sum(a!=b for a,b in zip(predictions["bm25_max"],candidate_baseline,strict=True))
        raise ValueError(f"Exact BM25 Top16 ranking parity failure: {bad}")
    labels = sorted(set(yfit))
    groups = {
        "tail": [c for c in labels if support[c]<10],
        "mid": [c for c in labels if 10<=support[c]<100],
        "head": [c for c in labels if support[c]>=100],
    }
    summary = {}
    for method, output in predictions.items():
        rescue=sum(p==y and b!=y for p,b,y in zip(output,candidate_baseline,truth,strict=True))
        breakage=sum(p!=y and b==y for p,b,y in zip(output,candidate_baseline,truth,strict=True))
        summary[method] = {
            "top1_percent":round(100*sum(p==y for p,y in zip(output,truth,strict=True))/len(cal),4),
            "macro_f1_percent":round(100*f1_score(truth,output,labels=labels,average="macro",zero_division=0),4),
            "rescued":rescue,"broken":breakage,
            **{
                f"{name}_macro_f1_percent":round(
                    100*f1_score(truth,output,labels=classes,average="macro",zero_division=0),4)
                for name,classes in groups.items()
            },
        }
        print(f"[pooling] {method:16}: {summary[method]}",flush=True)
    result={
        "scope":"Exploratory; original official TCM-SD Train Fit index -> grouped Cal, fixed BM25 Top16",
        "cal_previously_explored":True,
        "train":len(fit),"cal":len(cal),"labels":len(labels),
        "label_count_min_max":[int(counts.min()),int(counts.max())],
        "model":"same original BM25 IDF, k1, b, bigrams; only class pooling changed",
        "null_quantile":"query-specific empirical F^-1(1-1/n_c) over all Fit case BM25 scores",
        "variants":summary,"seconds":round(perf_counter()-t0,2),
        "dev_test_used":False,
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("[pooling] SAVED",OUT.relative_to(ROOT),flush=True)


if __name__=="__main__":
    main()
