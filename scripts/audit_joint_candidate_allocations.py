"""Dev-only fixed-16 candidate allocation study for long-tail TCM disease-syndrome pairs.

Compare direct BM25, D x S hierarchical pair slots, a disease-diverse hybrid,
and independent disease/syndrome BM25 cross-products. No Jeff GPU inference.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tcmdiscern.datasets import clinical_text, load_records
from tcmdiscern.disease_study import disease_name, pair_key, syndrome_name
from tcmdiscern.joint_candidates import (
    CANDIDATE_K,
    HIERARCHIES,
    crossed_pairs,
    hierarchical_pairs,
    hybrid_pairs,
)
from tcmdiscern.retrieval import CaseBM25

ROOT = Path(__file__).resolve().parents[1]


def aggregate(
    examples: list[dict], alternatives: list[list[str]], *,
    train_pair_counts: Counter,
) -> dict:
    if len(examples) != len(alternatives):
        raise ValueError("Must supply one full candidate set per official Dev row")
    gold = [pair_key(disease_name(row), syndrome_name(row)) for row in examples]
    if not all(len(c) == CANDIDATE_K and len(set(c)) == CANDIDATE_K for c in alternatives):
        raise ValueError("Every candidate allocation requires 16 unique options")
    recalled = [label in options for label, options in zip(gold, alternatives, strict=True)]
    correct_top1 = [label == options[0] for label, options in zip(gold, alternatives, strict=True)]
    seen = [train_pair_counts[label] > 0 for label in gold]
    tail = [train_pair_counts[label] <= 10 for label in gold]
    return {
        "n": len(gold),
        "top1_correct": sum(correct_top1),
        "top1": sum(correct_top1) / len(gold),
        "oracle16_correct": sum(recalled),
        "oracle16": sum(recalled) / len(gold),
        "seen_pair_oracle16": (
            sum(hit and is_seen for hit, is_seen in zip(recalled, seen, strict=True))
            / sum(seen)
        ),
        "tail_le10_oracle16": (
            sum(hit and is_tail for hit, is_tail in zip(recalled, tail, strict=True))
            / sum(tail)
        ),
        "pair_labels_not_seen_in_train": sum(not is_seen for is_seen in seen),
    }


def main() -> None:
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if len(train) != 43180 or len(dev) != 5486:
        raise ValueError("Official Train/Dev counts required")

    train_text = [clinical_text(row) for row in train]
    dev_text = [clinical_text(row) for row in dev]
    diseases = [disease_name(row) for row in train]
    syndromes = [syndrome_name(row) for row in train]
    pairs = [pair_key(d, s) for d, s in zip(diseases, syndromes, strict=True)]
    print("[RETRIEVAL] Build disease, syndrome and joint BM25 Train-only indexes", flush=True)
    d_bm25 = CaseBM25(train_text, diseases)
    s_bm25 = CaseBM25(train_text, syndromes)
    p_bm25 = CaseBM25(train_text, pairs)
    disease_rankings = d_bm25.rank(dev_text, max_k=16)
    syndrome_rankings = s_bm25.rank(dev_text, max_k=16)
    print("[RETRIEVAL] Rank entire paired-label index on official Dev", flush=True)
    all_pair_rankings = p_bm25.rank(dev_text, max_k=len(p_bm25.label_names))
    pair_counts = Counter(pairs)

    direct = [row[:CANDIDATE_K] for row in all_pair_rankings]
    variants = {"direct-pair-top16": direct}
    for d, s in HIERARCHIES:
        label = f"hierarchy-{d}d-{s}s"
        print(f"[CANDIDATES] {label}", flush=True)
        variants[label] = [
            hierarchical_pairs(dr, pr, disease_k=d, syndrome_k=s)
            for dr, pr in zip(disease_rankings, all_pair_rankings, strict=True)
        ]
    for flat, d, s in ((4, 4, 3), (8, 4, 2), (12, 4, 1)):
        label = f"hybrid-{flat}flat-{d}d-{s}s"
        print(f"[CANDIDATES] {label}", flush=True)
        variants[label] = [
            hybrid_pairs(dr, pr, flat_keep=flat, disease_k=d, syndrome_k=s)
            for dr, pr in zip(disease_rankings, all_pair_rankings, strict=True)
        ]
    for flat in (0, 8):
        label = f"crossed-{flat}flat-{16-flat}from-marginals"
        print(f"[CANDIDATES] {label}", flush=True)
        variants[label] = [
            crossed_pairs(dr, sr, pr, allowed_pairs=frozenset(pair_counts), flat_keep=flat)
            for dr, sr, pr in zip(
                disease_rankings, syndrome_rankings, all_pair_rankings, strict=True
            )
        ]

    report = {
        "dataset": "TCM-SD",
        "split": "official Dev 5486 rows",
        "training_only_retriever": True,
        "model_predictions_measured": False,
        "fixed_candidates": CANDIDATE_K,
        "train_pair_classes": len(pair_counts),
        "variants": {
            label: aggregate(dev, candidates, train_pair_counts=pair_counts)
            for label, candidates in variants.items()
        },
    }
    path = ROOT / "experiments/tcmsd_joint_candidate_allocations_dev.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
