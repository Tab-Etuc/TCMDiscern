"""Compute Train-only factorized 16x16 pair candidate upper bounds on official Dev.

Uses already produced disease and syndrome BM25 Dev candidates, no GPU or
new BM25 indexing and never inserts gold into a candidate set.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.fuse_joint16_logits import score_fusion
from tcmdiscern.datasets import load_records
from tcmdiscern.disease_study import disease_name, pair_key, syndrome_name
from tcmdiscern.joint_fusion import decode, factorized_oracles

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    disease_file = ROOT / "data/joint16/dev_disease_labels.jsonl"
    syndrome_file = ROOT / "data/candidates/bm25_dev.jsonl"
    diseases = [json.loads(line) for line in disease_file.read_text(encoding="utf-8").splitlines()]
    syndromes = [json.loads(line) for line in syndrome_file.read_text(encoding="utf-8").splitlines()]
    train = load_records(ROOT / "data/tcm-sd/train.json")
    dev = load_records(ROOT / "data/tcm-sd/dev.json")
    if not (len(diseases) == len(syndromes) == len(dev) == 5486):
        raise ValueError("All official Dev rows required")
    for i, (d, s, case) in enumerate(zip(diseases, syndromes, dev, strict=True)):
        if (
            d["case_id"] != f"dev-{i:06d}"
            or s["case_id"] != f"dev-{i:06d}"
            or len(d["candidates"]) != 16
            or len(s["candidate_labels"]) < 16
            or d["gold"] != case["lcd_name"]
            or s["gold_labels"] != [case["norm_syndrome"]]
        ):
            raise ValueError(f"Corrupt disease / syndrome original BM25 row {i}")
    gold = [pair_key(disease_name(case), syndrome_name(case)) for case in dev]
    known = frozenset(pair_key(disease_name(case), syndrome_name(case)) for case in train)
    result = factorized_oracles(
        gold,
        [item["candidates"] for item in diseases],
        [item["candidate_labels"][:16] for item in syndromes],
        known_pairs=known,
    )
    # Rank-only marginal decisions are separate CPU baselines, never Jeff scores.
    independent = []
    constrained = []
    constrained_fallbacks = 0
    for d, s in zip(diseases, syndromes, strict=True):
        d_labels = d["candidates"]
        s_labels = s["candidate_labels"][:16]
        rank_scores = [-float(i) for i in range(16)]
        raw, _ = decode(
            d_labels, s_labels, rank_scores, rank_scores, disease_weight=0.5
        )
        restricted, fallback = decode(
            d_labels, s_labels, rank_scores, rank_scores, known_pairs=known,
            disease_weight=0.5,
        )
        independent.append(raw)
        constrained.append(restricted)
        constrained_fallbacks += fallback
    result["bm25_factorized_baselines"] = {
        "unconstrained": score_fusion(
            train, dev, independent, fallback_count=0, constrained=False,
        ),
        "train_constrained": score_fusion(
            train, dev, constrained, fallback_count=constrained_fallbacks,
            constrained=True,
        ),
    }
    result["status"] = "FACTORIZED_BM25_DEV_ORACLE_AND_RANK_ONLY_BASELINES"
    result["gold_in_model_prompts"] = False
    path = ROOT / "experiments/joint16_factorized_oracle_dev.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
