"""K=16 single-row Chinese disease and disease-syndrome pair tasks for Jeff.

All labels are drawn from official Train; inference never injects a gold label.
Pairs are encoded as unambiguous JSON arrays in the model's option id and are
shown to Jeff as separate disease/syndrome names in option descriptions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .disease_study import disease_name, pair_key, pair_parts, syndrome_name
from .jeff_format import LETTERS, decision_row

K = 16
MODES = ("disease", "pair")
QUESTION = {
    "disease": "根據患者主訴、現病史與四診資訊，選出最符合的中醫病名。",
    "pair": "根據患者主訴、現病史與四診資訊，選出最符合的中醫病名與證型組合。",
}


def gold_label(record: Mapping[str, Any], mode: str) -> str:
    if mode == "disease":
        return disease_name(record)
    if mode == "pair":
        return pair_key(disease_name(record), syndrome_name(record))
    raise ValueError("Unknown K16 label space")


def choice_prompt(
    case: Mapping[str, Any], *, case_id: str, labels: Sequence[str], mode: str
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("Unknown K16 label space")
    if len(labels) != K or len(set(labels)) != K:
        raise ValueError("Jeff joint study requires exactly sixteen unique labels")
    row = decision_row(case, case_id=case_id, candidates=labels)
    row["question"] = QUESTION[mode]
    if mode == "pair":
        row["options"] = [
            {
                "id": label,
                "description": f"中醫病名：{d}；證型：{s}",
            }
            for label in labels
            for d, s in (pair_parts(label),)
        ]
    return row


def train_example(
    case: Mapping[str, Any], *, index: int, oof_labels: Sequence[str], mode: str
) -> tuple[dict, bool]:
    if mode not in MODES or len(set(oof_labels)) != len(oof_labels):
        raise ValueError("Invalid OOF candidate source")
    gold = gold_label(case, mode)
    retrieved = gold in oof_labels[:K]
    negatives = [name for name in oof_labels if name != gold]
    if len(negatives) < K - 1:
        raise ValueError("Not enough unique train-only OOF negatives")
    candidates = [gold, *negatives[: K - 1]]
    Random(20261113 + index * 131 + K * 10007).shuffle(candidates)
    row = choice_prompt(
        case, case_id=f"train-{index:06d}", labels=candidates, mode=mode,
    )
    idx = candidates.index(gold)
    item = {"row": row, "target_index": idx, "target_letter": LETTERS[idx]}
    return item, retrieved


def eval_example(
    case: Mapping[str, Any], *, index: int, bm25_labels: Sequence[str],
    mode: str, split: str,
) -> tuple[dict, dict]:
    if split not in ("dev", "test"):
        raise ValueError("Only official Dev/Test are allowed")
    row = choice_prompt(
        case, case_id=f"{split}-{index:06d}", labels=list(bm25_labels[:K]), mode=mode
    )
    meta = {
        "case_id": row["id"],
        "gold": gold_label(case, mode),
        "candidates": list(bm25_labels[:K]),
    }
    return row, meta
