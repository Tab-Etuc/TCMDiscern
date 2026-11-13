"""Disease-conditioned TCM-SD Jeff study: fixed row-level 5-choice tasks.

Train builds all priors from TRAIN only. Dev/Test gold is only used by the scorer.
The official split and every original row are kept intact.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .jeff_format import LETTERS, decision_row

K = 5
SEED = 20261113
MODES = ("global", "conditioned", "pair")
CONDITIONED_QUESTION = "已知中醫病名，根據患者主訴、病史及四診，選出該病名下最符合的證型。"
PAIR_QUESTION = "根據患者主訴、病史及四診，選出最符合的中醫病名與證型配對。"


def disease_name(case: Mapping[str, Any]) -> str:
    name = str(case.get("lcd_name") or "").strip()
    if not name:
        raise ValueError("TCM-SD row has no lcd_name")
    return name


def syndrome_name(case: Mapping[str, Any]) -> str:
    name = str(case.get("norm_syndrome") or "").strip()
    if not name:
        raise ValueError("TCM-SD row has no norm_syndrome")
    return name


def pair_key(disease: str, syndrome: str) -> str:
    if not disease or not syndrome:
        raise ValueError("Both disease and syndrome must be nonempty")
    return json.dumps([disease, syndrome], ensure_ascii=False, separators=(",", ":"))


def pair_parts(key: str) -> tuple[str, str]:
    value = json.loads(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("Invalid disease-syndrome pair key")
    return value[0], value[1]


def gold_for(case: Mapping[str, Any], mode: str) -> str:
    if mode not in MODES:
        raise ValueError("Invalid study mode")
    syndrome = syndrome_name(case)
    return pair_key(disease_name(case), syndrome) if mode == "pair" else syndrome


def allowed_syndromes(cases: Sequence[dict]) -> dict[str, frozenset[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        result[disease_name(case)].add(syndrome_name(case))
    return {name: frozenset(labels) for name, labels in result.items()}


def prioritize_disease(
    ranked: Sequence[str], *, disease: str, allow: Mapping[str, frozenset[str]]
) -> list[str]:
    """Place TRAIN-observed disease syndromes first; backfill the rest to fixed K."""
    if len(set(ranked)) != len(ranked):
        raise ValueError("BM25 rankings contain duplicate syndrome IDs")
    allowed = allow.get(disease, frozenset())
    return [label for label in ranked if label in allowed] + [
        label for label in ranked if label not in allowed
    ]


def study_decision(
    case: Mapping[str, Any], *, case_id: str, ranked: Sequence[str], mode: str
) -> dict[str, Any]:
    """Only declared patient evidence and (for conditioned task) supplied disease."""
    if mode not in MODES:
        raise ValueError("Invalid study mode")
    if len(ranked) != K or len(set(ranked)) != K:
        raise ValueError("Study requires five distinct alternatives")
    row = decision_row(case, case_id=case_id, candidates=ranked)
    if mode == "conditioned":
        row["state"]["lcd_name"] = disease_name(case)
        row["question"] = CONDITIONED_QUESTION
    elif mode == "pair":
        row["question"] = PAIR_QUESTION
        row["options"] = [
            {"id": label, "description": "中醫病名：" + d + "；證型：" + s}
            for label in ranked
            for d, s in (pair_parts(label),)
        ]
    return row


def train_item(
    case: Mapping[str, Any], *, index: int, ranked: Sequence[str],
    mode: str, seed: int = SEED, allow: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Always include the training gold; mine the other four only from its OOF index."""
    gold = gold_for(case, mode)
    ordered = [label for label in ranked if label != gold]
    if len(ordered) < K - 1 or len(set(ordered)) != len(ordered):
        raise ValueError("Insufficient unique OOF negative labels")
    selected = [gold, *ordered[: K - 1]]
    Random(seed + index * 131 + K * 10_007).shuffle(selected)
    row = study_decision(case, case_id=f"train-{index:06d}", ranked=selected, mode=mode)
    gold_index = selected.index(gold)
    result = {"row": row, "target_index": gold_index, "target_letter": LETTERS[gold_index]}
    if mode == "conditioned":
        if allow is None:
            raise ValueError("Conditioned examples need a fold-specific disease allowlist")
        in_disease = allow.get(disease_name(case), frozenset())
        # The first two BM25-ranked same-disease negatives form the hard-pair loss.
        negatives = [label for label in ordered if label in in_disease]
        chosen = [name for name in negatives if name in selected][:2]
        result["hard_negative_indices"] = [selected.index(name) for name in chosen]
    return result


def eval_item(
    case: Mapping[str, Any], *, index: int, ranked: Sequence[str], mode: str,
    split: str,
) -> tuple[dict, dict]:
    if split not in ("dev", "test"):
        raise ValueError("Only official Dev and Test may be evaluated")
    if len(ranked) < K:
        raise ValueError("Ranked alternatives shorter than fixed K")
    options = list(ranked[:K])
    row = study_decision(case, case_id=f"{split}-{index:06d}", ranked=options, mode=mode)
    # Gold and candidate labels live outside model-visible SemIf input.
    expected = {"case_id": row["id"], "gold": gold_for(case, mode), "candidates": options}
    return row, expected


def disease_only_predictions(
    train: Sequence[dict], cases: Sequence[dict]
) -> list[str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    prior: Counter = Counter()
    for record in train:
        disease, syndrome = disease_name(record), syndrome_name(record)
        counts[disease][syndrome] += 1
        prior[syndrome] += 1
    default = min(prior, key=lambda s: (-prior[s], s))
    winners = {
        disease: min(freq, key=lambda s: (-freq[s], s))
        for disease, freq in counts.items()
    }
    return [winners.get(disease_name(row), default) for row in cases]


def score_predictions(
    records: Sequence[dict], source: Sequence[dict], predictions: Sequence[str],
    *, mode: str, train_labels: Sequence[str],
) -> dict:
    """Report original row-wise Top-1 and Macro-F1; pairs have an extra disease score."""
    from sklearn.metrics import accuracy_score, f1_score

    if not (len(records) == len(source) == len(predictions)):
        raise ValueError("Predictions, records and metadata have different row counts")
    if mode not in MODES:
        raise ValueError("Invalid study mode")
    gold = [gold_for(row, mode) for row in records]
    all_classes = sorted(set(train_labels) | set(gold))
    for i, (metadata, predicted) in enumerate(zip(source, predictions, strict=True)):
        if metadata["case_id"] != f"{metadata['case_id'].split('-')[0]}-{i:06d}":
            raise ValueError("Case IDs are out of order")
        if metadata["gold"] != gold[i]:
            raise ValueError("Metadata gold disagrees with original official row")
        if predicted not in metadata["candidates"]:
            raise ValueError("Jeff selected a label outside five BM25 candidates")
    score = {
        "n": len(gold),
        "top1": float(accuracy_score(gold, predictions)),
        "macro_f1": float(f1_score(gold, predictions, labels=all_classes, average="macro",
                                  zero_division=0)),
        "oracle_at_5": sum(
            label in item["candidates"] for label, item in zip(gold, source, strict=True)
        ) / len(gold),
        "classes": len(all_classes),
    }
    if mode == "pair":
        gold_pairs = [pair_parts(p) for p in gold]
        predicted_pairs = [pair_parts(p) for p in predictions]
        score.update(
            disease_top1=float(accuracy_score(
                [p[0] for p in gold_pairs], [p[0] for p in predicted_pairs]
            )),
            syndrome_top1=float(accuracy_score(
                [p[1] for p in gold_pairs], [p[1] for p in predicted_pairs]
            )),
        )
    return score



def study_torch_loss(logits: Any, item: dict, *, pairwise_weight: float, margin: float):
    """Five-option Jeff CE plus same-disease hardest-negative margin objective."""
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() != K:
        raise ValueError("Disease study expects exactly five decision logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Decision logits must be finite")
    gold = item["target_index"]
    if not isinstance(gold, int) or not 0 <= gold < K:
        raise ValueError("Invalid supervised gold position")
    values = logits.float()
    ce = torch.logsumexp(values, dim=0) - values[gold]
    if pairwise_weight == 0:
        return ce
    if pairwise_weight < 0 or margin < 0:
        raise ValueError("Invalid pairwise objective coefficients")
    negative_positions = item.get("hard_negative_indices", [])
    if not negative_positions:
        return ce
    if (
        len(set(negative_positions)) != len(negative_positions)
        or any(not isinstance(i, int) or i == gold or not 0 <= i < K
               for i in negative_positions)
    ):
        raise ValueError("Invalid same-disease hard-negative positions")
    hard = values[negative_positions]
    return ce + pairwise_weight * torch.nn.functional.softplus(
        torch.logsumexp(hard, dim=0) - values[gold] + margin
    )

def baseline_scores(
    train: Sequence[dict], cases: Sequence[dict], sources: Mapping[str, Sequence[dict]]
) -> dict:
    report = {}
    for mode in MODES:
        rows = sources[mode]
        pred = [item["candidates"][0] for item in rows]
        report[mode + "_bm25"] = score_predictions(
            cases, rows, pred, mode=mode,
            train_labels=[gold_for(row, mode) for row in train],
        )
    default_source = sources["conditioned"]
    disease_preds = disease_only_predictions(train, cases)
    # The disease-only baseline predicts directly from the 148-class syndrome space,
    # so it may select a label outside the BM25 Top-5. Score independently.
    from sklearn.metrics import accuracy_score, f1_score
    truth = [syndrome_name(row) for row in cases]
    labels = sorted({syndrome_name(row) for row in train} | set(truth))
    report["disease_only"] = {
        "n": len(cases),
        "top1": float(accuracy_score(truth, disease_preds)),
        "macro_f1": float(f1_score(truth, disease_preds, labels=labels,
                                  average="macro", zero_division=0)),
        "classes": len(labels),
        "uses_bm25_candidates": False,
    }
    assert len(default_source) == len(cases)
    return report
