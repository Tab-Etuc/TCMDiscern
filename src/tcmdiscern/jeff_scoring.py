"""Jeff K16 decision-logit scoring against the official single-gold Dev labels.

Real model inference must export (case_id, raw 16 logits) to ignored data/.
The scorer NEVER adds gold labels to candidates, renumbers options or uses
gold labels to construct model input. Baseline dry run uses fixed rank-1 logits.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .evaluation import Prediction


def score_k(
    rows: Sequence[Prediction],
    raw_logits: Sequence[Sequence[float]] | np.ndarray,
    *,
    k: int,
    all_classes: Sequence[str],
) -> dict:
    """Score fixed BM25 candidate sets, including grouped Jeff decisions above K=16."""
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 35:
        raise ValueError("K must be an integer from 1 to 35")
    if not rows:
        raise ValueError("Need all official Dev rows")
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.shape != (len(rows), k) or not np.isfinite(logits).all():
        raise ValueError(f"Requires one finite {k}-slot decision score vector per Dev row")
    if len({row.case_id for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs in model predictions")
    if len(set(all_classes)) != 148:
        raise ValueError("Must evaluate all 148 original TCM-SD syndrome classes")

    gold = []
    pred = []
    bm25 = []
    covered = []
    for i, row in enumerate(rows):
        if row.case_id != f"dev-{i:06d}":
            raise ValueError(f"Dev row {i} is out of official order")
        if len(row.gold_labels) != 1 or len(row.candidate_labels) < k:
            raise ValueError(f"Requires original single gold and {k} BM25 candidates")
        if len(set(row.candidate_labels[:k])) != k:
            raise ValueError("Duplicate decision candidates")
        if row.predicted_label is not None:
            raise ValueError("Input retrieval predictions must not be prefilled")
        gold_label = row.gold_labels[0]
        pred.append(row.candidate_labels[int(np.argmax(logits[i]))])
        bm25.append(row.candidate_labels[0])
        gold.append(gold_label)
        covered.append(gold_label in row.candidate_labels[:k])

    n = len(rows)
    correct = np.array([a == b for a, b in zip(pred, gold, strict=True)])
    base_correct = np.array([a == b for a, b in zip(bm25, gold, strict=True)])
    covered_arr = np.array(covered)
    if int(covered_arr.sum()) == 0:
        raise ValueError("No gold label was retrieved in top16")
    rescue = int((correct & ~base_correct).sum())
    broken = int((~correct & base_correct).sum())
    return {
        "n_cases": n,
        "label_classes": len(all_classes),
        "top1_accuracy": int(correct.sum()) / n,
        "top1_correct_n": int(correct.sum()),
        "macro_f1": float(
            f1_score(gold, pred, labels=list(all_classes), average="macro", zero_division=0)
        ),
        "bm25_top1_accuracy": int(base_correct.sum()) / n,
        "bm25_top1_correct_n": int(base_correct.sum()),
        "bm25_macro_f1": float(
            f1_score(gold, bm25, labels=list(all_classes), average="macro", zero_division=0)
        ),
        "bm25_to_jeff_rescued_cases": rescue,
        "bm25_to_jeff_broken_cases": broken,
        "net_correct_change": rescue - broken,
        "k": k,
        "gold_in_topk_n": int(covered_arr.sum()),
        "gold_absent_topk_n": int((~covered_arr).sum()),
        "conditional_top1_when_gold_in_topk": float(correct[covered_arr].sum() / covered_arr.sum()),
        "predictions_from_supplied_candidates_only": True,
        "official_single_gold_scoring": True,
    }


def score_k16(
    rows: Sequence[Prediction],
    raw_logits: Sequence[Sequence[float]] | np.ndarray,
    *,
    all_classes: Sequence[str],
) -> dict:
    """Preserve the published K=16 scoring report interface."""
    scored = score_k(rows, raw_logits, k=16, all_classes=all_classes)
    for key in ("gold_in_topk_n", "gold_absent_topk_n", "conditional_top1_when_gold_in_topk"):
        scored[key.replace("topk", "top16")] = scored.pop(key)
    scored.pop("k")
    return scored


def read_model_logits(path: Path, *, num_cases: int) -> np.ndarray:
    """Strictly pair the model output with the original Dev case_id order."""
    vectors: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= num_cases:
                raise ValueError("More model predictions than original Dev cases")
            value = json.loads(line)
            if set(value) != {"case_id", "logits"}:
                raise ValueError("Model output must contain only case_id and raw logits")
            if value["case_id"] != f"dev-{index:06d}":
                raise ValueError(f"Out-of-order Jeff logit results at Dev row {index}")
            vectors.append(value["logits"])
    if len(vectors) != num_cases:
        raise ValueError(f"Expected {num_cases} Dev predictions, received {len(vectors)}")
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape != (num_cases, 16) or not np.isfinite(matrix).all():
        raise ValueError("Logit file must have exactly 16 finite numeric values per case")
    return matrix
