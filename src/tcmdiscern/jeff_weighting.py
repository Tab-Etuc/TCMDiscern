"""Frozen Train-only BM25 difficulty weighting for controlled Jeff ablations.

The official Train OOF gold rank, not any Dev/Test label, defines difficulty.
Only *recoverable* BM25 mistakes (gold rank 2..16) are upweighted x2.
Gold missing from the first 16 is not emphasized, since a K=16 deployment
cannot recover it without changing candidate retrieval.

Each original Train row still appears once per epoch; weights affect only
the supervised loss. No resampling, new labels or extra optimizer steps.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

WEIGHTING_CHOICES = ("uniform", "recoverable_oof_x2")
TOP_K = 16
RANKED_OPTIONS = 50
RECOVERABLE_MULTIPLIER = 2.0


def load_train_oof_weights(
    examples: Sequence[dict],
    *,
    source: Path,
) -> tuple[list[float], dict]:
    """Validate an entire Train-only grouped OOF ranking and build weights."""
    if not examples:
        raise ValueError("Training examples must not be empty")
    weights: list[float] = []
    rank_counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i >= len(examples):
                raise ValueError("OOF has more rows than Train examples")
            item = examples[i]
            case_id = item["row"]["id"]
            original_gold = item["row"]["options"][item["target_index"]]["id"]
            oof = json.loads(line)
            if case_id != f"train-{i:06d}" or oof.get("case_id") != case_id:
                raise ValueError(f"OOF/Train row ID/order disagree at {i}")
            if oof.get("gold_labels") != [original_gold]:
                raise ValueError(f"OOF/Train gold label disagree at {i}")
            candidates = oof.get("candidate_labels")
            if (
                not isinstance(candidates, list)
                or len(candidates) != RANKED_OPTIONS
                or len(set(candidates)) != RANKED_OPTIONS
                or any(not isinstance(s, str) or not s for s in candidates)
            ):
                raise ValueError(f"OOF candidates invalid at {i}")
            if original_gold == candidates[0]:
                category = "rank1"
            elif original_gold in candidates[1:TOP_K]:
                category = "rank2_to_16"
            elif original_gold in candidates[TOP_K:]:
                category = "rank17_to_50"
            else:
                category = "rank_over50"
            rank_counts[category] += 1
            weights.append(RECOVERABLE_MULTIPLIER if category == "rank2_to_16" else 1.0)
    if len(weights) != len(examples):
        raise ValueError(f"OOF incomplete: {len(weights)} vs {len(examples)}")
    return weights, {
        "policy": "recoverable_oof_x2",
        "source": source.name,
        "original_train_rows": len(weights),
        "original_rows_removed": 0,
        "rank_categories": {
            k: rank_counts[k] for k in ("rank1", "rank2_to_16", "rank17_to_50", "rank_over50")
        },
        "upweighted_original_rows": rank_counts["rank2_to_16"],
        "rank2_to_16_multiplier": RECOVERABLE_MULTIPLIER,
        "mean_raw_weight": sum(weights) / len(weights),
    }


def normalized_batch_weights(
    batch: Sequence[dict],
    *,
    scheme: str,
    oof_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Mean of weighted per-case losses; same gradient scale as uniform CE."""
    if scheme not in WEIGHTING_CHOICES:
        raise ValueError("Unknown Train-only weighting method")
    if not batch:
        raise ValueError("Empty accumulation update")
    if scheme == "uniform":
        return tuple(1.0 / len(batch) for _ in batch)
    if oof_weights is None:
        raise ValueError("Missing frozen OOF weights")
    selected = []
    for item in batch:
        name = item["row"]["id"]
        if not isinstance(name, str) or not name.startswith("train-"):
            raise ValueError("Difficulty weighting is permitted only on Train rows")
        try:
            index = int(name.removeprefix("train-"))
        except ValueError as exc:
            raise ValueError("Invalid Train row ID") from exc
        if name != f"train-{index:06d}" or not 0 <= index < len(oof_weights):
            raise ValueError("Invalid Train source index")
        selected.append(float(oof_weights[index]))
    if any(v <= 0 or not math.isfinite(v) for v in selected):
        raise ValueError("Nonpositive or NaN sample weight")
    total = sum(selected)
    if total <= 0:
        raise ValueError("Invalid batch weight normalization")
    return tuple(value / total for value in selected)
