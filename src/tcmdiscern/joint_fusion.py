"""Factorized disease + syndrome Top16 joint decoding using native Jeff logits.

Uses one disease decision and one independent syndrome decision, then optionally
limits cross-product candidates to Train-observed disease/syndrome pairs.
No gold disease name is supplied during inference.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .disease_study import pair_key


def log_softmax(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Nonfinite or missing decision logits")
    top = max(values)
    normalizer = top + math.log(sum(math.exp(v - top) for v in values))
    return [float(v - normalizer) for v in values]


def pair_candidates(
    disease_options: Sequence[str],
    syndrome_options: Sequence[str],
    *,
    known_pairs: frozenset[str] | None = None,
) -> list[tuple[int, int, str]]:
    if len(disease_options) != 16 or len(syndrome_options) != 16:
        raise ValueError("Requires two independent K16 candidate sets")
    if len(set(disease_options)) != 16 or len(set(syndrome_options)) != 16:
        raise ValueError("Duplicate label in factorized options")
    result = [
        (i, j, pair_key(d, s))
        for i, d in enumerate(disease_options)
        for j, s in enumerate(syndrome_options)
    ]
    if known_pairs is not None:
        result = [option for option in result if option[2] in known_pairs]
    return result


def decode(
    diseases: Sequence[str],
    syndrome_labels: Sequence[str],
    disease_logits: Sequence[float],
    syndrome_logits: Sequence[float],
    *,
    known_pairs: frozenset[str] | None = None,
    disease_weight: float = 0.5,
) -> tuple[str, bool]:
    """Pick best pair by weighted independent log probabilities; deterministic ties."""
    if len(disease_logits) != 16 or len(syndrome_logits) != 16:
        raise ValueError("Jeff logits must have 16 positions per label space")
    if not 0 < disease_weight < 1:
        raise ValueError("Blend weight must be strictly between zero and one")
    choices = pair_candidates(diseases, syndrome_labels, known_pairs=known_pairs)
    used_fallback = False
    if not choices:
        choices = pair_candidates(diseases, syndrome_labels)
        used_fallback = True
    p_d = log_softmax(disease_logits)
    p_s = log_softmax(syndrome_logits)
    _, _, pair = max(
        choices,
        key=lambda item: (
            disease_weight * p_d[item[0]] + (1.0 - disease_weight) * p_s[item[1]],
            -item[0], -item[1],
        ),
    )
    return pair, used_fallback


def factorized_oracles(
    gold_pairs: Sequence[str],
    disease_alternatives: Sequence[Sequence[str]],
    syndrome_alternatives: Sequence[Sequence[str]],
    *,
    known_pairs: frozenset[str],
) -> dict:
    if not (len(gold_pairs) == len(disease_alternatives) == len(syndrome_alternatives)):
        raise ValueError("Aligned official evaluation rows required")
    if not gold_pairs:
        raise ValueError("At least one evaluation row required")
    # No pair candidate is built using gold; gold only enters these metrics.
    from .disease_study import pair_parts

    rows = [
        (
            pair_parts(gold),
            set(diseases),
            set(syndromes),
            gold in known_pairs,
        )
        for gold, diseases, syndromes in zip(
            gold_pairs, disease_alternatives, syndrome_alternatives, strict=True
        )
    ]
    marginal_hits = sum(
        d in disease_set and s in syndrome_set
        for (d, s), disease_set, syndrome_set, _ in rows
    )
    constrained_hits = sum(
        d in disease_set and s in syndrome_set and seen
        for (d, s), disease_set, syndrome_set, seen in rows
    )
    return {
        "n_cases": len(gold_pairs),
        "factorized_oracle16x16": marginal_hits / len(gold_pairs),
        "factorized_oracle16x16_hits": marginal_hits,
        "train_pair_constrained_oracle16x16": constrained_hits / len(gold_pairs),
        "train_pair_constrained_oracle16x16_hits": constrained_hits,
        "unseen_pairs": sum(not seen for _, _, _, seen in rows),
    }
