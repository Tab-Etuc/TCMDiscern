"""Fixed-width hierarchical disease-syndrome retrieval, no gold-aware candidate injection.

One disease-label BM25 index and one pair-label BM25 index are built from
official Train. For each Dev/Test query, disease BM25 chooses D possible
diseases; pair BM25 scores the Train-supported pairs within each selected
disease, taking up to S pairs per disease. Exactly K=D*S candidates are
returned (backfilled by global pair ranks when a disease has few pairs).
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from .disease_study import pair_key, pair_parts

CANDIDATE_K = 16
HIERARCHIES = ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1))


@cache
def _disease(pair_label: str) -> str:
    return pair_parts(pair_label)[0]


def hierarchical_pairs(
    disease_ranking: Sequence[str],
    pair_ranking: Sequence[str],
    *,
    disease_k: int,
    syndrome_k: int,
    k: int = CANDIDATE_K,
) -> list[str]:
    """Fixed k, distinct candidates, ordered by original BM25 pair rank."""
    if not 1 <= disease_k <= len(disease_ranking) or not 1 <= syndrome_k <= k:
        raise ValueError("Invalid disease/syndrome slot allocation")
    if disease_k * syndrome_k != k:
        raise ValueError("Disease x syndrome slots must equal fixed candidate width")
    if len(set(disease_ranking)) != len(disease_ranking):
        raise ValueError("Duplicate disease label in retrieval ranking")
    if len(pair_ranking) < k or len(set(pair_ranking)) != len(pair_ranking):
        raise ValueError("Pair ranking is not complete or unique")

    selected_diseases = disease_ranking[:disease_k]
    by_disease: dict[str, list[str]] = {d: [] for d in selected_diseases}
    for pair in pair_ranking:
        disease = _disease(pair)
        if disease in by_disease and len(by_disease[disease]) < syndrome_k:
            by_disease[disease].append(pair)
    included = {pair for pairs in by_disease.values() for pair in pairs}
    for pair in pair_ranking:
        if len(included) >= k:
            break
        included.add(pair)
    if len(included) != k:
        raise ValueError("Unable to form exactly 16 joint candidates")
    return [pair for pair in pair_ranking if pair in included]




def crossed_pairs(
    disease_ranking: Sequence[str],
    syndrome_ranking: Sequence[str],
    pair_ranking: Sequence[str],
    *,
    allowed_pairs: frozenset[str],
    k: int = CANDIDATE_K,
    flat_keep: int = 0,
) -> list[str]:
    """Compose Train-observed pairs from two independent BM25 label rankings.

    Optional flat_keep reserves highest-ranked direct-pair BM25 candidates;
    other slots are filled using disease/syndrome rank-sum, then pair BM25.
    """
    if not 0 <= flat_keep < k or len(pair_ranking) < k:
        raise ValueError("Invalid candidate width or flat baseline")
    if len(set(disease_ranking)) != len(disease_ranking):
        raise ValueError("Duplicate diseases in independent ranking")
    if len(set(syndrome_ranking)) != len(syndrome_ranking):
        raise ValueError("Duplicate syndromes in independent ranking")
    chosen: set[str] = set(pair_ranking[:flat_keep])
    crossed: list[tuple[int, int, str]] = []
    for disease_rank, disease in enumerate(disease_ranking):
        for syndrome_rank, syndrome in enumerate(syndrome_ranking):
            pair = pair_key(disease, syndrome)
            if pair in allowed_pairs:
                crossed.append((disease_rank + syndrome_rank, disease_rank, pair))
    for _, _, pair in sorted(crossed):
        if len(chosen) >= k:
            break
        chosen.add(pair)
    for pair in pair_ranking:
        if len(chosen) >= k:
            break
        chosen.add(pair)
    return [pair for pair in pair_ranking if pair in chosen]

def hybrid_pairs(
    disease_ranking: Sequence[str],
    pair_ranking: Sequence[str],
    *,
    flat_keep: int,
    disease_k: int,
    syndrome_k: int,
    k: int = CANDIDATE_K,
) -> list[str]:
    """Keep best flat candidates and reserve remainder for disease diversity."""
    if not 1 <= flat_keep < k or disease_k * syndrome_k != k - flat_keep:
        raise ValueError("Invalid flat/diverse slot allocation")
    if len(pair_ranking) < k or len(set(pair_ranking)) != len(pair_ranking):
        raise ValueError("Insufficient unique pair labels")
    selected: set[str] = set(pair_ranking[:flat_keep])
    grouped = {d: [] for d in disease_ranking[:disease_k]}
    for pair in pair_ranking:
        disease = _disease(pair)
        if disease in grouped and len(grouped[disease]) < syndrome_k:
            grouped[disease].append(pair)
    for disease in disease_ranking[:disease_k]:
        for pair in grouped[disease]:
            if len(selected) >= k:
                break
            selected.add(pair)
    # Deduplication can reduce intended width; fill with strongest unseen pairs.
    for pair in pair_ranking:
        if len(selected) >= k:
            break
        selected.add(pair)
    return [pair for pair in pair_ranking if pair in selected]
