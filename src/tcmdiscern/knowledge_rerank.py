"""Diagnostic knowledge-to-candidate reranking that preserves original BM25 Top-16.

This is deliberately an *untrained* CPU ablation. It never changes frozen
candidate lists, writes patient text, opens Test or produces a Jeff result.
"""

from __future__ import annotations

from collections.abc import Sequence


def rerank_fixed_candidates(
    bm25_labels: Sequence[str], knowledge_labels: Sequence[str], *, k: int = 16
) -> dict[str, list[str]]:
    """Rank the exact same K labels using knowledge alone or fixed RRF(60)."""
    if not 1 <= k <= len(bm25_labels):
        raise ValueError("Invalid candidate width")
    if len(bm25_labels) != len(set(bm25_labels)):
        raise ValueError("BM25 ranking contains duplicate syndrome labels")
    if len(knowledge_labels) != len(set(knowledge_labels)):
        raise ValueError("Knowledge ranking contains duplicate syndrome labels")
    if not set(bm25_labels).issubset(set(knowledge_labels)):
        raise ValueError("Knowledge ranking must cover every frozen BM25 candidate")
    candidates = list(bm25_labels[:k])
    knowledge_position = {name: i for i, name in enumerate(knowledge_labels)}
    baseline = {name: i for i, name in enumerate(candidates)}
    by_knowledge = sorted(candidates, key=lambda name: (knowledge_position[name], baseline[name]))
    # Fixed RRF(60) is a pre-declared, label-blind illustrative control.
    by_rrf = sorted(
        candidates,
        key=lambda name: (
            -(1 / (60 + baseline[name] + 1) + 1 / (60 + knowledge_position[name] + 1)),
            baseline[name],
        ),
    )
    if set(by_knowledge) != set(candidates) or set(by_rrf) != set(candidates):
        raise RuntimeError("A reranker must not add, remove or replace a candidate")
    return {"bm25": candidates, "knowledge_within_k": by_knowledge, "rrf60_within_k": by_rrf}
