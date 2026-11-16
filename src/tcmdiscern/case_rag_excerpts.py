"""Source-faithful query-window case excerpts under the frozen character budget.

A deterministic extractive-only control for Case-RAG, not a learned model.
The original prefix-only supports remain the primary frozen baseline.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from .case_rag import CASE_CHARS
from .datasets import CLINICAL_FIELDS
from .retrieval import chinese_bigrams


def normalize_source(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def focused_window(text: str, query_terms: set[str], budget: int) -> str:
    """Select one contiguous source substring; use ellipses to mark truncation."""
    text = normalize_source(text)
    if budget < 5:
        raise ValueError("Clinical excerpt budget must be >=5 chars")
    if len(text) <= budget:
        return text
    # Reserve punctuation to explicitly mark a shortened source.
    available = budget - 2
    last_start = len(text) - available
    candidates: set[int] = {0, last_start}
    for position, char in enumerate(text):
        if char in "，。；、,;:： " and position + 1 <= last_start:
            candidates.add(position + 1)
    for term in query_terms:
        if not 1 <= len(term) <= 6:
            continue
        start = text.find(term)
        while start >= 0:
            for offset in (0, 8, 16):
                candidates.add(max(0, min(last_start, start - offset)))
            start = text.find(term, start + 1)
    best_start = 0
    best = -1
    for start in sorted(candidates):
        excerpt = text[start : start + available]
        # Distinct matched query bigrams; repetitive boilerplate gives no reward.
        overlap = len(query_terms.intersection(chinese_bigrams(excerpt)))
        if overlap > best:
            best = overlap
            best_start = start
    window = text[best_start : best_start + available]
    return ("…" if best_start else "") + window + (
        "…" if best_start + available < len(text) else ""
    )


def focused_case_view(
    row: Mapping[str, object],
    query: Mapping[str, object],
) -> dict[str, str]:
    """Select true source spans using only the current patient's clinical text."""
    query_terms = set(chinese_bigrams(
        " ".join(str(query.get(field) or "") for field in CLINICAL_FIELDS)
    ))
    return {
        field: focused_window(
            str(row.get(field) or ""), query_terms, CASE_CHARS[field],
        )
        for field in CLINICAL_FIELDS
    }


def verify_extractive(
    source: Mapping[str, object], excerpt: Mapping[str, str],
) -> None:
    """Fail closed if any clinical phrase was synthesized or budget exceeded."""
    if set(excerpt) != set(CLINICAL_FIELDS):
        raise ValueError("Clinical source fields must remain unchanged")
    for field in CLINICAL_FIELDS:
        source_text = normalize_source(source.get(field))
        value = excerpt[field]
        if not isinstance(value, str) or len(value) > CASE_CHARS[field]:
            raise ValueError("Excerpt exceeds the original character budget")
        if value.strip("…") not in source_text:
            raise ValueError("Excerpt includes content absent from the source")
