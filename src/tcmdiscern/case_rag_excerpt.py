"""Query-anchored, source-faithful clinical excerpts from retrieved cases.

Selects a contiguous substring of the actual historical case; never creates
new symptoms or interprets missing symptoms as negative findings.
This is an extractive RAG ablation, not an established new clinical method.
"""
from __future__ import annotations

import re
from random import Random

from .case_rag import CASE_CHARS
from .datasets import CLINICAL_FIELDS
from .retrieval import chinese_bigrams

EXCERPT_ARMS = ("baseline", "paired2", "querywindow2", "randomwindow2", "fullcase2")
BOUNDARY = re.compile(r"[，,。；;！？!?：:\n]")


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def candidate_starts(source: str, width: int) -> list[int]:
    """Prefer clinical clause boundaries, with spaced fallback for long notes."""
    if len(source) <= width:
        return [0]
    maximum = len(source) - width
    starts = {0, maximum}
    starts.update(min(maximum, match.end()) for match in BOUNDARY.finditer(source))
    # Recover relevance in long clauses with sparse punctuation.
    starts.update(range(width // 2, maximum + 1, max(1, width // 2)))
    return sorted(starts)


def excerpt(
    source: object, patient_query: str, *, width: int, mode: str,
    seed: int = 0,
) -> str:
    """Return one original contiguous fragment under a fixed character budget."""
    original = normalize(source)
    if mode == "full":
        return original
    if mode not in {"prefix", "query", "random"}:
        raise ValueError("Unsupported case evidence mode")
    if len(original) <= width:
        return original
    candidates = candidate_starts(original, width)
    if mode == "prefix":
        start = 0
    elif mode == "random":
        start = Random(seed).choice(candidates)
    else:
        q = set(chinese_bigrams(patient_query))
        start = min(
            candidates,
            key=lambda i: (
                -len(q & set(chinese_bigrams(original[i:i + width]))),
                i,
            ),
        )
    return original[start:start + width]


def evidence_case(
    train_row: dict, original_label: str, patient_query: str, *,
    mode: str, seed: int,
) -> dict:
    """Preserve source label; replace only extracted original clinical text."""
    return {
        "candidate_syndrome": original_label,
        **{
            field: excerpt(
                train_row.get(field), patient_query, width=CASE_CHARS[field],
                mode=mode, seed=seed + j * 9176,
            )
            for j, field in enumerate(CLINICAL_FIELDS)
        },
    }


def validate_excerpts(original: dict, chosen: dict, source: dict, *, mode: str) -> None:
    if original["candidate_syndrome"] != chosen["candidate_syndrome"]:
        raise ValueError("Source label changed during evidence extraction")
    for field in CLINICAL_FIELDS:
        raw = normalize(source.get(field))
        value = chosen[field]
        if (not raw and value) or (raw and value not in raw):
            raise ValueError("Extracted evidence is not a verbatim source substring")
        if mode != "full" and len(value) > CASE_CHARS[field]:
            raise ValueError("Extracted evidence exceeded original character budget")
