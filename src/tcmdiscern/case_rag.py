"""Query-conditioned clinical-case RAG controls for Jeff/TCM-SD.

All retrieval exemplars come from the official Train index. Retrieval never
reads query gold labels, and provenance is kept outside the model prompt.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

import numpy as np

from .datasets import CLINICAL_FIELDS, normalized_input
from .jeff_format import decision_row
from .retrieval import CaseBM25, chinese_bigrams

CONDITIONS = ("baseline", "global4", "balanced4", "paired4", "random4")
CONDITIONS_2 = ("baseline", "global2", "balanced2", "paired2", "random2")
CASE_CHARS = {"chief_complaint": 58, "description": 90, "detection": 70}
TOP_SUPPORTS = 16
MAX_POOL_PER_LABEL = 32  # Covers random4's 32-case pool and every other arm.
N_EVIDENCE = 4
RETRIEVAL_INSTRUCTION = (
    "以下為從訓練庫檢索的已標註歷史醫案，標註可能包含雜訊。"
    "請獨立判斷目前患者，僅從原始16個候選證型中選擇。"
)


@dataclass(frozen=True)
class Support:
    position: int
    label: str
    group: int
    score: float


def transitive_group_ids(rows: Sequence[dict]) -> list[int]:
    """Connect nonempty patient IDs and identical normalized clinical text."""
    parent = list(range(len(rows)))
    seen_user: dict[str, int] = {}
    seen_text: dict[str, int] = {}

    def find(x: int) -> int:
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: int, b: int) -> None:
        x, y = find(a), find(b)
        if x != y:
            parent[max(x, y)] = min(x, y)

    for i, row in enumerate(rows):
        uid = str(row.get("user_id") or "").strip()
        text = normalized_input(row, full=True)
        for value, mapping in ((uid, seen_user), (text, seen_text)):
            if value:
                if value in mapping:
                    join(i, mapping[value])
                mapping[value] = i
    return [find(i) for i in range(len(rows))]


def _view(row: dict) -> dict[str, str]:
    """A fixed char allowance per field; excludes demographics and diagnoses."""
    return {
        field: re.sub(r"\s+", " ", str(row.get(field) or "")).strip()[:CASE_CHARS[field]]
        for field in CLINICAL_FIELDS
    }


class CaseRAGIndex:
    def __init__(self, train_rows: Sequence[dict], *, indices: Sequence[int] | None = None):
        if not train_rows:
            raise ValueError("Need official training rows")
        self.rows = train_rows
        self.indices = list(range(len(train_rows))) if indices is None else list(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Need unique training index positions")
        self.groups = transitive_group_ids(train_rows)
        self.model = CaseBM25(
            ["\n".join(f"{f}: {str(train_rows[i].get(f) or '').strip()}" for f in CLINICAL_FIELDS)
             for i in self.indices],
            [str(train_rows[i]["norm_syndrome"]) for i in self.indices],
        )
        self.labels = [str(train_rows[i]["norm_syndrome"]) for i in self.indices]
        group_labels: dict[int, set[str]] = defaultdict(set)
        for pos in self.indices:
            group_labels[self.groups[pos]].add(str(train_rows[pos]["norm_syndrome"]))
        self.ambiguous_groups = {g for g, names in group_labels.items() if len(names) > 1}
        self._positions = np.asarray(self.indices, dtype=np.int32)
        label_to_code = {name: j for j, name in enumerate(self.model.label_names)}
        self._label_codes = np.asarray([label_to_code[name] for name in self.labels], dtype=np.int16)
        self._groups = np.asarray([self.groups[pos] for pos in self.indices], dtype=np.int32)
        by_label: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            if int(self._groups[index]) not in self.ambiguous_groups:
                by_label[label].append(index)
        self._eligible_by_label = {
            label: np.asarray(positions, dtype=np.int32)
            for label, positions in by_label.items()
        }
        self.text_groups = {
            normalized_input(train_rows[i], full=True): self.groups[i] for i in self.indices
        }
        self.patient_groups = {
            str(train_rows[i].get("user_id") or "").strip(): self.groups[i]
            for i in self.indices if str(train_rows[i].get("user_id") or "").strip()
        }
        self._lexical: dict[int, set[str]] = {}

    def _terms(self, pos: int) -> set[str]:
        if pos not in self._lexical:
            row = self.rows[pos]
            text = " ".join(str(row.get(f) or "") for f in CLINICAL_FIELDS)
            self._lexical[pos] = set(chinese_bigrams(text))
        return self._lexical[pos]

    def _case_scores(self, query: dict) -> np.ndarray:
        source = "\n".join(f"{f}: {str(query.get(f) or '').strip()}" for f in CLINICAL_FIELDS)
        q = self.model.vectorizer.transform([source])
        q.data[:] = 1
        return np.asarray((q @ self.model.weights_t).toarray()).ravel()

    def rank_and_supports(
        self, query: dict, *, max_k: int = 16,
        query_train_index: int | None = None,
    ) -> tuple[list[str], list[Support]]:
        """Share one BM25 query pass between the original shortlist and Case-RAG."""
        if not 2 <= max_k <= len(self.model.label_names):
            raise ValueError("Invalid shortlist width")
        scores = self._case_scores(query)
        label_scores = np.maximum.reduceat(
            scores[self.model.ordered_doc_ids], self.model.group_starts,
        )
        ranked = np.argsort(-label_scores, kind="stable")[:max_k]
        names = [self.model.label_names[int(i)] for i in ranked]
        return names, self._supports(
            query, names, query_train_index=query_train_index,
            case_scores=scores,
        )

    def _supports(
        self,
        query: dict,
        candidates: Sequence[str],
        *,
        query_train_index: int | None = None,
        case_scores: np.ndarray | None = None,
    ) -> list[Support]:
        scores = self._case_scores(query) if case_scores is None else case_scores
        if scores.shape != (len(self.indices),):
            raise ValueError("BM25 case score alignment failed")
        allowed = set(candidates)
        forbidden_groups = set()
        if query_train_index is not None:
            forbidden_groups.add(self.groups[query_train_index])
        text_group = self.text_groups.get(normalized_input(query, full=True))
        if text_group is not None:
            forbidden_groups.add(text_group)
        patient_group = self.patient_groups.get(str(query.get("user_id") or "").strip())
        if patient_group is not None:
            forbidden_groups.add(patient_group)
        # Preserve the original per-(patient group, label) maximum exactly,
        # including stable source-position tie-breaking. Vectorized sorting
        # avoids scanning/sorting tens of thousands of Python Support objects.
        eligible = [self._eligible_by_label[label] for label in allowed
                    if label in self._eligible_by_label]
        if not eligible:
            return []
        doc_ids = np.concatenate(eligible)
        if forbidden_groups:
            doc_ids = doc_ids[~np.isin(self._groups[doc_ids], list(forbidden_groups))]
        if doc_ids.size == 0:
            return []
        # Primary sort: group. Within group: score descending, position ascending.
        by_group = np.lexsort((
            self._positions[doc_ids], -scores[doc_ids], self._groups[doc_ids],
        ))
        sorted_ids = doc_ids[by_group]
        unique = np.r_[True, self._groups[sorted_ids[1:]] != self._groups[sorted_ids[:-1]]]
        winners = sorted_ids[unique]
        # Global order remains identical to the original sorted(best.values()).
        by_score = np.lexsort((self._positions[winners], -scores[winners]))
        ranked = winners[by_score]
        # Only 32 unique supports per syndrome can ever affect these five arms.
        # Filter *after* group de-duplication and global score ordering, so all
        # selected source cases remain byte-for-byte identical to the legacy run.
        codes = self._label_codes[ranked]
        keep = np.zeros(len(ranked), dtype=np.bool_)
        for code in np.unique(codes):
            keep[np.flatnonzero(codes == code)[:MAX_POOL_PER_LABEL]] = True
        return [
            Support(
                int(self._positions[j]), self.labels[int(j)],
                int(self._groups[j]), float(scores[j]),
            )
            for j in ranked[keep]
        ]

    @staticmethod
    def _top_by_label(items: list[Support], label: str, n: int = TOP_SUPPORTS) -> list[Support]:
        return [item for item in items if item.label == label][:n]

    @staticmethod
    def _unique(items: Sequence[Support], required: int = N_EVIDENCE) -> bool:
        return len(items) == required and len({s.group for s in items}) == required

    def _paired(self, items: list[Support], first: str, second: str, *, n_pairs: int = 2) -> list[Support]:
        a = self._top_by_label(items, first)
        b = self._top_by_label(items, second)
        if len(a) < n_pairs or len(b) < n_pairs:
            raise ValueError("Insufficient support groups for a paired comparison")
        max_score = max(1.0, items[0].score)
        ranked: list[tuple[float, int, int, Support, Support]] = []
        for i, left in enumerate(a):
            for j, right in enumerate(b):
                if left.group == right.group:
                    continue
                terms_l, terms_r = self._terms(left.position), self._terms(right.position)
                union = len(terms_l | terms_r)
                overlap = len(terms_l & terms_r) / union if union else 0.0
                # Match similarly presenting cases while preserving query relevance.
                score = (left.score + right.score) / (2 * max_score) + 0.30 * overlap
                ranked.append((-score, i, j, left, right))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))
        result = []
        used_groups: set[int] = set()
        for _, _, _, left, right in ranked:
            if left.group not in used_groups and right.group not in used_groups:
                result.extend((left, right))
                used_groups.update((left.group, right.group))
                if len(result) == 2 * n_pairs:
                    return result
        raise ValueError("Cannot select two independent matched pairs")

    def choose(
        self,
        query: dict,
        candidates: Sequence[str],
        *,
        condition: str,
        seed: int,
        query_train_index: int | None = None,
        precomputed_items: list[Support] | None = None,
    ) -> list[Support]:
        if condition not in CONDITIONS and condition not in CONDITIONS_2:
            raise ValueError(f"Unknown RAG condition: {condition}")
        if len(candidates) != 16 or len(set(candidates)) != 16:
            raise ValueError("Expected original distinct 16 candidates")
        if condition == "baseline":
            return []
        evidence_budget = int(condition[-1])
        pairs = evidence_budget // 2
        top2 = candidates[:2]
        items = (
            precomputed_items if precomputed_items is not None
            else self._supports(query, candidates, query_train_index=query_train_index)
        )
        if condition.startswith("global"):
            chosen = []
            used = set()
            for item in items:
                if item.group not in used:
                    chosen.append(item)
                    used.add(item.group)
                if len(chosen) == evidence_budget:
                    break
        elif condition.startswith("balanced"):
            first = self._top_by_label(items, top2[0], pairs)
            second = self._top_by_label(items, top2[1], pairs)
            chosen = [item for pair in zip(first, second) for item in pair]
        elif condition.startswith("paired"):
            chosen = self._paired(items, *top2, n_pairs=pairs)
        else:
            rnd = Random(seed)
            first = self._top_by_label(items, top2[0], MAX_POOL_PER_LABEL)
            second = self._top_by_label(items, top2[1], MAX_POOL_PER_LABEL)
            if len(first) < pairs or len(second) < pairs:
                raise ValueError("Insufficient support for the random control")
            chosen = [
                item for pair in zip(rnd.sample(first, pairs), rnd.sample(second, pairs))
                for item in pair
            ]
        if not self._unique(chosen, evidence_budget):
            raise ValueError("RAG support groups are missing or repeated")
        return chosen

    def make_row(
        self,
        query: dict,
        candidates: Sequence[str],
        *,
        case_id: str,
        condition: str,
        seed: int = 20261116,
        query_train_index: int | None = None,
        precomputed_items: list[Support] | None = None,
    ) -> tuple[dict, dict]:
        chosen = self.choose(
            query, candidates, condition=condition, seed=seed,
            query_train_index=query_train_index, precomputed_items=precomputed_items,
        )
        row = decision_row(query, case_id=case_id, candidates=candidates)
        if chosen:
            row["state"]["retrieval_note"] = RETRIEVAL_INSTRUCTION
            row["state"]["retrieved_cases"] = [
                {"candidate_syndrome": s.label, **_view(self.rows[s.position])}
                for s in chosen
            ]
        diagnostic = {
            "condition": condition,
            "source_train_positions": [s.position for s in chosen],
            "support_labels": [s.label for s in chosen],
            "support_scores": [round(s.score, 4) for s in chosen],
            "support_groups": [s.group for s in chosen],
            "prompt_chars": len(json.dumps(row, ensure_ascii=False)),
            "supported_options": len({s.label for s in chosen}),
        }
        return row, diagnostic
