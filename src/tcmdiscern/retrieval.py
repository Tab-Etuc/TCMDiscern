"""Case-based BM25 retrieval ranked by distinct TCM-SD syndrome labels.

This is a standalone *sparse* baseline, not the hybrid retriever. Training
labels are used only to group scores for training cases. Query labels are never
read by the retriever, and validation/test rows are never indexed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

WORD_RE = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+", flags=re.IGNORECASE)


def chinese_bigrams(text: str) -> list[str]:
    """CJK character bigrams, non-CJK alphanumeric tokens, no punctuation."""
    result: list[str] = []
    for segment in WORD_RE.findall(text.lower()):
        if "\u3400" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                result.append(segment)
            else:
                result.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            result.append(segment)
    return result


class CaseBM25:
    """BM25 exact per-label max over training case documents."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        min_df: int = 2,
    ) -> None:
        if not texts or len(texts) != len(labels):
            raise ValueError("Training texts and labels must be nonempty and aligned")
        if any(not label for label in labels):
            raise ValueError("Training labels must not be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")

        self.label_names = sorted(set(labels))
        label_to_index = {name: i for i, name in enumerate(self.label_names)}
        label_ids = np.asarray([label_to_index[label] for label in labels], dtype=np.int32)
        self.ordered_doc_ids = np.argsort(label_ids, kind="stable")
        ordered_labels = label_ids[self.ordered_doc_ids]
        self.group_starts = np.flatnonzero(np.r_[True, ordered_labels[1:] != ordered_labels[:-1]])

        self.vectorizer = CountVectorizer(analyzer=chinese_bigrams, dtype=np.float32, min_df=min_df)
        term_counts: sparse.csr_matrix = self.vectorizer.fit_transform(texts).tocsr()
        if term_counts.nnz == 0:
            raise ValueError("No searchable tokens in training documents")
        n_docs = term_counts.shape[0]
        lengths = np.asarray(term_counts.sum(axis=1)).ravel()
        avg_length = float(lengths.mean())
        if avg_length <= 0:
            raise ValueError("Empty corpus")
        doc_freq = np.asarray(term_counts.getnnz(axis=0)).ravel().astype(np.float32)
        idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        row_ids = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(term_counts.indptr))
        tf = term_counts.data
        length_norm = k1 * (1 - b + b * lengths[row_ids] / avg_length)
        term_counts.data = (idf[term_counts.indices] * (tf * (k1 + 1)) / (tf + length_norm)).astype(
            np.float32
        )
        self.weights_t = term_counts.transpose().tocsr()

    def rank(
        self, texts: Sequence[str], *, max_k: int = 50, batch_size: int = 24
    ) -> list[list[str]]:
        """Rank all labels with a bounded-memory sparse matrix product."""
        if not 1 <= max_k <= len(self.label_names):
            raise ValueError("max_k must not exceed the available label count")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output: list[list[str]] = []
        for start in range(0, len(texts), batch_size):
            queries = self.vectorizer.transform(texts[start : start + batch_size])
            queries.data[:] = 1.0  # unique query terms, standard BM25 query weighting
            case_scores = (queries @ self.weights_t).toarray()
            label_scores = np.maximum.reduceat(
                case_scores[:, self.ordered_doc_ids],
                self.group_starts,
                axis=1,
            )
            rankings = np.argsort(-label_scores, axis=1, kind="stable")[:, :max_k]
            output.extend([self.label_names[int(j)] for j in row] for row in rankings)
        return output
