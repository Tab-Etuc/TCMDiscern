"""Candidate coverage and end-to-end Top-1 evaluation.

The gold label is never inserted into the candidate list by this evaluator.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    case_id: str
    gold_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    predicted_label: str | None = None

    @classmethod
    def from_mapping(cls, row: dict[str, object]) -> Prediction:
        case_id = row.get("case_id")
        gold = row.get("gold_labels")
        candidates = row.get("candidate_labels")
        predicted = row.get("predicted_label")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(gold, list) or not gold or any(
            not isinstance(x, str) or not x for x in gold
        ):
            raise ValueError(f"{case_id}: gold_labels must be a non-empty string list")
        if not isinstance(candidates, list) or any(
            not isinstance(x, str) or not x for x in candidates
        ):
            raise ValueError(f"{case_id}: candidate_labels must be a string list")
        if len(set(gold)) != len(gold):
            raise ValueError(f"{case_id}: duplicate gold labels")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"{case_id}: duplicate candidate labels")
        if predicted is not None and (
            not isinstance(predicted, str) or predicted not in candidates
        ):
            raise ValueError(f"{case_id}: predicted label must be a candidate")
        return cls(case_id, tuple(gold), tuple(candidates), predicted)


def evaluate(rows: Sequence[Prediction], ks: Sequence[int] = (5, 10, 25)) -> dict:
    """Evaluate coverage and accuracy without modifying retrieval results."""
    if not rows:
        raise ValueError("At least one prediction is required")
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in ks):
        raise ValueError("K must contain positive integers")
    identifiers = [row.case_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate case_id")
    n = len(rows)
    recall = {
        str(k): sum(
            bool(set(row.gold_labels).intersection(row.candidate_labels[:k]))
            for row in rows
        )
        / n
        for k in sorted(set(ks))
    }
    evaluated = [row for row in rows if row.predicted_label is not None]
    result: dict = {"n_cases": n, "recall_any_at_k": recall}
    if evaluated:
        if len(evaluated) != n:
            raise ValueError("All or none of the cases must have predicted_label")
        correct = sum(row.predicted_label in row.gold_labels for row in rows)
        result["top1_accuracy"] = correct / n
        result["top1_correct"] = correct
        for k in sorted(set(ks)):
            covered = [
                row for row in rows
                if set(row.gold_labels).intersection(row.candidate_labels[:k])
            ]
            if covered and all(row.predicted_label in row.candidate_labels[:k] for row in rows):
                result.setdefault("conditional_accuracy_at_k", {})[str(k)] = (
                    sum(row.predicted_label in row.gold_labels for row in covered)
                    / len(covered)
                )
    return result


def read_predictions(path: Path) -> list[Prediction]:
    rows: list[Prediction] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("record must be an object")
                rows.append(Prediction.from_mapping(payload))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate label-level candidate coverage")
    parser.add_argument("predictions", type=Path, help="JSONL of predicted candidates")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 7, 10, 15, 20, 25])
    args = parser.parse_args()
    print(json.dumps(evaluate(read_predictions(args.predictions), args.ks), indent=2))


if __name__ == "__main__":
    main()
