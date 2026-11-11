"""Ensure the published K=1..35 result contains only real BM25 evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.export_bm25_k_grid import evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/bm25_k1_35_dev_2026-11-11.json"
CSV = ROOT / "experiments/bm25_k1_35_dev_2026-11-11.csv"


def test_exhaustive_35_k_bm25_only_result() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "BM25_ONLY_COMPLETE_K1_TO_K35"
    assert result["excluded_records"] == 0
    assert [row["k"] for row in result["results"]] == list(range(1, 36))
    assert len(result["per_syndrome_recall"]) == 148
    assert all(row["cases"] == 5486 for row in result["results"])
    assert all("nb" not in method.lower() for method in result["evaluation_methods"])


def test_bm25_top1_is_constant_and_coverage_is_monotone() -> None:
    rows = json.loads(RESULT.read_text(encoding="utf-8"))["results"]
    assert {row["bm25_top1_accuracy"] for row in rows} == {4366 / 5486}
    assert len({row["bm25_top1_macro_f1"] for row in rows}) == 1
    assert [row["candidate_hit_n"] for row in rows] == sorted(
        row["candidate_hit_n"] for row in rows
    )
    assert sum(row["new_candidate_hits_at_k"] for row in rows) == rows[-1]["candidate_hit_n"]
    assert rows[4]["candidate_hit_n"] == 5237
    assert rows[16]["candidate_hit_n"] == 5412
    assert rows[34]["candidate_hit_n"] == 5461


def test_csv_and_json_agree_exactly() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    with CSV.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 35
    for raw, expected in zip(rows, result["results"], strict=True):
        for key, value in expected.items():
            assert float(raw[key]) == value


def test_recalculate_from_official_dev_and_saved_bm25_candidates() -> None:
    fresh = evaluate(ROOT)
    frozen = json.loads(RESULT.read_text(encoding="utf-8"))
    assert fresh == frozen
