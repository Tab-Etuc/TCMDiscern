"""Regression tests for contradictory labels in Jeff training hard negatives."""

from __future__ import annotations

import json

import pytest

import scripts.export_jeff_train as exporter
from scripts.verify_jeff_train import validate_file


def fixture_records() -> tuple[list[dict[str, str]], list[list[str]]]:
    records = [
        {
            "norm_syndrome": "證型A",
            "chief_complaint": "完全相同的病歷",
            "description": "病情一致",
            "detection": "四診一致",
        },
        {
            "norm_syndrome": "證型B",
            "chief_complaint": "完全相同的病歷",
            "description": "病情一致",
            "detection": "四診一致",
        },
    ]
    records.extend(
        {
            "norm_syndrome": f"證型{i + 2}",
            "chief_complaint": f"獨立病例{i + 2}",
            "description": "不同的病情",
            "detection": "脈滑",
        }
        for i in range(16)
    )
    candidates = ["證型A", "證型B", *(f"候選{j}" for j in range(48))]
    return records, [candidates.copy() for _ in records]


def test_filter_preserves_cases_and_prevents_confirmed_contradictory_negatives(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, ranked = fixture_records()
    monkeypatch.setattr(exporter, "TRAIN_ROWS", len(records))
    filtered = tmp_path / "filtered.jsonl"
    original = tmp_path / "original.jsonl"
    safe_report = exporter.write_examples(records, ranked, path=filtered, k=16, epoch=0, seed=18)
    old_report = exporter.write_examples(
        records,
        ranked,
        path=original,
        k=16,
        epoch=0,
        seed=18,
        exclude_ambiguous_negatives=False,
    )
    assert safe_report["cases"] == old_report["cases"] == len(records)
    assert safe_report["cases_removed"] == 0
    assert safe_report["exclude_ambiguous_negatives"] is True
    assert safe_report["cases_with_confirmed_contradiction_prevented"] == 2
    assert safe_report["confirmed_ambiguous_negatives_replaced"] == 2
    assert old_report["confirmed_ambiguous_negatives_replaced"] == 0

    rows = [json.loads(line) for line in filtered.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(records)
    for i, row in enumerate(rows):
        alternatives = {"證型B"} if i == 0 else {"證型A"} if i == 1 else set()
        actual = {opt["id"] for opt in row["row"]["options"]}
        assert len(actual) == 16
        assert not actual.intersection(alternatives)
        assert row["row"]["options"][row["target_index"]]["id"] == records[i]["norm_syndrome"]
        assert "norm_syndrome" not in row["row"]["state"]

    audit = validate_file(
        records,
        ranked,
        filtered,
        expected_k=16,
        exclude_ambiguous_negatives=True,
    )
    assert audit["cases"] == len(records)
    assert audit["alternative_gold_excluded_from_hard_negatives"] is True
    with pytest.raises(ValueError, match="expected BM25 hard negatives"):
        validate_file(
            records,
            ranked,
            original,
            expected_k=16,
            exclude_ambiguous_negatives=True,
        )


def test_direct_filter_never_accepts_alt_gold_as_negative() -> None:
    records, ranked = fixture_records()
    sample = exporter.make_train_example(
        records[0],
        train_index=0,
        candidates=ranked[0],
        gold="證型A",
        k=16,
        epoch=3,
        seed=18,
        forbidden_negatives=frozenset({"證型B"}),
    )
    options = {opt["id"] for opt in sample["row"]["options"]}
    assert "證型A" in options
    assert "證型B" not in options
    assert len(options) == 16
