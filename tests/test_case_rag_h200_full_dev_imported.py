"""Regression audit for H200 imported aggregate and local full-Dev recomputations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(scope="module")
def reports():
    names = {
        "h200_raw": "case_rag_cpu_first_2026-11-17.json",
        "local_raw": "case_rag_cpu_first_local_recomputed_2026-11-17.json",
        "h200_fusion": "case_rag_strong_sgd_fusion_h200_imported_2026-11-17.json",
        "local_fusion": "case_rag_strong_sgd_all_arms_local_2026-11-17.json",
        "import": "h200_archive_import_2026-11-17.json",
    }
    return {
        key: json.loads((ROOT / name).read_text(encoding="utf-8")) for key, name in names.items()
    }


def test_imported_raw_logits_recompute_identical_prior_scores(reports):
    old = reports["h200_raw"]
    new = reports["local_raw"]
    assert old["dev_cases"] == new["dev_cases"] == 5486
    assert old["official_test_used"] is new["official_test_used"] is False
    assert set(old["metrics"]) < set(new["metrics"])
    for key in old["metrics"]:
        assert old["metrics"][key] == new["metrics"][key]
    for key, value in old["paired_error_decomposition"].items():
        actual = new["paired_error_decomposition"][key]
        for field, expected in value.items():
            if field == "mcnemar_exact_p_unadjusted":
                assert actual[field] == pytest.approx(expected, abs=1e-13)
            else:
                assert actual[field] == expected
    assert {"s3_random2", "s3_global2_labelonly"} <= set(new["metrics"])


def test_imported_fusion_reports_are_self_consistent(reports):
    h200 = reports["h200_fusion"]
    local = reports["local_fusion"]
    assert h200["official_test_used"] is local["official_test_used"] is False
    assert h200["dev_cases"] == local["dev_cases"] == 5486
    assert h200["new_gpu_forwards"] == local["new_gpu_forwards"] == 0
    assert h200["methods"]["s2_global2"]["metrics"]["correct"] == 4680
    assert not h200["historical_s1_reproduction"]["matched"]
    assert local["historical_s1_reproduction"]["matched"]
    assert local["methods"]["s1_base"]["metrics"]["correct"] == 4667
    assert local["methods"]["s3_global2_labelonly"]["metrics"]["correct"] == 4676
    assert len(local["control_logit_sources"]) == 6
    assert len(local["methods"]) == 12


def test_import_manifest_tracks_only_completed_scores(reports):
    report = reports["import"]
    assert report["status"] == "H200_CASE_RAG_ARCHIVE_IMPORT_COMPLETED"
    assert report["official_test_used"] is False
    assert len(report["verified_frozen_inputs"]) == 5
    assert len(report["skipped_incomplete"]) == 2
    assert all(path.endswith(".partial") for path in report["skipped_incomplete"])
    assert len(report["imported"]) + len(report["identical"]) == 22
