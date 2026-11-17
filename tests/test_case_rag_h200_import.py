"""Archive import regression tests with synthetic, nonclinical records."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from scripts.import_case_rag_h200_archive import (
    SCORES_PREFIX,
    _is_importable,
    import_archive,
    validate_jsonl,
)


def test_import_allowlist_filters_partial_scores_and_code():
    assert _is_importable(SCORES_PREFIX + "paper-s2-full/global2.jsonl")
    assert _is_importable(SCORES_PREFIX + "paper-s1-full/run_manifest.json")
    assert _is_importable(SCORES_PREFIX + "paper-s3-full/global2.timings.jsonl")
    assert not _is_importable(SCORES_PREFIX + "paper-s1-full/global2.jsonl.partial")
    assert not _is_importable("data/tcm-sd/test.json")
    assert not _is_importable("scripts/evaluate_case_rag_strong_sgd_fusion_cpu.py")


def test_score_and_timing_jsonl_validate_without_reading_patient_text(tmp_path):
    path = tmp_path / "score.jsonl"
    cases = [{"case_id": f"dev-{i:06d}", "logits": [0.5] * 16} for i in range(2)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in cases),
        encoding="utf-8",
    )
    validate_jsonl(path, case_count=2)
    short = _write_rows(
        tmp_path / "score_short.jsonl",
        [{"case_id": "dev-000000", "logits": [0.5] * 15}],
    )
    with pytest.raises(ValueError, match="Invalid 16-way logits"):
        validate_jsonl(short, case_count=1)
    timing = _write_rows(
        tmp_path / "timings.jsonl",
        [{"case_id": f"dev-{i:06d}", "wall_seconds": 0.25} for i in range(2)],
    )
    validate_jsonl(timing, case_count=2, timing=True)
    with pytest.raises(ValueError, match="Invalid 16-way logits"):
        validate_jsonl(timing, case_count=2, timing=False)


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_jsonl_rejects_missing_and_misaligned_cases(tmp_path):
    path = _write_rows(
        tmp_path / "scores.jsonl",
        [{"case_id": "dev-000001", "logits": [1.0] * 16}],
    )
    with pytest.raises(ValueError, match="Case ID mismatch"):
        validate_jsonl(path, case_count=1)
    path = _write_rows(
        tmp_path / "scores.jsonl",
        [{"case_id": "dev-000000", "logits": [1.0] * 16}],
    )
    with pytest.raises(ValueError, match="Incomplete file"):
        validate_jsonl(path, case_count=2)


def test_malicious_archive_path_is_rejected_without_extraction(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    payload = b"should not be extracted"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("../outside")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        import_archive(archive, root=tmp_path / "project")
    assert not (tmp_path / "outside").exists()
