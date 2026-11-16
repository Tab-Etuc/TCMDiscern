"""Regression tests for full-Dev no-wasted-weights preflight and resume."""

from __future__ import annotations

import json

import pytest

from scripts import run_case_rag_paper_h200 as paper


def test_preflight_refuses_corrupted_partial_logits_before_gpu(tmp_path):
    rows = [{"id": "dev-000000"}, {"id": "dev-000001"}]
    file = tmp_path / "global2.jsonl.partial"
    file.write_text(
        json.dumps({"case_id": "dev-999999", "logits": [0.0] * 16}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case"):
        paper._saved_progress(tmp_path, "global2", rows)


def test_preflight_accepts_crash_tail_and_reports_true_resume_count(tmp_path):
    rows = [{"id": f"dev-{i:06d}"} for i in range(3)]
    scores = tmp_path / "global2.jsonl.partial"
    timings = tmp_path / "global2.timings.jsonl.partial"
    scores.write_bytes(
        (json.dumps({"case_id": "dev-000000", "logits": [0.0] * 16}) + "\n").encode()
        + b'{"case_id": "dev-000001", "logits": '
    )
    timings.write_bytes(
        (
            json.dumps(
                {
                    "case_id": "dev-000000",
                    "input_tokens": 900,
                    "wall_seconds": 0.2,
                    "peak_cuda_bytes": 0,
                }
            )
            + "\n"
        ).encode()
        + b'{"case_id": "dev-000001", '
    )
    assert paper._saved_progress(tmp_path, "global2", rows) == 1
    assert scores.exists() and timings.exists()


def test_preflight_rejects_orphan_partial_timings(tmp_path):
    (tmp_path / "global2.timings.jsonl.partial").write_text(
        '{"case_id":"dev-000000"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Orphan"):
        paper._saved_progress(tmp_path, "global2", [{"id": "dev-000000"}])


def test_completed_logits_with_partial_timing_blocks_gpu_before_load(tmp_path):
    row = {"id": "dev-000000"}
    (tmp_path / "global2.jsonl").write_text(
        json.dumps({"case_id": row["id"], "logits": [0.0] * 16}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "global2.timings.jsonl.partial").write_text(
        '{"case_id":"dev-000000","input_tokens":900',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Incomplete timing"):
        paper._saved_progress(tmp_path, "global2", [row])


@pytest.mark.parametrize("bad_tokens", [2, 8193])
def test_true_token_preflight_refuses_invalid_token_count(tmp_path, monkeypatch, bad_tokens):
    from tcmdiscern import case_rag_tokenizer as token_helpers

    folder = tmp_path / "data/case_rag" / paper.TAG
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "samples": 5486,
                "test_used": False,
                "input_tokens": {
                    name: {"minimum": 100, "maximum": 100, "mean": 100}
                    for name in ("baseline", "global2", "random2")
                },
            }
        ),
        encoding="utf-8",
    )
    (folder / "paper_ablations.json").write_text(
        json.dumps(
            {
                "samples": 5486,
                "test_used": False,
                "input_tokens": {
                    name: {"minimum": 100, "maximum": 100, "mean": 100} for name in paper.ARMS
                },
            }
        ),
        encoding="utf-8",
    )
    original = tmp_path / "data/jeff/dev_k16.jsonl"
    original.parent.mkdir(parents=True)
    cases = [{"id": f"dev-{index:06d}"} for index in range(5486)]
    with original.open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case) + "\n")
    monkeypatch.setattr(
        token_helpers,
        "load_jeff_token_counter",
        lambda *args, **kwargs: (object(), tmp_path / "tokenizer"),
    )
    monkeypatch.setattr(
        token_helpers,
        "count_jeff_prompt_tokens",
        lambda *args: bad_tokens,
    )
    collection = {name: cases for name in ("baseline", "global2", "random2", *paper.ARMS)}
    with pytest.raises(ValueError, match="tokens"):
        paper._verify_frozen_prompts(tmp_path, tmp_path / "model", collection)
