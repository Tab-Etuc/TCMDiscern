"""CPU tests for atomic Jeff Case-RAG inference and partial-file resume."""
from __future__ import annotations

import json

import pytest

from tcmdiscern.case_rag_runtime import run_resumable


class StubDecider:
    def __init__(self, *, fail_after: int | None = None, shortened: bool = False):
        self.calls = 0
        self.fail_after = fail_after
        self.shortened = shortened
        self.last = {}

    def decide(self, row):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("Simulated H200 interruption")
        self.calls += 1
        self.last = {"shortened": self.shortened, "thought": None, "output_tokens": 0}
        return [1 / 16] * 16, [float(self.calls)] * 16, 42, None


def rows(n: int = 5):
    return [{"id": f"dev-{i:06d}"} for i in range(n)]


def test_run_recovers_valid_prefix_and_preserves_atomic_output(tmp_path):
    dest = tmp_path / "paired4.jsonl"
    before = StubDecider(fail_after=2)
    with pytest.raises(RuntimeError, match="interruption"):
        run_resumable(rows(), before, dest, progress_every=2)
    assert not dest.exists()
    partial = tmp_path / "paired4.jsonl.partial"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 2
    resumed = StubDecider()
    report = run_resumable(rows(), resumed, dest, progress_every=2)
    assert report["resumed_cases"] == 2
    assert report["new_forwards"] == 3
    assert report["new_input_tokens"] == 126
    assert resumed.calls == 3
    assert not partial.exists()
    assert [json.loads(s)["case_id"] for s in dest.read_text().splitlines()] == [
        x["id"] for x in rows()
    ]
    stable = StubDecider(fail_after=0)
    assert run_resumable(rows(), stable, dest)["new_forwards"] == 0
    assert stable.calls == 0


def test_partial_with_changed_case_id_refuses_resume(tmp_path):
    dest = tmp_path / "balanced4.jsonl"
    bad = tmp_path / "balanced4.jsonl.partial"
    bad.write_text(json.dumps({"case_id": "other", "logits": [1.0] * 16}) + "\n")
    with pytest.raises(ValueError, match="case IDs"):
        run_resumable(rows(), StubDecider(), dest)
    assert not dest.exists()


def test_interrupted_last_line_is_recovered(tmp_path):
    dest = tmp_path / "paired4.jsonl"
    partial = tmp_path / "paired4.jsonl.partial"
    good = json.dumps({"case_id": "dev-000000", "logits": [0.0] * 16})
    partial.write_bytes((good + "\n" + '{"case_id": "dev-000001"').encode("utf-8"))
    resumed = StubDecider()
    report = run_resumable(rows(3), resumed, dest)
    assert report["resumed_cases"] == 1
    assert resumed.calls == 2
    assert len(dest.read_text().splitlines()) == 3


def test_input_over_budget_does_not_publish_or_advance(tmp_path):
    dest = tmp_path / "global4.jsonl"
    decider = StubDecider()
    with pytest.raises(ValueError, match="context budget"):
        run_resumable(rows(1), decider, dest, max_tokens=41)
    assert not dest.exists()
    partial = tmp_path / "global4.jsonl.partial"
    assert partial.exists() and not partial.read_text(encoding="utf-8")


def test_shortened_input_is_rejected(tmp_path):
    dest = tmp_path / "random4.jsonl"
    with pytest.raises(ValueError, match="shortened"):
        run_resumable(rows(1), StubDecider(shortened=True), dest)
    assert not dest.exists()



def test_cpu_timing_sidecar_is_complete_and_resumable(tmp_path):
    dest = tmp_path / "paired2.jsonl"
    decider = StubDecider()
    result = run_resumable(rows(3), decider, dest)
    assert result["timing"]["latency_sample_n"] == 3
    assert result["timing"]["mean_input_tokens"] == 42
    assert result["timing"]["peak_cuda_allocated_bytes"] is None
    timing = tmp_path / "paired2.timings.jsonl"
    records = [json.loads(line) for line in timing.read_text().splitlines()]
    assert [x["case_id"] for x in records] == [x["id"] for x in rows(3)]
    assert all(x["wall_seconds"] >= 0 for x in records)
    assert run_resumable(rows(3), StubDecider(fail_after=0), dest)["new_forwards"] == 0


def test_unmatched_extra_logit_is_replayed_for_full_timing(tmp_path):
    dest = tmp_path / "paired2.jsonl"
    partial = tmp_path / "paired2.jsonl.partial"
    timing_partial = tmp_path / "paired2.timings.jsonl.partial"
    valid_score = lambda i: json.dumps({
        "case_id": f"dev-{i:06d}", "logits": [1.0] * 16
    }) + "\n"
    partial.write_text(valid_score(0) + valid_score(1), encoding="utf-8")
    timing_partial.write_text(json.dumps({
        "case_id": "dev-000000", "input_tokens": 19,
        "wall_seconds": 0.02, "peak_cuda_bytes": None,
    }) + "\n", encoding="utf-8")
    restarted = StubDecider()
    result = run_resumable(rows(3), restarted, dest)
    assert result["resumed_cases"] == 1
    assert restarted.calls == 2
    assert result["timing"]["latency_sample_n"] == 3
    assert len(dest.read_text().splitlines()) == 3


def test_legacy_partial_tracks_missing_old_latencies_without_faking_them(tmp_path):
    dest = tmp_path / "global2.jsonl"
    partial = tmp_path / "global2.jsonl.partial"
    partial.write_text(json.dumps({
        "case_id": "dev-000000", "logits": [1.0] * 16,
    }) + "\n", encoding="utf-8")
    result = run_resumable(rows(2), StubDecider(), dest)
    assert result["resumed_cases"] == 1
    assert result["timing"]["unmeasured_legacy_cases"] == 1
    assert result["timing"]["latency_sample_n"] == 1
    assert len((tmp_path / "global2.timings.jsonl").read_text().splitlines()) == 2


def test_corrupt_timing_source_case_fails_closed(tmp_path):
    dest = tmp_path / "paired2.jsonl"
    partial = tmp_path / "paired2.jsonl.partial"
    timing_partial = tmp_path / "paired2.timings.jsonl.partial"
    partial.write_text(json.dumps({
        "case_id": "dev-000000", "logits": [1.0] * 16,
    }) + "\n", encoding="utf-8")
    timing_partial.write_text(json.dumps({
        "case_id": "wrong", "input_tokens": 42,
        "wall_seconds": 0.01, "peak_cuda_bytes": None,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="case IDs"):
        run_resumable(rows(2), StubDecider(), dest)
    assert not dest.exists()

def test_completed_file_must_match_exact_row_order(tmp_path):
    dest = tmp_path / "baseline.jsonl"
    run_resumable(rows(2), StubDecider(), dest)
    with pytest.raises(ValueError, match="case IDs"):
        run_resumable(list(reversed(rows(2))), StubDecider(), dest)
