"""Zero-shot Jeff runner smoke tests without loading any model weights."""

from __future__ import annotations

import json

import pytest

from scripts.run_jeff_zeroshot import iter_dev_rows, run_decider
from tcmdiscern.evaluation import Prediction
from tcmdiscern.jeff_format import TCM_QUESTION


def fake_candidates() -> list[Prediction]:
    return [
        Prediction(
            case_id=f"dev-{i:06d}",
            gold_labels=("原始官方標籤",),
            candidate_labels=tuple(f"候選{j}" for j in range(50)),
        )
        for i in range(5486)
    ]


def fake_row(i: int) -> dict:
    return {
        "id": f"dev-{i:06d}",
        "state": {"chief_complaint": "主訴", "description": "病況", "detection": "診察"},
        "question": TCM_QUESTION,
        "options": [{"id": f"候選{j}", "description": f"候選{j}"} for j in range(16)],
        "qtype": "choice",
    }


def test_complete_dev_mapping_and_no_gold_in_model_prompt(tmp_path) -> None:
    source = tmp_path / "dev.jsonl"
    with source.open("w", encoding="utf-8") as stream:
        for i in range(5486):
            stream.write(json.dumps(fake_row(i), ensure_ascii=False) + "\n")
    rows = list(iter_dev_rows(source, fake_candidates(), limit=2))
    assert len(rows) == 2
    for row in rows:
        assert set(row) == {"id", "state", "question", "options", "qtype"}
        assert "原始官方標籤" not in json.dumps(row, ensure_ascii=False)


def test_dev_mapping_rejects_missing_or_reordered_candidates(tmp_path) -> None:
    source = tmp_path / "incomplete.jsonl"
    source.write_text(json.dumps(fake_row(0), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        list(iter_dev_rows(source, fake_candidates(), limit=1))

    source = tmp_path / "wrong.jsonl"
    wrong = fake_row(0)
    wrong["options"][0]["id"] = "不在BM25"
    source.write_text(json.dumps(wrong, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="out of order"):
        list(iter_dev_rows(source, fake_candidates(), limit=1))


class FakeDecider:
    def __init__(self, *, shortened: bool = False):
        self.shortened = shortened
        self.last = {}

    def decide(self, row: dict):
        self.last = {"shortened": self.shortened, "thought": False, "output_tokens": 0}
        return [1 / 16] * 16, [float(i) for i in range(16)], 300, 0.002


def test_raw_logits_are_written_by_id_and_not_overwritten(tmp_path) -> None:
    out = tmp_path / "logits.jsonl"
    total, tokens, elapsed = run_decider([fake_row(0), fake_row(1)], FakeDecider(), out)
    assert (total, tokens) == (2, 600)
    assert elapsed >= 0
    saved = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert saved[0] == {"case_id": "dev-000000", "logits": [float(i) for i in range(16)]}
    assert saved[1]["case_id"] == "dev-000001"
    assert "主訴" not in out.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_decider([fake_row(0)], FakeDecider(), out)


def test_real_runner_rejects_unexpected_prompt_condensation(tmp_path) -> None:
    out = tmp_path / "new.jsonl"
    with pytest.raises(ValueError, match="shortened"):
        run_decider([fake_row(0)], FakeDecider(shortened=True), out)
    assert not out.exists()
