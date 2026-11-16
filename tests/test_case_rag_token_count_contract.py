"""Regression for Transformers 5.x BatchEncoding input_ids counting."""

from __future__ import annotations

import pytest

from tcmdiscern.case_rag_tokenizer import (
    _TokenCounterCompatibility,
    count_jeff_prompt_tokens,
)

ROW = {
    "id": "dev-000001",
    "state": {"chief_complaint": "頭暈", "description": "頭痛三日"},
    "question": "辨證",
    "options": [
        {"id": "A", "description": "肝陽上亢證"},
        {"id": "B", "description": "氣血兩虛證"},
    ],
}


class Stub:
    def __init__(self, mode: str):
        self.mode = mode

    def apply_chat_template(self, *_args, **_kwargs):
        ids = list(range(120))
        if self.mode == "dict":
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.mode == "nested":
            return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        if self.mode == "tiny":
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}
        return ids


@pytest.mark.parametrize("mode", ["dict", "nested", "list"])
def test_true_input_ids_length(mode):
    codec = _TokenCounterCompatibility(Stub(mode))
    assert count_jeff_prompt_tokens(codec, ROW) == 120
    assert len(codec.apply_chat_template([], tokenize=True)) == 120


def test_rejects_two_token_object():
    with pytest.raises(ValueError, match="invalid token sequence"):
        count_jeff_prompt_tokens(_TokenCounterCompatibility(Stub("tiny")), ROW)
