"""Jeff 1.1/SemIf format and label-leakage regression tests."""

from __future__ import annotations

import json
import random

import pytest

from tcmdiscern.jeff_format import (
    DIRECT_SYSTEM,
    LETTERS,
    TCM_QUESTION,
    decision_row,
    render_jeff_prompt,
    semif_messages,
    train_example,
)


@pytest.fixture
def sample_case() -> dict[str, str]:
    return {
        "chief_complaint": "咳嗽",
        "description": "胸悶痰多",
        "detection": "苔白膩、脈滑",
        "norm_syndrome": "痰濕蘊肺",
        "syndrome": "某原始證型",
        "lcd_id": "123",
        "lcd_name": "某診斷",
        "user_id": "private-user-id",
    }


def test_jeff_row_uses_only_patient_fields(sample_case: dict) -> None:
    row = decision_row(
        sample_case,
        case_id="dev-000000",
        candidates=("風寒襲肺", "痰濕蘊肺", "肺氣虛"),
    )
    assert row["state"] == {
        "chief_complaint": "咳嗽",
        "description": "胸悶痰多",
        "detection": "苔白膩、脈滑",
    }
    assert row["qtype"] == "choice"
    assert row["question"] == TCM_QUESTION
    assert row["options"] == [
        {"id": "風寒襲肺", "description": "風寒襲肺"},
        {"id": "痰濕蘊肺", "description": "痰濕蘊肺"},
        {"id": "肺氣虛", "description": "肺氣虛"},
    ]
    encoded = json.dumps(row, ensure_ascii=False)
    assert "private-user-id" not in encoded
    assert "某原始證型" not in encoded
    assert '"norm_syndrome"' not in encoded
    assert '"lcd_id"' not in encoded
    assert '"lcd_name"' not in encoded


def test_exact_semif_system_and_user_json(sample_case: dict) -> None:
    row = decision_row(sample_case, case_id="example", candidates=("痰濕蘊肺", "風寒襲肺"))
    messages = semif_messages(row)
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": DIRECT_SYSTEM}
    assert DIRECT_SYSTEM == (
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning."
    )
    expected_payload = {
        "evidence": {
            "chief_complaint": "咳嗽",
            "description": "胸悶痰多",
            "detection": "苔白膩、脈滑",
        },
        "criterion": TCM_QUESTION,
        "options": [
            {"letter": "A", "description": "痰濕蘊肺"},
            {"letter": "B", "description": "風寒襲肺"},
        ],
    }
    assert messages[1] == {
        "role": "user",
        "content": json.dumps(expected_payload, ensure_ascii=False),
    }
    assert LETTERS == "ABCDEFGHIJKLMNOP"


def test_prompt_uses_original_jeff_chat_template_flags(sample_case: dict) -> None:
    calls = []

    class StubTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return "templated"

    row = decision_row(sample_case, case_id="example", candidates=("A", "B"))
    assert render_jeff_prompt(StubTokenizer(), row) == "templated"
    assert calls[0][0] == semif_messages(row)
    assert calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_train_examples_include_gold_only_in_candidates_and_target(
    sample_case: dict,
) -> None:
    rng = random.Random(12)
    positions = []
    for _ in range(24):
        example = train_example(
            sample_case,
            case_id="train-000000",
            gold_label="痰濕蘊肺",
            hard_negatives=["風寒襲肺", "肺氣虛", "肺熱"],
            k=4,
            rng=rng,
        )
        row = example["row"]
        index = example["target_index"]
        assert row["options"][index]["id"] == "痰濕蘊肺"
        assert example["target_letter"] == LETTERS[index]
        assert len({option["id"] for option in row["options"]}) == 4
        assert "target_index" not in json.dumps(row, ensure_ascii=False)
        assert "target_letter" not in json.dumps(row, ensure_ascii=False)
        assert "norm_syndrome" not in row["state"]
        positions.append(index)
    assert len(set(positions)) > 1  # shuffled across draws/epochs


def test_training_requires_valid_hard_negatives(sample_case: dict) -> None:
    with pytest.raises(ValueError, match="gold"):
        train_example(
            sample_case,
            case_id="train-0",
            gold_label="痰濕蘊肺",
            hard_negatives=["痰濕蘊肺", "肺氣虛"],
            k=2,
            rng=random.Random(1),
        )
    with pytest.raises(ValueError, match="Insufficient"):
        train_example(
            sample_case,
            case_id="train-0",
            gold_label="痰濕蘊肺",
            hard_negatives=["肺氣虛"],
            k=5,
            rng=random.Random(1),
        )
    with pytest.raises(ValueError, match="2..16"):
        train_example(
            sample_case,
            case_id="train-0",
            gold_label="痰濕蘊肺",
            hard_negatives=["肺氣虛"],
            k=17,
            rng=random.Random(1),
        )


def test_k17_groups_only_in_jeff_inference(sample_case: dict) -> None:
    labels = [f"證型{i}" for i in range(17)]
    with pytest.raises(ValueError, match="2..16"):
        decision_row(sample_case, case_id="example", candidates=labels)
    row = decision_row(sample_case, case_id="example", candidates=labels, allow_grouped=True)
    assert len(row["options"]) == 17
    with pytest.raises(ValueError, match="2..16"):
        semif_messages(row)  # must go through official decide_wide


def test_no_gold_needed_for_real_dev_inference(sample_case: dict) -> None:
    without_gold = {
        key: sample_case[key] for key in ("chief_complaint", "description", "detection")
    }
    candidates = ["痰濕蘊肺", "風寒襲肺"]
    row1 = decision_row(sample_case, case_id="dev-1", candidates=candidates)
    row2 = decision_row(without_gold, case_id="dev-1", candidates=candidates)
    assert row1 == row2


def test_reject_empty_or_duplicate_candidates(sample_case: dict) -> None:
    with pytest.raises(ValueError, match="unique"):
        decision_row(sample_case, case_id="x", candidates=("肺虛", "肺虛"))
    with pytest.raises(ValueError, match="nonempty"):
        decision_row(sample_case, case_id="x", candidates=("", "肺虛"))
    with pytest.raises(ValueError, match="2..16"):
        decision_row(sample_case, case_id="x", candidates=("肺虛",))
