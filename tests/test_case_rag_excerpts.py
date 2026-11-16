"""Clinical evidence excerpts must be deterministic, extractive, and gold-blind."""
from __future__ import annotations

from tcmdiscern.case_rag import CASE_CHARS
from tcmdiscern.case_rag_excerpts import (
    focused_case_view,
    focused_window,
    verify_extractive,
)
from tcmdiscern.retrieval import chinese_bigrams


def test_query_focused_window_recovers_relevant_late_source():
    original = "無明顯症狀，" * 25 + "舌紅苔黃口苦胸脅脹痛" + "食欲不振。" * 15
    terms = set(chinese_bigrams("舌紅苔黃口苦胸脅脹痛"))
    selected = focused_window(original, terms, 70)
    assert "舌紅苔黃口苦胸脅脹痛" in selected
    assert len(selected) <= 70
    assert selected.strip("…") in original


def test_focused_case_preserves_original_field_budget_and_no_gold_used():
    clinical = {
        "chief_complaint": "腹痛",
        "description": "一般病史" * 50 + "口乾煩渴腹痛",
        "detection": "舌淡脈沉" * 25 + "舌紅苔黃",
        "norm_syndrome": "金標絕不能進入摘要",
        "user_id": "secret-user",
    }
    query = {
        "chief_complaint": "腹痛",
        "description": "煩渴口乾",
        "detection": "舌紅苔黃",
        "norm_syndrome": "另一種金標",
    }
    selected = focused_case_view(clinical, query)
    verify_extractive(clinical, selected)
    assert len(selected["description"]) <= CASE_CHARS["description"]
    assert len(selected["detection"]) <= CASE_CHARS["detection"]
    assert "金標" not in str(selected)
    assert "secret-user" not in str(selected)
    assert "舌紅苔黃" in selected["detection"]
    assert selected == focused_case_view(clinical, {**query, "norm_syndrome": "改變金標"})


def test_short_case_remains_identical_and_empty_fields_supported():
    source = {"chief_complaint": "頭痛", "description": "", "detection": "脈浮"}
    assert focused_case_view(source, {"chief_complaint": "頭痛"}) == source


def test_excerpt_does_not_cut_immediately_after_negation():
    text = "病史正常。" * 25 + "不痛且無口渴" + "眠差。" * 12
    excerpt = focused_window(text, set(chinese_bigrams("痛口渴")), 30)
    assert not excerpt.lstrip("…").startswith("痛")
    assert excerpt.strip("…") in text


def test_invalid_budget_raises():
    import pytest

    with pytest.raises(ValueError, match="budget"):
        focused_window("very long source", set(), 4)
