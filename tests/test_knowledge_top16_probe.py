"""Knowledge BM25 experiments may reorder but never replace frozen Jeff candidates."""

from __future__ import annotations

import pytest

from scripts.probe_syndrome_knowledge_top16 import label_knowledge
from tcmdiscern.knowledge_rerank import rerank_fixed_candidates


def test_fixed_candidates_preserved_without_gold_input():
    original = ("甲證", "乙證", "丙證", "丁證")
    knowledge = ("丙證", "乙證", "甲證", "丁證")
    ranked = rerank_fixed_candidates(original, knowledge, k=3)
    assert ranked["bm25"] == ["甲證", "乙證", "丙證"]
    assert ranked["knowledge_within_k"] == ["丙證", "乙證", "甲證"]
    assert all(set(choice) == {"甲證", "乙證", "丙證"} for choice in ranked.values())
    assert all(len(choice) == 3 for choice in ranked.values())


def test_reranker_refuses_missing_or_duplicated_labels():
    with pytest.raises(ValueError, match="cover every frozen"):
        rerank_fixed_candidates(["a", "b"], ["a", "c"], k=2)
    with pytest.raises(ValueError, match="duplicate"):
        rerank_fixed_candidates(["a", "a"], ["a", "b"], k=2)
    with pytest.raises(ValueError, match="Invalid"):
        rerank_fixed_candidates(["a", "b"], ["a", "b"], k=3)


def test_syndrome_knowledge_requires_every_official_label_once():
    descriptions = [
        {"Name": "甲證", "Definition": "甲知識", "Typical_performance": ""},
        {"Name": "乙證", "Definition": "乙知識", "Typical_performance": "乙表現"},
        {"Name": "其他", "Definition": "第三個條目", "Typical_performance": "無關"},
    ]
    actual = label_knowledge({"甲證", "乙證"}, descriptions)
    assert actual == {"甲證": "甲證 甲知識", "乙證": "乙證 乙知識 乙表現"}
    with pytest.raises(ValueError, match="Duplicated"):
        label_knowledge({"甲證", "乙證"}, [*descriptions, descriptions[0]])
    with pytest.raises(ValueError, match="differ"):
        label_knowledge({"甲證", "乙證", "丙證"}, descriptions)
