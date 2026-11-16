"""Case-RAG retrieval invariants and Jeff baseline preservation."""
from collections import Counter

from tcmdiscern.case_rag import CONDITIONS, CaseRAGIndex, transitive_group_ids
from tcmdiscern.datasets import clinical_text
from tcmdiscern.jeff_format import decision_row, semif_messages


def sample(label, i, *, user=None):
    return {
        "chief_complaint": f"發熱咳嗽{i}日",
        "description": f"患者胸悶，症狀第{i}次，痰色不同",
        "detection": f"舌苔脈象{i}",
        "norm_syndrome": label,
        "user_id": user or f"p{i}",
        "lcd_name": "保密診斷",
    }


def index_fixture():
    labels = [f"證型{j}" for j in range(16)]
    cases = [sample(label, j*7+k) for j, label in enumerate(labels) for k in range(7)]
    return CaseRAGIndex(cases), labels


def test_all_conditions_preserve_original_options_and_patient_data():
    index, labels = index_fixture()
    query = sample("實際金標不應被使用", 1000, user="secret-identity")
    baseline = decision_row(query, case_id="dev-000123", candidates=labels)
    for condition in CONDITIONS:
        row, trace = index.make_row(
            query, labels, case_id="dev-000123", condition=condition, seed=20261116,
        )
        assert [entry["id"] for entry in row["options"]] == labels
        assert row["question"] == baseline["question"]
        assert row == baseline if condition == "baseline" else row != baseline
        assert "secret-identity" not in str(row)
        assert "保密診斷" not in str(row)
        assert "實際金標不應被使用" not in str(row)
        assert "norm_syndrome" not in str(row)
        assert len(trace["support_groups"]) == len(set(trace["support_groups"]))
        assert len(trace["support_groups"]) == (0 if condition == "baseline" else 4)
        assert len(semif_messages(row)) == 2


def test_paired_and_balanced_have_same_candidate_label_exposure():
    index, labels = index_fixture()
    query = sample("金標不可用", 900)
    for condition in ("balanced4", "paired4", "random4"):
        _, trace = index.make_row(query, labels, case_id="dev-0",
                                  condition=condition, seed=123)
        assert Counter(trace["support_labels"]) == Counter({labels[0]: 2, labels[1]: 2})
        assert len(set(trace["source_train_positions"])) == 4


def test_source_gold_field_cannot_change_retrieval():
    index, labels = index_fixture()
    query_a = sample("金標甲", 50_000)
    query_b = dict(query_a, norm_syndrome="金標乙")
    for condition in CONDITIONS:
        row_a, trace_a = index.make_row(
            query_a, labels, case_id="dev-1", condition=condition, seed=7
        )
        row_b, trace_b = index.make_row(
            query_b, labels, case_id="dev-1", condition=condition, seed=7
        )
        assert row_a == row_b
        assert trace_a == trace_b




def test_shared_bm25_ranking_and_supports_match_separate_passes():
    index, labels = index_fixture()
    query = sample("隱藏的查詢正解", 900)
    independent = index.model.rank([clinical_text(query)], max_k=16)[0]
    combined, pool = index.rank_and_supports(query, max_k=16)
    assert combined == independent
    assert set(combined) == set(labels)
    assert pool == index._supports(query, combined)


def test_same_patient_and_same_text_are_excluded_from_supports():
    index, labels = index_fixture()
    stored = index.rows[0]
    copy = dict(stored, norm_syndrome="診斷測試改標")
    changed_text = dict(stored, description="完全不同的病況描述")
    for query in (copy, changed_text):
        _, metadata = index.make_row(
            query, labels, case_id="dev-100", condition="paired4", seed=19
        )
        assert index.groups[0] not in metadata["support_groups"]


def test_blank_patient_identifier_not_a_shared_group():
    rows = [
        sample("A", 12, user=""),
        sample("A", 13, user=""),
    ]
    for item in rows:
        item["user_id"] = ""
    group_ids = transitive_group_ids(rows)
    assert group_ids[0] != group_ids[1]


def test_patient_text_links_and_ambiguous_groups():
    rows = [
        sample("A", 1, user="one"),
        sample("A", 2, user="one"),
        sample("A", 2, user="two"),
        sample("B", 4, user="different"),
    ]
    groups = transitive_group_ids(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
