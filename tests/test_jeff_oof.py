"""Train-only five-fold retrieval and Jeff supervision regression tests."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from scripts.export_jeff_train import make_train_example, write_examples
from scripts.mine_jeff_train_oof import cross_fold_exact_text_counts
from tcmdiscern.oof import (
    assign_grouped_folds,
    assign_stratified_folds,
    rank_statistics,
    rank_train_oof,
)


def fake_train(n_per_label: int = 6) -> tuple[list[str], list[str]]:
    texts = []
    labels = []
    for label in ("氣滯", "血瘀", "痰濕"):
        for i in range(n_per_label):
            texts.append(f"症狀{label}第{i}例")
            labels.append(label)
    return texts, labels


def test_five_fold_partition_covers_all_rows_without_duplicating_indices() -> None:
    _, labels = fake_train()
    result = assign_stratified_folds(labels, folds=5, seed=12)
    assert len(result) == len(labels)
    assert set(result) == {0, 1, 2, 3, 4}
    assert sum(Counter(result).values()) == 18
    # Stratification gives at most one-row difference between fold sizes.
    assert max(Counter(result).values()) - min(Counter(result).values()) <= 1
    assert result == assign_stratified_folds(labels, folds=5, seed=12)


def test_oof_never_searches_a_train_case_against_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    texts, labels = fake_train()
    calls = []

    class SpyBM25:
        def __init__(self, index_texts, index_labels, *, min_df):
            assert min_df == 1
            self.docs = set(index_texts)
            self.label_names = sorted(set(index_labels))
            calls.append(("index", len(index_texts)))

        def rank(self, query_texts, *, max_k, batch_size):
            assert batch_size == 4
            assert all(query not in self.docs for query in query_texts)
            calls.append(("query", len(query_texts)))
            return [self.label_names[:max_k] for _ in query_texts]

    monkeypatch.setattr("tcmdiscern.oof.CaseBM25", SpyBM25)
    oof = rank_train_oof(texts, labels, folds=5, seed=12, max_k=3, batch_size=4, min_df=1)
    assert len(oof.rankings) == len(texts)
    assert len(oof.fold_seconds) == 5
    assert len([item for item in calls if item[0] == "index"]) == 5
    assert sum(item[1] for item in calls if item[0] == "query") == len(texts)
    assert all(len(options) == 3 for options in oof.rankings)
    report = rank_statistics(labels, oof, ks=(1, 2, 3))
    assert report["total_train_rows"] == 18
    assert report["original_rows_deleted"] == 0
    assert report["bm25_oof_candidate_micro_recall"]["3"] == 1.0


def test_real_bm25_five_fold_ranking_retains_all_rows() -> None:
    texts, labels = fake_train()
    result = rank_train_oof(texts, labels, folds=5, seed=12, max_k=3, min_df=1, batch_size=4)
    assert len(result.rankings) == len(labels)
    assert len(result.fold_train_sizes) == len(result.fold_query_sizes) == 5
    assert all(len(result.rankings[i]) == 3 for i in range(len(labels)))


def test_rare_class_guard() -> None:
    with pytest.raises(ValueError, match="Rarest"):
        assign_stratified_folds(["A"] * 10 + ["B"] * 4, folds=5)


def test_exact_duplicates_are_audited_but_never_dropped() -> None:
    texts = ["症狀甲", "症狀甲", "症狀乙", "症狀乙"]
    labels = ["A", "B", "A", "A"]
    folds = (0, 1, 0, 1)
    report = cross_fold_exact_text_counts(texts, labels, folds)
    assert report["query_rows_with_identical_text_in_another_fold"] == 4
    assert report["such_rows_without_same_label_in_other_fold"] == 2
    assert len(texts) == 4  # No modification to original rows.


def test_k5_and_k16_training_rows_do_not_reveal_gold_in_evidence(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.export_jeff_train as exporter

    monkeypatch.setattr(exporter, "TRAIN_ROWS", 24)
    cases = [
        {
            "norm_syndrome": f"證型{i % 3}",
            "chief_complaint": f"主訴{i}",
            "description": "",
            "detection": "",
            "user_id": "private",
            "lcd_id": "secret",
        }
        for i in range(24)
    ]
    ranks = [[f"證型{j}" for j in range(30)] for _ in cases]
    p5 = tmp_path / "train_k05.jsonl"
    p16 = tmp_path / "train_k16.jsonl"
    for k, path in ((5, p5), (16, p16)):
        summary = write_examples(cases, ranks, path=path, k=k, epoch=0, seed=100)
        assert summary["cases"] == 24
        assert sum(summary["k_counts"].values()) == 24
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 24
        for i, item in enumerate(rows):
            assert item["row"]["id"] == f"train-{i:06d}"
            assert len(item["row"]["options"]) == k
            assert item["row"]["options"][item["target_index"]]["id"] == cases[i]["norm_syndrome"]
            assert "norm_syndrome" not in item["row"]["state"]
            assert "private" not in str(item["row"])
            assert "secret" not in str(item["row"])
            assert "target_index" not in str(item["row"])


def test_recreated_epoch_shuffles_train_options() -> None:
    case = {
        "norm_syndrome": "正解",
        "chief_complaint": "真實主訴",
        "description": "症狀",
        "detection": "舌脈",
    }
    options = ["錯誤" + str(i) for i in range(30)]
    a = make_train_example(
        case, train_index=0, candidates=options, gold="正解", k=16, epoch=0, seed=20
    )
    b = make_train_example(
        case, train_index=0, candidates=options, gold="正解", k=16, epoch=1, seed=20
    )
    again = make_train_example(
        case, train_index=0, candidates=options, gold="正解", k=16, epoch=0, seed=20
    )
    assert a == again
    assert a["row"]["options"] != b["row"]["options"]
    assert len({item["id"] for item in a["row"]["options"]}) == 16
    assert a["row"]["options"][a["target_index"]]["id"] == "正解"


def test_exact_text_groups_prevent_crossfold_self_match_without_dedup() -> None:
    texts, labels = fake_train(n_per_label=10)
    texts[1] = texts[0]
    texts[11] = texts[10]
    folds = assign_grouped_folds(labels, texts, folds=5, seed=7)
    assert len(folds) == 30
    assert folds[1] == folds[0]
    assert folds[11] == folds[10]
    assert set(folds) == set(range(5))
    assert sum(Counter(folds).values()) == 30
    assert cross_fold_exact_text_counts(texts, labels, folds) == {
        "query_rows_with_identical_text_in_another_fold": 0,
        "such_rows_without_same_label_in_other_fold": 0,
    }
    result = rank_train_oof(
        texts,
        labels,
        folds=5,
        seed=7,
        max_k=3,
        min_df=1,
        fold_policy="exact_text_grouped",
    )
    assert len(result.rankings) == len(texts)
    assert sum(result.fold_query_sizes) == len(texts)
    assert all(len(set(rank)) == 3 for rank in result.rankings)


def test_grouped_oof_requires_valid_text_and_policy() -> None:
    texts, labels = fake_train(n_per_label=10)
    with pytest.raises(ValueError, match="Aligned"):
        assign_grouped_folds(labels, texts[:-1], folds=5)
    with pytest.raises(ValueError, match="Unknown OOF"):
        rank_train_oof(texts, labels, min_df=1, max_k=3, fold_policy="unknown")


def test_mixed_epoch_contains_varied_k(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.export_jeff_train as exporter

    monkeypatch.setattr(exporter, "TRAIN_ROWS", 30)
    cases = [{"norm_syndrome": "真", "chief_complaint": f"病情{i}"} for i in range(30)]
    rank = ["負例" + str(i) for i in range(50)]
    result = write_examples(
        cases,
        [rank] * 30,
        path=tmp_path / "mixed.jsonl",
        k=None,
        epoch=0,
        seed=42,
    )
    assert len(result["k_counts"]) > 1
    assert sum(result["k_counts"].values()) == 30
    assert result["cases_removed"] == 0
