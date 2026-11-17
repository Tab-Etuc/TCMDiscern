"""CPU-only regression checks for frozen per-case Case-RAG analysis."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_case_rag_cpu_first as analysis
from scripts.analyze_case_rag_cpu_first import (
    build_pairwise_comparisons,
    divergent_tail_choices,
    fixed_frequency_override_diagnostics,
    paired_errors,
    split_by_gold_frequency,
)


@pytest.mark.skipif(
    not (analysis.ROOT / "data/tcm-sd/train.json").is_file(),
    reason="Full-Dev original TCM-SD data unavailable in this checkout",
)
def test_full_dev_strict_global2_with_saved_logits_uses_both_oracle_keys(monkeypatch):
    """Reproduce the exact H200 branch without needing to copy patient data."""
    from tcmdiscern.case_rag_artifacts import frozen_paths
    from tcmdiscern.jeff_scoring import read_model_logits

    real_is_file = Path.is_file

    def simulate_synced_global2(path):
        if (
            path.name == "global2.jsonl"
            and path.parent.name.startswith("paper-")
            and "scores" in path.parts
        ):
            return True
        return real_is_file(path)

    def use_original_valid_logits(path, rows):
        method = path.parent.name.split("-")[1]
        _, original = frozen_paths(analysis.ROOT, method)
        return read_model_logits(original, num_cases=5486)

    monkeypatch.setattr(Path, "is_file", simulate_synced_global2)
    monkeypatch.setattr(analysis, "read_condition_logits", use_original_valid_logits)
    output = analysis.run(strict_global2=True)
    assert output["missing_case_rag_global2"] == []
    assert {"s1_global2", "s2_global2", "s3_global2"} <= set(output["case_rag_logits_available"])
    pairs = output["paired_error_decomposition"]
    for method in ("s1", "s2", "s3"):
        assert f"{method}_base_vs_bm25" in pairs
        assert f"{method}_global2_vs_bm25" in pairs
    assert output["oracle_ceilings_descriptive_only"]["bm25_or_s3_baseline_top1_percent"] == 88.4251
    assert output["oracle_ceilings_descriptive_only"]["bm25_or_s3_global2_top1_percent"] == 88.4251


def test_global2_presence_preserves_s3_baseline_pairs_and_oracle():
    gold = np.asarray(["a", "b", "c", "d"])
    bm25 = np.asarray(["a", "x", "c", "x"])
    source = np.asarray(["a", "b", "x", "x"])
    original = np.asarray(["a", "b", "x", "x"])
    global2 = np.asarray(["x", "b", "c", "d"])
    models = {"bm25": bm25, "source_label": source}
    for method in ("s1", "s2", "s3"):
        models[f"{method}_base"] = original
        models[f"{method}_global2"] = global2
    models["s2_global2_labelonly"] = np.asarray(["a", "b", "x", "x"])
    comparison = build_pairwise_comparisons(gold, models)
    assert len(comparison) == 16
    assert comparison["s3_base_vs_bm25"]["oracle_pair_top1_percent"] == 75.0
    assert comparison["s3_global2_vs_bm25"]["oracle_pair_top1_percent"] == 100.0
    assert comparison["s3_global2_vs_base"]["rescue"] == 2
    assert "s2_global2_vs_global2_labelonly" in comparison


def test_missing_global2_keeps_baseline_oracle_consistent():
    gold = np.asarray(["a", "b"])
    models = {
        "bm25": np.asarray(["a", "x"]),
        "source_label": np.asarray(["a", "b"]),
        **{f"{method}_base": np.asarray(["x", "b"]) for method in ("s1", "s2", "s3")},
    }
    comparison = build_pairwise_comparisons(gold, models)
    assert len(comparison) == 6
    assert comparison["s3_base_vs_bm25"]["oracle_pair_top1_percent"] == 100.0
    assert not any("global2" in key for key in comparison)


def test_paired_rescue_break_and_oracle():
    gold = np.asarray(["a", "b", "c", "d"])
    ref = np.asarray(["a", "x", "c", "x"])
    challenger = np.asarray(["a", "b", "x", "y"])
    report = paired_errors(gold, ref, challenger)
    assert report["both_correct"] == 1
    assert report["reference_only_correct"] == 1
    assert report["candidate_only_correct"] == 1
    assert report["both_wrong"] == 1
    assert report["rescue"] == report["break"] == 1
    assert report["oracle_pair_top1_percent"] == 75.0
    assert report["delta_top1_pp"] == 0.0
    assert report["mcnemar_exact_p_unadjusted"] == 1.0


def test_paired_shape_mismatch_fails():
    with pytest.raises(ValueError, match="identical shapes"):
        paired_errors(np.array(["a", "b"]), np.array(["a"]), np.array(["a", "b"]))


def test_gold_train_frequency_groups_are_predefined_and_exhaustive():
    gold = np.array(["a", "b", "c", "d"])
    counts = Counter({"a": 1, "b": 12, "c": 55, "d": 201})
    prediction = {"bm25": np.array(["a", "c", "c", "x"])}
    grouped = split_by_gold_frequency(gold, counts, prediction)
    assert [entry["dev_cases"] for entry in grouped] == [1, 1, 1, 1]
    assert [entry["top1_percent"]["bm25"] for entry in grouped] == [
        100.0,
        0.0,
        100.0,
        0.0,
    ]


def test_frequency_override_is_a_fixed_prediction_only_rule():
    gold = np.asarray(["rare", "head", "head"])
    models = {
        "s3_base": np.asarray(["head", "head", "head"]),
        "bm25": np.asarray(["rare", "rare", "head"]),
        "source_label": np.asarray(["rare", "rare", "head"]),
    }
    frequency = Counter({"rare": 5, "head": 300})
    results = fixed_frequency_override_diagnostics(
        gold,
        models,
        frequency,
        ["head", "rare"],
        {"rare"},
        np.ones(3, dtype=bool),
    )
    guarded = next(
        row
        for row in results
        if row["model"] == "s3_base"
        and row["alternative"] == "bm25"
        and row["train_frequency_cutoff"] == 9
    )
    assert guarded["switched_cases"] == 2
    assert guarded["rescued"] == 1
    assert guarded["broken"] == 1
    assert guarded["delta_top1_pp"] == 0.0


def test_source_tail_disagreement_does_not_infer_gold_from_tail_status():
    gold = np.array(["rare", "common", "common"])
    jeff = np.array(["common", "rare", "common"])
    source = np.array(["rare", "common", "rare"])
    comparison = divergent_tail_choices(gold, jeff, source, {"rare"})
    right = comparison["competitor_tail_jeff_not_tail"]
    assert right["n"] == 2
    assert right["competitor_correct"] == 1
    left = comparison["jeff_tail_competitor_not_tail"]
    assert left["n"] == 1
    assert left["jeff_correct"] == 0
