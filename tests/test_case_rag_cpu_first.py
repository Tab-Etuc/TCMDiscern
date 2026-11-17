"""CPU-only regression checks for frozen per-case Case-RAG analysis."""

from collections import Counter

import numpy as np
import pytest

from scripts.analyze_case_rag_cpu_first import (
    divergent_tail_choices,
    fixed_frequency_override_diagnostics,
    paired_errors,
    split_by_gold_frequency,
)


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
