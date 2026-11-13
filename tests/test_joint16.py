"""TCM-SD K16 disease/pair preparation, choice, scoring and plan tests."""

from __future__ import annotations

import json

import pytest

from scripts import evaluate_joint16
from scripts.joint16_h200 import checkpoint, registered_runs, train_args
from scripts.select_joint16 import MODES, run_id, summarize
from tcmdiscern.disease_study import pair_key
from tcmdiscern.h200.plan import SEEDS
from tcmdiscern.jeff_epoch import reshuffle_training_example
from tcmdiscern.jeff_schedule import check_training_example
from tcmdiscern.joint_candidates import crossed_pairs, hierarchical_pairs, hybrid_pairs
from tcmdiscern.joint_study16 import K, eval_example, train_example


def case(disease="消渴", syndrome="氣陰兩虛"):
    return {
        "chief_complaint": "口渴",
        "description": "倦怠",
        "detection": "舌紅",
        "lcd_name": disease,
        "norm_syndrome": syndrome,
    }


def test_disease16_options_never_disclose_target():
    original = case()
    candidates = [f"中醫病名{i}" for i in range(K)]
    candidates[9] = "消渴"
    entry, in_top16 = train_example(original, index=0, oof_labels=candidates, mode="disease")
    assert in_top16 is True
    check_training_example(entry, row_index=0, multi=False, expected_k=K, study_mode="disease16")
    assert "lcd_name" not in entry["row"]["state"]
    assert entry["row"]["question"].endswith("中醫病名。")
    assert entry["row"]["options"][entry["target_index"]]["id"] == "消渴"
    after_shuffle = reshuffle_training_example(entry, epoch=1, seed=20261112)
    assert after_shuffle["row"]["options"][after_shuffle["target_index"]]["id"] == "消渴"


def test_pair16_options_show_separate_disease_and_syndrome():
    label = pair_key("消渴", "氣陰兩虛")
    ranked = [pair_key(f"病{i}", f"證{i}") for i in range(K + 5)]
    entry, in_top16 = train_example(case(), index=4, oof_labels=ranked, mode="pair")
    assert in_top16 is False
    check_training_example(entry, row_index=4, multi=False, expected_k=K, study_mode="pair16")
    assert entry["row"]["options"][entry["target_index"]] == {
        "id": label,
        "description": "中醫病名：消渴；證型：氣陰兩虛",
    }
    assert "norm_syndrome" not in entry["row"]["state"]
    row, meta = eval_example(case(), index=0, bm25_labels=ranked, mode="pair", split="dev")
    assert row["id"] == meta["case_id"] == "dev-000000"
    assert meta["gold"] not in meta["candidates"]
    assert meta["gold"] not in row["state"]
    assert "target_index" not in row


def test_hierarchical_cross_and_flat_keep_use_sixteen_unique_legal_pairs():
    diseases = [f"病{i}" for i in range(16)]
    syndromes = [f"證{i}" for i in range(16)]
    ranked = [pair_key(d, s) for i, d in enumerate(diseases) for s in syndromes]
    for d, s in ((1, 16), (2, 8), (4, 4), (8, 2), (16, 1)):
        options = hierarchical_pairs(diseases, ranked, disease_k=d, syndrome_k=s)
        assert len(set(options)) == K
        assert set(options) <= set(ranked)
    mixed = hybrid_pairs(diseases, ranked, flat_keep=8, disease_k=4, syndrome_k=2)
    assert len(set(mixed)) == K
    crossed = crossed_pairs(
        diseases,
        syndromes,
        ranked,
        allowed_pairs=frozenset(ranked),
        flat_keep=8,
    )
    assert len(set(crossed)) == K
    assert set(crossed) <= set(ranked)


def test_joint_scoring_is_full_row_and_checks_predictions(monkeypatch):
    monkeypatch.setattr(evaluate_joint16, "FULL_COUNT", 2)
    cases = [case("病甲", "證甲"), case("病乙", "證乙")]
    a = pair_key("病甲", "證甲")
    b = pair_key("病乙", "證乙")
    wrong = pair_key("病甲", "證乙")
    records = [
        {"gold": a, "candidates": [a, b, wrong, *[pair_key("病", str(i)) for i in range(13)]]},
        {"gold": b, "candidates": [wrong, b, a, *[pair_key("病", str(i)) for i in range(13)]]},
    ]
    scores = evaluate_joint16.score(
        train=cases,
        eval_cases=cases,
        metadata=records,
        predicted=[a, wrong],
        mode="pair",
    )
    assert scores["top1_accuracy"] == 0.5
    assert scores["disease_top1"] == 0.5
    assert scores["syndrome_top1"] == 1.0
    assert scores["oracle16"] == 1.0
    with pytest.raises(ValueError, match="sixteen candidates"):
        evaluate_joint16.score(
            train=cases,
            eval_cases=cases,
            metadata=records,
            predicted=[a, pair_key("未收錄", "新證")],
            mode="pair",
        )


def test_model_selection_uses_three_seeds_on_dev_and_fixed_test_seed(tmp_path):
    folder = tmp_path / "experiments"
    folder.mkdir()
    for mode in MODES:
        for epoch in (1, 2):
            for seed in SEEDS:
                name = f"{run_id(mode, seed)}-epoch{epoch}"
                data = {
                    "kind": "FULL_EVALUATION",
                    "split": "dev",
                    "mode": mode,
                    "run_id": name,
                    "attention_backend": "eager",
                    "scores": {
                        "n_cases": 5486,
                        "top1_accuracy": (0.5 + epoch * 0.1 + seed % 10 * 0.001),
                        "macro_f1": (0.2 + epoch * 0.05),
                    },
                }
                (folder / f"joint16_dev_{mode}_{name}.json").write_text(
                    json.dumps(data), encoding="utf-8"
                )
    for mode in MODES:
        (folder / f"joint16_dev_{mode}_zero.json").write_text(
            json.dumps(
                {
                    "kind": "FULL_EVALUATION",
                    "mode": mode,
                    "split": "dev",
                    "run_id": "zero",
                    "attention_backend": "eager",
                    "scores": {"n_cases": 5486},
                }
            ),
            encoding="utf-8",
        )
    selection = summarize(tmp_path)
    assert selection["status"] == "JOINT16_DEV_SELECTED_TEST_UNOPENED"
    assert all(selection["selected"][mode]["selected_epoch"] == 2 for mode in MODES)
    assert all(selection["selected"][mode]["fixed_test_seed"] == SEEDS[0] for mode in MODES)


def test_h200_expanded_joint_plan_has_six_extra_trainings():
    assert len(registered_runs()) == 6
    for mode, seed in registered_runs():
        args = train_args(mode, seed, checkpoint(mode, seed, 1).parent)
        assert args[args.index("--study-mode") + 1] == f"{mode}16"
        assert args[args.index("--steps") + 1] == "1350"
        assert args[args.index("--accumulation") + 1] == "64"
