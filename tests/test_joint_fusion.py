"""K16 factorized independent Jeff scores and Train-compatible pair constraints."""
from __future__ import annotations

import json

import pytest

from scripts.fuse_joint16_logits import read_candidates, read_logits, score_fusion
from tcmdiscern.disease_study import pair_key
from tcmdiscern.joint_fusion import decode, factorized_oracles, log_softmax


def candidates(prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(16)]


def record(disease: str, syndrome: str) -> dict:
    return {"lcd_name": disease, "norm_syndrome": syndrome}


def test_decode_two_marginals_and_train_constraint():
    d = candidates("病")
    s = candidates("證")
    logits_d = [10.0, 9.0, *[0.0] * 14]
    logits_s = [10.0, 9.0, *[0.0] * 14]
    independent, fallback = decode(d, s, logits_d, logits_s)
    assert independent == pair_key(d[0], s[0])
    assert not fallback
    known = frozenset([pair_key(d[1], s[1])])
    restricted, fallback = decode(d, s, logits_d, logits_s, known_pairs=known)
    assert restricted == pair_key(d[1], s[1])
    assert not fallback
    unseen, fallback = decode(d, s, logits_d, logits_s, known_pairs=frozenset())
    assert unseen == independent
    assert fallback


def test_oracle_compares_both_label_sets_and_seen_pair_condition():
    d = candidates("病")
    s = candidates("證")
    gold = [pair_key(d[3], s[4]), pair_key("不在候選", s[1])]
    known = frozenset([gold[0]])
    result = factorized_oracles(
        gold, [d, d], [s, s], known_pairs=known
    )
    assert result["factorized_oracle16x16"] == 0.5
    assert result["train_pair_constrained_oracle16x16"] == 0.5
    assert result["unseen_pairs"] == 1


def test_normalization_and_unseen_pair_rate_in_aggregate():
    train = [record("病A", "證A"), record("病B", "證B")]
    cases = [record("病A ", "證A"), record("病B", "證B")]
    predictions = [pair_key("病A", "證A"), pair_key("病A", "證B")]
    result = score_fusion(train, cases, predictions, fallback_count=0, constrained=False)
    assert result["label_classes"] == 2
    assert result["pair_top1"] == 0.5
    assert result["predicted_train_unseen_pair_count"] == 1
    assert result["gold_train_unseen_pair_count"] == 0
    assert result["disease_top1"] == 0.5


def test_logits_and_candidates_require_exact_official_row_order(tmp_path):
    bad_logits = tmp_path / "logits.jsonl"
    bad_logits.write_text(
        json.dumps({"case_id": "dev-000001", "logits": [0] * 16}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unaligned"):
        read_logits(bad_logits, n=1, split="dev")
    good_logits = tmp_path / "logits-good.jsonl"
    good_logits.write_text(
        json.dumps({"case_id": "dev-000000", "logits": [0] * 16}) + "\n",
        encoding="utf-8",
    )
    assert read_logits(good_logits, n=1, split="dev") == [[0.0] * 16]
    cand = tmp_path / "cand.jsonl"
    cand.write_text(
        json.dumps({"case_id": "dev-000000", "candidates": candidates("病")}) + "\n",
        encoding="utf-8",
    )
    assert read_candidates(cand, n=1, split="dev", disease=True)[0] == candidates("病")


def test_invalid_inputs_rejected():
    d = candidates("病")
    s = candidates("證")
    with pytest.raises(ValueError, match="finite"):
        log_softmax([float("nan"), 1.0])
    with pytest.raises(ValueError, match="Blend"):
        decode(d, s, [0.0] * 16, [0.0] * 16, disease_weight=1.0)
