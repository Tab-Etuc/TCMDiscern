"""CPU-only multi-positive training and decision-loss mathematical regression."""

from __future__ import annotations

from random import Random

import numpy as np
import pytest

from tcmdiscern.datasets import clinical_text
from tcmdiscern.jeff_format import LETTERS, semif_messages
from tcmdiscern.jeff_losses import numpy_loss_gradient
from tcmdiscern.jeff_multi import build_multi_example, observed_label_sets


def fake_rows() -> list[dict[str, str]]:
    return [
        {
            "chief_complaint": "完全相同的主訴",
            "description": "相同的現病史",
            "detection": "相同的舌脈",
            "norm_syndrome": label,
        }
        for label in ("甲證", "乙證", "丙證", "甲證")
    ] + [
        {
            "chief_complaint": "另一個病人的主訴",
            "description": "另一段病情",
            "detection": "另一種舌脈",
            "norm_syndrome": "丁證",
        }
    ]


def rank50() -> list[str]:
    return ["甲證", "乙證", "丙證", *(f"BM25相似證型{i}" for i in range(47))]


def make_multi(index: int, *, epoch: int = 3) -> dict:
    rows = fake_rows()
    sets = observed_label_sets(rows)
    return build_multi_example(
        rows[index],
        case_id=f"train-{index:06d}",
        case_index=index,
        observed_labels=sets[clinical_text(rows[index])],
        ranked_candidates=rank50(),
        epoch=epoch,
        seed=20261111,
    )


def test_multiple_observed_labels_use_train_only_identical_evidence() -> None:
    rows = fake_rows()
    observed = observed_label_sets(rows)
    assert len(observed) == 2
    assert observed[clinical_text(rows[0])] == ("丙證", "乙證", "甲證") or observed[
        clinical_text(rows[0])
    ] == ("甲證", "丙證", "乙證")  # sorted Unicode code-point order
    assert set(observed[clinical_text(rows[0])]) == {"甲證", "乙證", "丙證"}
    assert observed[clinical_text(rows[4])] == ("丁證",)


def test_three_observed_positives_and_thirteen_negatives() -> None:
    item = make_multi(0)
    row = item["row"]
    labels = [option["id"] for option in row["options"]]
    assert len(labels) == len(set(labels)) == 16
    assert len(item["positive_indices"]) == 3
    assert {labels[i] for i in item["positive_indices"]} == {"甲證", "乙證", "丙證"}
    assert labels[item["target_index"]] == "甲證"
    assert item["target_letter"] == LETTERS[item["target_index"]]
    assert set(labels) == {"甲證", "乙證", "丙證", *(f"BM25相似證型{i}" for i in range(13))}
    assert sum(item["positive_mask"]) == 3
    assert sum(item["target_distribution"]) == 1.0
    assert sorted(set(item["target_distribution"])) == [0.0, 1 / 3]
    assert "norm_syndrome" not in row["state"]


def test_identical_evidence_different_gold_shares_positive_set() -> None:
    a, b, c = [make_multi(i) for i in range(3)]
    for item in (a, b, c):
        opts = [o["id"] for o in item["row"]["options"]]
        assert {opts[i] for i in item["positive_indices"]} == {"甲證", "乙證", "丙證"}
        assert len(item["positive_indices"]) == 3
        assert opts[item["target_index"]] in {"甲證", "乙證", "丙證"}
    assert a["row"]["state"] == b["row"]["state"] == c["row"]["state"]


def test_single_positive_reduces_to_one_hot() -> None:
    item = make_multi(4)
    assert len(item["positive_indices"]) == 1
    assert item["positive_indices"] == [item["target_index"]]
    assert item["target_distribution"][item["target_index"]] == 1.0
    assert sum(item["target_distribution"]) == 1


def test_prompt_does_not_see_target_position_or_distribution() -> None:
    obj = make_multi(0)
    prompt = semif_messages(obj["row"])[1]["content"]
    for key in (
        "target_index",
        "target_letter",
        "positive_indices",
        "positive_mask",
        "target_distribution",
        "norm_syndrome",
    ):
        assert key not in prompt


def test_shuffle_reproducible_per_epoch_and_changes_between_epochs() -> None:
    a = make_multi(0, epoch=3)
    same = make_multi(0, epoch=3)
    b = make_multi(0, epoch=4)
    assert a == same
    assert a["row"]["options"] != b["row"]["options"]
    assert {o["id"] for o in a["row"]["options"]} == {o["id"] for o in b["row"]["options"]}
    for item in (a, b):
        options = [o["id"] for o in item["row"]["options"]]
        assert options[item["target_index"]] == "甲證"
        assert {options[j] for j in item["positive_indices"]} == {"甲證", "乙證", "丙證"}


@pytest.mark.parametrize("m", [1, 2, 3])
def test_losses_and_exact_gradients_match_finite_difference(m: int) -> None:
    rng = np.random.default_rng(77 + m)
    logits = rng.normal(size=16)
    positive = list(range(m))
    gold = 0
    for method in ("s1", "s2", "s3", "s4"):
        if method == "s1" and m != 1:
            continue
        loss, grad = numpy_loss_gradient(
            logits, positive_indices=positive, gold_index=gold, method=method
        )
        assert np.isfinite(loss)
        assert grad.shape == (16,)
        assert abs(grad.sum()) < 1e-10
        assert loss >= -1e-12
        for index in range(16):
            plus = logits.copy()
            minus = logits.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            lplus, _ = numpy_loss_gradient(
                plus, positive_indices=positive, gold_index=gold, method=method
            )
            lminus, _ = numpy_loss_gradient(
                minus, positive_indices=positive, gold_index=gold, method=method
            )
            assert grad[index] == pytest.approx((lplus - lminus) / 2e-6, abs=1e-7)


def test_single_positive_all_four_losses_identical() -> None:
    logits = np.linspace(-4, 4, 16)
    vals = [
        numpy_loss_gradient(logits, positive_indices=[7], gold_index=7, method=method)
        for method in ("s1", "s2", "s3", "s4")
    ]
    for loss, grad in vals[1:]:
        assert loss == pytest.approx(vals[0][0], abs=1e-12)
        np.testing.assert_allclose(grad, vals[0][1], atol=1e-12)


def test_masked_ce_has_exact_zero_gradient_on_other_observed_gold() -> None:
    logits = np.linspace(-3, 3, 16)
    _, gradient = numpy_loss_gradient(
        logits, positive_indices=[2, 9, 12], gold_index=9, method="s2"
    )
    assert gradient[2] == gradient[12] == 0.0
    assert gradient[9] < 0.0


def test_set_loss_and_soft_ce_have_distinct_behavior() -> None:
    logits = np.full(16, -12.0)
    logits[0] = 12.0
    logits[1] = -2.0
    set_loss, _ = numpy_loss_gradient(logits, positive_indices=[0, 1], gold_index=0, method="s3")
    soft_loss, _ = numpy_loss_gradient(logits, positive_indices=[0, 1], gold_index=0, method="s4")
    assert set_loss < 0.001  # one observed label gets essentially all positive mass
    assert soft_loss > 1.0  # soft CE still pushes the other observed label up


@pytest.mark.parametrize("method", ["s2", "s3", "s4"])
def test_loss_invariant_to_synchronized_candidate_permutation(method: str) -> None:
    rng = Random(43)
    logits = np.linspace(-8, 7, 16)
    pos = (1, 5, 8)
    gold = 5
    perm = list(range(16))
    rng.shuffle(perm)
    shuffled_logits = logits[perm]
    shuffled_pos = [perm.index(i) for i in pos]
    shuffled_gold = perm.index(gold)
    original_loss, original_grad = numpy_loss_gradient(
        logits, positive_indices=pos, gold_index=gold, method=method
    )
    new_loss, new_grad = numpy_loss_gradient(
        shuffled_logits,
        positive_indices=shuffled_pos,
        gold_index=shuffled_gold,
        method=method,
    )
    assert new_loss == pytest.approx(original_loss)
    np.testing.assert_allclose(new_grad, original_grad[perm])


@pytest.mark.parametrize("method", ["s1", "s2", "s3", "s4"])
def test_extreme_logits_are_numerically_stable(method: str) -> None:
    z = np.linspace(-2000.0, 2000.0, 16)
    y = [0] if method == "s1" else [0, 8, 15]
    loss, grad = numpy_loss_gradient(z, positive_indices=y, gold_index=0, method=method)
    assert np.isfinite(loss)
    assert np.isfinite(grad).all()


def test_loss_rejects_invalid_targets() -> None:
    logits = np.zeros(16)
    kwargs = {"positive_indices": [1, 4], "gold_index": 1, "method": "s3"}
    for wrong in (
        {"positive_indices": []},
        {"positive_indices": [1, 1]},
        {"positive_indices": [1, 16]},
        {"positive_indices": [1, True]},
        {"gold_index": 2},
        {"gold_index": True},
        {"method": "unknown"},
    ):
        with pytest.raises(ValueError):
            numpy_loss_gradient(logits, **(kwargs | wrong))
    with pytest.raises(ValueError):
        numpy_loss_gradient(logits[:15], **kwargs)
    with pytest.raises(ValueError):
        numpy_loss_gradient(np.full(16, np.inf), **kwargs)
