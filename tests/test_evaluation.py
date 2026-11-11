import pytest

from tcmdiscern.evaluation import Prediction, evaluate


def case(id_, gold, candidates, prediction=None):
    return Prediction.from_mapping(
        {
            "case_id": id_,
            "gold_labels": gold,
            "candidate_labels": candidates,
            "predicted_label": prediction,
        }
    )


def test_recall_any_and_top1():
    rows = [
        case("1", ["A"], ["B", "A"], "A"),
        case("2", ["C"], ["D", "E"], "D"),
        case("3", ["X", "Y"], ["Y", "Z"], "Y"),
    ]
    result = evaluate(rows, [1, 2])
    assert result["recall_any_at_k"] == {"1": 1 / 3, "2": 2 / 3}
    assert result["top1_accuracy"] == 2 / 3
    assert "1" not in result["conditional_accuracy_at_k"]
    assert result["conditional_accuracy_at_k"]["2"] == 1.0


def test_reject_duplicate_candidates():
    with pytest.raises(ValueError, match="duplicate candidate"):
        case("1", ["A"], ["A", "A"])


def test_reject_gold_outside_prediction():
    with pytest.raises(ValueError, match="must be a candidate"):
        case("1", ["A"], ["B"], "A")


def test_reject_duplicate_case_ids():
    rows = [case("1", ["A"], ["A"]), case("1", ["B"], ["B"])]
    with pytest.raises(ValueError, match="Duplicate case_id"):
        evaluate(rows)


def test_reject_mixed_prediction_coverage():
    rows = [case("1", ["A"], ["A"], "A"), case("2", ["B"], ["B"])]
    with pytest.raises(ValueError, match="All or none"):
        evaluate(rows)


def test_reject_invalid_ks():
    with pytest.raises(ValueError, match="positive integers"):
        evaluate([case("1", ["A"], ["A"])], [0])
