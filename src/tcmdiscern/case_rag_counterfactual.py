"""Counterfactual case-label attribution controls for retrieval-augmented diagnosis.

The controls reuse frozen clinical evidence from the official Train. Only the
retrieved-case labels/order/instruction change; patient evidence, fixed 16
candidate options and case text stay identical.

Label-swap intentionally falsifies *historical source labels* for a stress
test only; it must never be used for actual clinical inference.
"""
from __future__ import annotations

from copy import deepcopy

ARMS = ("baseline", "paired2", "labelswap2", "labelhide2", "orderflip2", "doubleflip2", "sourceguard2")
HIDDEN_LABEL = "未提供"
SOURCE_GUARD = (
    "以下是其他患者的歷史醫案，僅供辨認各證型的鑑別特徵。"
    "歷史醫案中的症狀不能視為本患者已觀察到的症狀；"
    "請優先依本患者主訴、病情及四診作答，僅從原本16個候選證型選擇。"
)


def variants(baseline: dict, paired: dict) -> dict[str, dict]:
    """Produce patient-gold-blind, evidence-isolated intervention prompts."""
    if paired["id"] != baseline["id"] or paired["options"] != baseline["options"]:
        raise ValueError("The paired input changed the original decision")
    patient = baseline["state"]
    if any(paired["state"].get(k) != v for k, v in patient.items()):
        raise ValueError("Retrieval altered patient clinical evidence")
    supports = paired["state"].get("retrieved_cases")
    if not isinstance(supports, list) or len(supports) != 2:
        raise ValueError("Counterfactual label audit requires exactly two support cases")
    labels = [r.get("candidate_syndrome") for r in supports]
    original_options = [o["id"] for o in baseline["options"]]
    if labels != original_options[:2] or labels[0] == labels[1]:
        raise ValueError("Support labels must be the original Top-2 competitors")
    if any(not all(k in r for k in ("chief_complaint", "description", "detection"))
           for r in supports):
        raise ValueError("All support cases must retain real clinical observations")
    group = {"baseline": deepcopy(baseline), "paired2": deepcopy(paired)}
    inverted = deepcopy(paired)
    inverted["state"]["retrieved_cases"][0]["candidate_syndrome"] = labels[1]
    inverted["state"]["retrieved_cases"][1]["candidate_syndrome"] = labels[0]
    group["labelswap2"] = inverted
    unlabeled = deepcopy(paired)
    for row in unlabeled["state"]["retrieved_cases"]:
        row["candidate_syndrome"] = HIDDEN_LABEL
    group["labelhide2"] = unlabeled
    reversed_rows = deepcopy(paired)
    reversed_rows["state"]["retrieved_cases"].reverse()
    group["orderflip2"] = reversed_rows
    doubly_altered = deepcopy(inverted)
    doubly_altered["state"]["retrieved_cases"].reverse()
    group["doubleflip2"] = doubly_altered
    guarded = deepcopy(paired)
    guarded["state"]["retrieval_note"] = SOURCE_GUARD
    group["sourceguard2"] = guarded
    return group


def assert_only_allowed_changes(records: dict[str, dict]) -> None:
    """Reject altered patient symptoms, option order, or support case content."""
    if tuple(records) != ARMS:
        raise ValueError("Unexpected counterfactual conditions")
    original = records["baseline"]
    paired = records["paired2"]
    expected_cases = paired["state"]["retrieved_cases"]
    for arm, row in records.items():
        if any(row[k] != original[k] for k in ("id", "options", "question", "qtype")):
            raise ValueError(f"{arm} changed the diagnostic decision")
        if any(row["state"].get(k) != v for k, v in original["state"].items()):
            raise ValueError(f"{arm} changed original patient observations")
        if arm == "baseline":
            if row != original:
                raise ValueError("The original Jeff baseline was modified")
            continue
        cases = row["state"]["retrieved_cases"]
        if len(cases) != 2:
            raise ValueError("Counterfactual cases changed evidence count")
        if arm == "orderflip2":
            if cases != expected_cases[::-1]:
                raise ValueError("Order control changed clinical evidence")
        elif arm == "doubleflip2":
            if cases != records["labelswap2"]["state"]["retrieved_cases"][::-1]:
                raise ValueError("Factorial control changed clinical evidence")
        else:
            for source, intervention in zip(expected_cases, cases, strict=True):
                if any(source[k] != intervention[k] for k in (
                    "chief_complaint", "description", "detection"
                )):
                    raise ValueError(f"{arm} changed historical clinical evidence")
        if arm != "sourceguard2" and row["state"]["retrieval_note"] != paired["state"]["retrieval_note"]:
            raise ValueError(f"{arm} unexpectedly changed attribution instructions")
    if [
        row["candidate_syndrome"]
        for row in records["labelswap2"]["state"]["retrieved_cases"]
    ] != [expected_cases[1]["candidate_syndrome"], expected_cases[0]["candidate_syndrome"]]:
        raise ValueError("The intended source-label intervention is absent")
    if any(
        row["candidate_syndrome"] != HIDDEN_LABEL
        for row in records["labelhide2"]["state"]["retrieved_cases"]
    ):
        raise ValueError("The intended hidden-label control is absent")
    if records["sourceguard2"]["state"]["retrieval_note"] != SOURCE_GUARD:
        raise ValueError("The source-attribution instruction has changed")
