"""Convert TCM-SD cases to Jeff's SemIf decision-row and prompt format.

Verified against j-geuter/Jeff/src/jeff/prompting.py (main, 2026-11-11).
The real Jeff trainer should use Jeff's tokenizer + render_prompt; this
module has no GPU/torch dependency and never embeds a gold label in the prompt.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .datasets import CLINICAL_FIELDS

# Keep these strings byte-compatible with Jeff's prompting.py.
LETTERS = "ABCDEFGHIJKLMNOP"
MAX_DIRECT_OPTIONS = len(LETTERS)
MAX_GROUPED_OPTIONS = MAX_DIRECT_OPTIONS**2
PROMPT_VERSION = "direct-options-v1"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
TCM_QUESTION = "根據患者主訴、病情與四診資訊，選出最符合的中醫證型。"


def patient_state(case: Mapping[str, Any]) -> dict[str, str]:
    """Whitelisted clinical fields only; do not include syndrome, ICD or user_id."""
    state = {field: str(case.get(field) or "").strip() for field in CLINICAL_FIELDS}
    if not any(state.values()):
        raise ValueError("Clinical case has no nonempty input fields")
    return state


def decision_row(
    case: Mapping[str, Any],
    *,
    case_id: str,
    candidates: Sequence[str],
    allow_grouped: bool = False,
) -> dict[str, Any]:
    """Jeff API row: id/state/question/options, with optional qtype=choice.

    For K>16, Jeff's Decider.decide_wide groups options into <=16 slots;
    semif_messages deliberately refuses >16 direct options.
    """
    limit = MAX_GROUPED_OPTIONS if allow_grouped else MAX_DIRECT_OPTIONS
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a nonempty string")
    if not 2 <= len(candidates) <= limit:
        raise ValueError(f"Jeff requires 2..{limit} candidate options")
    if any(not isinstance(label, str) or not label.strip() for label in candidates):
        raise ValueError("Every candidate must be a nonempty label string")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Candidate labels must be unique")

    return {
        "id": case_id,
        "state": patient_state(case),
        "question": TCM_QUESTION,
        "options": [{"id": label, "description": label} for label in candidates],
        "qtype": "choice",
    }


def train_example(
    case: Mapping[str, Any],
    *,
    case_id: str,
    gold_label: str,
    hard_negatives: Sequence[str],
    k: int,
    rng: Random,
) -> dict[str, Any]:
    """Build one gold + (K-1) OOF BM25 hard negatives; shuffle every epoch.

    A training pipeline must supply hard_negatives retrieved using TRAIN-only
    Out-of-Fold indexes. No train example may retrieve itself. Gold supervision
    stays outside row, so it cannot leak into the Jeff prompt.
    """
    if not 2 <= k <= MAX_DIRECT_OPTIONS:
        raise ValueError("Train K must be 2..16 (Jeff's direct decision-head slots)")
    if not isinstance(gold_label, str) or not gold_label.strip():
        raise ValueError("gold_label must be a nonempty string")
    if len(set(hard_negatives)) != len(hard_negatives):
        raise ValueError("Hard negatives must be distinct")
    if gold_label in hard_negatives:
        raise ValueError("Hard-negative pool must not contain the gold syndrome")
    if len(hard_negatives) < k - 1:
        raise ValueError("Insufficient hard negatives to construct K options")
    # Input hard negatives must already be rank-ordered by OOF BM25.
    selected = [gold_label, *hard_negatives[: k - 1]]
    rng.shuffle(selected)
    position = selected.index(gold_label)
    return {
        "row": decision_row(case, case_id=case_id, candidates=selected),
        "target_index": position,
        "target_letter": LETTERS[position],
    }


def semif_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Byte-for-byte equivalent message payload to Jeff.prompting.messages.

    The tokenizer later applies the *model's own* chat template with
    add_generation_prompt=True and enable_thinking=False.
    """
    if not {"id", "state", "question", "options"} <= row.keys():
        raise ValueError("Jeff row requires id, state, question and options")
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_DIRECT_OPTIONS:
        raise ValueError("A direct Jeff decision must contain 2..16 options")
    if not isinstance(row["state"], (dict, str, list)) or not row["state"]:
        raise ValueError("Jeff state must be a nonempty object, list or text")
    if not isinstance(row["question"], str) or not row["question"]:
        raise ValueError("Jeff question must be nonempty")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError("Jeff row id must be nonempty")
    if any(
        not isinstance(opt, dict)
        or not isinstance(opt.get("id"), str)
        or not isinstance(opt.get("description"), str)
        for opt in options
    ):
        raise ValueError("Options need string id and description")
    if len({opt["id"] for opt in options}) != len(options):
        raise ValueError("Options must have unique identifiers")

    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[i], "description": option["description"]}
            for i, option in enumerate(options)
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def render_jeff_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    """Use Jeff's original chat-template flags; do not train via chat completion."""
    return tokenizer.apply_chat_template(
        semif_messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
