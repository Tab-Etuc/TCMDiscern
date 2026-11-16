"""Use the same pinned Jeff tokenizer on CPU and on a clean H200 clone.

The local CPU experiments kept a tokenizer-only copy while run_h200.sh
downloads the released full model instead. Both have the published vocabulary
and chat_template.jinja; do not require a second HF download.
"""

from __future__ import annotations

import os
from pathlib import Path


def locate_jeff_tokenizer(
    root: Path,
    *,
    model_dir: Path | None = None,
) -> Path:
    candidates = []
    if model_dir is not None:
        candidates.append(model_dir)
    if os.environ.get("H200_MODEL_DIR"):
        candidates.append(Path(os.environ["H200_MODEL_DIR"]))
    candidates.extend(
        (
            root / "data/models/Jeff-1.1-Large-tokenizer",
            root / "data/models/Jeff-1.1-Large",
        )
    )
    for folder in candidates:
        if (folder / "tokenizer.json").is_file() and (folder / "chat_template.jinja").is_file():
            return folder.resolve()
    raise FileNotFoundError(
        "Jeff tokenizer.json/chat_template.jinja missing. "
        "On H200, run 'bash run_h200.sh pilot' to prepare the pinned released "
        "model, or pass H200_MODEL_DIR to the model download directory."
    )


def load_jeff_token_counter(
    root: Path,
    *,
    model_dir: Path | None = None,
):
    """Return the frozen SemIf-compatible token counter and its source path."""
    from transformers import PreTrainedTokenizerFast

    folder = locate_jeff_tokenizer(root, model_dir=model_dir)
    codec = PreTrainedTokenizerFast(
        tokenizer_file=str(folder / "tokenizer.json"),
        chat_template=(folder / "chat_template.jinja").read_text(encoding="utf-8"),
    )
    return _TokenCounterCompatibility(codec), folder


class _TokenCounterCompatibility:
    """Preserve list-of-token-IDs semantics for legacy token-count call sites."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        from collections.abc import Mapping

        result = self._tokenizer.apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", False):
            if isinstance(result, Mapping):
                result = result.get("input_ids")
            if (
                isinstance(result, (list, tuple))
                and len(result) == 1
                and isinstance(result[0], (list, tuple))
            ):
                result = result[0]
            if not isinstance(result, (list, tuple)) or len(result) < 32:
                raise ValueError("Jeff tokenizer returned an invalid token sequence")
        return result


def count_jeff_prompt_tokens(tokenizer, row: dict) -> int:
    """Count real input IDs across Transformers 4.x/5.x chat-template APIs.

    Transformers 5.x may return BatchEncoding with input_ids and attention_mask.
    len(BatchEncoding) counts keys (usually 2), not model input tokens.
    """
    from collections.abc import Mapping

    from .jeff_format import semif_messages

    encoded = tokenizer.apply_chat_template(
        semif_messages(row),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Jeff tokenizer response has no input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "shape"):
        shape = tuple(encoded.shape)
        if len(shape) == 1:
            n_tokens = shape[0]
        elif len(shape) == 2 and shape[0] == 1:
            n_tokens = shape[1]
        else:
            raise ValueError("Expected a single unpadded Jeff token sequence")
    elif isinstance(encoded, (list, tuple)):
        if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        if not encoded or not isinstance(encoded[0], int):
            raise ValueError("Jeff tokenizer input_ids are malformed")
        n_tokens = len(encoded)
    else:
        raise ValueError("Unsupported Jeff tokenizer input_ids representation")
    if n_tokens < 32:
        raise ValueError(f"Jeff prompt has implausibly few tokens: {n_tokens}")
    return int(n_tokens)
