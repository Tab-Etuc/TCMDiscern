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
    candidates.extend((
        root / "data/models/Jeff-1.1-Large-tokenizer",
        root / "data/models/Jeff-1.1-Large",
    ))
    for folder in candidates:
        if (
            (folder / "tokenizer.json").is_file()
            and (folder / "chat_template.jinja").is_file()
        ):
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
    return codec, folder
