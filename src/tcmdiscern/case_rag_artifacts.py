"""Locate frozen Jeff full-Dev artifacts across the H200 checkout migration.

The writable code/checkpoint runner lives in TCMDiscern-git. The original
full-Dev reports and logits may still live in sibling TCMDiscern-main.
Read existing artifacts in place; never copy or modify legacy results.
"""

from __future__ import annotations

import os
from pathlib import Path

EPOCH_RUN = "seed20261112-e2"


def artifact_roots(root: Path) -> tuple[Path, ...]:
    """Prefer this checkout, then explicit archive and original sibling."""
    current = root.expanduser().resolve()
    candidates = [current]
    if os.environ.get("TCMDISCERN_LEGACY_ROOT"):
        candidates.append(Path(os.environ["TCMDISCERN_LEGACY_ROOT"]).expanduser().resolve())
    candidates.append(current.parent / "TCMDiscern-main")
    if current.name.endswith("-git"):
        candidates.append(current.with_name(current.name[:-4] + "-main"))
    result: list[Path] = []
    for path in candidates:
        if path not in result:
            result.append(path)
    return tuple(result)


def frozen_paths(root: Path, method: str) -> tuple[Path, Path]:
    """Resolve report/logits without copying files; provenance is checked by preflight."""
    if method not in {"s1", "s2", "s3"}:
        raise ValueError(f"Unknown frozen Jeff checkpoint method: {method}")
    report = Path("experiments") / f"jeff_{method}-{EPOCH_RUN}-epoch2_k16_dev.json"
    logits = Path("data/jeff") / f"dev_{method}-{EPOCH_RUN}-epoch2_k16_5486.jsonl"
    candidates = artifact_roots(root)
    for candidate in candidates:
        if (candidate / report).is_file() and (candidate / logits).is_file():
            return candidate / report, candidate / logits
    # During migration one legacy asset may already have been linked into the
    # Git checkout. Accept split roots only when the caller verifies the study
    # metadata AND recalculates accuracy/F1 from the 5486 logits before GPU.
    study = next((folder / report for folder in candidates if (folder / report).is_file()), None)
    scores = next((folder / logits for folder in candidates if (folder / logits).is_file()), None)
    if study is not None and scores is not None:
        return study, scores
    attempted = "\n".join(
        f"  {folder}: report={(folder / report).is_file()} logits={(folder / logits).is_file()}"
        for folder in candidates
    )
    raise FileNotFoundError(
        f"{method.upper()}: original full-Dev report/logits are missing before the "
        f"31B weight load. Checked:\n{attempted}\n"
        "Existing files can remain in TCMDiscern-main. If elsewhere, set "
        "TCMDISCERN_LEGACY_ROOT=/absolute/path/to/training/root."
    )


def checkpoint_path(root: Path, method: str, choose) -> Path:
    """Find LoRA and Head without copying the model or training output."""
    failures = []
    for candidate in artifact_roots(root):
        if not candidate.is_dir():
            continue
        try:
            return choose(candidate, method, None)
        except FileNotFoundError as exc:
            failures.append(str(exc))
    raise FileNotFoundError(
        f"{method.upper()}: missing complete LoRA/Head checkpoint in "
        f"{[str(path) for path in artifact_roots(root)]}. "
        + ("; ".join(failures[-2:]) if failures else "")
    )
