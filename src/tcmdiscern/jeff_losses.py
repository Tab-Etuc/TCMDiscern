"""Numerically stable Jeff decision losses, independent of the 31B backbone.

NumPy implements a CPU validation oracle and analytic gradient. The optional
PyTorch implementation accepts Jeff head logits and preserves autograd; torch
is imported lazily and is NOT a dependency of the CPU data preparation tools.

S1: one-hot CE; S2: masked CE; S3: multi-positive set loss;
S4: uniform soft-target CE. S2/S3/S4 share identical 16-choice prompts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

VALID_METHODS = frozenset({"s1", "s2", "s3", "s4"})


def validate_supervision(
    num_options: int,
    positives: Sequence[int],
    gold: int,
    method: str,
) -> tuple[int, ...]:
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown Jeff training method: {method}")
    if num_options != 16:
        raise ValueError("This experiment requires exactly 16 Jeff decision logits")
    if not isinstance(gold, int) or isinstance(gold, bool) or not 0 <= gold < num_options:
        raise ValueError("The original gold index must be an integer in [0, 15]")
    if (
        not positives
        or any(not isinstance(i, int) or isinstance(i, bool) for i in positives)
        or len(set(positives)) != len(positives)
        or any(i < 0 or i >= num_options for i in positives)
    ):
        raise ValueError("Observed positive indices must be distinct integers in [0, 15]")
    if gold not in positives:
        raise ValueError("Original gold index must be among observed positives")
    if method == "s1" and len(positives) != 1:
        raise ValueError("S1 single-positive CE requires exactly one gold index")
    return tuple(sorted(positives))


def _logsumexp(values: np.ndarray) -> float:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logsumexp requires a nonempty 1D vector")
    shift = float(values.max())
    return shift + float(np.log(np.exp(values - shift).sum()))


def numpy_loss_gradient(
    logits: Sequence[float] | np.ndarray,
    *,
    positive_indices: Sequence[int],
    gold_index: int,
    method: str,
) -> tuple[float, np.ndarray]:
    """Return scalar loss and the exact dLoss/d(logits) analytical gradient."""
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (16,) or not np.isfinite(z).all():
        raise ValueError("Expected 16 finite decision logits")
    positives = validate_supervision(z.size, positive_indices, gold_index, method)
    pos = np.array(positives, dtype=np.int64)
    logz = _logsumexp(z)
    probs = np.exp(z - logz)
    grad = probs.copy()

    if method == "s1":
        loss = logz - float(z[gold_index])
        grad[gold_index] -= 1.0
    elif method == "s2":
        # Other observed positives are ignored, not treated as negatives.
        allowed = np.ones(16, dtype=bool)
        for i in positives:
            if i != gold_index:
                allowed[i] = False
        masked_logz = _logsumexp(z[allowed])
        grad = np.zeros(16, dtype=np.float64)
        grad[allowed] = np.exp(z[allowed] - masked_logz)
        grad[gold_index] -= 1.0
        loss = masked_logz - float(z[gold_index])
    elif method == "s3":
        pos_logz = _logsumexp(z[pos])
        loss = logz - pos_logz
        grad[pos] -= np.exp(z[pos] - pos_logz)
    else:  # s4
        loss = logz - float(z[pos].mean())
        grad[pos] -= 1.0 / len(positives)

    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise RuntimeError("Loss/gradient became non-finite")
    return float(loss), grad


def torch_loss(
    logits: object,
    *,
    positive_indices: Sequence[int],
    gold_index: int,
    method: str,
):
    """Autograd-compatible PyTorch reference; requires CPU/GPU torch at training time.

    In H200 training, call this on the 16 released Jeff Decision Head logits.
    The result is scalar FP32 for BF16 numerical stability.
    """
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() != 16:
        raise ValueError("Expected a single 16-choice PyTorch logit tensor")
    positives = validate_supervision(16, positive_indices, gold_index, method)
    if not torch.isfinite(logits).all().item():
        raise ValueError("Decision logits must be finite")

    z = logits.float()
    logz = torch.logsumexp(z, dim=0)
    pos = torch.tensor(positives, dtype=torch.long, device=z.device)
    if method == "s1":
        return logz - z[gold_index]
    if method == "s2":
        keep = torch.ones(16, dtype=torch.bool, device=z.device)
        for i in positives:
            if i != gold_index:
                keep[i] = False
        return torch.logsumexp(z[keep], dim=0) - z[gold_index]
    if method == "s3":
        return logz - torch.logsumexp(z.index_select(0, pos), dim=0)
    return logz - z.index_select(0, pos).mean()
