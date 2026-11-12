"""Reproducible CPU-only PyTorch autograd parity for all four Jeff losses.

Run without modifying project dependencies:
  uv run --with 'torch==2.9.1+cpu' \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    python -m scripts.check_jeff_torch_cpu

This checks S1/S2/S3/S4 math, not released Jeff logits or H200 training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcmdiscern.jeff_losses import numpy_loss_gradient, torch_loss


def main() -> None:
    import torch

    cases = [
        ("s1", [5], 5),
        ("s2", [0, 5, 8], 5),
        ("s3", [0, 5, 8], 5),
        ("s4", [0, 5, 8], 5),
        ("s2", [5], 5),
        ("s3", [5], 5),
        ("s4", [5], 5),
    ]
    stats = []
    for method, positive, original_gold in cases:
        scores = np.linspace(-4.0, 5.0, 16)
        logits = torch.tensor(scores, dtype=torch.float32, requires_grad=True)
        actual_loss = torch_loss(
            logits, positive_indices=positive, gold_index=original_gold, method=method
        )
        actual_loss.backward()
        oracle_loss, oracle_grad = numpy_loss_gradient(
            scores, positive_indices=positive, gold_index=original_gold, method=method
        )
        if not np.allclose(actual_loss.item(), oracle_loss, atol=1e-5):
            raise ValueError(f"CPU PyTorch {method} loss differs from NumPy reference")
        grad = logits.grad.detach().cpu().numpy()
        if not np.allclose(grad, oracle_grad, atol=1e-6):
            raise ValueError(f"CPU PyTorch {method} autograd differs from reference")
        stats.append(
            {
                "method": method,
                "observed_positives": len(positive),
                "loss": float(actual_loss.item()),
                "max_abs_gradient_difference": float(np.abs(grad - oracle_grad).max()),
                "passed": True,
            }
        )
    report = {
        "date": "2026-11-12",
        "status": "CPU_TORCH_AUTOGRAD_LOSS_PARITY_VERIFIED",
        "torch_version": torch.__version__,
        "device": "cpu",
        "validated": len(stats),
        "all_passed": True,
        "model_weights_loaded": False,
        "jeff_head_logits_verified": False,
        "h200_step_zero_training_verified": False,
        "comparisons": stats,
    }
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/jeff_cpu_autograd_loss_2026-11-12.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
