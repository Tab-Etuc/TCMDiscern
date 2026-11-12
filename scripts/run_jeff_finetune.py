"""H200-only, explicit Jeff LoRA + original Head training command.

Without --steps, performs exactly one optimizer-step smoke experiment.
All outputs stay under git-ignored data/runs/. No model is downloaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.jeff_train_loop import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Jeff-1.1-Large on H200")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("s1", "s2", "s3", "s4"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        type=Path,
        help="A prior resume-NNNNNNN checkpoint inside the same --run-id; keeps the exact optimizer budget",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
