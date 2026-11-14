"""Single source of truth for pre-registered Jeff H200 training runs.

All run plans are data. Nothing in this module loads a GPU or starts work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import ceil

TRAIN_ROWS = 43180
DEV_ROWS = 5486
PRIMARY_K = 16
ACCUMULATION = 64
SEEDS = (20261112, 20261113, 20261114)
UPDATES_PER_EPOCH = ceil(TRAIN_ROWS / ACCUMULATION)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_LR = 2e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
WARMUP_STEPS = 40
CHECKPOINT_EVERY = 300


@dataclass(frozen=True)
class Experiment:
    method: str
    seed: int
    epochs: int
    weighting: str = "uniform"

    def __post_init__(self) -> None:
        if self.method not in {"s1", "s2", "s3", "s4"}:
            raise ValueError("Unknown registered loss method")
        if self.seed not in SEEDS or self.epochs not in (1, 2):
            raise ValueError("Only registered seeds and epoch budgets are valid")
        if self.weighting not in ("uniform", "recoverable_oof_x2"):
            raise ValueError("Unknown difficulty weighting")
        if self.weighting != "uniform" and (
            self.method != "s3" or self.seed != SEEDS[0] or self.epochs != 1
        ):
            raise ValueError("Only S3 one-epoch first-seed hard-weight ablation is registered")

    @property
    def run_id(self) -> str:
        name = f"{self.method}-seed{self.seed}-e{self.epochs}"
        if self.weighting != "uniform":
            name += "-hard2"
        return name

    @property
    def updates(self) -> int:
        return self.epochs * UPDATES_PER_EPOCH

    def train_arguments(self, *, model_dir: str) -> list[str]:
        # Explicit opt-in until real 31B batch-2 throughput and VRAM are known.
        # Preserves the pinned historical B=1 contract and exact resume settings.
        micro_batch = int(os.environ.get("TCMDISCERN_MICRO_BATCH_SIZE", "1"))
        if not 1 <= micro_batch <= 8:
            raise ValueError("TCMDISCERN_MICRO_BATCH_SIZE must be in [1, 8]")
        return [
            "-m",
            "scripts.run_jeff_finetune",
            "--model-dir",
            model_dir,
            "--method",
            self.method,
            "--run-id",
            self.run_id,
            "--steps",
            str(self.updates),
            "--epochs",
            str(self.epochs),
            "--accumulation",
            str(ACCUMULATION),
            "--micro-batch-size",
            str(micro_batch),
            "--seed",
            str(self.seed),
            "--rank",
            str(LORA_RANK),
            "--alpha",
            str(LORA_ALPHA),
            "--lora-lr",
            str(LORA_LR),
            "--head-lr",
            str(HEAD_LR),
            "--weight-decay",
            str(WEIGHT_DECAY),
            "--clip-norm",
            str(CLIP_NORM),
            "--warmup-steps",
            str(WARMUP_STEPS),
            "--checkpoint-every",
            str(CHECKPOINT_EVERY),
            "--weighting",
            self.weighting,
        ]


def registered_main() -> tuple[Experiment, ...]:
    return tuple(
        Experiment(method, seed, epochs=2) for method in ("s1", "s2", "s3") for seed in SEEDS
    )


def registered_exploratory() -> tuple[Experiment, ...]:
    return (
        Experiment("s4", SEEDS[0], epochs=1),
        Experiment("s3", SEEDS[0], epochs=1, weighting="recoverable_oof_x2"),
    )
