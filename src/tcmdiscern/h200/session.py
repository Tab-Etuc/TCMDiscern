"""One resident released Jeff 31B for multiple independent H200 stages.

Only the original merged released weights are retained in GPU memory. Each
new training run starts with a freshly initialized LoRA and the original Head;
evaluation loads the requested saved adapter/Head after unloading the prior
adapter. No extra base weight copies or LoRA merges are permitted.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tcmdiscern.jeff_checkpoint import reset_released_backbone


class JeffGPUSession:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self._released: Any | None = None
        self.model_weight_loads = 0

    def released(self):
        """Lazy load: fully completed stages do not allocate a 31B model."""
        if self._released is None:
            from scripts.train_jeff_lora import load_original_checkpoint

            print("[SESSION] Loading released Jeff-1.1-Large (one time per process)", flush=True)
            self._released = load_original_checkpoint(self.model_dir)
            self.model_weight_loads += 1
        return self._released

    def original(self):
        """Return the model to its original published adapter-free state."""
        released = self.released()
        decider = released.decider
        if hasattr(decider.base, "peft_config"):
            reset_released_backbone(decider, self.model_dir)
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if hasattr(decider.base, "peft_config"):
            raise RuntimeError("Previous training adapter still attached after GPU reset")
        return released

    def train_cli(self, command: Sequence[str]) -> dict:
        """Use the exactly registered CLI flags, including resumable checkpoint."""
        if tuple(command[:2]) != ("-m", "scripts.run_jeff_finetune"):
            raise ValueError("GPU session only accepts registered Jeff training commands")
        from scripts.run_jeff_finetune import build_parser
        from tcmdiscern.jeff_train_loop import run

        args = build_parser().parse_args(command[2:])
        if args.model_dir.resolve() != self.model_dir:
            raise ValueError("GPU session model path differs from training configuration")
        report = run(args, released=self.original())
        print(
            json.dumps(
                {
                    "session_stage": "training",
                    "run_id": args.run_id,
                    "global_update": report["global_update"],
                    "loss_method": report["loss_method"],
                    "peak_cuda_bytes": report["peak_cuda_bytes"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return report

    def syndrome_dev(
        self,
        *,
        root: Path,
        checkpoint: Path | None = None,
        run_id: str | None = None,
    ) -> dict:
        from scripts.run_jeff_zeroshot import run

        released = self.original()
        result = run(
            self.model_dir,
            root=root,
            limit=5486,
            checkpoint=checkpoint,
            run_id=run_id,
            released=released,
        )
        print(
            json.dumps(
                {
                    "session_stage": "syndrome_dev",
                    "run_id": run_id or "released",
                    "accuracy": result["scores"]["top1_accuracy"],
                    "macro_f1": result["scores"]["macro_f1"],
                    "model_weight_loads": self.model_weight_loads,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return result

    def joint_dev(
        self,
        *,
        mode: str,
        split: str,
        run_id: str,
        checkpoint: Path | None = None,
        limit: int | None = None,
    ) -> dict:
        from scripts.evaluate_joint16 import evaluate

        result = evaluate(
            mode=mode,
            split=split,
            model_dir=self.model_dir,
            run_id=run_id,
            checkpoint=checkpoint,
            limit=limit,
            released=self.original(),
        )
        print(
            json.dumps(
                {
                    "session_stage": f"joint16_{mode}_{split}",
                    "run_id": run_id,
                    "kind": result["kind"],
                    "model_weight_loads": self.model_weight_loads,
                    "scores": result.get("scores"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return result
