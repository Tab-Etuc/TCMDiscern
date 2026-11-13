"""One Python entry for data preparation, GPU validation and registered Jeff runs.

Call using ./run_h200.sh. Importing or using --dry-run never loads model
weights, fetches patient records or runs any GPU training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tcmdiscern.h200.plan import (
    ACCUMULATION,
    SEEDS,
    TRAIN_ROWS,
    UPDATES_PER_EPOCH,
    registered_exploratory,
    registered_main,
)
from tcmdiscern.h200.workflow import ROOT, H200Runner


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="H200 one-entry reproducible Jeff/BM25 study",
    )
    parser.add_argument(
        "mode",
        choices=("status", "prepare", "pilot", "screen", "full", "plan"),
        help="plan/status are safe and never run GPU work",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("H200_MODEL_DIR", str(ROOT / "data/models/Jeff-1.1-Large"))),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print phases only, no writes or GPU"
    )
    parser.add_argument(
        "--confirm-screen", action="store_true",
        help="Required before three full first-seed two-epoch H200 trainings",
    )
    parser.add_argument(
        "--confirm-full",
        action="store_true",
        help="Required to begin expensive multi-seed full GPU training",
    )
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Also run S4 soft-target and S3 recoverable-hard weighting explorations",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.include_exploratory and args.mode != "full":
        parser.error("--include-exploratory only applies to 'full'")
    if args.mode == "screen" and not (args.confirm_screen or args.dry_run):
        parser.error(
            "screen requires --confirm-screen. It runs three 31B-model two-epoch trainings."
        )
    if args.mode == "full" and not (args.confirm_full or args.dry_run):
        parser.error(
            "full requires --confirm-full. It runs nine 31B-model "
            "two-epoch trainings, plus a GPU pilot and complete Dev evaluations."
        )
    runner = H200Runner(root=ROOT, model_dir=args.model_dir, dry_run=args.dry_run)
    if args.mode == "plan":
        main_runs = registered_main()
        optional = registered_exploratory()
        print(
            json.dumps(
                {
                    "status": "PRE_REGISTERED_GPU_NOT_STARTED",
                    "official_train": TRAIN_ROWS,
                    "primary_k": 16,
                    "accumulation": ACCUMULATION,
                    "updates_per_epoch": UPDATES_PER_EPOCH,
                    "seeds": list(SEEDS),
                    "main_runs": [
                        {
                            "run_id": e.run_id,
                            "method": e.method,
                            "epochs": e.epochs,
                            "steps": e.updates,
                            "weighting": e.weighting,
                        }
                        for e in main_runs
                    ],
                    "optional_runs": [
                        {
                            "run_id": e.run_id,
                            "method": e.method,
                            "epochs": e.epochs,
                            "steps": e.updates,
                            "weighting": e.weighting,
                        }
                        for e in optional
                    ],
                    "full_confirmation_required": True,
                    "test_data_used": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.mode == "status":
        print(json.dumps(runner.status(), ensure_ascii=False, indent=2))
    elif args.mode == "prepare":
        runner.prepare()
    elif args.mode == "pilot":
        runner.pilot()
    elif args.mode == "screen":
        runner.screen()
    else:
        runner.full(include_exploratory=args.include_exploratory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
