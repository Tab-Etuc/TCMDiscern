"""Freeze full-Dev Jeff experiment selection without ever opening Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.h200.results import save_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate complete registered Jeff Dev evaluations"
    )
    parser.add_argument("--include-exploratory", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = save_selection(root, include_exploratory=args.include_exploratory)
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "saved_report": str(path.relative_to(root)),
                "selected": summary["selected"],
                "test_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
