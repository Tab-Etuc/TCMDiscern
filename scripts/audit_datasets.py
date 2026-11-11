"""Audit the official TCM-SD release and save aggregate-only statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcmdiscern.datasets import audit_tcmsd, load_records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate official TCM-SD splits and overlap")
    parser.add_argument("--data", type=Path, default=root / "data/tcm-sd")
    parser.add_argument("--output", type=Path, default=root / "data/reports/tcmsd_audit.json")
    args = parser.parse_args()
    report = audit_tcmsd(args.data)
    secondary = root / "data/secondary/syndrome819/cases.json"
    if secondary.is_file():
        rows = load_records(secondary)
        labels = {label for row in rows for label in row.get("patterns", [])}
        report["syndrome819"] = {
            "cases": len(rows),
            "labels": len(labels),
            "source": "separate local research cohort",
        }
        if (len(rows), len(labels)) != (819, 78):
            raise ValueError("Unexpected secondary syndrome819 cohort size or label count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
