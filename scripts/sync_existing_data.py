"""Copy selected existing research corpora into TCMDiscern's ignored local data folder.

No writes to the original CMU_AITCM or SynologyDrive data.
No files inside data/ are committed to Git.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT.parent / "cmu_aitcm/data/private/external-syndrome"
SECONDARY = ROOT / "data/secondary"
PRIVATE_819 = Path("/home/tabetuc/SynologyDrive/CMU_RAG/syndrome_819_cases.json")

SOURCES = {
    "mtcmb/diagnosis.jsonl": EXTERNAL / "mtcmb/diagnosis.jsonl",
    "mtcmb/msdd.jsonl": EXTERNAL / "mtcmb/msdd.jsonl",
    "robust-sdt/reference.jsonl": EXTERNAL / "robust/reference.jsonl",
    "robust-sdt/incomplete.jsonl": EXTERNAL / "robust/incomplete.jsonl",
    "robust-sdt/oral.jsonl": EXTERNAL / "robust/oral.jsonl",
    "robust-sdt/conflict.jsonl": EXTERNAL / "robust/conflict.jsonl",
    "robust-sdt/longitudinal.json": EXTERNAL / "robust/longitudinal.json",
    "best4sdt/cases.json": EXTERNAL / "best4sdt/cases.json",
    "syndrome819/cases.json": PRIVATE_819,
}


def main() -> None:
    report = {"copied": [], "unchanged": [], "unavailable": []}
    for relative, source in SOURCES.items():
        destination = SECONDARY / relative
        if not source.is_file():
            report["unavailable"].append(relative)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size:
                raise ValueError(f"Existing destination differs in size: {relative}")
            report["unchanged"].append(relative)
            continue
        shutil.copy2(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Incomplete copy: {relative}")
        report["copied"].append({"path": relative, "bytes": destination.stat().st_size})
    outfile = ROOT / "data/reports/secondary_sources.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
