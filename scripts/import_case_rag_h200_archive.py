"""Safely import only archived H200 inference outputs into the active workspace.

The archive also contains user-private clinical records and historical
source code. Leave all of those untouched: verify canonical input equality,
then atomically import just score JSONLs, timing JSONLs, score manifests,
and the completed strong-SGD-fusion aggregate. Never import .partial files.

Usage:
  .venv/bin/python -m scripts.import_case_rag_h200_archive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path

from scripts.prepare_case_rag_paper_ablation import ROOT, TAG

ARCHIVE = ROOT / "TCMDiscern-H200-research-20261117.tar.gz"
SCORES_PREFIX = f"data/case_rag/{TAG}/scores/"
AGGREGATE = "experiments/case_rag_strong_sgd_fusion_cpu_2026-11-17.json"
FROZEN_INPUTS = (
    "data/tcm-sd/train.json",
    "data/tcm-sd/dev.json",
    "data/candidates/bm25_dev.jsonl",
    "data/jeff/dev_k16.jsonl",
    f"data/case_rag/{TAG}/global2.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_jsonl(path: Path, *, case_count: int = 5486, timing: bool = False) -> None:
    """Validate ordered score rows without writing or exporting patient text."""
    with path.open(encoding="utf-8") as stream:
        count = 0
        for count, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("case_id") != f"dev-{count - 1:06d}":
                raise ValueError(f"Case ID mismatch: {path.name}: row {count}")
            if timing:
                continue
            logits = row.get("logits")
            if not isinstance(logits, list) or len(logits) != 16:
                raise ValueError(f"Invalid 16-way logits: {path.name}: row {count}")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in logits):
                raise ValueError(f"Nonfinite score: {path.name}: row {count}")
    if count != case_count:
        raise ValueError(f"Incomplete file {path.name}: {count} != {case_count}")


def _is_importable(name: str) -> bool:
    if name == AGGREGATE:
        return True
    if not name.startswith(SCORES_PREFIX) or name.endswith(".partial"):
        return False
    return name.endswith((".jsonl", "run_manifest.json"))


def import_archive(archive: Path = ARCHIVE, *, root: Path = ROOT) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    imported = []
    identical = []
    skipped_partial = []
    verified = []
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        if any(
            (
                m.name.startswith("/")
                or ".." in Path(m.name).parts
                or (not m.isfile() and not m.isdir())
            )
            for m in members.values()
        ):
            raise ValueError("Unsafe archive member")
        for name in FROZEN_INPUTS:
            local = root / name
            member = members.get(name)
            if member is None or not member.isfile() or not local.is_file():
                raise FileNotFoundError(f"Missing immutable input {name}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archived {name}")
            with stream:
                if sha256_stream(stream) != sha256_file(local):
                    raise ValueError(f"H200 input differs from local: {name}")
            verified.append(name)

        for name, member in members.items():
            if name.startswith(SCORES_PREFIX) and name.endswith(".partial"):
                skipped_partial.append(name)
                continue
            if not member.isfile() or not _is_importable(name):
                continue
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                stream = tar.extractfile(member)
                with stream:
                    if sha256_stream(stream) != sha256_file(dest):
                        raise FileExistsError(f"Refusing to replace different local file: {name}")
                identical.append(name)
                continue

            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Invalid archived member: {name}")
            temp_name = ""
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=dest.parent, prefix=".h200_import_", delete=False
                ) as tmp:
                    temp_name = tmp.name
                    while chunk := stream.read(1024 * 1024):
                        tmp.write(chunk)
                source = Path(temp_name)
                if source.stat().st_size != member.size:
                    raise ValueError(f"Truncated output {name}")
                if name.endswith(".jsonl"):
                    validate_jsonl(source, timing=name.endswith(".timings.jsonl"))
                elif name == AGGREGATE:
                    report = json.loads(source.read_text(encoding="utf-8"))
                    if (
                        report.get("status") != "CASE_RAG_STRONG_SGD_CPU_FUSION_COMPLETED"
                        or report.get("official_test_used") is not False
                        or report.get("dev_cases") != 5486
                        or set(report.get("global2_logit_sources", {})) != {"s1", "s2", "s3"}
                    ):
                        raise ValueError("Incorrect strong fusion report")
                os.replace(source, dest)
                imported.append(name)
            finally:
                stream.close()
                if temp_name and Path(temp_name).exists():
                    Path(temp_name).unlink()

    if not all(
        (root / f"{SCORES_PREFIX}paper-{method}-full/global2.jsonl").is_file()
        for method in ("s1", "s2", "s3")
    ):
        raise ValueError("Required Global2 scores were not imported")
    if not (root / AGGREGATE).is_file():
        raise ValueError("H200 strong fusion aggregate missing")
    return {
        "status": "H200_CASE_RAG_ARCHIVE_IMPORT_COMPLETED",
        "verified_frozen_inputs": verified,
        "imported": sorted(imported),
        "identical": sorted(identical),
        "skipped_incomplete": sorted(skipped_partial),
        "full_dev_case_count": 5486,
        "official_test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/h200_archive_import_2026-11-17.json"
    )
    args = parser.parse_args()
    report = import_archive(args.archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report["status"])
    print("FROZEN INPUTS VERIFIED:", len(report["verified_frozen_inputs"]))
    print("SCORE/MANIFEST FILES IMPORTED:", len(report["imported"]))
    print("ALREADY IDENTICAL:", len(report["identical"]))
    print("PARTIAL FILES SKIPPED:", len(report["skipped_incomplete"]))
    print("REPORT:", args.report)


if __name__ == "__main__":
    main()
