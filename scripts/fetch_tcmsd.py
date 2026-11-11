"""Download the official TCM-SD train/dev package and labeled test split.

Source: https://github.com/Borororo/ZY-BERT (CC BY-NC-SA 4.0 for dataset).
The publisher uses .json for newline-delimited JSON records.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "https://raw.githubusercontent.com/Borororo/ZY-BERT/main"
EXPECTED_MEMBERS = (
    "train.json",
    "dev.json",
    "syndrome_knowledge.json",
    "syndrome_vocab.txt",
)


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "TCMDiscern-research"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not partial.stat().st_size:
            raise ValueError(f"Empty downloaded file from {url}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def unpack_selected(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipfile_obj:
        members: dict[str, zipfile.ZipInfo] = {}
        for item in zipfile_obj.infolist():
            if item.is_dir():
                continue
            basename = Path(item.filename.replace("\\", "/")).name
            if basename in EXPECTED_MEMBERS:
                if basename in members:
                    raise ValueError(f"Ambiguous ZIP member: {basename}")
                members[basename] = item
        missing = set(EXPECTED_MEMBERS) - members.keys()
        if missing:
            raise ValueError(f"Official archive missing members: {sorted(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in EXPECTED_MEMBERS:
            output = destination / name
            if output.is_file() and output.stat().st_size:
                continue
            partial = destination / (name + ".partial")
            partial.unlink(missing_ok=True)
            try:
                with zipfile_obj.open(members[name]) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if not partial.stat().st_size:
                    raise ValueError(f"Empty member: {name}")
                os.replace(partial, output)
            finally:
                partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch original TCM-SD release")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    archive = args.data_root / "TCM_SD_train_dev.zip"
    target = args.data_root / "tcm-sd"
    download(f"{REPO}/TCM_SD_train_dev.zip", archive)
    unpack_selected(archive, target)
    download(f"{REPO}/test.json", target / "test.json")
    print(f"Official TCM-SD train/dev/test + knowledge + vocab: {target}")


if __name__ == "__main__":
    main()
