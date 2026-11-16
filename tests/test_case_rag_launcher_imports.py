"""Regression: a symlinked H200 venv must import the current Git checkout."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("flock") is None, reason="requires flock")
def test_launcher_prefers_current_source_with_reused_venv(tmp_path: Path):
    root = tmp_path / "fresh-checkout"
    prior = tmp_path / "old-checkout"
    for checkout, marker in ((root, "NEW_CHECKOUT"), (prior, "OLD_CHECKOUT")):
        package = checkout / "src/tcmdiscern"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "case_rag.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")

    old_python = prior / ".venv/bin/python"
    old_python.parent.mkdir(parents=True)
    old_python.write_text(
        "#!/bin/sh\n"
        + "exec "
        + shlex.quote(sys.executable)
        + " -c 'from tcmdiscern.case_rag import MARKER; print(MARKER)'\n",
        encoding="utf-8",
    )
    old_python.chmod(0o755)
    (root / ".venv").symlink_to(prior / ".venv", target_is_directory=True)
    launcher = Path(__file__).resolve().parents[1] / "run_case_rag_h200.sh"
    shutil.copy2(launcher, root / launcher.name)

    # Simulate the editable-install path retained by the reused Python venv.
    env = {**os.environ, "PYTHONPATH": str(prior / "src")}
    completed = subprocess.run(
        ["bash", str(root / launcher.name), "s2", "--prepare-only"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "NEW_CHECKOUT"
