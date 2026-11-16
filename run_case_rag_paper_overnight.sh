#!/usr/bin/env bash
# Run frozen 5,486 Dev paper suite safely after disconnect, with one 31B load.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${TCMDISCERN_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Project Python not found: $PY" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for exclusive H200 experiments." >&2
  exit 1
fi
mkdir -p data/h200
exec 9>data/h200/runner.lock
if ! flock -n 9; then
  echo "Another H200 task holds the runner lock." >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TCMDISCERN_GEMMA4_ATTENTION=hybrid
echo "===== PAPER FULL DEV PREPARE ====="
"$PY" -m scripts.run_case_rag_paper_h200 --prepare-only
echo "===== PAPER FULL DEV PREFLIGHT ====="
"$PY" -m scripts.run_case_rag_paper_h200 --check
echo "===== PAPER FULL DEV S2 + S1 + S3 SINGLE WEIGHT LOAD ====="
"$PY" -m scripts.run_case_rag_paper_h200
echo "===== PAPER FULL DEV COMPLETED ====="
