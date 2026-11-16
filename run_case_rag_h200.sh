#!/usr/bin/env bash
# Single-entry Jeff Case-RAG 300-patient Dev pilot; official Test stays sealed.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${TCMDISCERN_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Project Python is missing: $PY" >&2
  echo "On H200 run: bash run_h200.sh prepare" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock (util-linux) is required for exclusive H200 experiments." >&2
  exit 1
fi
mkdir -p "$ROOT/data/h200"
exec 9>"$ROOT/data/h200/runner.lock"
if ! flock -n 9; then
  echo "An H200 experiment already holds data/h200/runner.lock." >&2
  exit 1
fi
exec "$PY" -m scripts.run_case_rag_h200 "$@"
