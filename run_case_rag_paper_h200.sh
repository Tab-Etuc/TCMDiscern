#!/usr/bin/env bash
# Frozen 5,486 Dev Case-RAG paper experiments. One resident 31B Jeff load.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${TCMDISCERN_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Project Python missing: $PY" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for exclusive H200 experiments" >&2
  exit 1
fi
mkdir -p "$ROOT/data/h200"
exec 9>"$ROOT/data/h200/runner.lock"
if ! flock -n 9; then
  echo "Another H200 job already holds data/h200/runner.lock" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TCMDISCERN_GEMMA4_ATTENTION=hybrid
exec "$PY" -m scripts.run_case_rag_paper_h200 "$@"
