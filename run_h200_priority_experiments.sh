#!/usr/bin/env bash
# One-command, prioritized, resumable H200 research queue.
# The official Test remains sealed.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Project Python is missing: $PY" >&2
  echo "Prepare the pinned H200 environment before launching this queue." >&2
  exit 1
fi

for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    exec "$PY" -m scripts.run_h200_priority_experiments "$@"
  fi
done

if ! command -v flock >/dev/null 2>&1; then
  echo "util-linux flock is required for exclusive H200 research." >&2
  exit 1
fi
mkdir -p "$ROOT/data/h200"
exec 9>"$ROOT/data/h200/runner.lock"
if ! flock -n 9; then
  echo "Another H200 experiment holds the existing GPU research lock." >&2
  exit 1
fi

echo "[PRIORITY] TCMDiscern ordered experiments; official Test sealed"
"$PY" -m scripts.run_h200_priority_experiments "$@"
