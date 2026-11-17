#!/usr/bin/env bash
# Batch all missing S1/S3 full-Dev attribution controls under one GPU lock.
# Seed 2/3 retraining is intentionally not scheduled here.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${TCMDISCERN_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "Project Python missing: $PY" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for the exclusive H200 runner" >&2
  exit 1
fi

mkdir -p data/h200
exec 9>data/h200/runner.lock
if ! flock -n 9; then
  echo "Another H200 run is active; follow-up was not started" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TCMDISCERN_GEMMA4_ATTENTION=hybrid
exec "$PY" -m scripts.run_case_rag_followup_h200 "$@"
