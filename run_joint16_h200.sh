#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
MODE="${1:-plan}"
if [ "$#" -gt 0 ]; then shift; fi

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi

DRY=0
for argument in "$@"; do
  if [ "$argument" = "--dry-run" ]; then DRY=1; fi
done

case "$MODE" in
  plan|status)
    PYTHONPATH="$ROOT/src:$ROOT" "$PY" -m scripts.joint16_h200 "$MODE" "$@"
    ;;
  prepare)
    bash "$ROOT/run_h200.sh" prepare
    "$ROOT/.venv/bin/python" -m scripts.joint16_h200 prepare "$@"
    ;;
  pilot|screen|full|test)
    if [ "$DRY" -eq 1 ]; then
      PYTHONPATH="$ROOT/src:$ROOT" "$PY" -m scripts.joint16_h200 "$MODE" "$@"
      exit 0
    fi
    if [ "$MODE" = "test" ]; then
      if [ ! -f "$ROOT/experiments/joint16_dev_selection.json" ]; then
        echo "Finish Dev model selection before official Test." >&2
        exit 2
      fi
      "$ROOT/.venv/bin/python" -m scripts.fetch_tcmsd --include-test
    else
      bash "$ROOT/run_h200.sh" pilot
    fi
    mkdir -p "$ROOT/data/h200"
    exec 9>"$ROOT/data/h200/joint16-runner.lock"
    flock -n 9
    "$ROOT/.venv/bin/python" -m scripts.joint16_h200 "$MODE" "$@"
    ;;
  *)
    echo "Usage: bash run_joint16_h200.sh {plan|status|prepare|pilot|screen|full|test} [--confirm] [--dry-run]" >&2
    exit 2
    ;;
esac
