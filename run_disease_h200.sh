#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
MODE="${1:-plan}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  plan|status)
    PY="$ROOT/.venv/bin/python"
    if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
    PYTHONPATH="$ROOT/src:$ROOT" "$PY" -m scripts.disease_h200 "$MODE" "$@"
    ;;
  prepare)
    bash "$ROOT/run_h200.sh" prepare
    "$ROOT/.venv/bin/python" -m scripts.disease_h200 prepare "$@"
    ;;
  pilot|full|test)
    DRY=0
    for arg in "$@"; do if [ "$arg" = "--dry-run" ]; then DRY=1; fi; done
    if [ "$DRY" -eq 1 ]; then
      PY="$ROOT/.venv/bin/python"
      if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
      PYTHONPATH="$ROOT/src:$ROOT" "$PY" -m scripts.disease_h200 "$MODE" "$@"
      exit 0
    fi
    bash "$ROOT/run_h200.sh" pilot
    if [ "$MODE" = "test" ]; then
      "$ROOT/.venv/bin/python" -m scripts.fetch_tcmsd --include-test
    fi
    exec 9>"$ROOT/data/h200/disease-runner.lock"
    flock -n 9
    "$ROOT/.venv/bin/python" -m scripts.disease_h200 "$MODE" "$@"
    ;;
  *)
    echo "Usage: bash run_disease_h200.sh {plan|status|prepare|pilot|full|test} [--confirm] [--selected-run ID] [--dry-run]" >&2
    exit 2
    ;;
esac
