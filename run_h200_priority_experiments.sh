#!/usr/bin/env bash
# One-command, prioritized, resumable H200 research queue.
# The official Test remains sealed.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# H200 may have a previously installed editable TCMDiscern package pointing to
# TCMDiscern-main. Always import the code from this exact Git checkout.
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$PY" ]; then
  echo "Project Python is missing: $PY" >&2
  echo "Prepare the pinned H200 environment before launching this queue." >&2
  exit 1
fi

if [ ! -f "$ROOT/src/tcmdiscern/priority_cpu_common.py" ]; then
  echo "This Git checkout is missing src/tcmdiscern/priority_cpu_common.py; run git pull --ff-only origin main." >&2
  exit 1
fi

# Fail before any expensive CPU fit or 31B GPU load if the venv sees an old
# installed distribution instead of the current src-layout checkout.
if ! "$PY" -c 'from pathlib import Path; import tcmdiscern; from scripts.run_h200_priority_cpu_ablation import arbitrate; root=Path.cwd().resolve(); actual=Path(tcmdiscern.__file__).resolve(); expected=root/"src"/"tcmdiscern"/"__init__.py"; assert actual==expected, f"TCMDiscern import comes from {actual}; expected {expected}"; print(f"[PREFLIGHT] Python imports current checkout: {actual}", flush=True)'; then
  echo "[PREFLIGHT] Python import failed. Inspect the checked-out src/ files and the pinned .venv." >&2
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
