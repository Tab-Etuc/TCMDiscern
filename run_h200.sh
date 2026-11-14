#!/usr/bin/env bash
# H200 setup and experiment entrypoint. A clone needs only Bash, Git, curl,
# outbound access to official sources and an H200 with an adequate driver.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-prepare}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
  prepare|micro|pilot|screen|full|plan|status) ;;
  *)
    echo "Usage: ./run_h200.sh {prepare|micro|pilot|screen|full|plan|status} [options]" >&2
    echo "To train all 3 seeds: ./run_h200.sh full --confirm-full" >&2
    exit 2
    ;;
esac

# Read-only inspection and dry-runs should never install packages or download data.
DRY_RUN=0
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then DRY_RUN=1; fi
done

if [ "$MODE" = "screen" ] && [ "$DRY_RUN" -eq 0 ]; then
  CONFIRMED=0
  for arg in "$@"; do
    if [ "$arg" = "--confirm-screen" ]; then CONFIRMED=1; fi
  done
  if [ "$CONFIRMED" -eq 0 ]; then
    echo "Screen requires --confirm-screen (3 two-epoch first-seed H200 trainings)." >&2
    exit 2
  fi
fi

if [ "$MODE" = "full" ] && [ "$DRY_RUN" -eq 0 ]; then
  CONFIRMED=0
  for arg in "$@"; do
    if [ "$arg" = "--confirm-full" ]; then CONFIRMED=1; fi
  done
  if [ "$CONFIRMED" -eq 0 ]; then
    echo "Full training requires --confirm-full (9 main trainings, 2 epochs each)." >&2
    exit 2
  fi
fi

UV_BIN=""
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
elif [ "$MODE" != "plan" ] && [ "$MODE" != "status" ] && [ "$MODE" != "micro" ] && [ "$DRY_RUN" -eq 0 ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the official uv installer." >&2
    exit 1
  fi
  echo "[SETUP] Installing uv from its official installer"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "uv installation failed. Install uv and retry." >&2
    exit 1
  fi
  UV_BIN="$HOME/.local/bin/uv"
fi

PY="$ROOT/.venv/bin/python"

# Fast real tiny Gemma 4 pipeline: no large checkpoint read, no package installation.
if [ "$MODE" = "micro" ]; then
  if [ ! -x "$PY" ]; then
    echo "Project .venv is missing. Run prepare before micro." >&2
    exit 1
  fi
  mkdir -p "$ROOT/data/h200"
  exec 9>"$ROOT/data/h200/runner.lock"
  if ! flock -n 9; then
    echo "Another H200 experiment is already running." >&2
    exit 1
  fi
  "$PY" -m scripts.smoke_tiny_jeff "$@"
  exit 0
fi

if [ "$MODE" = "plan" ] || [ "$MODE" = "status" ] || [ "$DRY_RUN" -eq 1 ]; then
  if [ ! -x "$PY" ]; then
    echo "[INFO] Virtual environment is not installed; safe plan/status requires Python 3."
    PY="$(command -v python3 || true)"
    if [ -z "$PY" ]; then
      echo "Python 3 is required." >&2
      exit 1
    fi
    PYTHONPATH="$ROOT/src:$ROOT" "$PY" -m scripts.h200 "$MODE" "$@"
  else
    "$PY" -m scripts.h200 "$MODE" "$@"
  fi
  exit 0
fi

# Keep only one writer of study artifacts / 31B GPU training at a time.
if ! command -v flock >/dev/null 2>&1; then
  echo "flock (util-linux) is required for exclusive H200 experiment execution." >&2
  exit 1
fi
mkdir -p "$ROOT/data/h200"
exec 9>"$ROOT/data/h200/runner.lock"
if ! flock -n 9; then
  echo "Another TCMDiscern H200 workflow already holds the experiment lock." >&2
  exit 1
fi

# Pin project environment once. Keep it: avoid 'uv run' later since that
# can resync away the separately installed upstream GPU packages.
if [ ! -x "$PY" ]; then
  echo "[SETUP] Creating Python 3.12 environment using frozen uv.lock"
  "$UV_BIN" sync --python 3.12 --locked --dev
fi
if ! "$PY" -c 'import tcmdiscern, scripts.h200' >/dev/null 2>&1; then
  echo "[SETUP] Installing project into the existing Python environment"
  "$UV_BIN" sync --python 3.12 --locked --dev
fi
if ! "$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  echo "This study requires Python 3.12. Move aside the old .venv and retry." >&2
  exit 1
fi

if [ "$MODE" = "prepare" ]; then
  if [ ! -s "$ROOT/data/tcm-sd/train.json" ] || [ ! -s "$ROOT/data/tcm-sd/dev.json" ] || [ ! -s "$ROOT/data/tcm-sd/syndrome_vocab.txt" ]; then
    echo "[DATA] Fetching official TCM-SD Train/Dev only (Test remains sealed)"
    "$PY" -m scripts.fetch_tcmsd
  fi
  "$PY" -m scripts.h200 prepare "$@"
  echo "[DONE] Official Train/Dev, BM25 OOF, Jeff S1-S4 and Dev prompts validated."
  exit 0
fi

# GPU modes require an actual H200-class machine. Installing these
# packages on the CPU developer machine is intentionally avoided.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver unavailable. Use 'prepare' on CPU; 'pilot/full' require H200." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git is required to fetch the pinned upstream Jeff release." >&2
  exit 1
fi
if ! command -v gcc >/dev/null 2>&1; then
  echo "A C compiler (gcc) is required by the upstream Jeff/Triton CUDA backend." >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '$1 >= 81920 { ok=1 } END { exit !ok }'; then
  echo "No GPU has at least 80 GiB VRAM; refuse to download a 31B model." >&2
  exit 1
fi
# Select a driver-compatible, project-local torch wheel; the host/container
# PyTorch installation is never modified. NVIDIA CUDA 12.x minor-version
# compatibility permits driver R525+; actual BF16 execution is smoke-tested
# before the 64 GB Jeff checkpoint is downloaded.
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | awk -F. 'NR == 1 { print $1 }')"
if [ "${DRIVER_MAJOR:-0}" -ge 580 ]; then
  TORCH_PROFILE="cu130"
elif [ "${DRIVER_MAJOR:-0}" -ge 525 ]; then
  TORCH_PROFILE="cu128"
else
  echo "NVIDIA driver is too old for CUDA 12.x (requires at least R525)." >&2
  exit 1
fi
echo "[GPU] NVIDIA driver R$DRIVER_MAJOR; selecting local PyTorch $TORCH_PROFILE"

JEFF_TAG="$("$PY" -c 'from tcmdiscern.jeff_config import JEFF_UPSTREAM_CODE_TAG; print(JEFF_UPSTREAM_CODE_TAG)')"
MODEL_REVISION="$("$PY" -c 'from tcmdiscern.jeff_config import JEFF_MODEL_REVISION; print(JEFF_MODEL_REVISION)')"
JEFF_SRC="$ROOT/data/upstream/Jeff"
if [ ! -f "$JEFF_SRC/requirements.txt" ]; then
  if [ -e "$JEFF_SRC" ]; then
    echo "Partial Jeff source folder exists. Resolve it before retrying: $JEFF_SRC" >&2
    exit 1
  fi
  mkdir -p "$ROOT/data/upstream"
  echo "[SETUP] Fetching pinned Jeff source v1.3.0"
  git clone --depth 1 --branch "$JEFF_TAG" https://github.com/j-geuter/Jeff "$JEFF_SRC"
fi
INSTALLED_JEFF_TAG="$(git -C "$JEFF_SRC" describe --tags --exact-match HEAD 2>/dev/null || true)"
if [ "$INSTALLED_JEFF_TAG" != "$JEFF_TAG" ]; then
  echo "Existing Jeff source is not the pinned release $JEFF_TAG: $JEFF_SRC" >&2
  exit 1
fi
if [ "$TORCH_PROFILE" = "cu128" ]; then
  echo "[SETUP] Installing isolated torch 2.11.0 CUDA 12.8 (ARM64-capable) and Jeff dependencies"
  # Jeff 1.3 declares torch>=2.11. The official CUDA 12.8 wheel meets that
  # minimum without requiring the newer R580 host driver.
  "$UV_BIN" pip install --python "$PY" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.11.0+cu128' 'transformers==5.17.0' \
    'safetensors==0.8.0' 'huggingface_hub==1.32.0' \
    'accelerate==1.15.0' 'peft==0.21.1'
else
  echo "[SETUP] Installing official Jeff CUDA 13.0 dependencies and PEFT"
  "$UV_BIN" pip install --python "$PY" -r "$JEFF_SRC/requirements.txt" 'peft==0.21.1'
fi
"$UV_BIN" pip install --python "$PY" --no-deps -e "$JEFF_SRC"

# Catch CUDA runtime, driver-minor-compatibility and Hopper BF16 failures
# before downloading ~64 GB of weights. This checks real device execution.
"$PY" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Hopper CUDA BF16 not available"; a=torch.ones((32,32), device="cuda", dtype=torch.bfloat16); assert torch.isfinite(a@a).all().item(), "BF16 GEMM failed"; torch.cuda.synchronize(); print(f"[GPU] BF16 GEMM PASS: torch={torch.__version__}, CUDA={torch.version.cuda}")'

MODEL_DIR="${H200_MODEL_DIR:-$ROOT/data/models/Jeff-1.1-Large}"
ARG_LIST=("$@")
for ((i=0; i<${#ARG_LIST[@]}; i++)); do
  if [ "${ARG_LIST[$i]}" = "--model-dir" ]; then
    if [ "$((i + 1))" -ge "${#ARG_LIST[@]}" ]; then
      echo "--model-dir requires a destination path." >&2
      exit 2
    fi
    MODEL_DIR="${ARG_LIST[$((i + 1))]}"
  fi
done
mkdir -p "$MODEL_DIR"
if ! "$PY" -c 'import sys; from pathlib import Path; from scripts.preflight_jeff_training import inspect_checkpoint; d=inspect_checkpoint(Path(sys.argv[1])); assert d["weights_present"] and d["decision_head_present"] and d["release_metadata_matches"] is True' "$MODEL_DIR" >/dev/null 2>&1; then
  FREE_KIB="$(df -Pk "$MODEL_DIR" | awk 'NR == 2 {print $4}')"
  if [ "${FREE_KIB:-0}" -lt 73400320 ]; then
    echo "[WARN] Less than 70 GiB free disk at model location. Download or training checkpoints may fail." >&2
  fi
  echo "[MODEL] Fetching/resuming released Jeff-1.1-Large pinned revision (~64 GB)"
  echo "[MODEL] A gated Hugging Face model requires prior 'hf auth login' on this host."
  "$ROOT/.venv/bin/hf" download jgeuter/Jeff-1.1-Large \
    --revision "$MODEL_REVISION" \
    --local-dir "$MODEL_DIR"
fi

# Pilot/full also need exactly the same private Train/Dev-derived input files.
if [ ! -s "$ROOT/data/tcm-sd/train.json" ] || [ ! -s "$ROOT/data/tcm-sd/dev.json" ] || [ ! -s "$ROOT/data/tcm-sd/syndrome_vocab.txt" ]; then
  "$PY" -m scripts.fetch_tcmsd
fi

# Gate every expensive 31B read on a complete tiny Gemma4/PEFT/Head smoke.
# This cannot prove 31B numerical safety, but catches software integration
# faults without another 42-minute failed checkpoint load.
echo "[MINI] Verifying miniature Gemma 4 pipeline before 31B weights"
"$PY" -m scripts.smoke_tiny_jeff --model-dir "$MODEL_DIR"
echo "[MINI] End-to-end tiny smoke PASS"
H200_MODEL_DIR="$MODEL_DIR" "$PY" -m scripts.h200 "$MODE" "$@"
