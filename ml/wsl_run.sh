#!/usr/bin/env bash
# Run a command inside the WSL training venv, from the repository root.
#   wsl -d Ubuntu -- bash /mnt/c/dev/ECG-App/ml/wsl_run.sh python ml/train_tracer.py ...
# Exists so Windows-side callers never have to quote through PowerShell and
# bash at once, and so the venv's CUDA library path (setup_wsl.sh) applies.
set -eo pipefail
source "${VENV:-$HOME/venvs/eacg-ml}/bin/activate"
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
exec "$@"
