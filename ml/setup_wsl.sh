#!/usr/bin/env bash
# Create the training environment in WSL2 Ubuntu (docs/ml_plan.md §2.4).
# Native Windows TensorFlow has no GPU support; WSL2 sees the RTX 3050.
# Prerequisite, once, with sudo:  sudo apt install -y python3.12-venv
#
#   bash /mnt/c/dev/ECG-App/ml/setup_wsl.sh
set -euo pipefail
VENV="${VENV:-$HOME/venvs/eacg-ml}"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "tensorflow[and-cuda]" opencv-python-headless scipy pandas wfdb pillow
# TF 2.21 finds most of the pip CUDA libraries itself but not
# libcusolver.so.11 (measured 2026-09-30: "Could not load dynamic library
# 'libcusolver.so.11'", and with one library missing it registers NO GPU).
# Put every nvidia/*/lib on the loader path, from the activate script.
if ! grep -q "eacg-ml nvidia libs" "$VENV/bin/activate"; then
  cat >> "$VENV/bin/activate" <<'SH'
# eacg-ml nvidia libs
for _d in "$VIRTUAL_ENV"/lib/python3*/site-packages/nvidia/*/lib; do
  LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
unset _d
SH
fi
source "$VENV/bin/activate"
"$VENV/bin/python" - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
PY
