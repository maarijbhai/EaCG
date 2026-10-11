#!/usr/bin/env bash
# The overnight queue for the diagnosis experiment (docs/classifier_training.md).
# Run INSIDE WSL, from the repository root, in the training venv:
#
#   cd /mnt/c/dev/ECG-App
#   source ~/venvs/eacg-ml/bin/activate
#   nohup bash ml/run_clf_overnight.sh > out/clf_runs/overnight.log 2>&1 &
#
# Each run is independent: one failing does not stop the others. The cache is
# built once (skipped if it exists). A 1-epoch smoke test runs first so a
# broken setup fails in minutes, not at 3 a.m.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/clf_runs
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
PY=python
START=$(date +%s)
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] git $(git rev-parse --short HEAD) $(git status --porcelain -- ml python | wc -l | tr -d ' ') uncommitted file(s) in ml/ python/"
$PY -c "import tensorflow as tf; print('tensorflow', tf.__version__, 'gpus', tf.config.list_physical_devices('GPU'))"

# 1. The cache (all 21,799 PTB-XL records at 100 Hz), once.
if [ ! -f "$HOME/data/ptbxl_clf/ptbxl100.npz" ]; then
  echo "[$(stamp)] building the PTB-XL cache (10-30 min, reading over /mnt/c)"
  $PY ml/clf_data.py || { echo "[$(stamp)] CACHE BUILD FAILED -- nothing else can run"; exit 1; }
fi

# 2. Smoke test: one epoch on 2,000 records. If this fails, stop.
echo "[$(stamp)] smoke test"
$PY ml/train_classifier.py --name smoke --task superclass --input printed --epochs 1 --limit 2000 \
  || { echo "[$(stamp)] SMOKE TEST FAILED -- see above; the queue is stopped"; exit 1; }

run() {   # run <name> <args...>
  local name="$1"; shift
  echo "[$(stamp)] START $name: $*"
  if $PY ml/train_classifier.py --name "$name" "$@" > "out/clf_runs/$name.log" 2>&1; then
    echo "[$(stamp)] DONE  $name: $(tail -2 "out/clf_runs/$name.log" | head -1)"
  else
    echo "[$(stamp)] FAILED $name (see out/clf_runs/$name.log)"
  fi
}

# 3. The experiment. Superclass task, multi-label, macro AUROC:
run full_resnet        --task superclass --input full    --arch resnet                       # standard PTB-XL reference
run printed_resnet     --task superclass --input printed --arch resnet                       # what a printout carries
run printed_resnet_aug --task superclass --input printed --arch resnet --augment digitiser   # robust to digitiser errors
# Mishra et al.'s task (5 single-label classes):
run mishra_arch        --task mishra --input mishra  --arch mishra                           # their set-up re-implemented
run mishra_resnet      --task mishra --input printed --arch resnet                           # same task, 12-lead printed view
# Seed repeats of the two printed models, for run-to-run spread:
run printed_resnet_s2      --task superclass --input printed --arch resnet --seed 2
run printed_resnet_aug_s2  --task superclass --input printed --arch resnet --augment digitiser --seed 2

# 4. Summary (validation for choosing, test for reporting).
$PY - <<'PY'
import json, glob, os
print("\nSUMMARY  (val / TEST macro AUROC; mishra task also accuracy and macro F1)")
for f in sorted(glob.glob("results/ml/clf_*.json")):
    j = json.load(open(f)); n = os.path.basename(f)[4:-5]
    if n == "smoke": continue
    v, t = j["val"], j["test"]
    extra = f"  acc {t['accuracy']:.3f}  F1 {t['macro_f1']:.3f}" if "accuracy" in t else ""
    print(f"  {n:24s} val {v['macro_auroc']:.4f}  TEST {t['macro_auroc']:.4f}{extra}"
          f"   ({j['epochs_run']} epochs, best {j['best_epoch']}, {j['minutes']} min)")
PY
echo "[$(stamp)] all done in $(( ($(date +%s) - START) / 60 )) min"
