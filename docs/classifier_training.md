# Training the diagnosis classifier — instruction manual

*Written 2026-10-11. Scripts: `ml/clf_data.py`, `ml/clf_models.py`,
`ml/train_classifier.py`, `ml/run_clf_overnight.sh`.*

## What it trains, and why

The thesis question for this chapter: **does a waveform digitised from a photographed
printout keep the diagnostic information?** The classifier is trained on PTB-XL signals
(folds 1–8), checked on fold 9, and tested **once** on fold 10. Later its predictions
on digitised signals are compared with its predictions on the true signals of the same
records (the degradation curve).

| Run | Task | Input | Model | What it shows |
|---|---|---|---|---|
| `full_resnet` | 5 superclasses (multi-label) | 12 leads × 10 s | ResNet | The standard PTB-XL reference |
| `printed_resnet` | 5 superclasses | each lead only in its printed 2.5 s window + lead II 10 s | ResNet | What a 3×4 printout can support |
| `printed_resnet_aug` | 5 superclasses | printed | ResNet + digitiser-like noise | Robustness to the digitiser's errors |
| `mishra_arch` | Mishra's 5 classes (single-label) | lead II, 400 samples | Mishra et al.'s network | Their set-up on open data, patient-safe folds |
| `mishra_resnet` | Mishra's 5 classes | printed | ResNet | Same task with the 12-lead printed view |
| `*_s2` | — | — | — | Repeats with another seed, for spread |

Expected time: roughly **2–4 hours** in total on the RTX 3050, plus 10–30 min for the
one-time cache. (An estimate; the log shows the real figure.)

## Before you go to bed (10 minutes)

**1. Stop Windows from sleeping.** If the laptop sleeps, WSL stops and the run dies.
Settings → System → Power → *Screen and sleep*: **"When plugged in, put my device to
sleep after: Never"**. Keep it **plugged in**. Closing the lid: Control Panel → Power
Options → *Choose what closing the lid does* → **Do nothing** (plugged in), or just
leave it open.

**2. Don't train anything else.** No tracer training or other GPU job at the same time
(the GPU has ~1.8 GB usable).

**3. The code is already committed** (so the results record a clean commit:
the JSON's `git_dirty` will be `false`). If you change any script before running,
commit again first.

**4. Open a WSL terminal** (Start → "Ubuntu"), *not* Claude Code, and run:
```
cd /mnt/c/dev/ECG-App
source ~/venvs/eacg-ml/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
You should see one GPU listed. If the list is empty, see Troubleshooting.

**5. Quick check that the data code works** (about a minute):
```
python ml/clf_data.py --check 50
```
It should print `views ok: ...` and label counts.

**6. Start the overnight queue:**
```
mkdir -p out/clf_runs
nohup bash ml/run_clf_overnight.sh > out/clf_runs/overnight.log 2>&1 &
```
`nohup ... &` keeps it running even if you close the terminal window.

**7. Watch the first few minutes** to be sure it got going:
```
tail -f out/clf_runs/overnight.log
```
Wait until you see either `building the PTB-XL cache` with progress lines
(`1000 / 21799 records ...`), or, if the cache already exists, `smoke test` followed
by an epoch line like `{"epoch": 1, ..., "val_macro_auroc": 0.7...}`. The smoke test
running to the end means the whole set-up works. Press **Ctrl+C** to stop *watching*
(the run keeps going). Then go to bed.

## In the morning

```
cd /mnt/c/dev/ECG-App
tail -30 out/clf_runs/overnight.log
```
The end of the log has a **SUMMARY** table: validation and test macro AUROC per run
(and accuracy / macro F1 for the Mishra task), and `all done in N min`. Then tell Claude
"the classifier runs finished" — it will read the results, commit them and add a
progress-log entry.

What was written:

| Where | What |
|---|---|
| `results/ml/clf_<run>.csv` | Per-epoch history (loss, validation AUROC) |
| `results/ml/clf_<run>.json` | Settings, git commit, **validation and test metrics per class**, confusion matrix (Mishra task) |
| `out/clf_runs/<run>/` | `best.weights.h5`, `model.keras`, `test_pred.npz` (test predictions, for the degradation curve), `train.log` |
| `out/clf_runs/<run>.log`, `overnight.log` | Full logs |

**How to read it.**
- **Macro AUROC:** 0.5 is chance and 1.0 is perfect. Published PTB-XL superclass models
  on the full 12 × 10 s input reach roughly the low 0.9s (check the exact figure in the
  paper you cite).
- **Use the validation numbers to choose between runs.** The test numbers are what you
  report; never pick a run because its test score is higher.
- **Expect `printed` below `full`.** It sees about a third of the signal, and the size
  of that drop is itself a result.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `gpus []` (no GPU) | `bash ml/setup_wsl.sh` once, then reopen the terminal and `source ~/venvs/eacg-ml/bin/activate` again |
| `FileNotFoundError ... ptbxl100.npz` | The cache was not built: `python ml/clf_data.py` (10–30 min) |
| `ResourceExhaustedError` / OOM | Add `--batch 64` to the runs in `ml/run_clf_overnight.sh` (each `run ... ` line) |
| `Killed` during the cache or data loading (out of RAM) | Close other programs; or give WSL more memory in `C:\Users\<you>\.wslconfig`: `[wsl2]` / `memory=12GB`, then `wsl --shutdown` in PowerShell and retry |
| `SMOKE TEST FAILED` | Nothing else ran. Send Claude the last 40 lines of `out/clf_runs/overnight.log` |
| One run `FAILED`, others `DONE` | Its log is `out/clf_runs/<run>.log`; the others are still valid |
| Want to stop everything | `pkill -f run_clf_overnight; pkill -f train_classifier` |

Re-running a single run later:
```
python ml/train_classifier.py --name printed_resnet --task superclass --input printed --arch resnet
```

## The rules these scripts keep (for the write-up)

- Folds 1–8 / 9 / 10, never random: no patient appears in two splits.
- Superclasses are **multi-label**: sigmoid and binary cross-entropy, never softmax.
  The Mishra task is single-label only because their paper is, and it uses only records
  with exactly one of their five classes.
- The test fold is evaluated once per run, with the checkpoint chosen on validation.
- Every result has a JSON sidecar with the git commit, settings and library versions.
