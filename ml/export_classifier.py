#!/usr/bin/env python3
"""Export the diagnosis classifier to LiteRT FP16 for the phone, with its thresholds.

THE DEPLOY MODEL. The chosen run's network (default `printed_resnet_aug_s2`:
the best VALIDATION macro AUROC of the printed-view superclass runs, 0.920 --
chosen on validation, never on test) with a sigmoid in the graph, so the
phone receives five probabilities (NORM, MI, STTC, CD, HYP) under the output
name "prob". Input "ecg": float32 (1, 1000, 13), the printed view
(`python/ecgproto/clf_input.py`). FP16 weights, float32 compute: the same
choice as the tracer (b3), where FP16 was exact.

THRESHOLDS, one per class, chosen on the VALIDATION fold (9) as the value
that maximises that class's F1 (a grid 0.01-0.99). The test fold (10) is
then evaluated once at those thresholds; choosing them on test would
inflate every number reported from it.

PARITY. Keras FP32 against the .tflite on the whole test fold: the largest
probability difference, decisions that change, and macro AUROC of each.

    bash ml/wsl_run.sh python ml/export_classifier.py

Writes android/app/src/main/assets/ml/clf_superclass_fp16.tflite and
clf_superclass.json (what the app reads), and results/ml/clf_export.csv +
.json (per class: threshold, validation F1, test sensitivity, specificity,
F1, AUROC Keras / TFLite).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

import clf_data as D  # noqa: E402
import clf_models as M  # noqa: E402
import train_classifier as TC  # noqa: E402

ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "ml"
ASSET_MODEL = "clf_superclass_fp16.tflite"
ASSET_META = "clf_superclass.json"
RESULTS = REPO / "results" / "ml"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def split(data: dict, folds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(view, labels, ecg ids) of the superclass-labelled records in `folds`."""
    keep = data["y_super"].sum(1) > 0
    idx = np.flatnonzero(keep & np.isin(data["fold"], list(folds)))
    X = D.views_batch("printed", data["X"][idx].astype(np.float32))
    return X, data["y_super"][idx].astype(np.float32), data["ecg_id"][idx]


def deploy_model(run: str, shape):
    """The run's network with the sigmoid in the graph; output "prob"."""
    import keras
    keras.mixed_precision.set_global_policy("float32")
    net = M.ARCHS["resnet"](shape, 5)
    net.load_weights(REPO / "out" / "clf_runs" / run / "best.weights.h5")
    inp = keras.Input(shape, dtype="float32", name="ecg")
    prob = keras.layers.Activation("sigmoid", name="prob")(net(inp))
    return keras.Model(inp, {"prob": prob})


def to_tflite_fp16(model) -> bytes:
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def tflite_predict(blob: bytes, X: np.ndarray) -> np.ndarray:
    """One record at a time, as on the phone."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=blob, num_threads=4)
    run = it.get_signature_runner()
    name = list(run.get_input_details())[0]
    return np.stack([run(**{name: x[None]})["prob"][0] for x in X]).astype(np.float64)


def confusion(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = y.astype(bool)
    d = p >= thr
    tp, fp = int((d & y).sum()), int((d & ~y).sum())
    fn, tn = int((~d & y).sum()), int((~d & ~y).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan")}


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(threshold, F1) maximising F1 on the grid; ties -> the lower threshold."""
    f1 = np.array([confusion(y, p, t)["f1"] for t in GRID])
    k = int(np.nanargmax(f1))
    return float(GRID[k]), float(f1[k])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="printed_resnet_aug_s2")
    ap.add_argument("--cache", type=Path, default=D.CACHE)
    args = ap.parse_args(argv)

    data = D.load(args.cache)
    names = [str(s) for s in data["superclasses"]]
    Xva, Yva, _ = split(data, (9,))
    Xte, Yte, _ = split(data, (10,))
    print(f"val {len(Xva)}, test {len(Xte)} records")

    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    model = deploy_model(args.run, Xva.shape[1:])
    pk_va = model.predict(Xva, batch_size=256, verbose=0)["prob"].astype(np.float64)
    pk_te = model.predict(Xte, batch_size=256, verbose=0)["prob"].astype(np.float64)

    blob = to_tflite_fp16(model)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / ASSET_MODEL).write_bytes(blob)
    print(f"wrote {ASSETS / ASSET_MODEL} ({len(blob) / 1e6:.2f} MB)")
    pt_te = tflite_predict(blob, Xte)
    pt_va = tflite_predict(blob, Xva)

    rows, thresholds = [], {}
    for k, n in enumerate(names):
        # Thresholds from the deployed (TFLite) model's validation outputs.
        thr, f1_val = best_threshold(Yva[:, k], pt_va[:, k])
        thresholds[n] = thr
        c = confusion(Yte[:, k], pt_te[:, k], thr)
        rows.append({"class": n, "threshold": thr, "val_f1": round(f1_val, 4),
                     "test_positives": int(Yte[:, k].sum()),
                     "test_sensitivity": round(c["sensitivity"], 4), "test_specificity": round(c["specificity"], 4),
                     "test_f1": round(c["f1"], 4),
                     "test_auroc_keras": round(TC.auroc(Yte[:, k], pk_te[:, k]), 4),
                     "test_auroc_tflite": round(TC.auroc(Yte[:, k], pt_te[:, k]), 4),
                     "max_abs_dp": float(np.abs(pk_te[:, k] - pt_te[:, k]).max()),
                     "decisions_changed": int(((pk_te[:, k] >= thr) != (pt_te[:, k] >= thr)).sum())})
        print(rows[-1])
    macro = {s: float(np.mean([r[f"test_auroc_{s}"] for r in rows])) for s in ("keras", "tflite")}
    val_macro = float(np.mean([TC.auroc(Yva[:, k], pt_va[:, k]) for k in range(5)]))
    print(f"test macro AUROC keras {macro['keras']:.4f}, tflite {macro['tflite']:.4f}; "
          f"max |dp| {max(r['max_abs_dp'] for r in rows):.2e}")

    commit = TC.git("rev-parse", "HEAD")
    dirty = bool(TC.git("status", "--porcelain", "--", "ml", "python"))
    meta = {"model": ASSET_MODEL, "run": args.run, "task": "superclass (multi-label, sigmoid)",
            "classes": names, "thresholds": thresholds,
            "threshold_rule": "per class, the F1-maximising value on the validation fold (9)",
            "input": {"name": "ecg", "shape": [1, 1000, 13], "fs_hz": 100,
                      "from_500hz": "mean of each block of 5 samples, NaN ignored",
                      "view": "printed: each lead in its 2.5 s window about its median, zero elsewhere; "
                              "channel 13 = lead II rhythm, all 10 s; NaN -> 0",
                      "lead_order": list(D.LEADS) + ["II_rhythm"]},
            "output": {"name": "prob", "shape": [1, 5]},
            "val_macro_auroc": round(val_macro, 4), "test_macro_auroc": round(macro["tflite"], 4),
            "git_commit": commit, "git_dirty": dirty}
    (ASSETS / ASSET_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "clf_export.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps({
        "generated_by": "ml/export_classifier.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty, "params": {"run": args.run, "threshold_grid": "0.01-0.99 step 0.01"},
        "n": {"val": int(len(Xva)), "test": int(len(Xte))},
        "test_macro_auroc": macro, "val_macro_auroc_tflite": val_macro,
        "model_bytes": len(blob), "asset": f"assets/ml/{ASSET_MODEL}",
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} and {ASSETS / ASSET_META}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
