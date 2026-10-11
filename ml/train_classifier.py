#!/usr/bin/env python3
"""Train one diagnosis classifier on PTB-XL (the diagnosis experiment).

One run = one (task, input view, architecture, augmentation). The overnight
queue is ml/run_clf_overnight.sh; the manual is docs/classifier_training.md.

  --task superclass  five PTB-XL superclasses, MULTI-LABEL: sigmoid + binary
                     cross-entropy (a record can be MI and STTC at once);
                     metric macro AUROC (the PTB-XL benchmark's).
  --task mishra      Mishra et al.'s five classes, SINGLE-LABEL: softmax +
                     cross-entropy, records with exactly one of the five;
                     metrics accuracy (what they report), macro F1 and macro
                     AUROC (one-vs-rest), because accuracy alone hides the
                     small classes. Class-balanced weights by default.
  --input full|printed|mishra   the view (ml/clf_data.py).
  --arch resnet|mishra          ml/clf_models.py.
  --augment none|digitiser      'digitiser': per-lead gain 0.8-1.1 (the
                     tracer reads peaks at ~0.9), Gaussian noise 0.01-0.03 mV,
                     and random 0.1-0.3 s gaps zeroed in ~20% of examples --
                     the errors a digitised printout carries.

SPLIT, never random: folds 1-8 train, 9 validation (checkpoint and early
stopping), 10 test -- evaluated ONCE, at the end, with the best checkpoint.
Choosing between runs must use validation numbers, not test numbers.

Writes out/clf_runs/<name>/ (best.weights.h5, model.keras, test_pred.npz,
train.log) and results/ml/clf_<name>.csv (per-epoch history) + .json
(provenance, validation and test metrics per class).

    python ml/train_classifier.py --name printed_resnet --task superclass --input printed
    python ml/train_classifier.py --name smoke --task superclass --input printed --epochs 1 --limit 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))

import clf_data as D  # noqa: E402
import clf_models as M  # noqa: E402

RUNS = REPO / "out" / "clf_runs"
RESULTS = REPO / "results" / "ml"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------- metrics (no sklearn needed)

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic; NaN if one class only."""
    y = np.asarray(y).astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    sv = np.asarray(s)[order]
    i = 0
    while i < len(sv):                       # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def multilabel_metrics(Y: np.ndarray, P: np.ndarray, names) -> dict:
    per = {n: auroc(Y[:, k], P[:, k]) for k, n in enumerate(names)}
    return {"macro_auroc": float(np.nanmean(list(per.values()))), "auroc": per}


def singlelabel_metrics(y: np.ndarray, P: np.ndarray, names) -> dict:
    pred = P.argmax(1)
    f1 = {}
    for k, n in enumerate(names):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        f1[n] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    conf = [[int(((y == a) & (pred == b)).sum()) for b in range(len(names))] for a in range(len(names))]
    per = {n: auroc(y == k, P[:, k]) for k, n in enumerate(names)}
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.nanmean(list(f1.values()))),
            "macro_auroc": float(np.nanmean(list(per.values()))), "f1": f1, "auroc": per,
            "confusion": conf}


# ----------------------------------------------------------------- data

def split_data(data: dict, task: str, view: str, limit: int):
    folds = data["fold"]
    if task == "superclass":
        keep = data["y_super"].sum(1) > 0               # benchmark convention: labelled records
        Y = data["y_super"].astype(np.float32)
    else:
        keep = data["y_mishra"] >= 0
        Y = data["y_mishra"].astype(np.int64)
    out = {}
    for split, fs in (("train", range(1, 9)), ("val", (9,)), ("test", (10,))):
        idx = np.flatnonzero(keep & np.isin(folds, list(fs)))
        if limit:
            idx = idx[:limit if split == "train" else max(200, limit // 5)]
        X = D.views_batch(view, data["X"][idx].astype(np.float32))
        out[split] = (X, Y[idx], data["ecg_id"][idx])
        print(f"  {split}: {len(idx)} records, input {X.shape[1:]}", flush=True)
    return out


def augment_fn(kind: str):
    import tensorflow as tf
    if kind == "none":
        return None

    def f(x, y):
        t, c = tf.shape(x)[0], tf.shape(x)[1]
        gain = tf.random.uniform([1, c], 0.8, 1.1)
        noise = tf.random.normal(tf.shape(x)) * tf.random.uniform([], 0.01, 0.03)
        x = x * gain + noise * tf.cast(tf.not_equal(x, 0.0), x.dtype)   # no noise where nothing is printed
        # A gap the digitiser left: 0.1-0.3 s zeroed in one random channel, 20% of examples.
        do_gap = tf.random.uniform([]) < 0.2
        ln = tf.random.uniform([], 10, 31, dtype=tf.int32)
        st = tf.random.uniform([], 0, t - ln, dtype=tf.int32)
        ch = tf.random.uniform([], 0, c, dtype=tf.int32)
        rows = tf.logical_and(tf.range(t) >= st, tf.range(t) < st + ln)
        mask = tf.logical_and(rows[:, None], tf.equal(tf.range(c), ch)[None, :])
        x = tf.where(tf.logical_and(do_gap, mask), tf.zeros_like(x), x)
        return x, y
    return f


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--task", choices=("superclass", "mishra"), default="superclass")
    p.add_argument("--input", choices=D.VIEWS, default="printed")
    p.add_argument("--arch", choices=tuple(M.ARCHS), default="resnet")
    p.add_argument("--augment", choices=("none", "digitiser"), default="none")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10, help="early stopping on validation macro AUROC")
    p.add_argument("--class-weight", choices=("balanced", "none"), default=None,
                   help="default: balanced for --task mishra, none for superclass")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="first N training records only (smoke test)")
    p.add_argument("--cache", type=Path, default=D.CACHE)
    args = p.parse_args(argv)
    cw_mode = args.class_weight or ("balanced" if args.task == "mishra" else "none")

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"run {args.name}: task {args.task}, input {args.input}, arch {args.arch}, "
          f"augment {args.augment}, class weights {cw_mode}", flush=True)
    data = D.load(args.cache)
    names = [str(s) for s in (data["superclasses"] if args.task == "superclass" else data["mishra_classes"])]
    sp = split_data(data, args.task, args.input, args.limit)
    Xtr, Ytr, _ = sp["train"]
    Xva, Yva, _ = sp["val"]
    Xte, Yte, Ite = sp["test"]

    import tensorflow as tf
    import keras
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.utils.set_random_seed(args.seed)

    multilabel = args.task == "superclass"
    model = M.ARCHS[args.arch](Xtr.shape[1:], len(names))
    print(f"  params {model.count_params():,}", flush=True)
    steps = int(np.ceil(len(Xtr) / args.batch))
    sched = keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=max(1, args.epochs * steps), alpha=0.02)
    loss = (keras.losses.BinaryCrossentropy(from_logits=True) if multilabel
            else keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4), loss=loss)

    class_weight = None
    if not multilabel and cw_mode == "balanced":
        counts = np.bincount(Ytr, minlength=len(names)).astype(float)
        class_weight = {k: float(len(Ytr) / (len(names) * max(1.0, c))) for k, c in enumerate(counts)}
        print(f"  class weights {class_weight}", flush=True)

    ds = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(len(Xtr), seed=args.seed,
                                                                reshuffle_each_iteration=True)
    aug = augment_fn(args.augment)
    if aug is not None:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch).prefetch(tf.data.AUTOTUNE)

    def predict(X):
        z = model.predict(X, batch_size=256, verbose=0)
        return (1 / (1 + np.exp(-z))) if multilabel else (np.exp(z - z.max(1, keepdims=True)) /
                                                          np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))

    def metrics(Y, P):
        return multilabel_metrics(Y, P, names) if multilabel else singlelabel_metrics(Y, P, names)

    history, best, best_epoch, waited = [], -np.inf, 0, 0
    log = open(run / "train.log", "a", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        h = model.fit(ds, epochs=1, verbose=0, class_weight=class_weight)
        mv = metrics(Yva, predict(Xva))
        row = {"epoch": epoch, "train_s": round(time.time() - t0, 1),
               "loss": round(float(h.history["loss"][-1]), 5),
               "val_macro_auroc": round(mv["macro_auroc"], 5)}
        if not multilabel:
            row.update(val_accuracy=round(mv["accuracy"], 5), val_macro_f1=round(mv["macro_f1"], 5))
        improved = mv["macro_auroc"] > best
        if improved:
            best, best_epoch, waited = mv["macro_auroc"], epoch, 0
            model.save_weights(run / "best.weights.h5")
        else:
            waited += 1
        row["best"] = improved
        history.append(row)
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
        if waited >= args.patience:
            print(f"  early stop: no validation gain for {args.patience} epochs", flush=True)
            break

    # The best checkpoint, evaluated once on validation (for choosing) and test (to report).
    model.load_weights(run / "best.weights.h5")
    model.save(run / "model.keras")
    val = metrics(Yva, predict(Xva))
    Pte = predict(Xte)
    test = metrics(Yte, Pte)
    np.savez(run / "test_pred.npz", ecg_id=Ite, y=Yte, p=Pte, classes=np.array(names))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"clf_{args.name}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    out.with_suffix(".json").write_text(json.dumps({
        "generated_by": "ml/train_classifier.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "ml", "python")),
        "params": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   "class_weight_mode": cw_mode, "classes": names, "model_params": int(model.count_params()),
                   "n": {s: int(len(sp[s][1])) for s in sp}},
        "best_epoch": best_epoch, "epochs_run": len(history),
        "minutes": round((time.time() - t_start) / 60, 1),
        "val": val, "test": test,
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__,
                     "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]},
    }, indent=2))
    print(f"best epoch {best_epoch}: val macro AUROC {val['macro_auroc']:.4f}, "
          f"TEST macro AUROC {test['macro_auroc']:.4f}"
          + ("" if multilabel else f", test accuracy {test['accuracy']:.4f}, macro F1 {test['macro_f1']:.4f}"),
          flush=True)
    print(f"wrote {out} and {out.with_suffix('.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
