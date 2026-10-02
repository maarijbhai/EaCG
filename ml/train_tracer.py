#!/usr/bin/env python3
"""Train the learned tracer: model B (row head) or model A (ink mask).

`docs/ml_plan.md` §2. Runs in WSL2 (`ml/setup_wsl.sh`, `ml/wsl_run.sh`);
the data are the bands from `make_strips.py`, copied to WSL's own disk,
augmented on the fly by `strips.augment` in a process pool.

THE MODEL. A small U-Net of MobileNetV2-style inverted-residual blocks
(depthwise-separable, BN, ReLU6), four stride-2 stages 24 -> 32 -> 64 -> 96
with two dilated blocks at 1/16 for vertical context (the row above and the
lead label must be told apart from the trace), and a decoder back to full
resolution with skips. Heads:

  row      (model B)  1x1 conv to 2 channels, reshaped to (H, 2W): two
                      output columns per input column, a softmax DOWN each
                      one. The waveform is the soft-argmax. Two columns, not
                      one, because one caps a perfect model's R-peak ratio
                      at 0.95 median (results/ml/label_ceiling.json).
  presence            per output column, from the decoder features pooled
                      over rows (mean and max) and a 1-D conv. Over the
                      pulse, the lead-in and past the strip end it is 0.
  mask     (model A)  1x1 conv, sigmoid: which pixels are trace. Read by the
                      classical column centroid; the ablation of §2.1.

LOSSES (§2.4). Row: cross-entropy per column against a sum of Gaussians
(sigma 1 px) at the rows of the samples in that column, built in-graph from
`strips.column_samples`, plus LAMBDA_L1 x |soft-argmax - mean truth row| --
the term that directly penalises an under-read R peak. Both only over
columns that hold a sample inside the window. Presence: BCE over all
columns. Mask: BCE weighted for the ~0.5% of pixels that are trace, plus
soft Dice.

VALIDATION is fold 9 only (§3.1), a fixed augmented set (one seeded draw
per strip) so epochs are comparable. Each epoch reports row MAE in pixels,
and -- reading the waveform back exactly as inference will -- PCC and the
R-peak ratio against the PTB-XL truth. The checkpoint kept is the one with
the lowest validation row MAE (model B) or highest PCC (model A).

    bash ml/wsl_run.sh python ml/train_tracer.py --head row --name b0 --epochs 30
    bash ml/wsl_run.sh python ml/train_tracer.py --head row --name smoke \\
        --epochs 1 --steps 20 --n-val 16            # smoke test

Writes out/ml_runs/<name>/ (checkpoints, gitignored) and
results/ml/train_<name>.csv + .json (per-epoch history and provenance).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import strips as S  # noqa: E402
from ecgproto import evaluate  # noqa: E402

K_SAMPLES = 4
SIGMA_PX = 1.0
LAMBDA_L1 = 0.1
RUNS = REPO / "out" / "ml_runs"
RESULTS = REPO / "results" / "ml"


# --------------------------------------------------------------------------
# data: runs in worker processes, no TensorFlow here
# --------------------------------------------------------------------------

_W = {}


def _worker_init(root: str):
    _W["root"] = Path(root)
    import cv2
    cv2.setNumThreads(1)


def make_example(job: tuple) -> dict:
    """(split, stem, seed, augment?) -> one example as numpy arrays."""
    split, stem, seed, aug = job
    st = S.load_strip(stem, split, root=_W["root"])
    ex = S.augment(st, np.random.default_rng(seed)) if aug else S.clean_canonical(st)
    rc, cc = ex["rows_c"], ex["cols_c"]
    tg = S.column_targets(rc, cc)
    return {
        "image": ex["image"][..., ::-1].copy(),          # BGR -> RGB
        "samples": S.column_samples(rc, cc, K_SAMPLES),
        "mean_row": np.nan_to_num(tg["mean_row"], nan=-1.0),
        "valid": tg["in_window"],
        "presence": tg["presence"],
        "mask": S.trace_mask(rc, cc),
        # For reading the waveform back against the truth.
        "t0c": np.float32(cc[0]),
        # Every sample's true position in the canonical frame: the tracer is
        # scored against these (image space), see `rows_at_samples`.
        "rows_c": rc.astype(np.float32),
        "cols_c": cc.astype(np.float32),
        "lead_mv": st.lead_mv.astype(np.float32),
        "fs": np.float32(st.fs),
    }


def pack(ex: dict, head: str, crop_w: int, rng: np.random.Generator):
    """Cut a random `crop_w`-column window from a full-width example.

    Training runs on crops because the full 384 x 2048 frame does not fit
    the RTX 3050's 4 GB even at batch 2 (measured: OOM in the first
    BatchNorm). The network is fully convolutional, so it is trained on
    crops and run on the full width; the only cross-column component, the
    presence head's 1-D conv, is local (5 columns). Output columns are cut
    at exactly OUT_UPSAMPLE x the input's, so labels stay aligned.
    """
    x0 = int(rng.integers(0, S.CANON_W - crop_w + 1)) if crop_w < S.CANON_W else 0
    u = S.OUT_UPSAMPLE
    xs, os_ = slice(x0, x0 + crop_w), slice(x0 * u, (x0 + crop_w) * u)
    x = ex["image"][:, xs]
    if head == "row":
        y_rows = np.concatenate([ex["samples"], ex["mean_row"][:, None],
                                 ex["valid"][:, None]], axis=1)[os_]
        return x, {"rows": y_rows.astype(np.float32),
                   "presence": ex["presence"][os_].astype(np.float32)}
    return x, {"mask": ex["mask"][:, xs].astype(np.float32)}


def make_pool(root: Path, workers: int):
    """The loader pool. MUST be created before TensorFlow is imported: a
    fork after CUDA and TF's thread pools start deadlocks every child
    (measured: 0% CPU and 0% GPU, indefinitely)."""
    return get_context("fork").Pool(workers, initializer=_worker_init,
                                    initargs=(str(root),))


def train_stream(pool, stems: list[str], head: str, seed: int, crop_w: int):
    """Endless augmented examples, each strip's draw seeded by (epoch, i)."""
    crop_rng = np.random.default_rng(seed + 1)

    def jobs():
        rng = np.random.default_rng(seed)
        epoch = 0
        while True:
            for i in rng.permutation(len(stems)):
                yield ("train", stems[i], seed * 7919 + epoch * 100_003 + int(i), True)
            epoch += 1

    # A bounded window of in-flight jobs. pool.imap over an endless job
    # generator has no backpressure: workers run ahead and finished examples
    # (~4 MB each) pile up until WSL's OOM killer ends the run (measured:
    # exit 137 before the first step).
    from collections import deque
    window = 3 * pool._processes
    it = jobs()
    q = deque(pool.apply_async(make_example, (next(it),)) for _ in range(window))
    while True:
        ex = q.popleft().get()
        q.append(pool.apply_async(make_example, (next(it),)))
        yield pack(ex, head, crop_w, crop_rng)


def val_set(pool, stems: list[str], seed: int) -> list[dict]:
    return pool.map(make_example,
                    [("val", s, seed + i, True) for i, s in enumerate(stems)])


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_model(head: str, h: int | None = None, w: int | None = None):
    """`w=None` builds a width-agnostic model (any multiple of 16): trained
    on crops, validated at full width. The LiteRT export will build it again
    at w = CANON_W and load the same weights."""
    import keras
    import tensorflow as tf
    from keras import layers as L
    h = S.CANON_H if h is None else h

    class ToColumns(L.Layer):
        """(B, H, W, u) -> (B, H, W*u), or (B, W, u) -> (B, W*u).
        Row-major, so the u channels of input column w become output
        columns u*w .. u*w+u-1 -- output column j = 2w + c."""
        def call(self, t):
            s = tf.shape(t)
            if len(t.shape) == 4:
                return tf.reshape(t, [s[0], s[1], s[2] * s[3]])
            return tf.reshape(t, [s[0], s[1] * s[2]])

    def cbr(x, ch, k=3, stride=1):
        x = L.Conv2D(ch, k, strides=stride, padding="same", use_bias=False)(x)
        return L.ReLU(6.0)(L.BatchNormalization()(x))

    def inv_res(x, out_ch, stride=1, expand=4, dilation=1):
        in_ch = x.shape[-1]
        y = L.Conv2D(in_ch * expand, 1, use_bias=False)(x)
        y = L.ReLU(6.0)(L.BatchNormalization()(y))
        y = L.DepthwiseConv2D(3, strides=stride, dilation_rate=dilation,
                              padding="same", use_bias=False)(y)
        y = L.ReLU(6.0)(L.BatchNormalization()(y))
        y = L.BatchNormalization()(L.Conv2D(out_ch, 1, use_bias=False)(y))
        return L.Add()([x, y]) if stride == 1 and in_ch == out_ch else y

    def up(x, skip, ch):
        x = L.UpSampling2D(2, interpolation="bilinear")(x)
        x = L.Concatenate()([x, skip])
        x = L.SeparableConv2D(ch, 3, padding="same", use_bias=False)(x)
        return L.ReLU(6.0)(L.BatchNormalization()(x))

    inp = keras.Input((h, w, 3), dtype="uint8", name="image")
    x = L.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    f0 = cbr(x, 12)                                   # 1
    f1 = inv_res(inv_res(f0, 24, 2, expand=2), 24)    # 1/2
    f2 = inv_res(inv_res(f1, 32, 2), 32)              # 1/4
    f3 = inv_res(inv_res(f2, 64, 2), 64)              # 1/8
    f4 = inv_res(inv_res(f3, 96, 2), 96)              # 1/16
    f4 = inv_res(f4, 96, dilation=2)
    f4 = inv_res(f4, 96, dilation=4)
    d = up(f4, f3, 64)
    d = up(d, f2, 32)
    d = up(d, f1, 24)
    d = up(d, f0, 16)                                 # full resolution

    if head == "mask":
        m = L.Conv2D(1, 1, dtype="float32")(d)
        m = L.Lambda(lambda t: t[..., 0], name="mask", dtype="float32")(m)
        return keras.Model(inp, {"mask": m}, name="tracer_mask")

    r = L.Conv2D(S.OUT_UPSAMPLE, 1, dtype="float32")(d)          # (H, W, 2)
    rows = ToColumns(name="rows", dtype="float32")(r)
    pm = L.Lambda(lambda t: keras.ops.concatenate(
        [keras.ops.mean(t, axis=1), keras.ops.max(t, axis=1)], axis=-1))(d)
    p = L.Conv1D(32, 5, padding="same", activation="relu")(pm)
    p = L.Conv1D(S.OUT_UPSAMPLE, 1, dtype="float32")(p)
    pres = ToColumns(name="presence", dtype="float32")(p)
    return keras.Model(inp, {"rows": rows, "presence": pres}, name="tracer_row")


# --------------------------------------------------------------------------
# losses and decoding
# --------------------------------------------------------------------------

def row_loss(y_true, logits):
    import tensorflow as tf
    s = y_true[..., :K_SAMPLES]                       # (B, W, K)
    mean = y_true[..., K_SAMPLES]
    valid = y_true[..., K_SAMPLES + 1]
    h = tf.cast(tf.shape(logits)[1], tf.float32)
    rr = tf.range(h)[None, :, None, None]             # (1, H, 1, 1)
    sv = s[:, None, :, :]                             # (B, 1, W, K)
    g = tf.exp(-0.5 * tf.square((rr - sv) / SIGMA_PX)) * tf.cast(sv >= 0, tf.float32)
    t = tf.reduce_sum(g, axis=-1)                     # (B, H, W)
    t = t / (tf.reduce_sum(t, axis=1, keepdims=True) + 1e-8)
    logp = tf.nn.log_softmax(logits, axis=1)
    ce = -tf.reduce_sum(t * logp, axis=1)             # (B, W)
    e = tf.reduce_sum(tf.exp(logp) * tf.range(h)[None, :, None], axis=1)
    l1 = tf.abs(e - mean)
    n = tf.reduce_sum(valid) + 1e-6
    return tf.reduce_sum(valid * (ce + LAMBDA_L1 * l1)) / n


def presence_loss(y_true, logits):
    import tensorflow as tf
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true, logits))


#: Trace is ~1.8% of a canonical crop's pixels (measured on training
#: batches), so a balanced weight is (1 - 0.018) / 0.018 ~ 54. At 20 the
#: background still dominated: after 450 steps no pixel was predicted trace.
MASK_POS_WEIGHT = 54.0


def mask_loss(y_true, logits):
    import tensorflow as tf
    bce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        y_true, logits, pos_weight=MASK_POS_WEIGHT))
    p = tf.sigmoid(logits)
    inter = tf.reduce_sum(p * y_true, axis=[1, 2])
    dice = 1.0 - (2 * inter + 1) / (tf.reduce_sum(p, [1, 2])
                                    + tf.reduce_sum(y_true, [1, 2]) + 1)
    return bce + tf.reduce_mean(dice)


#: Tiled inference: tile width and the margin discarded at each tile's inner
#: edges, canonical input columns. A full 512 x 2048 frame does not fit the
#: ~1.8 GB of GPU memory WSL leaves usable (measured: OOM in BatchNorm on the
#: first full-width validation of a 64 mm model), and the phone will need
#: the same for memory. 128 columns (16 mm) clears the network's horizontal
#: context: 4 stride-2 stages and two dilated 3x3 blocks at 1/16.
TILE_W = 1024
TILE_MARGIN = 128
#: Frames up to this many pixels run in one call, as a1/b1 were evaluated,
#: so their committed results reproduce exactly. Tiled against full width on
#: five b1 validation strips: mean row difference 0.0001-0.004 px, worst
#: column 0.88 px (0.1 mm), presence identical.
FULL_FRAME_MAX_PX = 384 * 2048


def predict_tiled(model, image: np.ndarray, tile_w: int = TILE_W,
                  margin: int = TILE_MARGIN) -> dict:
    """Run a fully convolutional tracer over overlapping width tiles.

    `image` is one canonical RGB uint8 frame (H, W, 3). Returns the same
    outputs a single full-width call would, as numpy without the batch
    axis: per output column for "rows"/"presence" (2 per input column),
    per input column for "mask". Each tile keeps only its middle, so every
    kept column saw at least `margin` columns of context either side
    (except at the frame's own ends, as a full-width call would).
    """
    h, w = image.shape[:2]
    if w <= tile_w or h * w <= FULL_FRAME_MAX_PX:
        out = model(image[None], training=False)
        return {k: np.asarray(v)[0] for k, v in out.items()}
    step = tile_w - 2 * margin
    starts = list(range(0, w - tile_w, step)) + [w - tile_w]
    acc = {}
    for x0 in starts:
        out = model(image[None, :, x0:x0 + tile_w], training=False)
        out = {k: np.asarray(v)[0] for k, v in out.items()}
        keep0 = 0 if x0 == 0 else margin
        keep1 = tile_w if x0 + tile_w >= w else tile_w - margin
        for k, v in out.items():
            u = v.shape[-1] // tile_w             # 2 for rows/presence, 1 for mask
            if k not in acc:
                acc[k] = np.zeros(v.shape[:-1] + (w * u,), v.dtype)
            a, b = (x0 + keep0) * u, (x0 + keep1) * u
            acc[k][..., a:b] = v[..., keep0 * u:keep1 * u]
    return acc


def decode_rows(logits: np.ndarray, pres_logits: np.ndarray) -> np.ndarray:
    """(H, W_out) logits -> per-output-column row (soft-argmax), NaN where
    presence < 0.5. The numpy twin of what the phone will do."""
    z = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=0, keepdims=True)
    row = (p * np.arange(logits.shape[0])[:, None]).sum(axis=0)
    row[pres_logits < 0.0] = np.nan
    return row


def decode_mask(logits: np.ndarray) -> np.ndarray:
    """(H, W) mask logits -> per-input-column ink centroid (the classical
    reading of a segmentation), NaN in empty columns."""
    m = (logits > 0.0).astype(np.float64)
    n = m.sum(axis=0)
    c = (m * np.arange(m.shape[0])[:, None]).sum(axis=0)
    return np.where(n > 0, c / np.maximum(n, 1), np.nan)


def rows_at_samples(row: np.ndarray, cols_c: np.ndarray, u: float) -> np.ndarray:
    """A per-output-column row trace read at each sample's own column.

    This is how the TRACER is scored: in image space, against where each
    sample was drawn. Reading time off the nominal 8 px/mm instead would
    charge the tracer for the augmentation's deliberate time-base error
    (up to 2% stretch, perspective), which is calibration's to answer for
    -- measured: an oracle with zero row error scored PCC 0.01-0.74 that
    way. End-to-end fidelity, calibration included, is the photo scorer's.
    Gaps are interpolated, as the classical `fill_nan` does.
    """
    j = np.flatnonzero(np.isfinite(row))
    return np.interp(S.output_columns(cols_c, u), j, row[j])


def score_val(model, val: list[dict], head: str, batch: int) -> dict:
    # One strip at a time, each result pulled to the host before the next:
    # model.predict concatenates every output on the GPU first, and 150
    # full-width row maps are 150 x 384 x 4096 floats, 0.9 GB (measured:
    # OOM in ConcatV2 at the end of epoch 1).
    mae, pccs, rrs, cov = [], [], [], []
    for i, v in enumerate(val):
        o = predict_tiled(model, v["image"])
        if head == "row":
            row = decode_rows(o["rows"], o["presence"])
            u = S.OUT_UPSAMPLE
            ok = (v["valid"] > 0) & np.isfinite(row)
            if ok.any():
                mae.append(float(np.abs(row[ok] - v["mean_row"][ok]).mean()))
            cov.append(float(np.isfinite(row)[v["presence"] > 0].mean()))
        else:
            row = decode_mask(o["mask"])
            u = 1
        if np.isfinite(row).sum() < 10:
            pccs.append(np.nan)
            rrs.append(np.nan)
            continue
        # Rows grow downward; negate so R peaks are maxima, as in mV.
        ref = -v["rows_c"].astype(np.float64)
        test = -rows_at_samples(row, v["cols_c"], u)
        pccs.append(evaluate.pcc(ref, test))
        rrs.append(evaluate.r_peak_ratio(ref, test, float(v["fs"]))[0])
    f = lambda a: float(np.nanmedian(a)) if len(a) else float("nan")
    return {"val_row_mae_px": f(mae), "val_pcc": f(pccs),
            "val_pcc_p10": float(np.nanpercentile(pccs, 10)),
            "val_r_ratio": f(rrs), "val_coverage": f(cov) if cov else ""}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", choices=("row", "mask"), default="row")
    p.add_argument("--name", required=True)
    p.add_argument("--data", type=Path, default=Path.home() / "data" / "ml_strips")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=500, help="steps per epoch")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--canon-h", type=int, default=S.CANON_H,
                   help="window height, canonical rows (384 = 48 mm for the "
                        "rhythm data, 512 = 64 mm for the four-row data)")
    p.add_argument("--crop-w", type=int, default=512,
                   help="training crop width, input columns (multiple of 16)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-val", type=int, default=150)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 3))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-mixed", action="store_true")
    p.add_argument("--init", default="",
                   help="start from another run's weights (e.g. b1): the "
                        "architecture does not depend on the window height")
    args = p.parse_args(argv)

    # Before the pool forks, so every worker inherits it.
    S.CANON_H = args.canon_h
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    train_stems = S.list_stems("train", root=args.data)
    val_stems = S.list_stems("val", root=args.data)
    # Fold discipline, asserted rather than assumed.
    assert all(s.split("_")[0] in {f"f{i}" for i in range(1, 9)} for s in train_stems)
    assert all(s.startswith("f9_") for s in val_stems)
    rng = np.random.default_rng(args.seed)
    val_stems = sorted(rng.choice(val_stems, min(args.n_val, len(val_stems)),
                                  replace=False))
    print(f"train {len(train_stems)} strips, val {len(val_stems)}; "
          f"{args.workers} loader workers", flush=True)

    pool = make_pool(args.data, args.workers)       # before TF: see make_pool
    t = time.perf_counter()
    val = val_set(pool, val_stems, seed=10_000)
    print(f"val set built in {time.perf_counter() - t:.0f} s", flush=True)

    import tensorflow as tf
    import keras
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    if not args.no_mixed:
        keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(args.seed)
    model = build_model(args.head)
    if args.init:
        model.load_weights(RUNS / args.init / "best.weights.h5")
        print(f"initialised from {args.init}", flush=True)
    print(f"params {model.count_params():,}", flush=True)

    h, w = S.CANON_H, args.crop_w
    wo = w * S.OUT_UPSAMPLE
    if args.head == "row":
        sig = (tf.TensorSpec((h, w, 3), tf.uint8),
               {"rows": tf.TensorSpec((wo, K_SAMPLES + 2), tf.float32),
                "presence": tf.TensorSpec((wo,), tf.float32)})
        losses = {"rows": row_loss, "presence": presence_loss}
        monitor, mode = "val_row_mae_px", "min"
    else:
        sig = (tf.TensorSpec((h, w, 3), tf.uint8),
               {"mask": tf.TensorSpec((h, w), tf.float32)})
        losses = {"mask": mask_loss}
        monitor, mode = "val_pcc", "max"
    ds = tf.data.Dataset.from_generator(
        lambda: train_stream(pool, train_stems, args.head, args.seed, w),
        output_signature=sig).batch(args.batch).prefetch(4)

    sched = keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=args.epochs * args.steps, alpha=0.02,
        warmup_target=args.lr, warmup_steps=min(500, args.steps))
    # jit_compile=False: under mixed precision, XLA rejects the bilinear
    # upsample's gradient ("Seen floating point types of different
    # precisions in Relu6Grad ... ResizeBilinearGrad"), TF 2.21.
    model.compile(optimizer=keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss=losses, jit_compile=False)

    def write_results(history, best):
        """After every epoch, so a run cut short still leaves its record."""
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"train_{args.name}.csv"
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(history[0]))
            wr.writeheader()
            wr.writerows(history)
        (RESULTS / f"train_{args.name}.json").write_text(json.dumps({
            "generated_by": "ml/train_tracer.py",
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "ml", "python")),
            "epochs_completed": len(history),
            "params": {**{k: str(v) for k, v in vars(args).items()},
                       "k_samples": K_SAMPLES, "sigma_px": SIGMA_PX,
                       "lambda_l1": LAMBDA_L1, "mask_pos_weight": MASK_POS_WEIGHT,
                   "params": model.count_params(),
                       "canon": [S.CANON_PX_PER_MM, S.CANON_H, S.CANON_W,
                                 S.OUT_UPSAMPLE],
                       "val_stems": list(val_stems)},
            "best": {monitor: best},
            "versions": {"python": platform.python_version(),
                         "tensorflow": tf.__version__, "keras": keras.__version__,
                         "numpy": np.__version__,
                         "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]},
        }, indent=2))
        return out

    history, best = [], None
    for epoch in range(args.epochs):
        t = time.perf_counter()
        hist = model.fit(ds, epochs=1, steps_per_epoch=args.steps, verbose=0)
        train_s = time.perf_counter() - t
        m = score_val(model, val, args.head, 1)        # full width, batch 1
        row = {"epoch": epoch + 1, "train_s": round(train_s, 1),
               **{k: round(float(v[-1]), 5) for k, v in hist.history.items()},
               **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}}
        cur = m[monitor]
        # An epoch with nothing scorable (NaN, e.g. an untrained mask head
        # marking no pixels) is never "best" and saves nothing.
        improved = bool(np.isfinite(cur)) and (
            best is None or (cur < best if mode == "min" else cur > best))
        if improved:
            best = cur
            model.save_weights(run / "best.weights.h5")
        row["best"] = improved
        history.append(row)
        print(json.dumps(row), flush=True)
        model.save_weights(run / "last.weights.h5")
        out = write_results(history, best)
    print(f"history {out}", flush=True)
    return 0


if __name__ == "__main__":
    # os._exit, not SystemExit: after an exception inside fit() the tf.data
    # generator thread and the loader pool keep the interpreter alive
    # indefinitely (measured: a GPU OOM left the run hung at 0% CPU).
    try:
        code = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
