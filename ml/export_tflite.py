#!/usr/bin/env python3
"""Export a row-head tracer to LiteRT (FP32 / FP16 / INT8) and check each.

`docs/ml_plan.md` §5: INT8 post-training quantisation, measured against FP32
before it is trusted; FP16 the fallback.

THE DEPLOY MODEL. The trained network at a FIXED tile shape (CANON_H x
--width; 512 x 1,024 by default, the tile `train_tracer.predict_tiled` uses,
since a 512 x 2,048 frame was too big for WSL's GPU and may be for the
phone), with the decode IN the graph: a softmax down each output column and
its expectation (the soft-argmax), so the phone receives two numbers per
output column -- `row` (canonical input rows, sub-pixel) and `presence`
(logit; < 0 means no trace) -- instead of a 512 x 2,048 probability map.
Input: float32 RGB 0-255, (1, H, W, 3); the model's own Rescaling layer
normalises.

CHECKS, per variant, on the same 60 clean + 60 augmented fold-9 row bands as
`eval_val_clean.py`, tiled exactly as on the phone would be: row error,
PCC and R-peak ratio in image space (train_tracer.rows_at_samples), coverage,
and the row difference against the Keras FP32 deploy model. Decision rule
(§5): INT8 is acceptable if within 0.02 PCC and 0.03 R of FP32; else FP16.

    bash ml/wsl_run.sh python ml/export_tflite.py --run b3

Writes out/ml_export/<run>_<h>x<w>_<variant>.tflite (gitignored) and
results/ml/export_<run>.csv + .json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import strips as S  # noqa: E402
import train_tracer as T  # noqa: E402
import score_12lead as S12  # noqa: E402
from ecgproto import evaluate  # noqa: E402

RESULTS = REPO / "results" / "ml"
OUT = REPO / "out" / "ml_export"
N_VAL, VAL_STEP = 60, 20


def deploy_model(run: str, h: int, w: int):
    """The trained net at (h, w) with the soft-argmax decode in the graph."""
    import keras
    import tensorflow as tf
    keras.mixed_precision.set_global_policy("float32")
    net = T.build_model("row", h=h, w=w)
    net.load_weights(REPO / "out" / "ml_runs" / run / "best.weights.h5")
    inp = keras.Input((h, w, 3), dtype="float32", name="image")
    out = net(inp)
    logits = out["rows"]                                   # (B, H, 2W)

    class SoftArgmax(keras.layers.Layer):
        def call(self, z):
            p = tf.nn.softmax(z, axis=1)
            r = tf.range(tf.shape(z)[1], dtype=z.dtype)[None, :, None]
            return tf.reduce_sum(p * r, axis=1)

    row = SoftArgmax(name="row")(logits)
    pres = keras.layers.Identity(name="presence")(out["presence"])
    return keras.Model(inp, {"row": row, "presence": pres})


def representative(n: int, w: int, seed: int = 3):
    """Augmented TRAINING row bands (folds 1-8), cut to tile width."""
    root = Path.home() / "data" / "ml_rows"
    T._worker_init(str(root))
    stems = S.list_stems("train", root)
    rng = np.random.default_rng(seed)
    pick = rng.choice(stems, size=n, replace=False)
    tiles = []
    for i, s in enumerate(pick):
        ex = T.make_example(("train", s, seed * 1000 + i, True))
        x0 = int(rng.integers(0, S.CANON_W - w + 1))
        tiles.append(ex["image"][:, x0:x0 + w].astype(np.float32))
    return tiles


class TfliteTracer:
    """A .tflite deploy model behind predict_tiled's model(x) interface."""

    def __init__(self, path: Path):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path), num_threads=4)
        # The signature keeps the Keras output names ("row", "presence");
        # the raw tensor names do not, and both outputs have one shape.
        self.run = self.it.get_signature_runner()
        self.in_name = list(self.run.get_input_details())[0]

    def __call__(self, x, training=False):
        out = self.run(**{self.in_name: np.asarray(x, np.float32)})
        return {k: np.asarray(v) for k, v in out.items()}


class KerasTracer:
    def __init__(self, model):
        self.m = model

    def __call__(self, x, training=False):
        o = self.m(np.asarray(x, np.float32), training=False)
        return {k: np.asarray(v) for k, v in o.items()}


def run_tiled(tracer, image: np.ndarray, w: int) -> np.ndarray:
    o = T.predict_tiled(tracer, image.astype(np.float32), tile_w=w)
    row = o["row"].astype(np.float64).copy()
    row[o["presence"] < 0] = np.nan
    return row


def metrics(rows: list[np.ndarray], val: list[dict], ref: list[np.ndarray] | None):
    mae, pcc, rr, cov, diff = [], [], [], [], []
    for row, v, rf in zip(rows, val, ref or [None] * len(rows)):
        ok = (v["valid"] > 0) & np.isfinite(row)
        if ok.any():
            mae.append(float(np.abs(row[ok] - v["mean_row"][ok]).mean()))
        cov.append(float(np.isfinite(row)[v["presence"] > 0].mean()))
        if rf is not None:
            both = np.isfinite(row) & np.isfinite(rf)
            if both.any():
                diff.append(float(np.abs(row[both] - rf[both]).mean()))
        if np.isfinite(row).sum() < 10:
            continue
        ref_img = -v["rows_c"].astype(np.float64)
        test = -T.rows_at_samples(row, v["cols_c"], S.OUT_UPSAMPLE)
        pcc.append(evaluate.pcc(ref_img, test))
        rr.append(evaluate.r_peak_ratio(ref_img, test, float(v["fs"]))[0])
    med = lambda a: round(float(np.nanmedian(a)), 4) if a else float("nan")
    return {"row_mae_px": med(mae), "pcc": med(pcc),
            "pcc_p10": round(float(np.nanpercentile(pcc, 10)), 4) if pcc else float("nan"),
            "r_ratio": med(rr), "coverage": med(cov),
            "diff_vs_keras_px": round(float(np.mean(diff)), 5) if diff else ""}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="b3")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--n-rep", type=int, default=200)
    p.add_argument("--variants", default="fp32,fp16,int8,int8f")
    p.add_argument("--reuse", action="store_true",
                   help="reuse .tflite files already in out/ml_export")
    args = p.parse_args(argv)

    rec = json.loads((RESULTS / f"train_{args.run}.json").read_text())["params"]
    h = int(rec.get("canon_h", 384))
    S.CANON_H = h
    w = args.width

    # Data first: worker-free here, but the val/representative builders use
    # cv2 and must not race TF's thread pools for memory on the small WSL VM.
    root = Path.home() / "data" / "ml_rows"
    T._worker_init(str(root))
    stems = S.list_stems("val", root)[::VAL_STEP][:N_VAL]
    val = {"clean": [T.make_example(("val", s, 0, False)) for s in stems],
           "augmented": [T.make_example(("val", s, 10_000 + i, True))
                         for i, s in enumerate(stems)]}
    rep = representative(args.n_rep, w)
    print(f"val {len(stems)} x2, representative {len(rep)} tiles of {h}x{w}", flush=True)

    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    model = deploy_model(args.run, h, w)

    OUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for v in args.variants.split(","):
        path = OUT / f"{args.run}_{h}x{w}_{v}.tflite"
        if path.exists() and args.reuse:
            files[v] = path
            print(f"{v}: reusing {path.name}", flush=True)
            continue
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if v == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif v == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: ([t[None]] for t in rep)
            # Float input/output and float fallback for any op without an
            # int8 kernel; the body (convolutions) is int8.
        t0 = time.perf_counter()
        if v == "int8f":
            # INT8 body, FLOAT decode. In plain int8 the soft-argmax output is
            # itself 8-bit: a 0-511 row quantised in ~2 px steps (measured:
            # row error 0.33 -> 1.07 px, 1.0-2.1 px from Keras, though PCC
            # and R stayed within the section 5 rule). The decode is a
            # softmax, a multiply and a sum per column; nearly all compute is
            # in the convolutions, which stay int8.
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            dbg_opts = tf.lite.experimental.QuantizationDebugOptions(
                denylisted_ops=["SOFTMAX", "MUL", "SUM"])
            dbg = tf.lite.experimental.QuantizationDebugger(
                converter=conv, debug_dataset=lambda: ([t[None]] for t in rep[:50]),
                debug_options=dbg_opts)
            conv.representative_dataset = lambda: ([t[None]] for t in rep)
            blob = dbg.get_nondebug_quantized_model()
        else:
            blob = conv.convert()
        path.write_bytes(blob)
        files[v] = path
        print(f"{v}: {len(blob)/1e6:.2f} MB in {time.perf_counter()-t0:.0f} s", flush=True)

    rows_out = []
    ref = {name: [run_tiled(KerasTracer(model), ex["image"], w) for ex in vs]
           for name, vs in val.items()}
    for name, vs in val.items():
        r = {"variant": "keras_fp32", "set": name, "size_mb": "",
             **metrics(ref[name], vs, None)}
        rows_out.append(r)
        print(r, flush=True)
    for v, path in files.items():
        tr = TfliteTracer(path)
        for name, vs in val.items():
            t0 = time.perf_counter()
            rows = [run_tiled(tr, ex["image"], w) for ex in vs]
            el = (time.perf_counter() - t0) / len(vs)
            r = {"variant": v, "set": name, "size_mb": round(path.stat().st_size / 1e6, 3),
                 "desktop_s_per_row": round(el, 3), **metrics(rows, vs, ref[name])}
            rows_out.append(r)
            print(r, flush=True)

    S12.RESULTS = RESULTS
    print(S12.persist(rows_out, {"run": args.run, "tile": [h, w], "n_rep": args.n_rep,
                                 "n_val": len(stems), "tf": tf.__version__,
                                 "note": "desktop_s_per_row is x86 CPU, not the phone"},
                      f"export_{args.run}"))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)
