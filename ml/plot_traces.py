#!/usr/bin/env python3
"""Figure: classical vs model A vs model B traces on a real photograph.

Reads the per-run trace files `eval_photos.py` saves (out/ml_photo/
<stem>_<run>_traces.npz) and draws each tracer against the PTB-XL truth, one
row per tracer: the full 10 s on the left, two beats on the right.

Each trace is shown after the same scale/lag alignment its aligned PCC was
scored with (`run_photo.best_alignment`), so what differs between rows is the
tracing, not the corner error all three share. Heights are mV relative to
each signal's median, which is how the R-peak ratio is measured.

    python ml/plot_traces.py ptbxl40_20260929_01 [--a a1 --b b1]

Writes out/ml_photo/<stem>_compare.png (regenerable, gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import run_photo as rp  # noqa: E402
from ecgproto import evaluate  # noqa: E402

DIAG = REPO / "out" / "ml_photo"
FS = 500.0

# dataviz reference palette, light mode, slots 1-3 (validated all-pairs);
# the truth is reference ink, not a series colour.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
COLOURS = {"classical": "#eb6834", "A": "#1baf7a", "B": "#2a78d6"}


def waveform(x, r, t0_col, mm):
    """Sheet (x, row) -> mV at 500 Hz from t0, as eval_photos.score does."""
    keep = np.isfinite(r)
    x, r = x[keep], r[keep]
    o = np.argsort(x)
    x, r = x[o], r[o]
    t = (x - t0_col) / (mm * 25.0)
    k0 = int(np.ceil(max(0.0, t[0]) * FS))
    n = int(np.floor(t[-1] * FS)) + 1
    ts = np.arange(k0, n) / FS
    return k0, -(np.interp(ts, t, r) - np.median(r)) / (mm * 10.0)


def aligned(truth, k0, mv):
    """Apply best_alignment; return (time s, truth seg, test seg, pcc, R, overlap)."""
    ref = truth[k0:k0 + len(mv)]
    pcc, scale, lag, n, _ = rp.best_alignment(ref, mv)
    res = np.interp(np.arange(int(round(len(mv) / scale))) * scale,
                    np.arange(len(mv)), mv)
    if lag >= 0:
        a, b, start = ref[lag:], res, k0 + lag
    else:
        a, b, start = ref, res[-lag:], k0
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    rr = evaluate.r_peak_ratio(a, b, FS)[0]
    return (start + np.arange(m)) / FS, a, b, pcc, rr, n / FS


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("stem")
    p.add_argument("--a", default="a1")
    p.add_argument("--b", default="b1")
    p.add_argument("--zoom", default="3.3,5.1", help="zoom window, seconds")
    args = p.parse_args(argv)
    za, zb = (float(v) for v in args.zoom.split(","))

    db = np.load(DIAG / f"{args.stem}_{args.b}_traces.npz")
    da = np.load(DIAG / f"{args.stem}_{args.a}_traces.npz")
    mm, t0 = float(db["px_per_mm"]), float(db["t0_col"])
    truth = db["truth_mv"]
    rows = [("classical", "Classical (threshold + column centroid)",
             db["classical_x"], db["classical_r"]),
            ("A", f"Model A ({args.a}): learned ink mask + column centroid",
             da["model_x"], da["model_r"]),
            ("B", f"Model B ({args.b}): learned row per column",
             db["model_x"], db["model_r"])]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, axes = plt.subplots(3, 2, figsize=(13, 7.2), sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.2]},
                             facecolor=SURFACE)
    for i, (key, title, x, r) in enumerate(rows):
        k0, mv = waveform(x, r, t0, mm)
        t, ref, test, pcc, rr, ovl = aligned(truth, k0, mv)
        ref = ref - np.median(ref)
        test = test - np.median(test)
        stat = (f"aligned PCC {pcc:.3f}   R-peak ratio {rr:.2f}"
                if ovl >= rp.MIN_OVERLAP_S else
                f"alignment spans only {ovl:.2f} s (< {rp.MIN_OVERLAP_S} s): not scorable")
        for j, (lo, hi) in enumerate([(t[0], t[-1]), (za, zb)]):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            k = (t >= lo) & (t <= hi)
            ax.plot(t[k], ref[k], color=INK, lw=1.0, alpha=0.55, label="truth (PTB-XL)")
            ax.plot(t[k], test[k], color=COLOURS[key], lw=1.4, label=title.split(":")[0])
            ax.set_xlim(lo, hi)
            ax.grid(True, color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_title(f"{title}  —  {stat}", loc="left", color=INK, fontsize=9.5)
                ax.set_ylabel("mV")
            else:
                ax.set_title("two beats", loc="left", color=INK2, fontsize=9)
        axes[i, 0].legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 1].set_xlabel("time (s)")
    fig.suptitle(f"Real A30s photograph {args.stem}: three tracers against the "
                 f"true waveform (same corners, warp and calibration)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = DIAG / f"{args.stem}_compare.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
