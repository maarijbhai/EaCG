#!/usr/bin/env python3
"""The digitised sheet: all 12 leads + rhythm from one photograph, vs the truth.

Runs the full-sheet pipeline exactly as `ml/eval_sheets.py` scores it --
registered corners, `segment.locate_sheet`, the separator-tick time scale
(`segment.tick_scale`, the calibration the phone would use), `segment_leads`,
then the learned tracer once per row and the classical tracer for
comparison -- and writes, per photograph:

    out/ml_sheet/<photo>_<run>_leads.csv   the digitised ECG at 500 Hz: one
        column per lead over the 10 s recording, each 3x4 lead filled only
        in its own 2.5 s window (that is all a 3x4 printout holds), plus the
        full 10 s rhythm lead II. mV relative to each cell's median.
    out/ml_sheet/<photo>_<run>.png         the sheet laid out as printed:
        three rows of four leads and the rhythm strip, truth in reference
        ink, the learned tracer in blue; per-lead as-is PCC in the label.
    out/ml_sheet/<photo>_classical.png     the same for the classical tracer.

No alignment search: what is drawn is what the pipeline produced, at the
calibration it measured. Regenerable; out/ is gitignored.

    bash ml/wsl_run.sh python ml/plot_sheet.py --photo ptbxl40_20260929_01 --run b3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
import score_12lead as S12  # noqa: E402
from ecgproto import evaluate, segment  # noqa: E402

OUT = REPO / "out" / "ml_sheet"
FS = 500
INK, MUTED, GRID, GRID5, SURFACE = "#0b0b0b", "#898781", "#f1d6d6", "#e6b3b3", "#fcfcfb"
COLOUR = {"model": "#2a78d6", "classical": "#eb6834"}


def digitise(sheet, target, tracer_fn):
    """-> {(lead, rhythm): mV array over its window}, the layout, the scale."""
    sl = segment.locate_sheet(sheet)
    ticks = segment.tick_scale(sheet, sl)
    px = ticks.px_per_mm if np.isfinite(ticks.px_per_mm) else sl.anchor_px_per_mm
    layout = segment.segment_leads(sheet, sl.as_strip_location(), px,
                                   ticks if np.isfinite(ticks.px_per_mm) else None)
    rows = {r: S12.on_sheet_columns(*tracer_fn(sheet, layout, r)) for r in range(4)}
    out = {}
    for cell in layout.cells:
        pos, x0 = rows[cell.row]
        out[(cell.name, cell.row == 3)] = S12.cell_signal(pos, x0, cell, px, px, layout)
    return out, layout, px, ticks


def write_csv(path: Path, sig: dict) -> None:
    n = int(segment.RECORDING_S * FS)
    cols = {}
    for r, names in enumerate(segment.LAYOUT_3X4):
        for k, name in enumerate(names):
            col = np.full(n, np.nan)
            a = int(k * segment.CELL_S * FS)
            v = sig[(name, False)]
            col[a:a + len(v)] = v
            cols[name] = col
    cols["II_rhythm"] = sig[("II", True)][:n]
    order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
             "II_rhythm"]
    with path.open("w", encoding="utf-8") as f:
        f.write("time_s," + ",".join(order) + "\n")
        for i in range(n):
            vals = [cols[c][i] if i < len(cols[c]) else np.nan for c in order]
            f.write(f"{i / FS:.3f}," + ",".join("" if not np.isfinite(v) else f"{v:.4f}"
                                              for v in vals) + "\n")


def draw(path: Path, sig: dict, truth, title: str, colour: str, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(16, 10.5), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.32})
    for r, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        # ECG paper: 0.04 s / 0.1 mV minor, 0.2 s / 0.5 mV major.
        for x in np.arange(0, 10.001, 0.04):
            ax.axvline(x, color=GRID5 if abs(x / 0.2 - round(x / 0.2)) < 1e-6 else GRID,
                       lw=0.6 if abs(x / 0.2 - round(x / 0.2)) < 1e-6 else 0.3, zorder=0)
        for y in np.arange(-2.0, 2.001, 0.1):
            ax.axhline(y, color=GRID5 if abs(y / 0.5 - round(y / 0.5)) < 1e-6 else GRID,
                       lw=0.6 if abs(y / 0.5 - round(y / 0.5)) < 1e-6 else 0.3, zorder=0)
        cells = ([(n, k * segment.CELL_S, False) for k, n in enumerate(segment.LAYOUT_3X4[r])]
                 if r < 3 else [("II", 0.0, True)])
        lo, hi = [], []
        for name, t0, rh in cells:
            s = sig[(name, rh)]
            tr = truth.leads[(name, rh)]["mv"]
            tr = tr - np.median(tr)
            t = t0 + np.arange(len(s)) / FS
            ax.plot(t0 + np.arange(len(tr)) / FS, tr, color=INK, lw=0.9, alpha=0.55,
                    zorder=2, label="truth (PTB-XL)" if (r == 0 and t0 == 0) else None)
            ax.plot(t, s, color=colour, lw=1.2, zorder=3,
                    label=label if (r == 0 and t0 == 0) else None)
            ok = np.isfinite(s)
            pcc = evaluate.pcc(tr[:len(s)][ok[:len(tr)]], s[ok]) if ok.sum() > 50 else float("nan")
            ax.text(t0 + 0.05, 1.0, f"{name}{' (rhythm)' if rh else ''}  PCC {pcc:.2f}",
                    transform=ax.get_xaxis_transform(), va="top", fontsize=9, color=INK)
            lo.append(min(np.nanmin(tr), np.nanmin(s) if ok.any() else 0))
            hi.append(max(np.nanmax(tr), np.nanmax(s) if ok.any() else 0))
        ax.set_xlim(0, 10)
        ax.set_ylim(min(lo) - 0.15, max(hi) + 0.35)
        for k in (1, 2, 3):
            if r < 3:
                ax.axvline(k * segment.CELL_S, color=MUTED, lw=1.0, zorder=1)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel("mV", color=MUTED, fontsize=8)
    axes[-1].set_xlabel("time (s)", color=MUTED)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, ncol=2,
                   bbox_to_anchor=(1.0, 1.32))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--photo", action="append", required=True)
    p.add_argument("--run", default="b3")
    args = p.parse_args(argv)

    import tensorflow as tf
    import keras
    import train_tracer as T
    import eval_sheets as ES
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    rec = json.loads((REPO / "results" / "ml" / f"train_{args.run}.json").read_text())["params"]
    h = int(rec.get("canon_h", 384))
    m = T.build_model(rec["head"], h=h)
    m.load_weights(REPO / "out" / "ml_runs" / args.run / "best.weights.h5")
    model_fn = ES.make_tracer(m, rec["head"], h)

    def classical_fn(sh, lay, r):
        pos, x0 = segment.trace_row(sh, lay, r)
        return x0 + np.arange(len(pos), dtype=np.float64), pos

    OUT.mkdir(parents=True, exist_ok=True)
    reg = skp.load_registry()["photos"]
    for stem in args.photo:
        e = reg[stem]
        t = skp.Target(e["target"])
        truth = S12.SheetTruth(t)
        sheet = skp.warp_to_sheet(rp.read_oriented(skp.PHOTOS / e["file"]),
                                  np.asarray(e["corners_px"], np.float32), t)
        for name, fn, colour, label in (
                (args.run, model_fn, COLOUR["model"], f"model B ({args.run})"),
                ("classical", classical_fn, COLOUR["classical"], "classical")):
            sig, layout, px, ticks = digitise(sheet, t, fn)
            title = (f"{stem} (PTB-XL {t.ecg_id}), digitised by {label}: all 12 leads + "
                     f"rhythm II, as-is at the tick calibration "
                     f"({px:.3f} px/mm from {ticks.n_found}/9 ticks; no alignment)")
            draw(OUT / f"{stem}_{name}.png", sig, truth, title, colour, label)
            if name == args.run:
                write_csv(OUT / f"{stem}_{name}_leads.csv", sig)
            print(OUT / f"{stem}_{name}.png", flush=True)
    return 0


if __name__ == "__main__":
    import os
    code = main()
    sys.stdout.flush()
    os._exit(code)
