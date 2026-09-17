#!/usr/bin/env python3
"""End-to-end demonstration: ecg-image-kit printout -> digitised waveform.

Runs every stage that currently exists, on a realistic generated printout,
and scores the result against the generator's own ground truth. Writes a
four-panel figure so the output can be looked at rather than only tabulated.

**One stage is faked, and it is the important one.** Lead segmentation does
not exist (CLAUDE.md limitation 1): `digitise` traces a single continuous
strip, and handed a 3x4 printout it would run straight across the cell
boundaries. This script therefore crops one lead using the *ground-truth*
`lead_bounding_box` from the generator's sidecar JSON. That is a stand-in
for a detector nobody has written, not a working segmenter, and every
number below is conditional on segmentation being solved perfectly. Read it
as "what the rest of the pipeline does once segmentation works", and not as
a pipeline that runs on an unlabelled photograph. It cannot.

The rhythm strip is used because it is the only cell on a 3x4 printout that
is a genuinely continuous 10 s trace; the other twelve are 2.5 s windows
and would each need the time-offset bookkeeping that CLAUDE.md describes.

The crop starts after the calibration pulse. `lead_bounding_box` includes
the pulse, `plotted_pixels` does not, so the x extent is taken from the
latter -- a 10 mm square wave at the head of the strip is a large, fast,
entirely artificial deflection and tracing it as signal would corrupt both
the baseline estimate and the first samples.

Usage::

    python python/demo_kit_pipeline.py --style poc_clean --record 00009_hr
    python python/demo_kit_pipeline.py --style poc_wrinkled --out fig.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb

from ecgproto import deskew as dsk
from ecgproto import digitise as dig
from ecgproto import evaluate

REPO = Path(__file__).resolve().parents[1]

#: Peak-selection threshold for `estimate_grid_pitch` on kit printouts.
#: The old default of 0.20 returned the 5 mm rule as though it were the
#: 1 mm rule on every one of these images -- the minor rules are resolved
#: but sit at 0.11-0.19 of the autocorrelation maximum against 0.72-0.95
#: for the majors, so it filtered them out. Measured on this image: 0.20
#: gives 41.30 px/mm, 0.03 gives 11.83 and 0.05 gives 11.84, against a
#: true 11.811.
#:
#: Now simply the shipped default, so that what this script demonstrates is
#: the pipeline as it stands rather than a locally tuned variant. Kept as a
#: named constant, and still exposed as --peak-fraction, because the
#: before-and-after against 0.20 is the point of step [4].
#: See CLAUDE.md limitation 6 and CHANGES.txt section 11.5.
KIT_PEAK_FRACTION = dig.DEFAULT_PEAK_FRACTION

#: Vertical padding around the lead box, in millimetres of paper. The box is
#: tight to the ink, and the tracer needs grid either side of the trace for
#: the baseline to sit inside the crop rather than on its edge.
PAD_MM = 6.0

#: Resolution matters more than it looks. `binarise_trace` finishes with a
#: 3x3 morphological opening, and erosion by a 3x3 kernel deletes any run of
#: ink that is not 3 px thick in both directions. Measured on the rhythm
#: strip of this record:
#:
#:     200 dpi            trace ~3 px    column coverage 100% -> 24.9%
#:     300 dpi            trace ~4 px    column coverage 100% -> 99.8%
#:     200 dpi upscaled 2x  trace ~5 px  column coverage 100% -> 100.0%
#:
#: At 200 dpi the opening destroys three quarters of the trace, digitise
#: returns a signal full of NaN, and every metric comes back NaN. This is
#: not the red-channel dependency that CLAUDE.md limitation 2 describes --
#: the thresholding is fine, raw coverage is 100% either way -- it is the
#: despeckling step eating a thin line. Hence the 300 dpi default, and
#: --upscale for lower-resolution sources.


def load_case(style: str, record: str):
    d = REPO / "data" / "images" / style
    png = d / f"{record}-0.png"
    meta = json.loads((d / f"{record}-0.json").read_text())
    image = cv2.imread(str(png), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read {png}")
    truth = wfdb.rdrecord(str(d / record))
    return image, meta, truth, png


def rhythm_lead(meta: dict) -> dict:
    """The full-duration strip: the entry spanning the whole record."""
    best = max(meta["leads"], key=lambda L: L["end_sample"] - L["start_sample"])
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--style", default="gt_300dpi")
    p.add_argument("--record", default="00009_hr")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--peak-fraction", type=float, default=KIT_PEAK_FRACTION)
    p.add_argument("--upscale", type=int, default=1,
                   help="enlarge the ROI before tracing; see the module "
                        "docstring on the 3x3 opening. 1 = off")
    args = p.parse_args(argv)

    out = args.out or (REPO / "out" / "demo" / f"{args.style}_{args.record}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    image, meta, truth_rec, png = load_case(args.style, args.record)
    fs = float(meta["sampling_frequency"])
    true_px_per_mm = meta["x_grid"] / 5.0      # sidecar records px per 5 mm box
    print(f"\n[1] loaded {png.name}  {image.shape[1]}x{image.shape[0]}  "
          f"fs={fs:.0f} Hz  true {true_px_per_mm:.4f} px/mm")

    # ---- 2. deskew --------------------------------------------------------
    d = dsk.deskew(image)
    m, _ = dsk.rotation_matrix(image.shape[:2], -d.angle_deg, expand=True)
    print(f"[2] deskew: {d.angle_deg:+.4f} deg removed  margin={d.margin:.1f}  "
          f"axis disagreement={d.axis_disagreement_deg:.4f} deg  "
          f"({d.elapsed_ms:.0f} ms, {d.interpolation})")

    # ---- 3. lead ROI, via ground-truth boxes (the faked stage) ------------
    lead = rhythm_lead(meta)
    pts = np.array(lead["plotted_pixels"])                  # [y, x]
    pts_xy = dsk.transform_points(m, pts[:, ::-1])          # -> deskewed frame
    box = np.array([lead["lead_bounding_box"][k] for k in "0123"])
    box_xy = dsk.transform_points(m, box[:, ::-1])

    # The lead-name label sits inside the padded box and is dark neutral ink,
    # so `binarise_trace` cannot tell it from trace and the tracer reads the
    # glyphs as a waveform. `--lead_name_bbox` gave us exactly the rectangles
    # needed to suppress it; a real pipeline would get them from the same
    # detector that finds the leads. Painted white because white is what
    # both the trace mask and the grid response treat as empty.
    page = d.image.copy()
    labelled = [L for L in meta["leads"] if "text_bounding_box" in L]
    if not labelled:
        print("    NOTE: this image was generated without --lead_name_bbox, so "
              "the lead labels cannot be suppressed and will be traced as "
              "signal wherever they fall inside the ROI")
    for other in labelled:
        tb = np.array([other["text_bounding_box"][k] for k in "0123"])
        tb_xy = dsk.transform_points(m, tb[:, ::-1])
        tx0, ty0 = np.floor(tb_xy.min(axis=0)).astype(int) - 2
        tx1, ty1 = np.ceil(tb_xy.max(axis=0)).astype(int) + 2
        page[max(0, ty0):ty1, max(0, tx0):tx1] = 255

    pad = PAD_MM * true_px_per_mm
    # Start one millimetre past the first plotted sample: the calibration
    # pulse ends exactly there, and its trailing vertical edge would
    # otherwise be the first thing the tracer sees.
    x0 = int(max(0, pts_xy[:, 0].min() + true_px_per_mm))
    x1 = int(min(page.shape[1], pts_xy[:, 0].max()))
    y0 = int(max(0, box_xy[:, 1].min() - pad))
    y1 = int(min(page.shape[0], box_xy[:, 1].max() + pad))
    crop = page[y0:y1, x0:x1]
    if args.upscale > 1:
        crop = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale,
                          interpolation=cv2.INTER_LANCZOS4)
    print(f"[3] lead '{lead['lead_name']}' samples {lead['start_sample']}-"
          f"{lead['end_sample']}  ROI {crop.shape[1]}x{crop.shape[0]} px "
          f"(x from plotted_pixels, so the calibration pulse is excluded"
          + (f", upscaled {args.upscale}x)" if args.upscale > 1 else ")"))

    # How much of the trace survives despeckling, reported because it is the
    # difference between a result and a column of NaN.
    _red = crop[:, :, 2]
    _, _raw = cv2.threshold((255 - _red).astype(np.uint8), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cov_raw = (_raw > 0).any(axis=0).mean()
    cov_open = dig.binarise_trace(crop).any(axis=0).mean()
    flag = "  <-- the 3x3 opening is eating the trace" if cov_open < 0.9 else ""
    print(f"    column coverage: {cov_raw:.1%} thresholded -> "
          f"{cov_open:.1%} after the 3x3 opening{flag}")

    # ---- 4. grid pitch, shown failing on the default then working ---------
    # Upscaling multiplies the pitch, so the truth to compare against moves
    # with it. Digitisation is unaffected -- px/mm, px/mV and px/s all scale
    # together -- but reporting the error against the unscaled figure would
    # show a spurious +100%.
    true_px_scaled = true_px_per_mm * args.upscale
    # 0.20 pinned, not left to the default: the default is 0.05 since
    # 2026-09-17 and this call exists to show what the old one did.
    bad = dig.estimate_grid_pitch(crop, peak_fraction=0.20)
    bad_chk = dig.check_calibration(
        bad, image_width_px=crop.shape[1],
        expected_duration_s=len(pts) / fs,
    )
    print(f"[4] grid pitch @ peak_fraction=0.20 (the old default) -> "
          f"{bad.px_per_mm:.4f} px/mm "
          f"({(bad.px_per_mm / true_px_scaled - 1) * 100:+.1f}% vs true) "
          f"-> calibration {'OK' if bad_chk.ok else 'REJECTED'}")
    for r in bad_chk.reasons:
        print(f"      reason: {r}")

    grid = dig.estimate_grid_pitch(crop, peak_fraction=args.peak_fraction)
    chk = dig.check_calibration(
        grid, image_width_px=crop.shape[1],
        expected_duration_s=len(pts) / fs,
    )
    err = (grid.px_per_mm / true_px_scaled - 1) * 100
    print(f"    grid pitch @ peak_fraction={args.peak_fraction} -> "
          f"{grid.px_per_mm:.4f} px/mm ({err:+.2f}% vs true) "
          f"-> calibration {'OK' if chk.ok else 'REJECTED'}  "
          f"implied {chk.implied_mm_per_s:.1f} mm/s")
    for r in chk.reasons:
        print(f"      reason: {r}")

    # ---- 5. digitise ------------------------------------------------------
    res = dig.digitise(crop, fs_out=fs, grid=grid, strict=False,
                       expected_duration_s=len(pts) / fs)
    print(f"[5] digitised: {len(res.signal_mv)} samples @ {res.fs:.0f} Hz "
          f"({len(res.signal_mv) / res.fs:.2f} s)  coverage={res.coverage:.1%}")

    # ---- 6. score against the generator's own ground truth ---------------
    lead_i = list(truth_rec.sig_name).index(lead["lead_name"])
    # The crop starts 1 mm in, so the reference has to start at the matching
    # sample or the comparison is offset by ~1 mm of paper (24 ms at 25 mm/s).
    skip = int(round(true_px_per_mm / (true_px_per_mm * 25.0) * fs))
    reference = truth_rec.p_signal[
        lead["start_sample"] + skip:lead["end_sample"], lead_i]
    n_nan = int(np.isnan(res.signal_mv).sum())
    if n_nan:
        print(f"    WARNING: {n_nan}/{len(res.signal_mv)} recovered samples are NaN "
              f"(gaps too long to bridge); metrics will be degenerate")
    metrics = evaluate.evaluate_all(reference, res.signal_mv, fs=fs)
    print(f"[6] vs companion .dat ({lead['lead_name']}, "
          f"{np.isfinite(reference).sum()} non-NaN samples):")
    print(f"      SNR {metrics.snr_db:6.2f} dB   shifted {metrics.shifted_snr_db:6.2f} dB")
    print(f"      RMSE {metrics.rmse_mv:.4f} mV   PCC {metrics.pcc:.4f}   "
          f"gain {metrics.gain_ratio:.4f}")
    print(f"      {metrics.diagnose()}")

    # ---- 7. figure --------------------------------------------------------
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 0.9, 0.9, 1.5], hspace=0.38)

    ax = fig.add_subplot(gs[0])
    ax.imshow(cv2.cvtColor(d.image, cv2.COLOR_BGR2RGB))
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#0b5fff", lw=2.5))
    ax.set_title(f"1-2. {png.name} after deskew ({d.angle_deg:+.3f}°)   "
                 f"blue = lead '{lead['lead_name']}' ROI from ground-truth bbox "
                 f"(segmentation is FAKED here)", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1])
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(f"3. cropped strip, {crop.shape[1]}x{crop.shape[0]} px", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[2])
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    mask = dig.binarise_trace(crop)
    ypx = dig.trace_positions(mask)
    ax.plot(np.arange(len(ypx)), ypx, color="#00c853", lw=1.0)
    ax.axhline(res.baseline_px, color="#d500f9", lw=1.0, ls="--")
    ax.set_title("4. recovered trace (green) and baseline (magenta), in pixels",
                 fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[3])
    t_ref = np.arange(len(reference)) / fs
    t_rec = np.arange(len(res.signal_mv)) / res.fs
    ax.plot(t_ref, reference, color="#111", lw=1.1, label="ground truth (.dat)")
    ax.plot(t_rec, res.signal_mv, color="#e53935", lw=0.9, alpha=0.85,
            label="digitised")
    ax.set_xlabel("seconds")
    ax.set_ylabel("mV")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"5-6. recovered vs truth   SNR {metrics.snr_db:.2f} dB   "
        f"RMSE {metrics.rmse_mv:.4f} mV   PCC {metrics.pcc:.4f}   "
        f"gain {metrics.gain_ratio:.4f}   |   grid {grid.px_per_mm:.3f} px/mm "
        f"({err:+.2f}%)", fontsize=10)

    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n[7] figure -> {out.relative_to(REPO)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
