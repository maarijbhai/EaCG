#!/usr/bin/env python3
"""Prove which PTB-XL record the bundled strip fixtures actually contain.

**Why this is not a paranoid exercise.** The app now reports a PCC against a
waveform committed as `strip_ii_truth.csv`, and that number is only meaningful
if the CSV really is the record the manifest names and the PNG really is a
picture of that CSV. Three things could be wrong independently: the CSV could
have come from a different record, the PNG could have been rendered from
something else, or the manifest could simply be mislabelled. None of those
would look wrong on the phone -- they would produce a confident PCC against
the wrong reference.

So this checks all three, and two of the checks are exact rather than visual:

  1. The committed CSV against the dataset, sample by sample. Exact.
  2. The rendered PNG against the CSV, by overlaying the trace position the
     renderer's own geometry implies on top of the actual pixels. If the ink
     and the overlay coincide, the image is a picture of that waveform.
  3. The 3x4 sheet fixture `gt_norm_150dpi.png` claims the SAME record
     (00009_hr). Its rhythm strip is lead II over the full 10 s, so it should
     carry the same beats. This one is visual: the sheet came from
     ecg-image-kit rather than from this repository, so there is no shared
     geometry to check against, only the morphology.

    python python/verify_strip_fixture.py

Writes out/verify_strip_fixture.png (gitignored -- it is reproducible) and
prints the record's database entry.
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

from ecgproto import digitise as dig
from ecgproto import ptbxl, synth

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "reference"
OUT = REPO / "out" / "verify_strip_fixture.png"

#: Seconds of the strip shown in the overlay panel. The whole 10 s at
#: 7.87 px/mm is 2047 px wide and individual beats are unreadable; two
#: seconds is about two beats, which is enough to see whether the overlay
#: sits on the ink or beside it.
OVERLAY_SECONDS = 2.0


def rhythm_strip_crop(sheet: np.ndarray) -> tuple[np.ndarray, str]:
    """Crop the bottom-most trace band from a 3x4 sheet.

    Found from the ink rather than from a fixed fraction of the page: a
    hardcoded crop would silently move if the fixture were ever re-rendered
    at a different margin. Reuses the same binarisation the tracer uses, so
    if this finds nothing that is itself informative about the sheet.
    """
    mask = dig.binarise_trace(sheet)
    rows = mask.sum(axis=1).astype(float)
    if rows.max() <= 0:
        return (sheet[int(sheet.shape[0] * 0.72):, :],
                "fixed 72% fallback", [], (0, 0))

    # Bands are runs of rows carrying appreciably more ink than the page
    # background. The threshold is a fraction of the busiest row rather than
    # an absolute count, so it does not depend on the render scale.
    # 0.06 rather than 0.10, and the difference is not cosmetic: at 0.10 a
    # cell row fragments into two bands wherever its four leads happen to be
    # near baseline together, and the rhythm strip -- a single flat-ish trace
    # carrying a quarter of a cell row's ink per row -- falls below the
    # threshold entirely. That yields six bands and a crop of the WRONG row.
    # At 0.06 the sheet returns exactly four, which is what a 3x4 layout has
    # and what QualityGate.tracesPerColumn independently counts.
    inked = rows > 0.06 * rows.max()
    bands: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(inked):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(inked)))

    # Ignore slivers: printed labels and tick marks also clear the threshold.
    bands = [b for b in bands if b[1] - b[0] >= 0.02 * sheet.shape[0]]
    if not bands:
        return (sheet[int(sheet.shape[0] * 0.72):, :],
                "fixed 72% fallback", [], (0, 0))

    lo, hi = bands[-1]
    pad = int(0.01 * sheet.shape[0])
    lo = max(0, lo - pad)
    hi = min(sheet.shape[0], hi + pad)
    return (sheet[lo:hi, :], f"{len(bands)} bands found, cropped rows {lo}-{hi}",
            bands, (lo, hi))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--overlay-dpi", type=int, default=200,
                   help="which strip the overlay panel uses")
    args = p.parse_args(argv)

    manifest = json.loads((ASSETS / "strip_fixtures.json").read_text(encoding="utf-8"))
    ecg_id = manifest["ecg_id"]
    lead = manifest["lead"]

    # ---- the database entry ---------------------------------------------
    df = ptbxl.load_metadata(args.root)
    row = df.loc[ecg_id]
    print(f"Manifest names ecg_id {ecg_id}, lead {lead}\n")
    print("PTB-XL database entry:")
    for key in ("patient_id", "age", "sex", "weight", "strat_fold",
                "recording_date", "scp_codes", "report",
                "filename_lr", "filename_hr"):
        if key in df.columns:
            print(f"  {key:15s} {row[key]}")
    print(f"\n  sex code 0 = male, 1 = female")
    print(f"  strat_fold {row['strat_fold']} -- "
          f"{'TEST fold, correct for a fixture' if row['strat_fold'] == 10 else 'NOT the test fold'}")

    # ---- check 1: the committed CSV against the dataset, exactly --------
    sig, meta = ptbxl.load_record(ecg_id, root=args.root,
                                  sampling_rate=int(manifest["fs_hz"]), df=df)
    truth_db = sig[:, ptbxl.LEAD_NAMES.index(lead)]
    truth_csv = np.array([
        float(v) for v in
        (ASSETS / manifest["truth_csv"]).read_text(encoding="utf-8").split()
    ])
    print(f"\nCheck 1 -- committed CSV against the dataset:")
    print(f"  dataset  {len(truth_db)} samples, first {truth_db[0]:+.4f} mV")
    print(f"  csv      {len(truth_csv)} samples, first {truth_csv[0]:+.4f} mV")
    if len(truth_db) != len(truth_csv):
        print("  MISMATCH in length -- the CSV is not this record")
        return 1
    max_diff = float(np.abs(truth_db - truth_csv).max())
    print(f"  max abs difference {max_diff:.3e} mV  "
          f"{'IDENTICAL' if max_diff == 0.0 else 'DIFFERS'}")

    # ---- check 2: the PNG against the CSV, via the render geometry ------
    strip_meta = next(s for s in manifest["strips"]
                      if s["dpi_nominal"] == args.overlay_dpi)
    strip = cv2.imread(str(ASSETS / strip_meta["file"]), cv2.IMREAD_COLOR)
    px_per_mm = strip_meta["px_per_mm_true"]
    baseline_px = strip_meta["baseline_px"]
    # The trace is inset by the renderer's margin. Omitting this shifts the
    # overlay by ~39 px at 200 dpi, which on a QRS upstroke reads as tens of
    # pixels of disagreement -- i.e. exactly like the wrong waveform.
    x0_px = strip_meta["x0_px"]
    mm_per_s = manifest["mm_per_s"]
    mm_per_mv = manifest["mm_per_mv"]

    # Where the renderer must have put the trace, from its own conventions.
    # Note the MINUS: rows grow down, millivolts grow up.
    n_show = int(OVERLAY_SECONDS * meta.fs)
    x_expected = x0_px + np.arange(n_show) / meta.fs * px_per_mm * mm_per_s
    y_expected = baseline_px - truth_csv[:n_show] * px_per_mm * mm_per_mv

    # Where the ink actually is, measured from the pixels with no knowledge
    # of the waveform at all.
    mask = dig.binarise_trace(strip)
    y_actual = dig.trace_positions(mask)

    crop_w = int(np.ceil(x_expected.max())) + 2
    # Compare only columns the overlay actually spans: the margin to the left
    # of x0 carries no trace, so including it would compare the waveform
    # against blank paper.
    cols = np.arange(int(np.floor(x0_px)), crop_w)
    inside = np.isfinite(y_actual[cols])
    resid = np.abs(np.interp(cols[inside], x_expected, y_expected)
                   - y_actual[cols][inside])
    print(f"\nCheck 2 -- {strip_meta['file']} against the CSV, "
          f"over the first {OVERLAY_SECONDS:g} s:")
    print(f"  px/mm {px_per_mm:.4f}, baseline {baseline_px:.1f} px, "
          f"{mm_per_s:g} mm/s, {mm_per_mv:g} mm/mV")
    print(f"  ink position vs waveform-implied position: "
          f"median {np.median(resid):.2f} px, max {resid.max():.2f} px")
    print(f"  (a couple of pixels is the trace's own line width; tens of "
          f"pixels would mean a different waveform)")

    # ---- check 3: the 3x4 sheet, which claims the same record -----------
    sheet_path = ASSETS / "gt_norm_150dpi.png"
    sheet = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)
    sheet_crop, how, bands, chosen = rhythm_strip_crop(sheet)
    print(f"\nCheck 3 -- {sheet_path.name} claims record 00009_hr, the same one.")
    print(f"  rhythm strip: {how}")
    print(f"  bands at rows: " + ", ".join(f"{a}-{b}" for a, b in bands))
    print(f"  NOTE a 3x4 sheet has three CELL rows plus the rhythm strip, so")
    print(f"  four trace bands is the expected count; more than that means the")
    print(f"  detector also caught printed labels. The crop is the LOWEST band,")
    print(f"  which is the rhythm strip only if the layout puts it last.")
    print(f"  This one is visual only: the sheet came from ecg-image-kit, not")
    print(f"  from this repository, so there is no shared geometry to check.")

    # ---- the figure ------------------------------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(13, 19),
                             gridspec_kw={"height_ratios": [2, 2, 1.4, 1.6, 4, 1.2]})

    t = np.arange(len(truth_db)) / meta.fs
    axes[0].plot(t, truth_db, lw=0.8, color="#1b3b6f")
    axes[0].set_title(
        f"1. PTB-XL record {ecg_id}, lead {lead}, from {row['filename_lr']} "
        f"-- patient {int(row['patient_id'])}, age {int(row['age'])}, "
        f"fold {int(row['strat_fold'])}, \"{row['report']}\"",
        fontsize=9, loc="left")
    axes[0].set_ylabel("mV")
    axes[0].set_xlim(0, t[-1])
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, truth_csv, lw=0.8, color="#8b1a1a")
    axes[1].set_title(
        f"2. The committed {manifest['truth_csv']} the app scores against "
        f"-- max difference from panel 1: {max_diff:.3e} mV",
        fontsize=9, loc="left")
    axes[1].set_ylabel("mV")
    axes[1].set_xlim(0, t[-1])
    axes[1].grid(alpha=0.3)

    axes[2].imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"3. The rendered asset {strip_meta['file']} "
        f"({strip.shape[1]}x{strip.shape[0]} px, {px_per_mm:.4f} px/mm)",
        fontsize=9, loc="left")
    axes[2].set_axis_off()

    crop = strip[:, :crop_w]
    axes[3].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), aspect="auto")
    axes[3].plot(x_expected, y_expected, lw=1.1, color="#00a0a0",
                 label="position implied by the committed waveform")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[3].set_title(
        f"4. First {OVERLAY_SECONDS:g} s of the same asset, with the trace "
        f"position the waveform implies drawn over it -- residual median "
        f"{np.median(resid):.2f} px. If these coincide, the PNG is a picture "
        f"of panel 2.",
        fontsize=9, loc="left")
    axes[3].set_xlim(0, crop_w)
    axes[3].set_ylim(strip.shape[0], 0)
    axes[3].set_axis_off()

    axes[4].imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
    for a, b in bands:
        axes[4].axhline(a, color="#00a0a0", lw=0.6)
        axes[4].axhline(b, color="#00a0a0", lw=0.6)
    if chosen != (0, 0):
        axes[4].axhspan(chosen[0], chosen[1], color="#ffcc00", alpha=0.22)
    axes[4].set_title(
        f"5. The whole 3x4 fixture {sheet_path.name}, which claims the SAME "
        f"record. Detected trace bands ruled in teal, the band cropped below "
        f"shaded. A 3x4 sheet has three cell rows plus one rhythm strip.",
        fontsize=9, loc="left")
    axes[4].set_axis_off()

    axes[5].imshow(cv2.cvtColor(sheet_crop, cv2.COLOR_BGR2RGB), aspect="auto")
    axes[5].set_title(
        f"6. That band, enlarged. If it is the rhythm strip it is lead {lead} "
        f"over the full 10 s and its beats should match panel 1. If it is a "
        f"cell row it is four DIFFERENT leads in four consecutive 2.5 s "
        f"windows, and it should not match.",
        fontsize=9, loc="left")
    axes[5].set_axis_off()

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
