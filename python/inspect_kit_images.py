#!/usr/bin/env python3
"""Audit the ecg-image-kit corpus under data/images, and prove its annotations.

**What these files are.** ecg-image-kit renders of PTB-XL records, each PNG
paired with a JSON sidecar. The three `gt_*dpi` sets are the fixtures the
Android app bundles; `poc_mild` and `poc_wrinkled` add photographic
degradation. They are the thing CLAUDE.md limitation 3 asks for -- real
printout rendering rather than `synth.render_printout` -- and they carry
annotation the synthetic renderer never produced.

**The annotation is the point, and it is worth more than the images.** Each
sidecar's `leads` list gives, per lead:

  lead_name           which lead
  lead_bounding_box   four corners of the drawn trace's extent
  text_bounding_box   where the printed label sits
  start_sample        the recording sample this cell begins at
  end_sample          and ends at
  plotted_pixels      the pixel the renderer put EVERY SAMPLE at

That is direct supervision for FR4 (lead segmentation, which does not exist
in any form), and exact ground truth for FR5 and FR6 -- better than anything
this repository can currently produce, because `synth.render_printout` knows
where it drew the trace but never wrote it down.

**The coordinate convention is a trap, and this script exists partly to prove
it.** `lead_bounding_box` and `plotted_pixels` are **(row, column)**, not
(x, y), despite the sidecar's own `width`/`height` keys being the other way
round. Read naively as (x, y) they are nonsense: the second coordinate runs
to 2086 in an image 1700 tall. Measured below by asking how many annotated
points actually land on ink under each reading.

    python python/inspect_kit_images.py

Writes out/kit_images_audit.png (gitignored; reproducible).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ecgproto import ptbxl

REPO = Path(__file__).resolve().parents[1]
IMAGES = REPO / "data" / "images"
OUT = REPO / "out" / "kit_images_audit.png"

#: How many annotated points to test the coordinate convention on. The
#: corpus has ~20,000 per image and the answer is unambiguous at any sample
#: size; this keeps the check instant.
CONVENTION_STRIDE = 37


def lead_box(lead: dict) -> tuple[int, int, int, int]:
    """(row0, col0, row1, col1) from a lead_bounding_box, in (row, col)."""
    bb = lead["lead_bounding_box"]
    rows = [bb[k][0] for k in bb]
    cols = [bb[k][1] for k in bb]
    return min(rows), min(cols), max(rows), max(cols)


def convention_hit_rate(img: np.ndarray, meta: dict) -> tuple[float, float]:
    """Fraction of annotated points landing on ink, read each way.

    The decisive test. A correct reading puts essentially every annotated
    point on a dark pixel, because the annotation IS where the renderer drew.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = gray < 128
    pts = np.array([p for lead in meta["leads"]
                    for p in lead["plotted_pixels"]], dtype=float)

    def rate(as_rowcol: bool) -> float:
        ok = tot = 0
        for a, b in pts[::CONVENTION_STRIDE]:
            r, c = (a, b) if as_rowcol else (b, a)
            r, c = int(round(r)), int(round(c))
            if 0 <= r < dark.shape[0] and 0 <= c < dark.shape[1]:
                tot += 1
                # A couple of pixels of tolerance: the trace has line width.
                if dark[max(0, r - 2):r + 3, max(0, c - 2):c + 3].any():
                    ok += 1
        return ok / tot if tot else float("nan")

    return rate(True), rate(False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--figure-set", default="gt_200dpi")
    p.add_argument("--figure-record", default="00009_hr-0")
    args = p.parse_args(argv)

    if not IMAGES.exists():
        print(f"no corpus at {IMAGES}")
        return 1

    # ---- what is actually here -------------------------------------------
    records: Counter = Counter()
    conditions: list[str] = []
    for d in sorted(IMAGES.iterdir()):
        if not d.is_dir():
            continue
        conditions.append(d.name)
        for j in sorted(d.glob("*.json")):
            records[j.stem.split("-")[0]] += 1

    print(f"Corpus at {IMAGES.relative_to(REPO)}")
    print(f"  {len(conditions)} conditions: {', '.join(conditions)}")
    print(f"  {len(records)} distinct records: {', '.join(sorted(records))}")
    print(f"  {sum(records.values())} images total\n")

    # ---- which records, and which fold -----------------------------------
    df = ptbxl.load_metadata(args.root)
    print("Provenance of each record:")
    folds = []
    for stem in sorted(records):
        ecg_id = int(stem.split("_")[0])
        row = df.loc[ecg_id]
        folds.append(int(row["strat_fold"]))
        print(f"  ecg_id {ecg_id:5d}  fold {int(row['strat_fold']):2d}  "
              f"{str(row['scp_codes'])[:44]:46s} {row['report'][:38]}")
    if folds and all(f == ptbxl.TEST_FOLD for f in folds):
        print(f"\n  ** EVERY record is in fold {ptbxl.TEST_FOLD}, the TEST fold. **")
        print("  These are evaluation images. Training on them destroys the")
        print("  held-out set, and the published split is the only thing")
        print("  keeping ~18,900 patients from leaking across the boundary.")

    # ---- the coordinate convention ---------------------------------------
    img_path = IMAGES / args.figure_set / f"{args.figure_record}.png"
    meta = json.loads((img_path.with_suffix(".json")).read_text(encoding="utf-8"))
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rowcol, xy = convention_hit_rate(img, meta)
    print(f"\nCoordinate convention, tested on {args.figure_set}/{args.figure_record}:")
    print(f"  image is {img.shape[1]} wide x {img.shape[0]} tall")
    print(f"  annotated points landing on ink, read as (row, col): {rowcol:6.1%}")
    print(f"  annotated points landing on ink, read as (x, y)    : {xy:6.1%}")
    print("  -> the sidecar's boxes and plotted_pixels are (row, col).")

    # ---- what the annotation contains ------------------------------------
    print(f"\nAnnotation per image: {len(meta['leads'])} entries "
          f"(12 cells + the rhythm strip)")
    fs = meta["sampling_frequency"]
    for lead in meta["leads"][:3]:
        r0, c0, r1, c1 = lead_box(lead)
        print(f"  {lead['lead_name']:4s} samples {lead['start_sample']:5d}-"
              f"{lead['end_sample']:5d} "
              f"({lead['start_sample'] / fs:.1f}-{lead['end_sample'] / fs:.1f} s)  "
              f"box rows {r0}-{r1}, cols {c0}-{c1}  "
              f"{len(lead['plotted_pixels'])} plotted samples")
    print("  ...")
    rhythm = meta["leads"][-1]
    print(f"  {rhythm['lead_name']:4s} samples {rhythm['start_sample']:5d}-"
          f"{rhythm['end_sample']:5d} "
          f"({rhythm['start_sample'] / fs:.1f}-{rhythm['end_sample'] / fs:.1f} s)"
          f"  <- the rhythm strip, the only lead spanning the whole recording")

    # ---- the figure ------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 15),
                             gridspec_kw={"height_ratios": [3, 1.5, 3]})

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for lead in meta["leads"]:
        r0, c0, r1, c1 = lead_box(lead)
        axes[0].add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0, fill=False, lw=1.0, color="#0077ff"))
        axes[0].text(c0, r0 - 8, lead["lead_name"], fontsize=7, color="#0077ff")
    axes[0].set_title(
        f"1. {args.figure_set}/{args.figure_record}.png with every "
        f"lead_bounding_box drawn, read as (row, col). This is FR4 supervision "
        f"-- lead segmentation does not exist in this repository in any form.",
        fontsize=9, loc="left")
    axes[0].set_axis_off()

    # A single cell, with the per-sample annotation on top of the ink.
    target = next(L for L in meta["leads"] if L["lead_name"] == "V2")
    r0, c0, r1, c1 = lead_box(target)
    pad = 20
    crop = img[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]
    pts = np.array(target["plotted_pixels"], dtype=float)
    axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), aspect="auto")
    axes[1].plot(pts[:, 1] - max(0, c0 - pad), pts[:, 0] - max(0, r0 - pad),
                 lw=0.8, color="#00c8c8", label="plotted_pixels")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(
        f"2. Lead {target['lead_name']} alone, with the renderer's own "
        f"per-sample pixel record laid over the ink "
        f"({rowcol:.0%} of annotated points land on ink). This is exact FR5/FR6 "
        f"ground truth -- synth.render_printout never wrote this down.",
        fontsize=9, loc="left")
    axes[1].set_axis_off()

    wrinkled = IMAGES / "poc_wrinkled" / f"{args.figure_record}.png"
    if wrinkled.exists():
        wimg = cv2.imread(str(wrinkled), cv2.IMREAD_COLOR)
        wmeta = json.loads(wrinkled.with_suffix(".json").read_text(encoding="utf-8"))
        axes[2].imshow(cv2.cvtColor(wimg, cv2.COLOR_BGR2RGB))
        axes[2].set_title(
            f"3. The same record from poc_wrinkled: rotate={wmeta['rotate']} deg, "
            f"{wmeta['number_of_creases_horizontally']}h/"
            f"{wmeta['number_of_creases_vertically']}v creases at "
            f"{wmeta['crease_angle']} deg, noise={wmeta['noise']}, "
            f"colour temperature {wmeta['temperature']} K. Still a render, not a "
            f"photograph -- the closed loop is narrowed, not broken.",
            fontsize=9, loc="left")
        axes[2].set_axis_off()

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100)
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
