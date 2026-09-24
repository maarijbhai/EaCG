#!/usr/bin/env python3
"""Render one PTB-XL lead to a print-ready PDF at TRUE physical scale.

This is the first half of the real-photograph experiment. Everything the
project has measured so far comes from a closed loop -- the image is drawn
by the same code, with the same constants, that later decodes it. Grid
pitch comes back at +0.00% error because the renderer drew rules at exactly
that integer pitch; the best time shift is exactly 0 samples because the
strip begins where the renderer said it begins. Neither gift survives a
camera, and until one is photographed nobody knows how far the bound sits
from reality.

**The whole experiment rests on the print being at true scale.** The
digitiser recovers calibration from the grid, and the grid is only a ruler
if 1 mm on the page really is 1 mm. If the printer scales the page -- which
is what "Fit to page" and "Shrink oversized pages" do, and both are on by
default in common print dialogs -- then the grid is a ruler of unknown
length, the recovered millivolts are wrong by that factor, and the
experiment measures nothing. There is no way to detect this after the fact
from the image alone, which is why this script prints physical rulers onto
the page and why `docs/photo_protocol.md` makes the human verify them with
a real ruler before photographing anything.

Scale assumptions, stated explicitly because they are the load-bearing ones:

* The PDF is written with an explicit resolution, so one image pixel is
  exactly 1/dpi inch and the page's MediaBox is the true physical size.
  Nothing downstream infers scale from DPI -- Reyna et al. are explicit
  that image DPI and physical inches do not correspond for camera capture.
  DPI is used *here*, at print time, where it does correspond, and then
  deliberately thrown away.
* `px_per_mm = dpi / 25.4`. At the 300 dpi default that is 11.811 px/mm,
  so a 1 mm minor grid rule is ~12 px on the page -- comfortably above the
  ~3 px/mm floor below which `estimate_grid_pitch` stops resolving minor
  rules and silently locks onto the 5 mm majors.
* Page geometry is rounded to whole pixels, so the strip's true width is
  recovered back from its pixel size and written to the manifest. The
  manifest value, not the nominal one, is the ground truth used for scoring.

What lands on the page, and why in this arrangement:

    [ header text ]        record identity, calibration, git commit
    [ THE STRIP   ]        the gridded region, and the only thing to crop to
    [ rulers      ]        100 mm horizontal, 50 mm vertical, 10 mm = 1 mV

The rulers and text sit **outside** the strip on purpose. `digitise` traces
every column it is given and has no way to reject non-trace ink, so a
calibration pulse or a label inside the strip would be digitised as signal.
Crop marks are drawn just outside each corner, their inner vertex exactly on
the strip corner, so the human has something precise to click without adding
ink inside the region.

Usage::

    python python/make_print_target.py --ecg-id 9 --lead II
    python python/make_print_target.py --ecg-id 9 --lead II --dpi 600
    python python/make_print_target.py --ecg-id 180 --lead V2 --duration 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from PIL import Image

from ecgproto import ptbxl, synth

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Page sizes in millimetres, (width, height). A 10 s strip at 25 mm/s is
#: 250 mm of trace plus margins, so only landscape A4 or Letter holds one.
PAGE_SIZES_MM = {
    "a4-landscape": (297.0, 210.0),
    "a4-portrait": (210.0, 297.0),
    "letter-landscape": (279.4, 215.9),
    "letter-portrait": (215.9, 279.4),
}

#: Printers cannot reach the paper edge. Content is kept this far in, which
#: is more than any consumer inkjet or laser needs, so nothing is cropped
#: and -- more importantly -- the driver is never tempted to rescale to fit.
PAGE_MARGIN_MM = 10.0

#: Physical sizes of the on-page references. 100 mm is long enough that a
#: 2% print scaling error shows up as a visible 2 mm discrepancy against a
#: household ruler; 10 mm would not.
RULER_LENGTH_MM = 100.0
RULER_VERTICAL_MM = 50.0

COLOUR_INK = (0, 0, 0)          # BGR: rulers, crop marks, text
TEXT_HEIGHT_MM = 3.0
RULE_THICKNESS_MM = 0.3         # so lines survive printing and photography


def git_commit() -> str:
    """Short commit plus a dirty flag, or 'unknown' outside a checkout.

    Recorded in the manifest because a reconstruction scored months later
    has to be traceable to the code that rendered the paper it came from.
    """
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        if rev.returncode != 0:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "python"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        dirty = "+dirty" if status.stdout.strip() else ""
        return rev.stdout.strip() + dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _mm(value_mm: float, px_per_mm: float) -> int:
    return int(round(value_mm * px_per_mm))


def _put_text(
    image: np.ndarray,
    text: str,
    x_mm: float,
    y_mm: float,
    px_per_mm: float,
    height_mm: float = TEXT_HEIGHT_MM,
) -> None:
    """Draw text with its baseline at (x_mm, y_mm), sized in millimetres."""
    # cv2's Hershey cap height is ~22 px at scale 1.0.
    scale = height_mm * px_per_mm / 22.0
    cv2.putText(
        image, text, (_mm(x_mm, px_per_mm), _mm(y_mm, px_per_mm)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, COLOUR_INK,
        max(1, _mm(RULE_THICKNESS_MM, px_per_mm)), cv2.LINE_AA,
    )


#: Height of the band below the strip that holds the printed references.
RULER_BAND_MM = 62.0


def _vertical_bar(
    image: np.ndarray,
    x_mm: float,
    top_mm: float,
    length_mm: float,
    px_per_mm: float,
    thick: int,
    cap: float = 2.0,
) -> None:
    """A capped vertical bar of exactly `length_mm`, hanging from `top_mm`."""
    cv2.line(image, (_mm(x_mm, px_per_mm), _mm(top_mm, px_per_mm)),
             (_mm(x_mm, px_per_mm), _mm(top_mm + length_mm, px_per_mm)),
             COLOUR_INK, thick, cv2.LINE_AA)
    for yy in (top_mm, top_mm + length_mm):
        cv2.line(image, (_mm(x_mm - cap, px_per_mm), _mm(yy, px_per_mm)),
                 (_mm(x_mm + cap, px_per_mm), _mm(yy, px_per_mm)),
                 COLOUR_INK, thick, cv2.LINE_AA)


def draw_rulers(
    image: np.ndarray,
    x_mm: float,
    band_top_mm: float,
    px_per_mm: float,
    mm_per_mv: float,
) -> None:
    """Draw the physical references, in a band hanging from `band_top_mm`.

    Three references, because they fail differently:

    * a 100 mm horizontal bar -- catches uniform print scaling, which is by
      far the most likely fault ("Fit to page");
    * a 50 mm vertical bar -- catches *anisotropic* scaling, which a
      horizontal check alone would miss and which would corrupt amplitude
      and time by different factors;
    * a 10 mm vertical bar labelled in millivolts -- the same ruler
      expressed in the unit the reconstruction is reported in, so a reader
      of the printout can sanity-check gain without arithmetic.

    Everything is laid out downward from `band_top_mm` and nothing is
    centred on it, because a bar centred on the band would extend upward
    into the strip. Black ink inside the strip is not cosmetic damage:
    `binarise_trace` would return it as trace and `trace_positions` would
    digitise it as signal.
    """
    thick = max(1, _mm(RULE_THICKNESS_MM, px_per_mm))

    _put_text(image,
              "Verify with a ruler BEFORE photographing. If either bar is the "
              "wrong length the print was scaled - reprint at 100%.",
              x_mm, band_top_mm + 4.0, px_per_mm, 2.6)

    # --- horizontal 100 mm, ticks every 10 mm, measured cap edge to cap edge
    y = band_top_mm + 14.0
    cv2.line(image, (_mm(x_mm, px_per_mm), _mm(y, px_per_mm)),
             (_mm(x_mm + RULER_LENGTH_MM, px_per_mm), _mm(y, px_per_mm)),
             COLOUR_INK, thick, cv2.LINE_AA)
    for i in range(0, int(RULER_LENGTH_MM) + 1, 10):
        tick = 2.5 if i in (0, int(RULER_LENGTH_MM)) else 1.2
        cv2.line(image, (_mm(x_mm + i, px_per_mm), _mm(y - tick, px_per_mm)),
                 (_mm(x_mm + i, px_per_mm), _mm(y + tick, px_per_mm)),
                 COLOUR_INK, thick, cv2.LINE_AA)
        if i % 50 == 0:
            _put_text(image, f"{i}", x_mm + i - 2.0, y + 7.5, px_per_mm, 2.6)
    _put_text(image, f"{RULER_LENGTH_MM:.1f} mm horizontal",
              x_mm, y + 14.0, px_per_mm, 2.8)

    # --- vertical 50 mm, hanging so it cannot reach the strip
    vx = x_mm + RULER_LENGTH_MM + 50.0
    v_top = band_top_mm + 8.0
    _vertical_bar(image, vx, v_top, RULER_VERTICAL_MM, px_per_mm, thick)
    _put_text(image, f"{RULER_VERTICAL_MM:.0f} mm vertical",
              vx + 5.0, v_top + RULER_VERTICAL_MM / 2, px_per_mm, 2.8)
    _put_text(image, "(catches uneven scaling)",
              vx + 5.0, v_top + RULER_VERTICAL_MM / 2 + 4.5, px_per_mm, 2.3)

    # --- 10 mm = 1 mV at the rendered gain
    ax = vx + 60.0
    amp_mm = mm_per_mv * 1.0
    _vertical_bar(image, ax, v_top, amp_mm, px_per_mm, thick)
    _put_text(image, f"{amp_mm:.0f} mm = 1.0 mV",
              ax + 5.0, v_top + amp_mm / 2, px_per_mm, 2.8)


def draw_crop_marks(
    image: np.ndarray,
    x0_mm: float,
    y0_mm: float,
    w_mm: float,
    h_mm: float,
    px_per_mm: float,
) -> None:
    """L-shaped marks whose inner vertex is exactly on each strip corner.

    They point away from the strip so no ink is added inside the region that
    gets digitised, and they give the human an unambiguous target to click
    in `score_photo.py` -- "the corner of the grid" is a judgement call at
    photograph resolution, a printed vertex is not.
    """
    thick = max(1, _mm(RULE_THICKNESS_MM, px_per_mm))
    arm, gap = 6.0, 1.0
    corners = (
        (x0_mm, y0_mm, -1, -1),
        (x0_mm + w_mm, y0_mm, 1, -1),
        (x0_mm + w_mm, y0_mm + h_mm, 1, 1),
        (x0_mm, y0_mm + h_mm, -1, 1),
    )
    for cx, cy, sx, sy in corners:
        ox, oy = cx + sx * gap, cy + sy * gap
        cv2.line(image, (_mm(ox, px_per_mm), _mm(oy, px_per_mm)),
                 (_mm(ox + sx * arm, px_per_mm), _mm(oy, px_per_mm)),
                 COLOUR_INK, thick, cv2.LINE_AA)
        cv2.line(image, (_mm(ox, px_per_mm), _mm(oy, px_per_mm)),
                 (_mm(ox, px_per_mm), _mm(oy + sy * arm, px_per_mm)),
                 COLOUR_INK, thick, cv2.LINE_AA)


def build_page(
    signal_mv: np.ndarray,
    fs: float,
    dpi: int,
    page: str,
    mm_per_s: float,
    mm_per_mv: float,
    height_mm: float,
    header_lines: list[str],
) -> tuple[np.ndarray, synth.RenderSpec, dict]:
    """Compose the full page. Returns (BGR page, strip spec, geometry dict)."""
    px_per_mm = dpi / 25.4
    page_w_mm, page_h_mm = PAGE_SIZES_MM[page]

    strip, spec = synth.render_printout(
        signal_mv, fs=fs, px_per_mm=px_per_mm,
        mm_per_s=mm_per_s, mm_per_mv=mm_per_mv,
        height_mm=height_mm, on_overflow="expand",
    )

    # Recover true physical size from the pixel size actually produced;
    # render_printout rounds to whole pixels and the rounded value is what
    # is on the paper, so it is what scoring must be told about.
    strip_w_mm = strip.shape[1] / px_per_mm
    strip_h_mm = strip.shape[0] / px_per_mm

    header_h_mm = 6.0 + TEXT_HEIGHT_MM * 1.9 * len(header_lines)
    needed_h = PAGE_MARGIN_MM * 2 + header_h_mm + strip_h_mm + RULER_BAND_MM
    needed_w = PAGE_MARGIN_MM * 2 + max(strip_w_mm, RULER_LENGTH_MM + 150.0)
    if needed_h > page_h_mm or needed_w > page_w_mm:
        # Name the binding dimension and the remedy that addresses *it*.
        # Width is driven by duration and paper speed; height by the signal's
        # amplitude and the gain, and telling someone to shorten a recording
        # that is too tall wastes their afternoon.
        fixes = []
        if needed_w > page_w_mm:
            fixes.append(
                f"width is over by {needed_w - page_w_mm:.0f} mm -- reduce "
                f"--duration (now {strip_w_mm / mm_per_s:.1f} s of paper at "
                f"{mm_per_s:g} mm/s) or --mm-per-s"
            )
        if needed_h > page_h_mm:
            fixes.append(
                f"height is over by {needed_h - page_h_mm:.0f} mm -- the signal "
                f"spans {spec.v_min_mv:+.2f}..{spec.v_max_mv:+.2f} mV and needs "
                f"{spec.required_height_mm:.0f} mm at {mm_per_mv:g} mm/mV. Use "
                f"--mm-per-mv 5, which is a real clinical setting for "
                f"large-amplitude records and which the digitiser has to cope "
                f"with anyway"
            )
        raise ValueError(
            f"content needs {needed_w:.0f} x {needed_h:.0f} mm but {page} is "
            f"{page_w_mm:.0f} x {page_h_mm:.0f} mm.\n  "
            + "\n  ".join(fixes)
            + "\n  Or choose a larger --page. Do NOT print this scaled down to "
              "fit: scaling destroys the experiment."
        )

    page_img = np.full(
        (_mm(page_h_mm, px_per_mm), _mm(page_w_mm, px_per_mm), 3), 255, np.uint8
    )

    for i, line in enumerate(header_lines):
        _put_text(page_img, line, PAGE_MARGIN_MM,
                  PAGE_MARGIN_MM + TEXT_HEIGHT_MM * (1.9 * i + 1), px_per_mm)

    x0_mm = PAGE_MARGIN_MM
    y0_mm = PAGE_MARGIN_MM + header_h_mm
    x0, y0 = _mm(x0_mm, px_per_mm), _mm(y0_mm, px_per_mm)
    page_img[y0: y0 + strip.shape[0], x0: x0 + strip.shape[1]] = strip

    draw_crop_marks(page_img, x0_mm, y0_mm, strip_w_mm, strip_h_mm, px_per_mm)
    # The band hangs from below the crop marks, which extend 7 mm past the
    # strip; starting any higher would put ruler ink inside the crop region.
    draw_rulers(page_img, PAGE_MARGIN_MM, y0_mm + strip_h_mm + 10.0,
                px_per_mm, mm_per_mv)

    geometry = {
        "page": page,
        "page_width_mm": page_w_mm,
        "page_height_mm": page_h_mm,
        "dpi": dpi,
        "px_per_mm_at_print": px_per_mm,
        "strip_origin_mm": [x0_mm, y0_mm],
        "strip_width_mm": strip_w_mm,
        "strip_height_mm": strip_h_mm,
        "strip_width_px": int(strip.shape[1]),
        "strip_height_px": int(strip.shape[0]),
        "ruler_length_mm": RULER_LENGTH_MM,
        "ruler_vertical_mm": RULER_VERTICAL_MM,
    }
    return page_img, spec, geometry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    parser.add_argument("--ecg-id", type=int, required=True)
    parser.add_argument("--lead", default="II",
                        help=f"one of {' '.join(ptbxl.LEAD_NAMES)}")
    parser.add_argument("--rate", type=int, choices=(100, 500), default=500,
                        help="source sampling rate; 500 avoids the "
                             "render-interpolation flattery of 100 Hz")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds of the record to print")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--page", choices=sorted(PAGE_SIZES_MM), default="a4-landscape")
    parser.add_argument("--mm-per-s", type=float, default=synth.MM_PER_S)
    parser.add_argument("--mm-per-mv", type=float, default=synth.MM_PER_MV)
    parser.add_argument("--height-mm", type=float, default=synth.DEFAULT_HEIGHT_MM)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "print_targets")
    parser.add_argument("--also-png", action="store_true",
                        help="write the page as PNG too, for on-screen checking")
    args = parser.parse_args(argv)

    if args.lead not in ptbxl.LEAD_NAMES:
        parser.error(f"unknown lead {args.lead!r}; expected {ptbxl.LEAD_NAMES}")
    if not args.root.exists():
        print(f"dataset not found at {args.root}", file=sys.stderr)
        return 2

    signal, meta = ptbxl.load_record(
        args.ecg_id, root=args.root, sampling_rate=args.rate
    )
    lead_idx = ptbxl.LEAD_NAMES.index(args.lead)
    n = int(round(args.duration * meta.fs))
    if n > signal.shape[0]:
        parser.error(
            f"record {args.ecg_id} is {meta.duration_s:g} s, cannot print "
            f"{args.duration:g} s"
        )
    lead = np.ascontiguousarray(signal[:n, lead_idx])

    commit = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = [
        f"PTB-XL {args.ecg_id}  lead {args.lead}  {meta.fs:g} Hz  "
        f"{args.duration:g} s   |   {args.mm_per_s:g} mm/s   "
        f"{args.mm_per_mv:g} mm/mV   |   {args.dpi} dpi   {args.page}",
        f"commit {commit}   rendered {stamp}   "
        f"PRINT AT 100% SCALE - NOT 'fit to page'. Verify the 100 mm bar below.",
    ]

    try:
        page_img, spec, geometry = build_page(
            lead, meta.fs, args.dpi, args.page,
            args.mm_per_s, args.mm_per_mv, args.height_mm, header,
        )
    except ValueError as exc:
        print(f"cannot lay out this page:\n  {exc}", file=sys.stderr)
        return 2

    name = f"ptbxl{args.ecg_id}_{args.lead}_{args.rate}hz_{args.dpi}dpi"
    out_dir: Path = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    npy_path = out_dir / f"{name}_truth.npy"
    manifest_path = out_dir / f"{name}_manifest.json"

    # Pillow sets the PDF MediaBox from the image size and this resolution,
    # so one pixel is exactly 1/dpi inch and the page is its true size.
    Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)).save(
        pdf_path, "PDF", resolution=float(args.dpi)
    )
    if args.also_png:
        cv2.imwrite(str(out_dir / f"{name}.png"), page_img)
    np.save(npy_path, lead)

    manifest = {
        "created_utc": stamp,
        "git_commit": commit,
        "source": {
            "dataset": "PTB-XL 1.0.3",
            "ecg_id": int(args.ecg_id),
            "lead": args.lead,
            "lead_index": lead_idx,
            "sampling_rate_hz": float(meta.fs),
            "duration_s": float(len(lead) / meta.fs),
            "n_samples": int(len(lead)),
            "units": "mV",
            "patient_id": meta.patient_id,
            "strat_fold": meta.strat_fold,
            "truth_npy": npy_path.name,
        },
        "calibration": {
            "mm_per_s": float(args.mm_per_s),
            "mm_per_mv": float(args.mm_per_mv),
        },
        "render": {
            "px_per_mm": float(spec.px_per_mm),
            "trace_thickness_px": 2,
            "on_overflow": "expand",
            "v_min_mv": float(spec.v_min_mv),
            "v_max_mv": float(spec.v_max_mv),
            "required_height_mm": float(spec.required_height_mm),
            "clipped": bool(spec.clipped),
            "baseline_px_in_strip": float(spec.baseline_px),
            "x0_px_in_strip": float(spec.x0_px),
        },
        "geometry": geometry,
        "photo": {
            "corners_px": None,
            "corners_order": "TL, TR, BR, BL of the gridded strip -- the "
                             "inner vertex of each printed crop mark",
            "note": "Fill corners_px in after photographing, or pass "
                    "--corners to score_photo.py. Coordinates are in the "
                    "photograph's pixels, origin top-left.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    px_per_mm = geometry["px_per_mm_at_print"]
    print(f"wrote {pdf_path}")
    print(f"      {npy_path.name}   ground truth, {len(lead)} samples in mV")
    print(f"      {manifest_path.name}")
    print()
    print(f"  page         {geometry['page']} "
          f"{geometry['page_width_mm']:.1f} x {geometry['page_height_mm']:.1f} mm "
          f"at {args.dpi} dpi ({px_per_mm:.3f} px/mm)")
    print(f"  strip        {geometry['strip_width_mm']:.2f} x "
          f"{geometry['strip_height_mm']:.2f} mm "
          f"({geometry['strip_width_px']} x {geometry['strip_height_px']} px)")
    print(f"  signal       {spec.v_min_mv:+.3f} .. {spec.v_max_mv:+.3f} mV, "
          f"needed {spec.required_height_mm:.1f} mm, clipped={spec.clipped}")
    print()
    print("  NEXT, and none of it is optional:")
    print("    1. Print at 100% scale. Turn OFF 'Fit to page' / 'Shrink to fit'.")
    print("    2. Print in COLOUR. The digitiser separates grid from trace by")
    print("       red-channel arithmetic; a greyscale print breaks it.")
    print(f"    3. Measure the {RULER_LENGTH_MM:.0f} mm bar with a real ruler "
          f"before photographing.")
    print("    4. Follow docs/photo_protocol.md for capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
