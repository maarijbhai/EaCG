#!/usr/bin/env python3
"""Generate print-ready ECG sheets from PTB-XL via ecg-image-kit, at TRUE scale.

This is the missing half of the real-photograph experiment. `run_photo.py`
measures what a camera costs, but every number it can produce today is
bounded by not knowing what was on the paper: the page photographed on
2026-09-21 turned out to be a print of ecg-image-kit's sample
`ecg00001.png`, a public-domain scan with no waveform behind it. This
script produces pages that do have a waveform behind them, and carries the
identity along so it can never be lost again.

WHAT COMES OUT, per record, and why each piece exists:

    <name>.pdf          two pages. Page 1 is the ECG sheet at true physical
                        scale and carries nothing this script added -- it is
                        the kit's pixels, cropped at most. Page 2 is a
                        verification sheet with printed rulers.
    <name>.png          page 1 as an image, for reference and for scoring
                        the photograph against the clean original the way
                        `run_photo.py --experiment source` does.
    <name>_kit.json     the kit's own sidecar. **This is the valuable
                        part.** It carries, per lead, `lead_name`,
                        `start_sample`, `end_sample` and `plotted_pixels`
                        -- the pixel position of every plotted sample. That
                        is direct supervision for FR4, which does not
                        exist, and exact ground truth for FR5 and FR6 *in
                        image space*, which is better than a waveform:
                        it separates a tracing error from a calibration
                        error without inferring either.
    <name>_truth.npy    the 12-lead PTB-XL signal in mV, (n_samples, 12).
    <name>.json         provenance: ecg_id, patient_id, strat_fold, the
                        kit's config, git commit, library versions, the
                        page geometry actually produced, and the crop.

WHY TWO PAGES. The kit renders edge to edge -- its grid fills all
279.4 x 215.9 mm of a Letter sheet with no margin anywhere. There is
nowhere to put a ruler without drawing on the grid, and black ink inside
the grid is not cosmetic: `binarise_trace` returns it as trace and
`trace_positions` digitises it as signal. So the references go on their
own page, printed in the same job with the same settings. If page 2's
100 mm bar measures 100 mm, page 1 was printed at the same scale.

THE WHOLE EXPERIMENT RESTS ON THE PRINT BEING AT TRUE SCALE. The digitiser
recovers calibration from the grid, and the grid is only a ruler if 1 mm on
the page really is 1 mm. "Fit to page" and "Shrink oversized pages" are on
by default in common print dialogs and both destroy this silently -- there
is no way to detect it afterwards from the image alone. Measure page 2
before photographing anything.

A4 NEEDS A CROP, AND CROPPING IS SAFE WHERE SCALING IS NOT. The kit emits
Letter (279.4 x 215.9 mm); A4 landscape is 297 x 210, so the sheet is
5.9 mm too tall. Cropping blank grid rows changes nothing about px/mm;
scaling changes everything. This script crops symmetrically and **refuses
if the crop would remove trace ink**, naming the remedy instead of
quietly trimming a waveform.

    python make_kit_target.py --ecg-id 9
    python make_kit_target.py --n-records 6 --page a4-landscape

With no --ecg-id it draws from fold 10 (the test fold), because a fixture
from the training folds is unusable the moment a classifier exists and
replacing fixtures later invalidates every number measured against them.

Writes into out/kit_targets/<name>/. `out/` is gitignored and everything
here is reproducible from the manifest.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ecgproto import ptbxl
from make_print_target import (
    PAGE_SIZES_MM,
    PAGE_MARGIN_MM,
    RULER_LENGTH_MM,
    TEXT_HEIGHT_MM,
    _put_text,
    _mm,
    draw_rulers,
    git_commit,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "kit_targets"

#: The vendored ecg-image-kit generator. Not a dependency of this
#: repository -- it is a third-party checkout -- so the path is a default
#: that `--kit` overrides, and it is recorded in every manifest.
DEFAULT_KIT = Path(
    r"C:\dev\vendor\ecg-image-kit\codes\ecg-image-generator")

#: Raster resolution to render at. Note this sets print QUALITY, not
#: physical size: the kit lays its figure out in inches (`x_grid_inch =
#: 5/25.4`) and renders at `resolution` dots per inch, so the sheet is
#: 11 x 8.5 inch at any dpi. 300 puts a 1 mm rule at 11.8 px on paper,
#: comfortably above the ~3 px/mm floor where `estimate_grid_pitch` stops
#: resolving minor rules.
DEFAULT_DPI = 300

#: How dark a pixel must be, IN EVERY CHANNEL, to count as trace ink when
#: checking whether a crop would destroy signal.
#:
#: It has to be every channel, and the reason is a trap worth recording.
#: The kit draws the trace at (12,12,12), the major grid at pure red
#: (255,0,0) and the minor grid at (254,223,219). A threshold on the green
#: channel alone looks like it separates trace from grid -- and it does
#: separate it from the *minor* grid -- but pure red has a green value of
#: ZERO, so every major rule is counted as trace. The whole sheet then
#: reads as inked edge to edge and the crop guard refuses a crop that is
#: perfectly safe. Taking the per-pixel maximum across channels is what
#: actually means "dark": trace 12, major grid 255, minor grid 254.
TRACE_MAX_ANY_CHANNEL = 120

#: What the 1 mm rules are redrawn in, RGB. The kit's red palette draws them
#: at (254,223,219), within 2-36 levels of white, and a real printer drops
#: them: the first photographed kit sheet (PTB-XL 40, 2026-09-28, an A30s
#: frame) carries the 5 mm rules and no 1 mm rules at all, so the grid
#: estimator read the 5 mm pitch -- correctly, for the paper it was given --
#: and calibrated 5.12x wrong. Clinical paper prints its minor rules visibly,
#: so this is a correction toward realism, not a convenience. Chosen well
#: inside what an office printer reproduces while staying clearly lighter
#: than the (255,0,0) majors, and far above the trace on 255-R: 13 levels
#: of inverted red, where the trace sits at 200+ and the paper near 0.
MINOR_GRID_RGB = (242, 160, 160)

#: Largest distance, in RGB levels, from the white-to-minor-colour line at
#: which a pixel still counts as a blend of paper and minor rule. Measured
#: on the kit's own renders, those blends sit within 2 of the line; the
#: trace, the major rules and their anti-aliased edges sit 20 or more away.
MINOR_BLEND_TOLERANCE = 6.0


def recolour_minor_grid(sheet: np.ndarray, minor_rgb_kit, target_rgb
                        ) -> tuple[np.ndarray, int]:
    """Redraw the kit's minor rules in `target_rgb`. Returns (sheet, pixels).

    Every pixel the kit wrote as a blend of white paper and its minor colour
    -- p = white - a * (white - minor), 0 < a <= 1, which covers the rule and
    its anti-aliased edge -- is moved to the same blend of white and the
    target, so line width and anti-aliasing are kept exactly and only the
    colour changes. Anything off that line (trace, major rules, text, their
    blends with the minor rule) is left untouched. Geometry is unchanged, so
    `plotted_pixels` and the truth stay valid.
    """
    idx, a = minor_grid_blend(sheet, minor_rgb_kit)
    return apply_minor_blend(sheet, idx, a, target_rgb), int(len(idx))


def minor_grid_blend(sheet: np.ndarray, minor_rgb_kit, dtype=np.float64
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The analysis half of `recolour_minor_grid`: which pixels are a blend
    of white and the minor colour (flat indices), and how much (a in [0,1]).

    Split out so a caller recolouring one sheet many times -- the ML
    augmentation, `ml/strips.py` -- pays for the projection once. float64
    is the default and what the printed targets use; the training loader
    passes float32, ~2x faster, and may differ at the tolerance edge by a
    pixel or two, which only changes an augmentation.
    """
    white = np.array([255.0, 255.0, 255.0], dtype)
    # The sheet is BGR; the kit records its colours as RGB.
    d = white - np.asarray(minor_rgb_kit, dtype=dtype)[::-1]
    q = white - sheet.reshape(-1, 3).astype(dtype)
    a = q @ d / float(d @ d)
    resid = np.linalg.norm(q - a[:, None] * d[None, :], axis=1)
    blend = (a > 0.02) & (a <= 1.05) & (resid <= MINOR_BLEND_TOLERANCE)
    return np.flatnonzero(blend), np.clip(a[blend], 0.0, 1.0)


def apply_minor_blend(sheet: np.ndarray, idx: np.ndarray, a: np.ndarray,
                      target_rgb) -> np.ndarray:
    """The apply half: move each blend pixel to the same blend of white and
    `target_rgb` (RGB). Returns a new BGR uint8 sheet."""
    white = np.array([255.0, 255.0, 255.0])
    t = np.asarray(target_rgb, dtype=np.float64)[::-1]
    out = sheet.reshape(-1, 3).copy()
    out[idx] = np.round(white - a[:, None] * (white - t)[None, :]).astype(np.uint8)
    return out.reshape(sheet.shape)


def numpy_shim() -> None:
    """Make ecg-image-kit importable under NumPy 2.x.

    `imgaug` reads `np.sctypes`, removed in NumPy 2.0, and it is imported at
    module scope by the kit's `helper_functions`, so the whole generator
    fails to import before any argument is parsed. Patched here rather than
    in the vendor tree: that checkout is third-party and must stay
    byte-identical to upstream or its provenance means nothing.

    Only the clean-render path is needed, so TensorFlow and spaCy (the
    handwritten-text generator) are never imported and need not be
    installed. `imutils`, `validators` and `html5lib` are, and they are in
    the kit's own requirements.
    """
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }


def run_kit(kit_dir: Path, rec_stem: Path, out_dir: Path, dpi: int,
            rhythm_lead: str, print_header: bool,
            extra_argv: tuple[str, ...] = ()) -> tuple[Path, Path]:
    """Render one record with the kit. Returns (png, sidecar json).

    Run in-process rather than as a subprocess so the shim applies and so a
    failure surfaces as a traceback rather than an exit code. The kit reads
    its templates and fonts relative to the working directory, hence the
    chdir. `extra_argv` is appended to the kit's own arguments; the ML data
    generator uses it to vary the grid style (`ml/make_strips.py`).
    """
    numpy_shim()
    if str(kit_dir) not in sys.path:
        sys.path.insert(0, str(kit_dir))
    import os
    cwd = os.getcwd()
    try:
        os.chdir(kit_dir)
        import gen_ecg_image_from_data as gen
        argv = [
            # The kit's loader switches on the file EXTENSION, so this must
            # be the .dat and not the record stem; given a stem it falls
            # through and raises UnboundLocalError on an unset variable.
            "-i", str(rec_stem) + ".dat",
            "-hea", str(rec_stem) + ".hea",
            "-o", str(out_dir),
            "-st", "0",
            "-r", str(dpi),
            # 2 stores the per-lead geometry including plotted_pixels.
            "--store_config", "2",
            "--full_mode", rhythm_lead,
            "--calibration_pulse", "1",
        ]
        if print_header:
            argv.append("--print_header")
        argv += list(extra_argv)
        args = gen.get_parser().parse_args(argv)
        gen.run_single_file(args)
    finally:
        os.chdir(cwd)

    pngs = sorted(out_dir.glob("*.png"))
    jsons = sorted(out_dir.glob("*.json"))
    if not pngs or not jsons:
        raise RuntimeError(
            f"kit produced no image in {out_dir}; got {list(out_dir.iterdir())}")
    return pngs[0], jsons[0]


def fit_to_page(sheet: np.ndarray, px_per_mm: float, page: str
                ) -> tuple[np.ndarray, dict]:
    """Crop blank grid so the sheet fits `page`. Never scales.

    Returns (cropped sheet, crop record). Raises if the crop needed would
    reach trace ink, because a silently shortened lead is exactly the kind
    of fault that shows up later as an unexplained fidelity number.
    """
    page_w_mm, page_h_mm = PAGE_SIZES_MM[page]
    # Printable area. Consumer printers cannot reach the sheet edge and a
    # driver that thinks content is out of bounds will offer to scale --
    # which is the one thing that must not happen.
    avail_w = page_w_mm - 2 * PAGE_MARGIN_MM
    avail_h = page_h_mm - 2 * PAGE_MARGIN_MM
    h, w = sheet.shape[:2]
    have_w, have_h = w / px_per_mm, h / px_per_mm

    dark = (sheet.max(axis=2) < TRACE_MAX_ANY_CHANNEL)

    # The kit draws a 1 px black frame around the whole sheet. Strip it
    # before anything else, for two reasons. It is a solid run of dark
    # pixels along every edge, so it defeats any "where is the trace"
    # measurement -- which is how it was found, as a crop guard insisting
    # that a blank sheet was inked edge to edge. And it must not reach the
    # paper regardless: `binarise_trace` returns a black frame as ink and
    # `trace_positions` digitises it as signal, so a photograph of a framed
    # sheet carries a false trace along its border.
    def frame_run(is_dark_line) -> int:
        n = 0
        for v in is_dark_line:
            if v > 0.5:
                n += 1
            else:
                break
        return n

    row_frac = dark.mean(axis=1)
    col_frac = dark.mean(axis=0)
    top = frame_run(row_frac)
    bottom = frame_run(row_frac[::-1])
    left = frame_run(col_frac)
    right = frame_run(col_frac[::-1])
    if top or bottom or left or right:
        h0, w0 = sheet.shape[:2]
        sheet = sheet[top:h0 - bottom, left:w0 - right]
        dark = dark[top:h0 - bottom, left:w0 - right]
        h, w = sheet.shape[:2]
        have_w, have_h = w / px_per_mm, h / px_per_mm

    trace = dark
    rows = np.where(trace.any(axis=1))[0]
    cols = np.where(trace.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("no trace ink found in the rendered sheet")

    def crop_axis(need_mm, have_mm, lo, hi, n, axis):
        if have_mm <= need_mm:
            return 0, n, 0.0
        drop = int(round((have_mm - need_mm) * px_per_mm))
        # Take from whichever side has more blank, keeping the traces
        # centred rather than drifting toward one edge.
        head, tail = int(lo), int(n - 1 - hi)
        if head + tail < drop:
            raise RuntimeError(
                f"cannot fit {have_mm:.1f} mm of {axis} into {need_mm:.1f} mm "
                f"without cutting trace ink: only {(head+tail)/px_per_mm:.1f} "
                f"mm of blank grid exists. Use --page letter-landscape, or "
                f"print on the larger sheet; do NOT let the printer scale.")
        take_head = min(head, int(round(drop * head / max(1, head + tail))))
        take_tail = drop - take_head
        if take_tail > tail:
            take_tail, take_head = tail, drop - tail
        return take_head, n - take_tail, drop / px_per_mm

    r0, r1, dh = crop_axis(avail_h, have_h, rows[0], rows[-1], h, "height")
    c0, c1, dw = crop_axis(avail_w, have_w, cols[0], cols[-1], w, "width")
    out = sheet[r0:r1, c0:c1]
    return out, {
        "page": page,
        "printable_mm": [round(avail_w, 2), round(avail_h, 2)],
        "kit_sheet_mm": [round(have_w, 2), round(have_h, 2)],
        "cropped_mm": [round(dw, 2), round(dh, 2)],
        "frame_px_stripped": {"top": top, "bottom": bottom,
                              "left": left, "right": right},
        "crop_px": {"top": r0, "bottom": h - r1, "left": c0, "right": w - c1},
        "final_mm": [round(out.shape[1] / px_per_mm, 2),
                     round(out.shape[0] / px_per_mm, 2)],
        # In FINAL image coordinates, not the pre-crop ones -- a bbox whose
        # y1 exceeds the image height is worse than no bbox.
        "trace_bbox_px": [int(cols[0]) - c0, int(rows[0]) - r0,
                          int(cols[-1]) - c0, int(rows[-1]) - r0],
        # THE ONE NUMBER A CONSUMER OF plotted_pixels CANNOT DO WITHOUT.
        # The kit's sidecar gives every sample's position in the ORIGINAL
        # rendered image. This sheet has had a frame stripped and blank grid
        # cropped, so those coordinates are offset by exactly this much.
        # Subtract it, or every ground-truth point lands in the wrong place
        # by ~25 mm and the error looks like a tracing fault.
        "plotted_pixels_offset_px": {"x": left + c0, "y": top + r0},
        "plotted_pixels_note": (
            "kit plotted_pixels are (row, column) -- NOT (x, y) -- in the "
            "original render. For this sheet: row_final = row_kit - "
            f"{top + r0}, col_final = col_kit - {left + c0}."),
    }


def verification_page(px_per_mm: float, page: str, lines: list[str],
                      mm_per_mv: float) -> np.ndarray:
    """Page 2: the printed rulers and the record's identity.

    Separate from the ECG sheet because the kit's grid reaches every edge,
    so anything drawn on page 1 lands inside a region that gets digitised.
    """
    page_w_mm, page_h_mm = PAGE_SIZES_MM[page]
    img = np.full((_mm(page_h_mm, px_per_mm), _mm(page_w_mm, px_per_mm), 3),
                  255, dtype=np.uint8)
    y = PAGE_MARGIN_MM + TEXT_HEIGHT_MM * 2.2
    _put_text(img, "PRINT VERIFICATION -- measure before photographing",
              PAGE_MARGIN_MM, y, px_per_mm, TEXT_HEIGHT_MM * 1.4)
    y += TEXT_HEIGHT_MM * 3.0
    for line in lines:
        _put_text(img, line, PAGE_MARGIN_MM, y, px_per_mm, TEXT_HEIGHT_MM)
        y += TEXT_HEIGHT_MM * 1.9
    y += TEXT_HEIGHT_MM * 1.5
    for line in (
        "If the 100 mm bar below is not 100 mm on a real ruler, the printer",
        "scaled the job and BOTH pages are void. Reprint with scaling off:",
        "'Actual size' / 100%, never 'Fit to page' or 'Shrink oversized'.",
    ):
        _put_text(img, line, PAGE_MARGIN_MM, y, px_per_mm, TEXT_HEIGHT_MM)
        y += TEXT_HEIGHT_MM * 1.9
    draw_rulers(img, PAGE_MARGIN_MM, y + 4.0, px_per_mm, mm_per_mv)
    return img


def provenance(ecg_id: int, meta, kit_cfg: dict, geom: dict,
               params: dict) -> dict:
    return {
        "generated_by": "python/make_kit_target.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "source": {
            "dataset": "PTB-XL 1.0.3",
            "ecg_id": int(ecg_id),
            "patient_id": float(getattr(meta, "patient_id", float("nan"))),
            "strat_fold": int(getattr(meta, "strat_fold", -1)),
            "sampling_rate_hz": float(meta.fs),
            "n_samples": int(meta.n_samples)
            if hasattr(meta, "n_samples") else None,
            "duration_s": float(getattr(meta, "duration_s", float("nan"))),
            "lead_names": list(ptbxl.LEAD_NAMES),
        },
        "renderer": {
            "tool": "ecg-image-kit",
            "kit_dir": params["kit"],
            "note": "third-party checkout, unmodified; NumPy 2.x import "
                    "shimmed in-process by make_kit_target.numpy_shim",
        },
        "kit_config": kit_cfg,
        "page": geom,
        "params": params,
        "calibration": {
            "px_per_mm_nominal": params["dpi"] / 25.4,
            "px_per_mm_from_kit_x_grid": kit_cfg.get("x_grid", 0) / 5.0,
            "mm_per_s": 25.0,
            "mm_per_mv": 10.0,
            "note": "px_per_mm holds ONLY if the sheet is printed at 100%. "
                    "It is a property of the print, never of a photograph "
                    "of it -- recover scale from the image content.",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }


def one_record(ecg_id: int, args) -> Path:
    signal, meta = ptbxl.load_record(ecg_id, root=args.root,
                                     sampling_rate=args.rate)
    fold = int(getattr(meta, "strat_fold", -1))
    if fold != 10:
        print(f"  WARNING ecg_id {ecg_id} is strat_fold {fold}, not 10. "
              f"A fixture from a training fold is unusable once a classifier "
              f"exists.")
    rec_stem = (Path(args.root) / f"records{args.rate}"
                / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_hr")
    if args.rate == 100:
        rec_stem = rec_stem.with_name(f"{ecg_id:05d}_lr")
    if not rec_stem.with_suffix(".dat").exists():
        raise FileNotFoundError(f"no record at {rec_stem}.dat")

    name = f"ptbxl{ecg_id}_kit_{args.rate}hz_{args.dpi}dpi"
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        png, sidecar = run_kit(Path(args.kit), rec_stem, Path(tmp), args.dpi,
                               args.rhythm_lead, args.print_header)
        sheet = cv2.imread(str(png), cv2.IMREAD_COLOR)
        kit_cfg = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        shutil.copy(sidecar, out_dir / f"{name}_kit.json")

    px_per_mm = args.dpi / 25.4
    # The kit records pixels per 5 mm box; it must agree with dpi/25.4 or
    # the sheet is not laid out in the physical units everything assumes.
    x_grid = float(kit_cfg.get("x_grid", 0.0))
    if x_grid and abs(x_grid / 5.0 / px_per_mm - 1.0) > 1e-3:
        raise RuntimeError(
            f"kit x_grid {x_grid} px/5mm implies {x_grid/5:.4f} px/mm but "
            f"dpi/25.4 is {px_per_mm:.4f}; the sheet is not at true scale")

    minor_px = 0
    if args.minor_grid != "kit":
        target = tuple(int(v) for v in args.minor_grid.split(","))
        sheet, minor_px = recolour_minor_grid(
            sheet, kit_cfg["grid_line_color_minor"], target)

    sheet, geom = fit_to_page(sheet, px_per_mm, args.page)
    cv2.imwrite(str(out_dir / f"{name}.png"), sheet)

    hdr = [
        f"PTB-XL {ecg_id}   fold {fold}   {meta.fs:g} Hz   "
        f"rhythm lead {args.rhythm_lead}",
        f"rendered by ecg-image-kit at {args.dpi} dpi  "
        f"({px_per_mm:.4f} px/mm, 25 mm/s, 10 mm/mV)",
        f"sheet {geom['final_mm'][0]:.1f} x {geom['final_mm'][1]:.1f} mm "
        f"on {args.page}   git {git_commit()[:12]}",
        f"ground truth: {name}_kit.json (plotted_pixels), {name}_truth.npy",
    ]
    verify = verification_page(px_per_mm, args.page, hdr, 10.0)

    pdf = out_dir / f"{name}.pdf"
    # Pillow takes the PDF MediaBox from the pixel size and this resolution,
    # so the page is physically correct and the driver has no reason to
    # rescale. This is the only place DPI is allowed to mean inches.
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(
        pdf, "PDF", resolution=float(args.dpi), save_all=True,
        append_images=[Image.fromarray(cv2.cvtColor(verify, cv2.COLOR_BGR2RGB))],
    )

    np.save(out_dir / f"{name}_truth.npy", signal.astype(np.float64))
    params = {"dpi": args.dpi, "page": args.page, "rate": args.rate,
              "rhythm_lead": args.rhythm_lead, "kit": str(args.kit),
              "print_header": bool(args.print_header),
              # The kit_config's grid_line_color_minor is what the kit drew;
              # this is what was printed. See MINOR_GRID_RGB.
              "minor_grid_rgb_printed": args.minor_grid,
              "minor_grid_pixels_recoloured": minor_px}
    (out_dir / f"{name}.json").write_text(
        json.dumps(provenance(ecg_id, meta, kit_cfg, geom, params), indent=2),
        encoding="utf-8")

    print(f"  {name}")
    print(f"    sheet   {geom['final_mm'][0]:.1f} x {geom['final_mm'][1]:.1f} mm"
          f"  (cropped {geom['cropped_mm'][0]:.1f} x "
          f"{geom['cropped_mm'][1]:.1f} mm of blank grid)")
    print(f"    leads   {len(kit_cfg.get('leads', []))} with plotted_pixels")
    print(f"    pdf     {pdf}")
    return pdf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ecg-id", type=int, action="append", default=None,
                   help="PTB-XL record; repeatable. Default: fold-10 records.")
    p.add_argument("--n-records", type=int, default=1,
                   help="how many fold-10 records when --ecg-id is not given")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--page", default="a4-landscape",
                   choices=sorted(PAGE_SIZES_MM))
    p.add_argument("--rate", type=int, choices=(100, 500), default=500)
    p.add_argument("--rhythm-lead", default="II")
    p.add_argument("--minor-grid", default=",".join(map(str, MINOR_GRID_RGB)),
                   help="R,G,B to redraw the 1 mm rules in, or 'kit' to keep "
                        "the kit's near-white (254,223,219), which printers drop")
    p.add_argument("--print-header", action="store_true",
                   help="let the kit print its ID/Name header band")
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    args = p.parse_args(argv)

    if not Path(args.kit).exists():
        print(f"ecg-image-kit generator not found at {args.kit}; pass --kit")
        return 2

    ids = args.ecg_id
    if not ids:
        df = ptbxl.load_metadata(args.root)
        _, _, test = ptbxl.split_folds(df)
        ids = [int(i) for i in test.index[: args.n_records]]
        print(f"no --ecg-id given; drawing {len(ids)} from fold 10: {ids}")

    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for ecg_id in ids:
        try:
            made.append(one_record(ecg_id, args))
        except Exception as exc:
            print(f"  ecg_id {ecg_id} FAILED: {type(exc).__name__}: {exc}")
    if not made:
        return 1
    print(f"\n{len(made)} sheet(s) in {OUT}")
    print("Print at 100% / 'Actual size'. Both pages, one job, same settings.")
    print("Measure the 100 mm bar on page 2 with a real ruler BEFORE "
          "photographing page 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
