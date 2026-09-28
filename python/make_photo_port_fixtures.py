#!/usr/bin/env python3
"""Golden vectors for the Kotlin port of the photograph path.

Covers `ecgproto/segment.py` (StripLocator.kt) and `ecgproto/page.py`
(PageCorners.kt), the two stages the phone needs to score a photograph of a
printed sheet and that did not exist when `make_port_fixtures.py` was
written. Kept in its own file and its own fixture so that regenerating it
cannot move a single number in the existing port suite.

Writes android/app/src/test/resources/photo_port_fixtures.txt, committed,
in the same flat `key v1 v2 ...` format `PortFixtures` reads.

LOCATOR CASES. Masks are shipped run-length encoded, one line per inked row
(`<case>.rle.<row> start len start len ...`): a trace mask is mostly empty
and a page-sized mask would otherwise be megabytes of zeros.

  synth   layout.render_12lead at 3 px/mm with a calibration pulse drawn
          before the rhythm strip, lead labels, and a "25mm/s 10mm/mV" line
          under it -- every feature the locator has a rule for.
  kit38   out/kit_targets/ptbxl38 downscaled to 40%. The real sheet, and
          the one whose trace leaves the pulse with a steep deflection (the
          "second cluster, not the last" rule). Skipped, with a note in the
          file, if out/ has not been generated.

CORNER CASE. The image is not shipped at all: it is DEFINED here and in
`PhotoPortFixturesTest.kt` by the same integer-and-double formula -- a
sheared gridded sheet on off-white paper with LCG noise -- so both sides
build it bit for bit and only the rough corners and the answers travel.

    python make_photo_port_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import signal as sps

from ecgproto import digitise as dig
from ecgproto import layout, page, segment, synth

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "android" / "app" / "src" / "test" / "resources" / "photo_port_fixtures.txt"
KIT = REPO / "out" / "kit_targets" / "ptbxl38_kit_500hz_300dpi" / "ptbxl38_kit_500hz_300dpi.png"


def fmt(v) -> str:
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(int(v))


def emit_mask(lines: list[str], case: str, mask: np.ndarray) -> None:
    h, w = mask.shape
    lines.append(f"{case}.shape {h} {w}")
    for r in range(h):
        row = mask[r]
        if not row.any():
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        parts = " ".join(f"{s} {e - s}" for s, e in zip(starts, ends))
        lines.append(f"{case}.rle.{r} {parts}")


def emit_location(lines: list[str], case: str, mask: np.ndarray) -> None:
    loc = segment.locate_rhythm_strip(np.zeros(mask.shape + (3,), np.uint8), mask=mask)
    lines.append(f"{case}.box {loc.x0} {loc.y0} {loc.x1} {loc.y1}")
    lines.append(f"{case}.peaks " + " ".join(str(p) for p in loc.row_peaks))
    lines.append(f"{case}.pulse_x1 {loc.pulse_x1 if loc.pulse_x1 is not None else -1}")
    lines.append(f"{case}.t0_x {fmt(loc.t0_x)}")
    iso = segment.isolate_trace(mask[loc.y0:loc.y1, loc.x0:loc.x1])
    lines.append(f"{case}.isolated_cols " + " ".join(str(int(v)) for v in iso.sum(axis=0)))


def synth_sheet() -> np.ndarray:
    return dig.binarise_trace(synth_sheet_image())


def synth_sheet_image() -> np.ndarray:
    fs = 500.0
    beats = synth.synthetic_ecg(duration_s=10.0, fs=fs, heart_rate_bpm=75.0, seed=3)
    sig = beats[:, None] * np.linspace(0.6, 1.2, 12)[None, :]
    px = 3.0
    image, spec = layout.render_12lead(sig, fs, px_per_mm=px, draw_labels=True)
    cell = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
    x1 = int(round(cell.x0_px))
    x0 = x1 - int(round(5 * px))
    base = int(round(cell.baseline_px))
    top = base - int(round(10 * px))
    pts = np.array([[x0 - 3, base], [x0, base], [x0, top], [x1, top], [x1, base]], np.int32)
    cv2.polylines(image, [pts], False, (0, 0, 0), 1)
    cv2.putText(image, "25mm/s  10mm/mV", (x1 + 60, int(cell.y_bottom_px) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return image


# --------------------------------------------------------------------------
# the procedurally defined corner image -- MUST match PhotoPortFixturesTest
# --------------------------------------------------------------------------

CORNER_W, CORNER_H = 520, 380
SHEET_X0, SHEET_Y0, SHEET_W, SHEET_H = 60.0, 45.0, 400.0, 280.0
SHEAR = 0.03
LCG_SEED, LCG_MULT, LCG_ADD, LCG_MASK = 12345, 1103515245, 12345, 0x7FFFFFFF


def corner_image() -> np.ndarray:
    """BGR uint8. Shared definition; see the module docstring."""
    img = np.zeros((CORNER_H, CORNER_W, 3), np.uint8)
    paper = (236, 241, 244)                 # B, G, R
    minor = (219, 223, 254)
    major = (0, 0, 255)
    state = LCG_SEED
    for y in range(CORNER_H):
        for x in range(CORNER_W):
            u = (x - SHEET_X0) + SHEAR * (y - SHEET_Y0)
            v = (y - SHEET_Y0) - SHEAR * (x - SHEET_X0)
            ink = None
            if 0.0 <= u < SHEET_W and 0.0 <= v < SHEET_H:
                iu, iv = int(np.floor(u)), int(np.floor(v))
                if iu % 40 == 0 or iv % 40 == 0:
                    ink = major
                elif iu % 8 == 0 or iv % 8 == 0:
                    ink = minor
            state = (state * LCG_MULT + LCG_ADD) & LCG_MASK
            noise = (state >> 16) % 5 - 2
            for ch in range(3):
                val = paper[ch] if ink is None else paper[ch] * ink[ch] // 255
                img[y, x, ch] = min(255, max(0, val + noise))
    return img


def main() -> int:
    lines = [
        "# generated by python/make_photo_port_fixtures.py -- do not edit",
        "# golden vectors for StripLocator.kt and PageCorners.kt",
    ]

    # ---- primitives the locator's answer depends on ----------------------
    rng = np.random.default_rng(7)
    x = np.round(rng.random(200) * 10) / 2.0          # plateaus and ties
    x[50:54] = 9.0
    lines.append("prim.x " + " ".join(fmt(v) for v in x))
    lines.append("prim.peaks_d7 " + " ".join(str(p) for p in segment.find_peaks_distance(x, 7)))
    lines.append("prim.box9 " + " ".join(fmt(v) for v in np.convolve(x, np.ones(9), mode="same") / 9))
    b = (rng.random(300) < 0.3).astype(np.uint8)
    b[:5] = 1
    lines.append("prim.bits " + " ".join(str(int(v)) for v in b))
    for k in (6, 7):
        c = cv2.morphologyEx(b.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)).ravel()
        lines.append(f"prim.close{k} " + " ".join(str(int(v)) for v in c))

    # ---- locator ---------------------------------------------------------
    m = synth_sheet()
    emit_mask(lines, "synth", m)
    emit_location(lines, "synth", m)
    # The same sheet with a calibration pulse at the start of the rhythm row
    # and a short thin fragment before it -- what a grid remnant at the strip
    # edge looks like on a photograph. Exercises the lead-in branch of the
    # pulse test (segment.PULSE_LEADIN_FRACTION), which synth alone does not.
    lm = m.copy()
    band = slice(291, 329)
    lm[band, 0:62] = False
    lm[317:320, 18:25] = True            # the lead-in fragment
    lm[295:320, 27:29] = True            # rising edge
    lm[295:297, 27:46] = True            # flat top
    lm[295:320, 44:46] = True            # falling edge
    row61 = int(np.flatnonzero(m[band, 62])[0]) + band.start if m[band, 62].any() else 318
    lm[min(317, row61):max(320, row61 + 1), 46:62] = True   # back to the trace
    emit_mask(lines, "leadin", lm)
    emit_location(lines, "leadin", lm)
    if KIT.exists():
        sheet = cv2.imread(str(KIT), cv2.IMREAD_COLOR)
        small = cv2.resize(sheet, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        km = dig.binarise_trace(small)
        emit_mask(lines, "kit38", km)
        emit_location(lines, "kit38", km)
        lines.append("cases synth kit38")
    else:
        lines.append(f"# kit38 skipped: {KIT} not generated")
        lines.append("cases synth")

    # ---- corners ---------------------------------------------------------
    img = corner_image()
    dark = page.ink_darkness(img)
    lines.append("corner.dark_probe " + " ".join(
        fmt(float(dark[y, x])) for y, x in ((0, 0), (50, 70), (100, 200), (379, 519))))
    true = []
    for (u, v) in ((0, 0), (SHEET_W, 0), (SHEET_W, SHEET_H), (0, SHEET_H)):
        # invert u = dx + s dy, v = dy - s dx for the corner (u, v)
        s = SHEAR
        dx = (u - s * v) / (1 + s * s)
        dy = (v + s * u) / (1 + s * s)
        true.append((SHEET_X0 + dx - 0.5, SHEET_Y0 + dy - 0.5))
    rough = np.array(true) + np.array([[4.0, -3.0], [-5.0, 2.0], [3.0, 4.0], [-2.0, -5.0]])
    lines.append("corner.rough " + " ".join(fmt(v) for v in rough.ravel()))
    r = page.refine_corners(img, rough)
    lines.append("corner.refined " + " ".join(fmt(v) for v in np.asarray(r.corners, np.float64).ravel()))
    lines.append("corner.edge_rmse " + " ".join(fmt(v) for v in r.edge_rmse_px))
    lines.append("corner.inliers " + " ".join(str(int(v)) for v in r.edge_inliers))
    lines.append("corner.true " + " ".join(fmt(v) for v in np.array(true).ravel()))

    # ---- binarisation threshold (midpoint rule) ---------------------------
    hists = {}
    if KIT.exists():
        sheet = cv2.imread(str(KIT), cv2.IMREAD_COLOR)
        hists["kitstrip"] = (255 - sheet[1780:2230, 175:3120, 2]).astype(np.uint8)
    hists["synthpage"] = (255 - synth_sheet_image()[:, :, 2]).astype(np.uint8)
    r = np.random.default_rng(11)
    hists["noise"] = np.clip(r.normal(40, 12, 20000), 0, 255).astype(np.uint8)
    hists["flat"] = np.full(1000, 7, np.uint8)
    for name, inv in hists.items():
        h = np.bincount(inv.ravel(), minlength=256)
        lines.append(f"thr.{name}.hist " + " ".join(str(int(v)) for v in h))
        lines.append(f"thr.{name}.t {dig.midpoint_threshold(inv)}")
    lines.append("thr.cases " + " ".join(hists))

    # ---- harmonic resolution ----------------------------------------------
    # The autocorrelation of real projections and the spacing the peak rule
    # reported. `simy` is a rectified simulated photograph whose y axis
    # locks at twice the pitch; `fixture` is one where nothing should move.
    import score_kit_photo as skp
    from score_photo import rectify
    hcases = {}
    fx = REPO / "android" / "app" / "src" / "main" / "assets" / "reference" / "gt_mi_200dpi.png"
    hcases["fixture"] = dig.redness(cv2.imread(str(fx))).mean(axis=0)
    if KIT.exists():
        t = skp.Target("ptbxl38_kit_500hz_300dpi")
        photo, true, _ = skp.simulate(t, skp.SIM_CONDITIONS[1], seed=1)
        rect, _ = rectify(photo, true)
        hcases["simy"] = dig.redness(rect).mean(axis=1)
    for name, prof in hcases.items():
        prof = dig.detrend(prof)
        spacing = dig._pitch_1d(prof, 3.0, 80.0, dig.DEFAULT_PEAK_FRACTION)[0]
        # Truncated to 1500 lags so the fixture stays small; the Kotlin is
        # handed exactly this array, and the Python answer below is
        # recomputed on it, so both sides see the same input.
        ac = dig._acf(prof)[:1500]
        pitch, div = dig._resolve_harmonic_acf(ac, spacing, 3.0)
        lines.append(f"harm.{name}.acf " + " ".join(fmt(v) for v in ac))
        lines.append(f"harm.{name}.spacing {fmt(spacing)}")
        lines.append(f"harm.{name}.out {fmt(pitch)} {div}")
        lines.append(f"harm.{name}.novelty " + " ".join(
            fmt(dig._novelty(ac, spacing, m)) for m in dig.HARMONIC_DIVISORS))
    lines.append("harm.cases " + " ".join(hcases))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} lines -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
