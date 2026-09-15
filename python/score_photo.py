#!/usr/bin/env python3
"""Score a photograph of a printed ECG strip against its stored ground truth.

The second half of the real-photograph experiment. `make_print_target.py`
rendered a PTB-XL lead to paper at true physical scale and saved the source
waveform beside it; a human printed and photographed it; this reads the
photograph back and reports how much of the waveform survived the round
trip through ink, paper, lens and sensor.

**Why this is the only measurement in the project that means anything about
field performance.** Everything measured so far comes from a closed loop --
the image is drawn by the same code, with the same constants, that decodes
it. Grid pitch returns +0.00% error because the renderer drew rules at
exactly that integer pitch. The best time shift is exactly 0 samples
because the strip begins where the renderer said it begins. Those are not
results, they are tautologies, and they set an upper bound rather than an
estimate. This script measures the distance between that bound and reality.

**Report per stage, not just end to end.** A single SNR figure cannot
distinguish a calibration failure from a tracing failure, and on a
photograph both are live for the first time. So each stage is reported
separately, and the grid estimate is checked against the truth implied by
the known print scale -- because audit finding B1 is that
`estimate_grid_pitch` can lock onto a harmonic and return confident,
self-consistent, wrong numbers, and nothing in `digitise` currently guards
against it. On a render that never happened. On a photograph it is the
first thing to suspect.

Rectification is **manual and deliberate**. Four corners are supplied, not
detected: automatic page detection and deskew are a later pipeline stage
(ML Kit Document Scanner is the intended capture-time fix) and building
them here would confound the measurement with the quality of a detector
nobody has written yet. The output size is derived only from the
photograph's own corner geometry -- the mean length of each pair of
opposite edges -- and never from the strip's known aspect ratio. Forcing
the true aspect would hand the estimator half the answer it is being tested
on, and would erase exactly the anisotropy that perspective introduces.

`digitise.py` was NOT modified for any of this.

Usage::

    python python/score_photo.py --photo photo.jpg --manifest m.json --pick
    python python/score_photo.py --photo photo.jpg --manifest m.json \
        --corners "412,388 3180,340 3210,1090 402,1130"
    python python/score_photo.py --photo photo.jpg --manifest m.json \
        --save-corners --diagnostics out/photo_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from ecgproto import digitise as dig
from ecgproto import evaluate

#: Ratios worth testing the grid estimate against. A comb autocorrelates at
#: every multiple of its pitch, and ECG paper has a 5:1 major/minor
#: structure, so these are the lags a failed estimate lands on.
HARMONICS = (0.2, 0.5, 1.0, 2.0, 5.0)


def parse_corners(text: str) -> np.ndarray:
    """Parse 'x,y x,y x,y x,y' into a (4, 2) float32 array, TL TR BR BL."""
    parts = text.replace(";", " ").split()
    if len(parts) != 4:
        raise ValueError(
            f"need exactly 4 corners as 'x,y x,y x,y x,y', got {len(parts)}"
        )
    pts = []
    for part in parts:
        xy = part.split(",")
        if len(xy) != 2:
            raise ValueError(f"corner {part!r} is not 'x,y'")
        pts.append([float(xy[0]), float(xy[1])])
    return np.asarray(pts, dtype=np.float32)


def pick_corners(image: np.ndarray) -> np.ndarray:
    """Click the four strip corners in a window. Returns (4, 2) float32.

    Click order is TL, TR, BR, BL -- the inner vertex of each printed crop
    mark. The window is scaled to fit the screen and the clicks are scaled
    back up, so the returned coordinates are in the photograph's own pixels.
    """
    max_side = 1100.0
    scale = min(1.0, max_side / max(image.shape[:2]))
    view = cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()
    picked: list[tuple[float, float]] = []
    labels = ("top-left", "top-right", "bottom-right", "bottom-left")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 4:
            picked.append((x, y))
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(len(picked)), (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    window = "click the 4 strip corners: TL, TR, BR, BL  (u = undo, enter = done)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        shown = view.copy()
        if len(picked) < 4:
            cv2.putText(shown, f"click the {labels[len(picked)]} corner",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(picked) == 4:
            break
        if key == ord("u") and picked:
            picked.pop()
            view = cv2.resize(image, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()
            for i, (px, py) in enumerate(picked, start=1):
                cv2.circle(view, (int(px), int(py)), 6, (0, 0, 255), -1)
                cv2.putText(view, str(i), (int(px) + 9, int(py) - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if key == 27:
            cv2.destroyWindow(window)
            raise SystemExit("cancelled")
    cv2.destroyWindow(window)
    return np.asarray(picked, dtype=np.float32) / scale


def rectify(
    image: np.ndarray, corners: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Perspective-warp the quad to a rectangle. Returns (image, geometry).

    Output size is the mean length of each pair of opposite edges, measured
    in the photograph. Nothing about the strip's true dimensions enters
    here; that knowledge is reserved for scoring the result.
    """
    tl, tr, br, bl = corners
    edge = lambda a, b: float(np.hypot(*(a - b)))
    top, bottom = edge(tl, tr), edge(bl, br)
    left, right = edge(tl, bl), edge(tr, br)

    out_w = int(round((top + bottom) / 2.0))
    out_h = int(round((left + right) / 2.0))
    if out_w < 16 or out_h < 16:
        raise ValueError(
            f"the four corners enclose only {out_w}x{out_h} px; "
            "they are probably in the wrong order or units"
        )

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    geometry = {
        "edge_top_px": top, "edge_bottom_px": bottom,
        "edge_left_px": left, "edge_right_px": right,
        # Opposite edges of a rectangle photographed square are equal; the
        # ratio is a direct read on how oblique the camera was.
        "keystone_horizontal": max(top, bottom) / min(top, bottom),
        "keystone_vertical": max(left, right) / min(left, right),
        "out_w": out_w, "out_h": out_h,
    }
    return warped, geometry


def harmonic_of(estimate: float, truth: float) -> tuple[float, float] | None:
    """Nearest harmonic the estimate sits on, if it is not the fundamental.

    Returns (ratio, relative error against that harmonic) or None when the
    estimate is closest to 1.0x. This is the check `digitise` does not do:
    a 2x or 5x lock returns a finite, positive, self-consistent pitch and
    halves or fifths every recovered amplitude without complaint.
    """
    if not np.isfinite(estimate) or truth <= 0:
        return None
    ratio = estimate / truth
    best = min(HARMONICS, key=lambda h: abs(np.log(ratio / h)))
    if best == 1.0:
        return None
    return best, ratio / best - 1.0


def report_grid(grid: dig.GridEstimate, true_x: float, true_y: float) -> None:
    print("\n[2] GRID  estimate_grid_pitch on the rectified strip")
    for axis, est, true in (("x", grid.px_per_mm_x, true_x),
                            ("y", grid.px_per_mm_y, true_y)):
        if not np.isfinite(est):
            print(f"    px_per_mm_{axis}   FAILED (NaN)   true {true:.4f}")
            continue
        err = (est - true) / true
        print(f"    px_per_mm_{axis}   {est:8.4f}   true {true:8.4f}   "
              f"error {err:+7.2%}")
        lock = harmonic_of(est, true)
        if lock is not None:
            ratio, residual = lock
            print(f"      >>> HARMONIC LOCK: sitting on {ratio:g}x the true "
                  f"pitch (within {residual:+.1%} of it).")
            print(f"      >>> Audit finding B1: digitise does not check this. "
                  f"Every recovered amplitude is wrong by ~{ratio:g}x.")
    print(f"    peaks found  x {grid.n_peaks_x}   y {grid.n_peaks_y}")
    print(f"    anisotropy   {grid.anisotropy:.4f}    "
          f"confidence {grid.confidence:.3f}")
    print("      (both are computed and then ignored by digitise -- B1)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corners", type=str, default=None,
                        help="'x,y x,y x,y x,y' as TL TR BR BL")
    parser.add_argument("--pick", action="store_true",
                        help="click the corners in a window")
    parser.add_argument("--save-corners", action="store_true",
                        help="write the corners back into the manifest")
    parser.add_argument("--diagnostics", type=Path, default=None,
                        help="directory for rectified / mask / overlay images")
    args = parser.parse_args(argv)

    if not args.photo.exists():
        print(f"no photograph at {args.photo}", file=sys.stderr)
        return 2
    if not args.manifest.exists():
        print(f"no manifest at {args.manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source, geom = manifest["source"], manifest["geometry"]
    truth_path = args.manifest.parent / source["truth_npy"]
    if not truth_path.exists():
        print(f"no ground truth at {truth_path}", file=sys.stderr)
        return 2
    reference = np.load(truth_path)
    fs = float(source["sampling_rate_hz"])

    photo = cv2.imread(str(args.photo), cv2.IMREAD_COLOR)
    if photo is None:
        print(f"could not decode {args.photo}", file=sys.stderr)
        return 2

    if args.corners:
        try:
            corners = parse_corners(args.corners)
        except ValueError as exc:
            print(f"bad --corners: {exc}", file=sys.stderr)
            return 2
    elif args.pick:
        try:
            corners = pick_corners(photo)
        except cv2.error as exc:
            print(f"--pick needs a GUI-capable OpenCV build: {exc}\n"
                  f"pass --corners 'x,y x,y x,y x,y' instead", file=sys.stderr)
            return 2
    elif manifest.get("photo", {}).get("corners_px"):
        corners = np.asarray(manifest["photo"]["corners_px"], dtype=np.float32)
    else:
        print("no corners: pass --corners, or --pick, or put corners_px in "
              "the manifest", file=sys.stderr)
        return 2

    print("=" * 74)
    print(f"PTB-XL {source['ecg_id']} lead {source['lead']}  "
          f"{fs:g} Hz  {source['duration_s']:g} s   "
          f"rendered at commit {manifest['git_commit']}")
    print(f"photograph {args.photo.name}  {photo.shape[1]}x{photo.shape[0]} px")
    print("=" * 74)

    print("\n[1] RECTIFY  manual four-corner perspective warp")
    print(f"    corners  " + "  ".join(f"({x:.0f},{y:.0f})" for x, y in corners))
    try:
        rectified, rgeom = rectify(photo, corners)
    except ValueError as exc:
        print(f"    FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"    edges px   top {rgeom['edge_top_px']:.1f}  "
          f"bottom {rgeom['edge_bottom_px']:.1f}  "
          f"left {rgeom['edge_left_px']:.1f}  right {rgeom['edge_right_px']:.1f}")
    print(f"    keystone   horizontal {rgeom['keystone_horizontal']:.4f}  "
          f"vertical {rgeom['keystone_vertical']:.4f}   (1.0 = square on)")
    print(f"    rectified  {rgeom['out_w']} x {rgeom['out_h']} px")

    strip_w_mm = float(geom["strip_width_mm"])
    strip_h_mm = float(geom["strip_height_mm"])
    true_x = rgeom["out_w"] / strip_w_mm
    true_y = rgeom["out_h"] / strip_h_mm
    print(f"    strip is   {strip_w_mm:.2f} x {strip_h_mm:.2f} mm of real paper")
    print(f"    => TRUE    {true_x:.4f} px/mm horizontally, "
          f"{true_y:.4f} px/mm vertically")
    print(f"       (printed at {geom['px_per_mm_at_print']:.3f} px/mm; the "
          f"photograph resolves it at {true_x / geom['px_per_mm_at_print']:.2f}x that)")
    if min(true_x, true_y) < 3.0:
        print("    !! below ~3 px/mm the 1 mm rules stop resolving and the "
              "pitch estimate silently returns the 5 mm spacing instead")

    grid = dig.estimate_grid_pitch(rectified)
    report_grid(grid, true_x, true_y)

    print("\n[3] BINARISE  binarise_trace, Otsu on the inverted red channel")
    mask = dig.binarise_trace(rectified)
    print(f"    ink        {mask.mean():.3%} of pixels")
    print(f"    columns    {(mask.any(axis=0)).mean():.1%} contain ink")

    print("\n[4] TRACE  trace_positions, per-column ink centroid")
    y_px = dig.trace_positions(mask)
    print(f"    NaN        {np.isnan(y_px).mean():.1%} of columns had no ink")

    print("\n[5] DIGITISE  full pipeline, reusing the grid estimate above")
    try:
        result = dig.digitise(rectified, fs_out=fs, grid=grid)
    except ValueError as exc:
        print(f"    FAILED: {exc}")
        print("\nNo waveform was recovered, so no metric can be reported.")
        return 1
    nan_frac = float(np.isnan(result.signal_mv).mean())
    print(f"    coverage   {result.coverage:.1%}")
    print(f"    baseline   row {result.baseline_px:.1f} of {rgeom['out_h']}")
    print(f"    output     {len(result.signal_mv)} samples at {result.fs:g} Hz "
          f"(reference has {len(reference)})")
    print(f"    range      {np.nanmin(result.signal_mv):+.3f} .. "
          f"{np.nanmax(result.signal_mv):+.3f} mV   "
          f"(reference {reference.min():+.3f} .. {reference.max():+.3f})")
    if nan_frac:
        print(f"    !! {nan_frac:.1%} of the output is NaN (audit finding B2: "
              f"digitise does not check, and the metrics will return NaN)")

    print("\n[6] METRICS  against the stored ground truth")
    metrics = evaluate.evaluate_all(reference, result.signal_mv, fs=fs)
    print(f"    SNR          {metrics.snr_db:8.2f} dB")
    print(f"    shifted SNR  {metrics.shifted_snr_db:8.2f} dB  "
          f"(best shift {metrics.best_shift_samples:+d} samples)")
    print(f"    RMSE         {metrics.rmse_mv:8.4f} mV")
    print(f"    PCC          {metrics.pcc:8.3f}")
    print(f"    gain ratio   {metrics.gain_ratio:8.3f}")
    print(f"    compared on  {metrics.n_compared} samples")
    print(f"\n    reading: {metrics.diagnose()}")

    print("\n[7] WHERE THE ERROR CAME FROM")
    pitch_err = (grid.px_per_mm_y - true_y) / true_y if np.isfinite(grid.px_per_mm_y) else float("nan")
    print(f"    calibration  vertical pitch is {pitch_err:+.2%} off truth, "
          f"which alone would scale amplitude by {1/(1+pitch_err):.3f}"
          if np.isfinite(pitch_err) else
          "    calibration  vertical pitch estimate failed")
    if np.isfinite(pitch_err) and np.isfinite(metrics.gain_ratio):
        residual = metrics.gain_ratio * (1 + pitch_err)
        print(f"    tracing      removing that, the tracer's own gain is "
              f"{residual:.3f}")
        print(f"                 (renders at 8-12 px/mm give 0.86-0.93; "
              f"that is the closed-loop bound to compare against)")

    if args.diagnostics:
        args.diagnostics.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.diagnostics / "rectified.png"), rectified)
        cv2.imwrite(str(args.diagnostics / "mask.png"), mask.astype(np.uint8) * 255)
        overlay = rectified.copy()
        cols = np.flatnonzero(np.isfinite(y_px))
        overlay[np.clip(y_px[cols], 0, rectified.shape[0] - 1).astype(int), cols] = (255, 0, 0)
        cv2.imwrite(str(args.diagnostics / "overlay.png"), overlay)
        print(f"\nwrote diagnostics to {args.diagnostics}")

    if args.save_corners:
        manifest.setdefault("photo", {})["corners_px"] = corners.tolist()
        manifest["photo"]["photo_file"] = args.photo.name
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"saved corners into {args.manifest.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
