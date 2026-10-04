#!/usr/bin/env python3
"""How accurately does each geometry stage put the page onto a regular grid?

Every pipeline stage after rectification assumes the page is a uniform
ruler: one px/mm everywhere, rows horizontal. This measures how far each
geometry variant is from that, against the truth for the test sheets -- the
homography that registers the photograph onto the printed PNG
(`corner_truth.find_true_corners`, an evaluation instrument only).

For each variant, a lattice of points every 5 mm over the printed sheet is
mapped sheet -> photograph (truth) -> output page (the variant), and the best
SIMILARITY (one scale, rotation, offset) from sheet mm to output px is
fitted. What remains is distortion no later stage can undo: a non-uniform
scale is a time base that drifts across the page. Reported in mm of paper:
RMS, 95th percentile and worst point, and the horizontal part separately
(it is the time axis: 1 mm = 40 ms at 25 mm/s).

Variants, all from the phone's 3,200 px decode:

    auto            page.find_sheet_corners -> sheet.rectify (the phone today)
    auto+grid       ... then gridfit.fit_grid on the rectified page
    truth           the registered corners -> sheet.rectify (the ceiling for
                    any four-corner method)
    truth+grid      ... then gridfit.fit_grid

    python run_geometry.py

Writes results/kitphoto/geometry.csv + .json.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import corner_truth, digitise as dig, gridfit, page, segment, sheet  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "kitphoto" / "geometry"
DECODE_PX = 3200
LATTICE_MM = 5.0
#: Points this close to the sheet's edge are left out: the trace region is
#: what matters, and the border is where every variant is worst.
EDGE_MM = 5.0


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity src -> dst (Umeyama). Returns (scale, R, t)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return scale, R, md - scale * R @ ms


def apply_h(H, pts):
    return cv2.perspectiveTransform(pts[None].astype(np.float64), H)[0]


def distortion(sheet_mm: np.ndarray, out_px: np.ndarray) -> dict:
    s, R, t = similarity(sheet_mm, out_px)
    pred = (s * (R @ sheet_mm.T)).T + t
    e = (out_px - pred) / s                      # mm of paper
    # Horizontal = along the page's rows in the output frame.
    ex = np.abs(e[:, 0])
    r = np.hypot(*e.T)
    return {"px_per_mm": round(s, 4),
            "rms_mm": round(float(np.sqrt(np.mean(r ** 2))), 3),
            "p95_mm": round(float(np.percentile(r, 95)), 3),
            "max_mm": round(float(r.max()), 3),
            "x_p95_mm": round(float(np.percentile(ex, 95)), 3),
            "x_p95_ms": round(float(np.percentile(ex, 95)) * 1000 / dig.MM_PER_S, 1)}


def main() -> int:
    reg = skp.load_registry()["photos"]
    rows = []
    for stem, e in sorted(reg.items()):
        t = skp.Target(e["target"])
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = DECODE_PX / max(photo.shape[:2])
        photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        guess = np.asarray(e["corners_px"], np.float64) * sc
        ct = corner_truth.find_true_corners(photo, t.sheet, guess, t.px_per_mm)
        quad = corner_truth.sheet_quad(t.sheet.shape[1], t.sheet.shape[0])
        H_true = cv2.getPerspectiveTransform(quad, ct.corners.astype(np.float32))
        # Lattice on the sheet, in its pixels and in mm.
        mm = t.px_per_mm
        gx = np.arange(EDGE_MM, t.width_mm - EDGE_MM + 1e-6, LATTICE_MM)
        gy = np.arange(EDGE_MM, t.height_mm - EDGE_MM + 1e-6, LATTICE_MM)
        lat_mm = np.array([(x, y) for y in gy for x in gx], np.float64)
        in_photo = apply_h(H_true, lat_mm * mm - 0.5)
        try:
            auto = page.find_sheet_corners(photo).corners
        except (RuntimeError, ValueError) as exc:
            auto = None
            print(f"{stem}: auto corners failed: {exc}")
        for name, corners in (("auto", auto), ("truth", ct.corners)):
            if corners is None:
                continue
            rect, _ = sheet.rectify(photo, corners)
            h, w = rect.shape[:2]
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
            P = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
            head = {"photo": stem, "target": t.name, "ecc": round(ct.ecc, 4)}
            rows.append({**head, "variant": name,
                         **distortion(lat_mm, apply_h(P, in_photo))})
            try:
                px_guess = segment.locate_sheet(rect).anchor_px_per_mm
                fit = gridfit.fit_grid(rect, px_guess)
            except (RuntimeError, ValueError) as exc:
                rows.append({**head, "variant": name + "+grid", "error": str(exc)})
                continue
            M = fit.H @ P
            rows.append({**head, "variant": name + "+grid",
                         **distortion(lat_mm, apply_h(M, in_photo)),
                         "fit_rms_mm": round(fit.rms_mm, 3),
                         "fit_rules": f"{fit.n_vertical}x{fit.n_horizontal}",
                         "fit_samples": fit.n_samples,
                         "fit_inliers": round(fit.inlier_fraction, 3)})
        for r in rows[-4:]:
            print(f"{r['photo']:22s} {r['variant']:11s} "
                  + (r.get("error") or f"rms {r['rms_mm']:.3f}  p95 {r['p95_mm']:.3f}  "
                     f"max {r['max_mm']:.3f} mm   x p95 {r['x_p95_ms']:.1f} ms"
                     + (f"   fit rms {r['fit_rms_mm']:.3f} mm, {r['fit_rules']} rules,"
                        f" inliers {r['fit_inliers']:.2f}" if "fit_rms_mm" in r else "")),
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_geometry.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"decode_px": DECODE_PX, "lattice_mm": LATTICE_MM, "edge_mm": EDGE_MM,
                   "gridfit": {k: getattr(gridfit, k) for k in (
                       "PITCH_MM", "BAND_MM", "BAND_STEP_MM", "EDGE_MARGIN_MM",
                       "MIN_REL_PROMINENCE", "MATCH_FRACTION", "HUBER_PX", "MIN_RULES")}},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
