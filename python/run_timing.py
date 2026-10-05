#!/usr/bin/env python3
"""Where does the whole-sheet pipeline's timing error come from?

Sliding a digitised lead against its truth (timemap.best_shift) says THAT a
lead is early or late, not WHY: the time map (pulse time zero, separator
ticks, page geometry) and the tracer both move a waveform in time. This
separates them, for the phone's path (3,200 px, automatic corners, grid
fit), with no tracer involved:

  1. The registration (corner_truth: the photo matched onto the printed
     PNG) maps every plotted sample of every lead -- whose true time we know
     -- into the photograph, and the pipeline's own homography (corners ->
     rectify -> grid fit) maps it onto the page the pipeline works on.
  2. The pipeline's time map (layout.time_at, from the pulse time zero and
     the tick knots) is read at that column. time_at(x) - true time is the
     TIME-MAP ERROR at that sample, in ms.

Per lead: the median time-map error over its samples (what the map alone
does to that lead), and its slope across the cell. Per photo: time zero
(pulse edge) and every tick against where the true boundary lands.

If a lead's time-map error matches its measured shift, the fault is the
map; if the map is right and the lead still shifts, the fault is in the
trace.

    python run_timing.py                       # time map only (no tracer)
    bash ml/wsl_run.sh python python/run_timing.py --b3   # + each lead's measured
                                               # shift with b3, to compare

Writes results/kitphoto/timing.csv + .json.
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
import score_12lead as S12  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import corner_truth, segment, sheet, timemap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "kitphoto" / ("timing_b3" if "--b3" in sys.argv else "timing")
DECODE_PX = 3200


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def no_trace(img, layout, row):
    return np.arange(10.0), np.full(10, np.nan)


def _shift(sig, truth) -> dict:
    """The lead's measured shift (b3), as in run_timemap: + = late."""
    sh = timemap.best_shift(truth, sig, S12.FS)
    return {"b3_shift_ms": round(sh.shift_ms, 1) if np.isfinite(sh.shift_ms) else "",
            "b3_shift_status": sh.status}


def photo_to_page(res, rect_shape) -> np.ndarray:
    """The pipeline's photo -> page homography (corners, then the grid fit)."""
    h, w = rect_shape[:2]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    P = cv2.getPerspectiveTransform(np.asarray(res.corners, np.float32), dst).astype(np.float64)
    return res.grid_fit.H @ P if res.geometry == "grid" else P


def b3_tracer():
    sys.path.insert(0, str(REPO / "ml"))
    from export_tflite import TfliteTracer
    tfl = TfliteTracer(REPO / "android/app/src/main/assets/ml/b3_512x1024_fp16.tflite")
    h = int(tfl.run.get_input_details()[tfl.in_name]["shape"][1])
    return sheet.model_row_tracer(tfl, h)


def main() -> int:
    use_b3 = "--b3" in sys.argv
    tracer = b3_tracer() if use_b3 else no_trace
    rows = []
    for stem, e in sorted(skp.load_registry()["photos"].items()):
        t = skp.Target(e["target"])
        truth = S12.SheetTruth(t)
        full = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = DECODE_PX / max(full.shape[:2])
        photo = cv2.resize(full, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        # Truth: printed sheet -> photo (3,200 px), by registration.
        ct = corner_truth.find_true_corners(photo, t.sheet,
                                            np.asarray(e["corners_px"], np.float64) * sc,
                                            t.px_per_mm)
        quad = corner_truth.sheet_quad(t.sheet.shape[1], t.sheet.shape[0])
        H_true = cv2.getPerspectiveTransform(quad, ct.corners.astype(np.float32)).astype(np.float64)
        res = sheet.digitise_sheet(photo, tracer, None, "grid")
        rect, _ = sheet.rectify(photo, res.corners)
        M = photo_to_page(res, rect.shape) @ H_true          # sheet -> page
        lay = res.layout
        head = {"photo": stem, "target": t.name, "ecc": round(ct.ecc, 4),
                "geometry": res.geometry, "ticks_found": res.ticks_found,
                "t0_source": res.t0_source, "pulses_found": int(np.isfinite(res.location.t0_per_row).sum())}
        for cell in lay.cells:
            rhythm = cell.row == 3
            tr = truth.leads[(cell.name, rhythm)]
            xy = np.stack([tr["cols"], tr["rows"]], 1).astype(np.float64)
            pg = cv2.perspectiveTransform(xy[None], M)[0]
            t_true = (tr["start"] + np.arange(len(xy))) / S12.FS
            err_ms = 1000 * (lay.time_at(cell.row, pg[:, 0]) - t_true)
            # Slope across the cell: a stretch, ms per second of recording.
            slope = np.polyfit(t_true - t_true.mean(), err_ms, 1)[0]
            rows.append({**head, "lead": cell.name, "rhythm": rhythm, "row": cell.row,
                         "col": cell.col,
                         "map_err_ms": round(float(np.median(err_ms)), 2),
                         "map_err_start_ms": round(float(np.median(err_ms[:50])), 2),
                         "map_err_end_ms": round(float(np.median(err_ms[-50:])), 2),
                         "map_slope_ms_per_s": round(float(slope), 3),
                         "page_y_drift_px": round(float(np.ptp(pg[:, 1] - pg[:, 1].mean())), 1),
                         **(_shift(res.leads[(cell.name, rhythm)], tr["mv"]) if use_b3 else {})})
        # Time zero and each tick against the true boundary position.
        for r in range(3):
            names = segment.LAYOUT_3X4[r]
            for k in range(4):
                tr = truth.leads[(names[k], False)]
                p0 = cv2.perspectiveTransform(
                    np.array([[[tr["cols"][0], tr["rows"][0]]]], np.float64), M)[0, 0]
                if k == 0:
                    det = res.location.t0_per_row[r]
                    kind = "pulse_t0"
                else:
                    det = {(rr, kk): x for rr, kk, x in res.ticks.positions}.get((r, k), np.nan)
                    kind = f"tick_{k}"
                pps = lay.px_per_s
                rows.append({**head, "lead": f"{kind}_row{r}", "row": r, "col": k,
                             "true_x": round(float(p0[0]), 2),
                             "detected_x": "" if not np.isfinite(det) else round(float(det), 2),
                             "detected_err_ms": "" if not np.isfinite(det)
                             else round(float(1000 * (det - p0[0]) / pps), 2),
                             "t0_used_err_ms": round(float(1000 * (lay.t0_x - p0[0]) / pps), 2)
                             if k == 0 else ""})
        cells = [x for x in rows if x["photo"] == stem and "map_err_ms" in x and not x["rhythm"]]
        by_col = [np.median([x["map_err_ms"] for x in cells if x["col"] == c]) for c in range(4)]
        t0e = [x["detected_err_ms"] for x in rows if x["photo"] == stem and str(x["lead"]).startswith("pulse")]
        print(f"{stem}: ECC {ct.ecc:.3f}  map error by column (ms) "
              + " ".join(f"{v:+.1f}" for v in by_col)
              + f"  | pulse t0 err (ms) {t0e}",
              flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for x in rows:
        keys += [k for k in x if k not in keys]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_timing.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"decode_px": DECODE_PX, "corners": "auto", "geometry": "grid", "b3": use_b3,
                   "truth": "corner_truth registration of the photo to the printed PNG"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
