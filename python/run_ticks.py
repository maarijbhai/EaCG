#!/usr/bin/env python3
"""Why does the separator-tick search find fewer than 9 ticks on some sheets?

A 3x4 sheet has three rows of four leads; a short thick bar marks each lead
change at 2.5, 5.0 and 7.5 s, so 3 rows x 3 boundaries = 9 ticks (the rhythm
strip has none). For every photograph, both geometries, and every one of the
nine, this reports what `segment.tick_scale` saw in that tick's search
window: the best column's score (the tallest ink run crossing the row's
centre, as the min over a tick-wide window), in mm, against the 4 mm a tick
must reach; where that column is relative to the predicted position; and,
for a missing tick, which rule refused it. The search is mirrored here
step for step and checked against tick_scale's own output on every sheet.

Also, per tick found: the origin it implies at the grid scale (x - k*2.5 s
* px/s), against the pulse edge -- the quantity timemap.grid_tickzero uses.

    python run_ticks.py

Writes results/kitphoto/ticks.csv + .json.
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
from ecgproto import digitise as dig, segment as S, sheet  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "kitphoto" / "ticks"
DECODE_PX = 3200


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def no_trace(img, layout, row):
    return np.arange(10.0), np.full(10, np.nan)


def mirror(image, sl, open_kernel=2):
    """segment.tick_scale, exposing every window. Returns (rows, positions)."""
    m = dig.binarise_trace(image, open_kernel=open_kernel)
    approx = sl.search_px_per_mm
    wt = max(2, int(round(S.TICK_WIDTH_MM * approx)))
    band = int(round(S.TICK_BASE_BAND_MM * approx))
    runs = {r: S._baseline_runs(m, sl.rows[r], band) for r in range(3)}
    score = {r: np.array([runs[r][x:x + wt].min() for x in range(len(runs[r]) - wt)])
             for r in range(3)}
    t0 = sl.t0_x
    cell_px = S.CELL_S * dig.MM_PER_S
    first = []
    pass1 = {}
    for r in range(3):
        xe = t0 + cell_px * approx
        lo = max(0, int(t0 + (1 - S.TICK_FIRST_WINDOW) * (xe - t0)))
        hi = min(len(score[r]), int(t0 + (1 + S.TICK_FIRST_WINDOW) * (xe - t0)))
        j = lo + int(np.argmax(score[r][lo:hi]))
        pass1[r] = score[r][j] / approx
        if score[r][j] >= S.TICK_MIN_HEIGHT_MM * approx:
            first.append((j + (wt - 1) / 2.0 - t0) / cell_px)
    search = approx
    if first:
        approx = float(np.median(first))
    out, pos = [], []
    for r in range(3):
        sc, run = score[r], runs[r]
        for k in (1, 2, 3):
            xe = t0 + k * cell_px * approx
            half = S.TICK_WINDOW * k * cell_px * approx
            lo, hi = max(0, int(xe - half)), min(len(sc), int(xe + half))
            j = lo + int(np.argmax(sc[lo:hi]))
            h_mm = sc[j] / approx
            found = sc[j] >= S.TICK_MIN_HEIGHT_MM * approx
            xc = j + (wt - 1) / 2.0
            if found:
                a0 = max(0, j - 3 * wt)
                cols = np.flatnonzero(run[a0:j + 4 * wt] >= 0.5 * sc[j]) + a0
                if cols.size:
                    segs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
                    xc = float(min(segs, key=lambda s: abs(s.mean() - xc)).mean())
                pos.append((r, k, round(xc, 1)))
            # Tallest crossing run anywhere near, to tell "faint" from "elsewhere".
            wide = sc[max(0, int(xe - 3 * half)):min(len(sc), int(xe + 3 * half))]
            out.append({"row": r, "k": k, "found": bool(found),
                        "best_score_mm": round(float(h_mm), 2),
                        "best_offset_mm": round(float((xc - xe) / approx), 2),
                        "window_half_mm": round(float(half / approx), 2),
                        "wider_best_score_mm": round(float(wide.max() / approx), 2) if wide.size else "",
                        "pass1_score_mm": round(float(pass1[r]), 2) if k == 1 else "",
                        "search_px_per_mm": round(float(search), 4),
                        "refined_px_per_mm": round(float(approx), 4),
                        "reason": "" if found else (
                            f"best column only {h_mm:.1f} mm tall (< {S.TICK_MIN_HEIGHT_MM:g} mm)")})
    return out, tuple(pos)


def main() -> int:
    rows = []
    for stem, e in sorted(skp.load_registry()["photos"].items()):
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = DECODE_PX / max(photo.shape[:2])
        photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        for geo in ("corners", "grid"):
            res = sheet.digitise_sheet(photo, no_trace, None, geo)
            ticks, positions = mirror(res.page, res.location)
            assert positions == res.ticks.positions, (stem, geo, positions, res.ticks.positions)
            pps = (res.grid_fit.px_per_mm * dig.MM_PER_S if res.grid_fit is not None
                   and res.geometry == "grid" else float("nan"))
            pmap = {(r, k): x for r, k, x in positions}
            for t in ticks:
                x = pmap.get((t["row"], t["k"]))
                origin = x - t["k"] * S.CELL_S * pps if x is not None and np.isfinite(pps) else None
                rows.append({"photo": stem, "geometry": res.geometry,
                             "t0_source": res.location.t0_source,
                             "pulse_t0_x": round(res.location.t0_x, 1),
                             "ticks_found": res.ticks.n_found, **t,
                             "x": "" if x is None else x,
                             "origin_minus_pulse_px": "" if origin is None
                             else round(origin - res.location.t0_x, 2)})
            n = res.ticks.n_found
            miss = [f"r{t['row']}k{t['k']}:{t['best_score_mm']}mm" for t in ticks if not t["found"]]
            print(f"{stem} {geo:7s} {n}/9 ticks  missing {miss or '-'}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_ticks.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"decode_px": DECODE_PX, "corners": "auto", "open_kernel": 2,
                   "tick_min_height_mm": S.TICK_MIN_HEIGHT_MM,
                   "tick_width_mm": S.TICK_WIDTH_MM, "tick_window": S.TICK_WINDOW,
                   "tick_first_window": S.TICK_FIRST_WINDOW},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
