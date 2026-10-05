#!/usr/bin/env python3
"""Which time marker is misplaced? The traces as the instrument.

Each 3x4 row's time map is pinned at the pulse edge (0 s) and the separator
ticks (2.5 / 5.0 / 7.5 s) and extended past the last tick. If marker k is
read e_k ms LATE, a feature at true time t inside cell k is labelled early
by the interpolated error, so the lead's measured shift there is

    shift(t) = -( e_k + (e_{k+1} - e_k) * (t - t_k) / 2.5 s )

with e_4 = 2 e_3 - e_2 (the extrapolated end). Measuring the shift of each
HALF cell (b3 on the phone's path, timemap.best_shift) gives 8 equations per
row for the 4 marker errors; a least-squares fit says which marker is off.
Unlike run_timing.py this needs no registration -- only the traces and the
PTB-XL truth, which are exact.

    bash ml/wsl_run.sh python python/run_halfshift.py           # the photographs
    bash ml/wsl_run.sh python python/run_halfshift.py --clean   # the printed PNGs,
                                         # no camera: what the drawing itself does

Writes results/kitphoto/halfshift.csv (per half cell) and
halfshift_markers.csv (fitted marker errors per photo and row), + .json.
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
from ecgproto import segment, sheet, timemap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "kitphoto"
HALF = int(segment.CELL_S * S12.FS / 2)


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def b3_tracer():
    sys.path.insert(0, str(REPO / "ml"))
    from export_tflite import TfliteTracer
    tfl = TfliteTracer(REPO / "android/app/src/main/assets/ml/b3_512x1024_fp16.tflite")
    h = int(tfl.run.get_input_details()[tfl.in_name]["shape"][1])
    return sheet.model_row_tracer(tfl, h)


def design(col: int, half: int) -> np.ndarray:
    """Row of the linear model: shift = -A @ e, e = (e0, e1, e2, e3)."""
    f = 0.25 if half == 0 else 0.75                   # half-cell centre, in cells
    w = np.zeros(5)
    w[col] += 1 - f
    w[col + 1] += f
    a = w[:4].copy()                                   # fold e4 = 2 e3 - e2
    a[3] += 2 * w[4]
    a[2] -= w[4]
    return -a


def write(rows, name, params):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    p = OUT / f"{name}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    p.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_halfshift.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": params,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    return p


def main() -> int:
    tracer = b3_tracer()
    halves, markers = [], []
    clean = "--clean" in sys.argv
    if clean:
        names = sorted({e["target"] for e in skp.load_registry()["photos"].values()})
        jobs = [(f"clean:{n}", n, None) for n in names]
    else:
        jobs = [(stem, e["target"], e) for stem, e in sorted(skp.load_registry()["photos"].items())]
    for stem, target, e in jobs:
        t = skp.Target(target)
        truth = S12.SheetTruth(t)
        if clean:
            # The printed image itself is the page: its own corners, no warp.
            h, w = t.sheet.shape[:2]
            res = sheet.digitise_sheet(t.sheet, tracer,
                                       np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float),
                                       "corners")
        else:
            photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
            sc = 3200 / max(photo.shape[:2])
            photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            res = sheet.digitise_sheet(photo, tracer, None, "grid")
        for r in range(3):
            A, y = [], []
            for k, name in enumerate(segment.LAYOUT_3X4[r]):
                sig = res.leads[(name, False)]
                tr = truth.leads[(name, False)]["mv"]
                for h in (0, 1):
                    sh = timemap.best_shift(tr[h * HALF:(h + 1) * HALF], sig[h * HALF:(h + 1) * HALF],
                                            S12.FS, max_shift_s=0.1)
                    halves.append({"photo": stem, "row": r, "col": k, "lead": name, "half": h,
                                   "shift_ms": round(sh.shift_ms, 1) if np.isfinite(sh.shift_ms) else "",
                                   "pcc": round(sh.pcc, 4) if np.isfinite(sh.pcc) else "",
                                   "status": sh.status})
                    if sh.status == "ok":
                        A.append(design(k, h))
                        y.append(sh.shift_ms)
            row = {"photo": stem, "row": r, "n_halves": len(y)}
            if len(y) >= 6:
                e_hat, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)
                res_ms = np.asarray(y) - np.asarray(A) @ e_hat
                row.update({f"e{k}_ms": round(float(v), 2) for k, v in enumerate(e_hat)})
                row["fit_rms_ms"] = round(float(np.sqrt(np.mean(res_ms ** 2))), 2)
                row["shift_rms_ms"] = round(float(np.sqrt(np.mean(np.square(y)))), 2)
            markers.append(row)
            print(stem, r, {k: v for k, v in row.items() if k.startswith(("e", "fit", "shift"))}, flush=True)
    params = {"decode_px": 3200, "corners": "auto", "geometry": "grid", "tracer": "b3 FP16",
              "max_shift_s": 0.1, "half_cell_s": segment.CELL_S / 2}
    suffix = "_clean" if clean else ""
    params["source"] = "printed PNGs" if clean else "photographs"
    print(write(halves, "halfshift" + suffix, params))
    print(write(markers, "halfshift_markers" + suffix, params))
    return 0


if __name__ == "__main__":
    import os
    code = main()
    sys.stdout.flush()
    os._exit(code)
