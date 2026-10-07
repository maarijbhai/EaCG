#!/usr/bin/env python3
"""The time and geometry references behind the FR8 sheet warnings, per photo.

For every registered photograph, at the phone's 3,200 px decode with
automatic corners and the grid fit (the phone's path), the quantities
`ecgproto.sheet_quality` judges -- none of which depends on the tracer:

  grid vs tick scale       ticks px/mm / grid px/mm - 1
  worst tick deviation     max over ticks of |x - median of the same tick
                           over the rows|, mm (TICK_OUTLIER)
  ticks found, pulses found (of 4), time-zero source, grid fit used

and the reference-level warnings they raise (lead-level checks need the
learned tracer's leads; those are in ml/eval_phone_path.py's `warnings`).
This is the evidence the thresholds in sheet_quality.py were set from.

    python run_sheet_warnings.py

Writes results/kitphoto/warnings.csv + .json.
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
from ecgproto import sheet, sheet_quality as Q  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "kitphoto" / "warnings.csv"
DECODE_PX = 3200


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def worst_tick_mm(positions, px_per_mm: float) -> float:
    pos = {(int(r), int(k)): float(x) for r, k, x in positions}
    worst = 0.0
    for (r, k), x in pos.items():
        same = [pos[(rr, k)] for rr in range(3) if (rr, k) in pos]
        if len(same) >= 2:
            worst = max(worst, abs(x - float(np.median(same))) / px_per_mm)
    return worst


def main() -> int:
    rows = []
    for stem, e in sorted(skp.load_registry()["photos"].items()):
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = DECODE_PX / max(photo.shape[:2])
        if sc < 1:
            photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        # The references do not depend on the tracer; the classical one is
        # used because it needs no TensorFlow.
        res = sheet.digitise_sheet(photo, sheet.classical_row_tracer(), None, "grid")
        used = res.geometry == "grid" and res.grid_fit is not None
        g = res.grid_fit.px_per_mm if used else float("nan")
        ws = [w for w in Q.assess_result(res, grid_requested=True)
              if not w.code.startswith("LEAD_")]
        row = {"photo": stem, "target": e["target"], "geometry": res.geometry,
               "grid_px_per_mm": round(g, 4), "calibration": res.calibration,
               "tick_px_per_mm": round(res.px_per_mm, 4),
               "scale_diff_pct": round(100 * (res.px_per_mm / g - 1), 3) if used else "",
               "ticks_found": res.ticks_found,
               "worst_tick_dev_mm": round(worst_tick_mm(res.ticks.positions, res.px_per_mm), 3),
               "pulses_found": int(np.isfinite(res.location.t0_per_row).sum()),
               "t0_source": res.t0_source, "quality_refs": Q.verdict(ws),
               "warnings_refs": "; ".join(
                   f"{w.code}/{w.severity}" + (f"/{','.join(w.leads)}" if w.leads else "")
                   for w in ws)}
        rows.append(row)
        print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_sheet_warnings.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"decode_px": DECODE_PX, "corners": "auto", "geometry": "grid",
                   "scale_disagree_fraction": Q.SCALE_DISAGREE_FRACTION,
                   "tick_mate_mm": Q.TICK_MATE_MM, "min_ticks": Q.MIN_TICKS,
                   "min_lead_coverage": Q.MIN_LEAD_COVERAGE},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2))
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
