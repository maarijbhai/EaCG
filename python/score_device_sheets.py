#!/usr/bin/env python3
"""Score the phone's whole-sheet exports against the truth, lead by lead.

Every `eacg_sheet_*.csv` in data/device_exports/ (the app's 12-lead export,
from the Scan tab or a record in the Records tab) whose header names a
printed test sheet is cut back into its 13 leads and scored with the same
function as every desktop result (score_12lead.score_lead: PCC, SNR,
shifted SNR, aligned PCC, R-peak ratio), plus the per-lead best shift
(timemap.best_shift). The header's run facts -- tracer, corners,
calibration, ticks found, time-zero source, timings -- are carried into
every row, so a bad run can be traced to the stage that failed.

    python score_device_sheets.py

Writes results/device/sheet_scores.csv + .json.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_12lead as S12  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import segment, timemap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXPORTS = REPO / "data" / "device_exports"
OUT = REPO / "results" / "device" / "sheet_scores"
CELL = int(segment.CELL_S * S12.FS)


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def read_export(path: Path) -> tuple[dict, dict]:
    """(header facts, column -> samples) of one export."""
    meta, rows, names = {}, [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].strip().partition(":")
            meta[k.strip()] = v.strip()
        elif names is None:
            names = line.split(",")
        elif line:
            rows.append([float(v) if v else np.nan for v in line.split(",")])
    a = np.array(rows, np.float64)
    return meta, {n: a[:, i] for i, n in enumerate(names)}


def main() -> int:
    out = []
    for path in sorted(EXPORTS.glob("eacg_sheet_*.csv")):
        meta, cols = read_export(path)
        target = meta.get("target", "none")
        if target == "none":
            continue
        t = skp.Target(target)
        truth = S12.SheetTruth(t)
        head = {"export": path.name, "local_time": meta.get("local_time"),
                "target": target, "ecg_id": t.ecg_id, "tracer": meta.get("tracer"),
                "corners": meta.get("corners"), "rect": meta.get("rect"),
                "calibration": meta.get("calibration"), "timings_ms": meta.get("timings_ms"),
                # native/java MB per stage (StageMemory.kt); absent before 2026-10-07
                "memory_peak_mb": meta.get("memory_peak_mb"),
                "pss_end_mb": meta.get("pss_end_mb")}
        for r, names in enumerate(tuple(segment.LAYOUT_3X4) + (("II",),)):
            for k, name in enumerate(names):
                rhythm = r == 3
                sig = cols["II_rhythm" if rhythm else name]
                if not rhythm:
                    sig = sig[k * CELL:(k + 1) * CELL]
                tr = truth.leads[(name, rhythm)]["mv"]
                sc = S12.score_lead(sig, tr, segment.RECORDING_S if rhythm else segment.CELL_S)
                sh = timemap.best_shift(tr, sig, S12.FS)
                out.append({**head, "lead": name, "rhythm": rhythm, "row": r, "col": k, **sc,
                            "shift_ms": round(sh.shift_ms, 1) if np.isfinite(sh.shift_ms) else "",
                            "shift_status": sh.status})
        cells = [x for x in out if x["export"] == path.name and not x["rhythm"]]
        med = lambda key: np.median([float(x[key]) for x in cells if x.get(key, "") != ""])
        print(f"{path.name}: {meta.get('calibration')}\n    median lead PCC {med('pcc_asis'):.3f}"
              f"  aligned {med('pcc_aligned'):.3f}  SNR {med('snr_db_asis'):.1f}"
              f" / shifted {med('shifted_snr_db_asis'):.1f} dB  R {med('r_ratio'):.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for x in out:
        keys += [k for k in x if k not in keys]
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/score_device_sheets.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python")),
        "params": {"exports": "data/device_exports/eacg_sheet_*.csv", "fs": S12.FS,
                   "max_shift_s": timemap.MAX_SHIFT_S, "min_pcc": timemap.MIN_PCC},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }, indent=2), encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
