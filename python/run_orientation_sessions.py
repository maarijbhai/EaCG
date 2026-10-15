#!/usr/bin/env python3
"""The automatic orientation, measured on every session photograph (progress_log §39).

Each photograph in data/kit_photos/{setB, setC_tilt, SetC_Shadow} is decoded
as the phone decodes it (EXIF applied, 3,200 px long side) and turned by
`orientation.quarter_turns` -- the app's rule: the corner outline decides
sideways, the pipeline's lead markers on the grid-fit page decide up from
down. The truth is `python/orientation_truth.json` (how each was
established is written there).

For comparison, the rule it replaces (7ffc92f): sideways -> a quarter turn
in the direction the user last confirmed, never a half turn. It depends on
that memory, so both directions are scored ("old_cw", "old_ccw").

    python run_orientation_sessions.py            (from python/)

Writes results/kitphoto/orientation_sessions.csv + .json.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from ecgproto import orientation as O, page  # noqa: E402
from run_orientation import load  # noqa: E402

FOLDERS = ("setB", "setC_tilt", "SetC_Shadow")
TRUTH = HERE / "orientation_truth.json"
OUT = REPO / "results" / "kitphoto" / "orientation_sessions"


def corners(img: np.ndarray) -> np.ndarray:
    return page.find_sheet_corners(img).corners


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["photos"]
    rows = []
    for fo in FOLDERS:
        for path in sorted((REPO / "data" / "kit_photos" / fo).glob("*.jpg")):
            bgr = np.ascontiguousarray(load(path)[:, :, ::-1])     # the pipeline is cv2 (BGR)
            want = truth[path.name]["upright_cw"]
            t0 = time.time()
            q, info = O.quarter_turns(bgr, corners)
            row = {"folder": fo, "photo": path.name, "truth_cw": want, "new_cw": q, "new_ok": q == want,
                   "aspect": info.get("aspect", ""), "geometry": info.get("geometry", ""),
                   "up_pulses": info["up"].pulses if "up" in info else "",
                   "up_ticks": info["up"].ticks if "up" in info else "",
                   "turned_pulses": info["turned"].pulses if "turned" in info else "",
                   "turned_ticks": info["turned"].ticks if "turned" in info else "",
                   "error": info.get("error", "")}
            side = info.get("aspect", 9) < O.SIDEWAYS_BELOW if "aspect" in info else False
            row["old_cw"] = 1 if side else 0
            row["old_ccw"] = 3 if side else 0
            row["old_cw_ok"] = row["old_cw"] == want
            row["old_ccw_ok"] = row["old_ccw"] == want
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print({k: row[k] for k in ("photo", "truth_cw", "new_cw", "old_cw", "old_ccw", "up_pulses",
                                       "up_ticks", "turned_pulses", "turned_ticks", "error")}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    n = len(rows)
    summary = {"photos": n,
               "new_correct": sum(r["new_ok"] for r in rows),
               "old_cw_correct": sum(r["old_cw_ok"] for r in rows),
               "old_ccw_correct": sum(r["old_ccw_ok"] for r in rows),
               "new_wrong": [f"{r['photo']} (truth {r['truth_cw']}, got {r['new_cw']}{', ' + r['error'] if r['error'] else ''})"
                             for r in rows if not r["new_ok"]],
               "truth_by_cw": {str(k): sum(r["truth_cw"] == k for r in rows) for k in range(4)}}
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/run_orientation_sessions.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python/ecgproto", "python/run_orientation_sessions.py",
                              "python/orientation_truth.json")),
        "params": {"decode_px": 3200, "folders": list(FOLDERS), "sideways_below": O.SIDEWAYS_BELOW,
                   "max_ticks_turned": O.MAX_TICKS_TURNED, "truth": "python/orientation_truth.json"},
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
