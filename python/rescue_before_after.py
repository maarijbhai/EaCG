#!/usr/bin/env python3
"""Set C Crumple before / after the guarded time-zero fallback (progress_log §42).

Each failed Crumple run (results/device/sheet_scores_setC_crumple.csv, the app
before the guard) beside its re-run from the gallery with the guard
(results/device/sheet_scores_setC_crumple_rescue.csv): same photograph, same
corners (data/device_exports/Crumple ResQ/README.txt). Reads only; the lead is
the unit (12 per run, the rhythm strip apart). The tick-vs-grid disagreement is
the phone's own (header calibration and geometry lines).

    python rescue_before_after.py          (from python/)

Writes results/device/setC_crumple_rescue_before_after.csv + .json.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEV = REPO / "results" / "device"
EXPORTS = REPO / "data" / "device_exports"
BEFORE = DEV / "sheet_scores_setC_crumple.csv"
AFTER = DEV / "sheet_scores_setC_crumple_rescue.csv"
OUT = DEV / "setC_crumple_rescue_before_after"

#: (photograph, original run's pair_id, [(rescue pair_id, note)]) -- Crumple ResQ/README.txt.
PAIRS = [("eacg_20261015_124720.jpg", "20261015_124807", [("20261015_153336", "guard used")]),
         ("eacg_20261015_124821.jpg", "20261015_124910", [("20261015_154119", "guard used")]),
         ("eacg_20261015_125056.jpg", "20261015_125142", [("20261015_154211", "guard used")]),
         ("eacg_20261015_125342.jpg", "20261015_125428",
          [("20261015_155451", "guard used"),
           ("20261015_154533", "second run: TR corner by hand; guard did not fire; scored against "
                               "PTB-XL 4016 (labelled 40, python/target_corrections.csv)")])]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def header(path: Path) -> dict:
    h = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        k, _, v = line[1:].partition(":")
        h[k.strip()] = v.strip()
    return h


def find(name: str) -> Path:
    hits = [p for p in EXPORTS.rglob(name)]
    if len(hits) != 1:
        raise RuntimeError(f"{name}: {len(hits)} files")
    return hits[0]


def runs_of(scores: pd.DataFrame, pair_id: str) -> dict:
    """tracer -> (export, lead rows) of the run(s) with this pair_id."""
    out = {}
    for exp, g in scores.groupby("export"):
        h = header(find(exp))
        if h.get("pair_id") == pair_id:
            out["classical" if str(h.get("tracer")).startswith("classical") else "learned"] = (exp, h, g)
    return out


def facts(exp: str, h: dict, g: pd.DataFrame) -> dict:
    cells = g[~g["rhythm"].astype(bool)]
    num = lambda k: pd.to_numeric(cells[k], errors="coerce")
    gp = re.search(r"grid \(.*?([\d.]+) px/mm", h.get("geometry", ""))
    tp = re.match(r"ticks ([\d.]+) px/mm", h.get("calibration", ""))
    d = 100 * (float(tp.group(1)) / float(gp.group(1)) - 1) if tp and gp else float("nan")
    rh = g[g["rhythm"].astype(bool)]
    return {"export": exp, "target": h.get("target", ""), "scored_against": g["target"].iloc[0],
            "ticks_found": int(g["ticks_found"].iloc[0]), "t0_source": g["t0_source"].iloc[0],
            "tick_vs_grid_pct": round(d, 1) if np.isfinite(d) else "",
            "quality": h.get("quality"), "t0_guard": h.get("t0_guard", ""),
            "median_pcc": round(float(num("pcc_asis").median()), 3),
            "median_pcc_aligned": round(float(num("pcc_aligned").median()), 3),
            "median_snr_db": round(float(num("snr_db_asis").median()), 2),
            "median_shifted_snr_db": round(float(num("shifted_snr_db_asis").median()), 2),
            "median_r_ratio": round(float(num("r_ratio").median()), 3),
            "leads_pcc_ge_0.9": int((num("pcc_asis") >= 0.9).sum()),
            "rhythm_pcc": rh["pcc_asis"].iloc[0] if len(rh) else ""}


def main() -> int:
    b, a = pd.read_csv(BEFORE), pd.read_csv(AFTER)
    rows = []
    for photo, orig, rescues in PAIRS:
        before = runs_of(b, orig)
        for rid, note in rescues:
            after = runs_of(a, rid)
            for tr in ("learned", "classical"):
                if tr not in before or tr not in after:
                    continue
                fb, fa = facts(*before[tr]), facts(*after[tr])
                rows.append({"photo": photo, "tracer": tr, "before_pair_id": orig, "after_pair_id": rid, "note": note,
                             **{f"before_{k}": v for k, v in fb.items()}, **{f"after_{k}": v for k, v in fa.items()},
                             "delta_median_pcc": round(fa["median_pcc"] - fb["median_pcc"], 3)})
    with OUT.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/rescue_before_after.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", str(BEFORE.relative_to(REPO)),
                              str(AFTER.relative_to(REPO)))),
        "params": {"before": BEFORE.relative_to(REPO).as_posix(), "after": AFTER.relative_to(REPO).as_posix(),
                   "pairs": "data/device_exports/Crumple ResQ/README.txt",
                   "note": "IN-SAMPLE: the guard was designed on these failures (progress_log §41)"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
