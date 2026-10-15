#!/usr/bin/env python3
"""Set C Crumple for the thesis: per run, and per tracer against Set B (ideal).

Reads (never rewrites) results/device/sheet_scores_setC_crumple.csv and, for
the reference, the committed results/device/sheet_scores.csv restricted to the
Set B exports of python/test_set.csv. The lead is the unit (12 leads per run;
the rhythm strip is reported apart), as in make_report_figures.py; an empty
PCC counts as a failure (in n, not >= 0.9). The photograph of each run is from
data/kit_photos/SetC_Crumple/README.txt (matched by the recorded corners).

    python summarise_crumple.py          (from python/)

Writes results/device/setC_crumple_runs.csv and setC_crumple_summary.csv (+ .json each).
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
CRUMPLE = DEV / "sheet_scores_setC_crumple.csv"
ALL = DEV / "sheet_scores.csv"
MANIFEST = HERE / "test_set.csv"

#: pair_id -> photograph (data/kit_photos/SetC_Crumple/README.txt).
PHOTO = {"20261015_124711": "eacg_20261015_124607.jpg", "20261015_124807": "eacg_20261015_124720.jpg",
         "20261015_124910": "eacg_20261015_124821.jpg", "20261015_125142": "eacg_20261015_125056.jpg",
         "20261015_125332": "eacg_20261015_125242.jpg", "20261015_125428": "eacg_20261015_125342.jpg",
         "20261015_125638": "eacg_20261015_125545.jpg"}
MEASURES = ["pcc_asis", "pcc_aligned", "snr_db_asis", "shifted_snr_db_asis", "rmse_mv_asis", "gain_ratio", "r_ratio"]


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


def tracer(t: str) -> str:
    return "classical" if str(t).startswith("classical") else "learned"


def summarise(leads: pd.DataFrame, group: str) -> list[dict]:
    out = []
    for tr, g in leads.groupby(leads["tracer"].map(tracer)):
        row = {"group": group, "tracer": tr, "runs": g["export"].nunique(), "leads": len(g)}
        for m in MEASURES:
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
            row.update({f"{m}_median": round(q[1], 3), f"{m}_q1": round(q[0], 3), f"{m}_q3": round(q[2], 3),
                        f"{m}_n": len(v)})
        pcc = pd.to_numeric(g["pcc_asis"], errors="coerce")
        row["leads_pcc_ge_0.9"] = int((pcc >= 0.9).sum())
        row["fraction_pcc_ge_0.9"] = round(float((pcc >= 0.9).sum() / len(g)), 3)
        out.append(row)
    return out


def write(rows: list[dict], name: str, params: dict) -> None:
    p = DEV / name
    with p.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    p.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/summarise_crumple.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", "results/device/sheet_scores_setC_crumple.csv",
                              "results/device/sheet_scores.csv")),
        "params": params,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(p.with_suffix(".csv"))


def main() -> int:
    c = pd.read_csv(CRUMPLE)
    exports = REPO / "data" / "device_exports" / "Set C Crumple"
    runs = []
    for exp, g in c.groupby("export", sort=False):
        h = header(exports / exp)
        cells = g[~g["rhythm"].astype(bool)]
        rh = g[g["rhythm"].astype(bool)]
        geo = h.get("geometry", "")
        inl = re.search(r"inliers ([\d.]+)", geo)
        rms = re.search(r"fit rms ([\d.]+) mm", geo)
        pcc = pd.to_numeric(cells["pcc_asis"], errors="coerce")
        med = lambda m: round(float(pd.to_numeric(cells[m], errors="coerce").median()), 3)
        runs.append({"pair_id": h.get("pair_id"), "photo": PHOTO.get(h.get("pair_id"), ""),
                     "record": h.get("target", "").split("_")[0], "tracer": tracer(h.get("tracer")), "export": exp,
                     "photo_rotation": h.get("photo_rotation", ""), "orient_ms": h.get("orient_ms", ""),
                     "quality": h.get("quality"), "warnings": h.get("warnings"),
                     "ticks_found": g["ticks_found"].iloc[0], "t0_source": g["t0_source"].iloc[0],
                     "calibration": h.get("calibration"),
                     "grid_inliers": float(inl.group(1)) if inl else "", "grid_rms_mm": float(rms.group(1)) if rms else "",
                     **{f"{m}_median": med(m) for m in MEASURES},
                     "leads_pcc_ge_0.9": int((pcc >= 0.9).sum()),
                     "rhythm_pcc_asis": rh["pcc_asis"].iloc[0] if len(rh) else ""})
    runs.sort(key=lambda r: (r["pair_id"], r["tracer"] != "learned"))
    write(runs, "setC_crumple_runs", {"scores": "results/device/sheet_scores_setC_crumple.csv",
                                      "photos": "data/kit_photos/SetC_Crumple/README.txt"})

    m = pd.read_csv(MANIFEST)
    setb = set(m.loc[m["set"] == "ideal", "export"])
    a = pd.read_csv(ALL)
    b = a[a["export"].isin(setb) & ~a["rhythm"].astype(bool)]
    leads = c[~c["rhythm"].astype(bool)]
    rows = summarise(leads, "C crumple (PTB-XL 40, 4016)")
    rows += summarise(b[b["ecg_id"].isin([40, 4016])], "B ideal, same records (PTB-XL 40, 4016)")
    rows += summarise(b, "B ideal, all 12 records")
    write(rows, "setC_crumple_summary", {"unit": "lead (12 per run; rhythm strip excluded)",
                                         "crumple": "results/device/sheet_scores_setC_crumple.csv",
                                         "set_B": "results/device/sheet_scores.csv x python/test_set.csv set=ideal"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
