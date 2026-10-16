#!/usr/bin/env python3
"""One workbook for the results that are not in the A/B/C statistics workbooks.

results/report_set_C/stats_extras.xlsx, a sheet per committed results file (copied
as is; nothing is recomputed):
  Set C runs              results/device/setC_runs.csv (one row per Set C and rescue export)
  Rescue before-after     results/device/setC_crumple_rescue_before_after.csv
  Rescue scores (leads)   results/device/sheet_scores_setC_crumple_rescue.csv + _setC_tilt_rescue.csv
  Orientation             results/kitphoto/orientation_sessions.csv (the shipped rule, 64/68)
  Orientation rule v1     results/kitphoto/orientation_sessions_rule_v1.csv (REJECTED, 59/68)
  Orientation sideways    results/kitphoto/orientation.csv (the quarter-turn test, 76/76)
  t0 guard replay         results/kitphoto/t0_guard_replay.csv (where the guard fires)
  t0 guard what-if b3     results/kitphoto/t0_guard_whatif_b3.csv (IN-SAMPLE)
  t0 guard what-if cl     results/kitphoto/t0_guard_whatif.csv (IN-SAMPLE)

    python make_extras_workbook.py          (from python/)
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results" / "report_set_C" / "stats_extras.xlsx"
SHEETS = [
    ("Set C runs", ["results/device/setC_runs.csv"],
     "One row per export of Set C (tilt, shadow, crumple) and of the rescue re-runs: photograph, record, tracer, "
     "verdict and warnings, ticks, time-zero source, t0 guard, grid-fit inliers, median scores. Facts per run; the "
     "statistics are in report_set_C/set_C_robustness/stats_set_C.xlsx."),
    ("Rescue before-after", ["results/device/setC_crumple_rescue_before_after.csv"],
     "The four failed crumple photographs: the original run (no guard) beside its re-run with the guarded "
     "time-zero fallback, same photograph and corners. IN-SAMPLE (the guard was designed on these failures)."),
    ("Rescue scores (leads)", ["results/device/sheet_scores_setC_crumple_rescue.csv",
                               "results/device/sheet_scores_setC_tilt_rescue.csv"],
     "Every lead of the rescue re-runs (Crumple ResQ, Tilt ResQ). Two Crumple ResQ exports are scored against "
     "their true record PTB-XL 4016 (labelled 40 by mistake): python/target_corrections.csv."),
    ("Orientation", ["results/kitphoto/orientation_sessions.csv"],
     "Automatic orientation on the 68 session photographs: the shipped rule (lead markers on the grid-fit page) "
     "against the old rule (remembered direction, both ways). Truth: python/orientation_truth.json."),
    ("Orientation rule v1", ["results/kitphoto/orientation_sessions_rule_v1.csv"],
     "REJECTED first rule ('more ticks wins' at <= 6 ticks): 59/68, five upright photographs turned over."),
    ("Orientation sideways", ["results/kitphoto/orientation.csv"],
     "The quarter-turn (sideways) test: every kit, in-app and clean image turned four ways, 76/76."),
    ("t0 guard replay", ["results/kitphoto/t0_guard_replay.csv"],
     "Where the guarded time-zero fallback fires: every phone run of Sets A, B, C, by the phone's own scales and "
     "on the desktop reference."),
    ("t0 guard what-if b3", ["results/kitphoto/t0_guard_whatif_b3.csv"],
     "IN-SAMPLE what-if, learned tracer: the firing runs under 9 conditions each, before and after the guard."),
    ("t0 guard what-if cl", ["results/kitphoto/t0_guard_whatif.csv"],
     "The same what-if with the classical tracer."),
]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    readme = pd.DataFrame([(n, "; ".join(f), w) for n, f, w in SHEETS], columns=["sheet", "source file(s)", "what it is"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as wr:
        readme.to_excel(wr, sheet_name="README", index=False)
        for name, files, _ in SHEETS:
            pd.concat([pd.read_csv(REPO / f) for f in files], ignore_index=True).to_excel(wr, sheet_name=name, index=False)
        for ws in wr.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(9, width + 2),
                                                                     110 if ws.title == "README" else 48)
    srcs = [f for _, fs, _ in SHEETS for f in fs]
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/make_extras_workbook.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", *srcs)),
        "sources": srcs,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
