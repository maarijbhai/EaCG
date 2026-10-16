#!/usr/bin/env python3
"""Inputs for running Set C through the SAME report scripts as Sets A and B.

make_report_figures.py and make_stats_workbook.py read one scores file and one
manifest. This writes, without changing either original:

  results/device/sheet_scores_all_sets.csv   the committed sheet_scores.csv rows (Sets A, B)
                                             + sheet_scores_setC_tilt / _shadow / _crumple
                                             (each file's rows unchanged, concatenated)
  python/test_set_all.csv                    python/test_set.csv (unchanged rows) + every
                                             Set C export as set "robustness", condition
                                             tilt / shadow / crumple, photo_id = its pair

The rescue re-runs (Crumple ResQ, Tilt ResQ) are left out: the report pools every
robustness condition into one table, and the rescues are in-sample (the t0 guard
was designed on those failures); they have their own before/after table.
Set C rows get no photo_path (the workbook's photo-quality view is Set B's and
assumes both tracers per photograph; three Set C runs are learned-only).

    python make_setC_inputs.py          (from python/)
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
DEV = REPO / "results" / "device"
EXPORTS = REPO / "data" / "device_exports"
SETC = {"tilt": ("sheet_scores_setC_tilt.csv", "Set C Tilt"),
        "shadow": ("sheet_scores_setC_shadow.csv", "Set C Shadow"),
        "crumple": ("sheet_scores_setC_crumple.csv", "Set C Crumple")}
OUT_SCORES = DEV / "sheet_scores_all_sets.csv"
OUT_MANIFEST = HERE / "test_set_all.csv"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def pair_id(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        if line.startswith("# pair_id:"):
            return line.split(":", 1)[1].strip()
    return ""


def main() -> int:
    base = pd.read_csv(DEV / "sheet_scores.csv", dtype=str, keep_default_na=False)
    parts, man_rows = [base], []
    for cond, (fname, folder) in SETC.items():
        s = pd.read_csv(DEV / fname, dtype=str, keep_default_na=False)
        clash = set(s["export"]) & set(base["export"])
        if clash:
            raise SystemExit(f"{fname}: exports already in sheet_scores.csv: {sorted(clash)}")
        parts.append(s)
        for exp in s["export"].unique():
            pid = pair_id(EXPORTS / folder / exp)
            man_rows.append({"export": exp, "photo_id": f"C_{cond}_{pid or exp}", "set": "robustness",
                             "condition": cond, "photo_path": ""})
    cols = list(base.columns)
    for p in parts[1:]:
        cols += [c for c in p.columns if c not in cols]
    allsc = pd.concat(parts, ignore_index=True)[cols]
    allsc.to_csv(OUT_SCORES, index=False, lineterminator="\n")
    man = pd.read_csv(HERE / "test_set.csv", dtype=str, keep_default_na=False)
    pd.concat([man, pd.DataFrame(man_rows)], ignore_index=True).to_csv(OUT_MANIFEST, index=False,
                                                                       lineterminator="\n")
    OUT_SCORES.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/make_setC_inputs.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", "results/device/sheet_scores.csv",
                              *[f"results/device/{v[0]}" for v in SETC.values()])),
        "params": {"concatenated": ["results/device/sheet_scores.csv"] + [f"results/device/{v[0]}" for v in SETC.values()],
                   "manifest": "python/test_set_all.csv",
                   "rows": {"sheet_scores.csv": len(base), **{v[0]: int(len(p)) for v, p in zip(SETC.values(), parts[1:])}},
                   "note": "rows copied unchanged; the rescue re-runs are not included (in-sample)"},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT_SCORES, len(allsc), "rows;", OUT_MANIFEST, len(man) + len(man_rows), "exports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
