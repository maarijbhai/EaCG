#!/usr/bin/env python3
"""Per-sheet and per-lead fidelity for the report's appendix.

results/report/appendix_per_sheet.csv, appendix_per_lead.csv (+ .json).
Presentation of committed results only: the per-lead scores of
results/device/sheet_scores_all_sets.csv with python/test_set_all.csv,
loaded with make_report_figures' own loader and summarised with its own
`summary` (median over available leads; an empty PCC counts in n and as a
failure). The twelve leads only; the rescue re-runs are not in the manifest.

    python make_appendix_tables.py          (from python/)
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import make_report_figures as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORES = REPO / "results" / "device" / "sheet_scores_all_sets.csv"
MANIFEST = HERE / "test_set_all.csv"
SHEETS = REPO / "results" / "kitphoto" / "test_sheets.csv"
OUT = REPO / "results" / "report"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def cell(d: pd.DataFrame, prefix: str) -> dict:
    s = R.summary(d["pcc_asis"])
    return {f"{prefix}_median_pcc": s["median"], f"{prefix}_pct_ge_09": s["pct_ge_09"],
            f"{prefix}_n_leads": s["n"], f"{prefix}_n_unavailable": s["n_unavailable"],
            f"{prefix}_runs": d["export"].nunique()}


def groups(df: pd.DataFrame):
    for set_name in ("clean", "ideal", "robustness"):
        for kind in ("learned", "classical"):
            yield f"{set_name}_{kind}", df[(df["set"] == set_name) & (df["tracer_kind"] == kind)]


def main() -> int:
    df, _ = R.load(SCORES, MANIFEST)
    df = R.leads12(df)
    sheets = pd.read_csv(SHEETS)

    rows = []
    for _, s in sheets.iterrows():
        r = {"ecg_id": int(s["ecg_id"]), "superclass": s["superclass"], "group": R.group_of(s["ecg_id"])}
        for key, d in groups(df):
            r.update(cell(d[d["ecg_id"] == s["ecg_id"]], key))
        rows.append(r)
    per_sheet = pd.DataFrame(rows)

    rows = []
    for lead in LEADS:
        r = {"lead": lead}
        for key, d in groups(df):
            r.update(cell(d[d["lead"] == lead], key))
        rows.append(r)
    per_lead = pd.DataFrame(rows)

    meta = {"generated_by": "python/make_appendix_tables.py",
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": R.git("rev-parse", "HEAD"),
            "git_dirty": bool(R.git("status", "--porcelain", "--", "python")),
            "params": {"scores": str(SCORES.relative_to(REPO)), "manifest": str(MANIFEST.relative_to(REPO)),
                       "leads": "12 (rhythm strip excluded)", "good_pcc": R.GOOD},
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    for name, t in (("appendix_per_sheet", per_sheet), ("appendix_per_lead", per_lead)):
        t.to_csv(OUT / f"{name}.csv", index=False)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote results/report/{name}.csv ({len(t)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
