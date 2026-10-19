#!/usr/bin/env python3
"""The 12 printed test sheets: their PTB-XL diagnoses, and every photograph of each.

From data/ptb-xl/1.0.3 (ptbxl_database.csv, scp_statements.csv): the SCP codes with
their likelihoods, the diagnostic superclass of each diagnostic code (as
example_physionet.py aggregates them), age, sex (PTB-XL: 0 male, 1 female),
strat_fold, and the cardiologist's report text as recorded (German, Swedish or
English). Photographs: Set A data/kit_photos/SetA_Clean/clean_ptbxl<id>.png; Set B
python/test_set.csv; Set C results/device/setC_runs.csv (learned runs).

    python list_test_sheets.py          (from python/)

Writes results/kitphoto/test_sheets.csv + .json.
"""

from __future__ import annotations

import ast
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PTBXL = REPO / "data" / "ptb-xl" / "1.0.3"
OUT = REPO / "results" / "kitphoto" / "test_sheets"
IDS = [9, 38, 40, 57, 160, 167, 1737, 3908, 4016, 8504, 5830, 7916]


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    db = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id")
    st = pd.read_csv(PTBXL / "scp_statements.csv", index_col=0)
    m = pd.read_csv(HERE / "test_set.csv")
    b = m[m["set"] == "ideal"].copy()
    b["ecg_id"] = b["export"].str.extract(r"ptbxl(\d+)_")[0].astype(int)
    c = pd.read_csv(REPO / "results" / "device" / "setC_runs.csv")
    c = c[c["tracer"] == "learned"].copy()
    c["ecg_id"] = c["record"].str.extract(r"(\d+)")[0].astype(int)
    rows = []
    for i in IDS:
        r = db.loc[i]
        codes = ast.literal_eval(r["scp_codes"])
        diag = [k for k in codes if k in st.index and st.loc[k, "diagnostic"] == 1]
        other = [k for k in codes if k not in diag]
        row = {"ecg_id": i, "strat_fold": int(r["strat_fold"]),
               "superclass": "/".join(sorted({st.loc[k, "diagnostic_class"] for k in diag})),
               "diagnostic_codes": "; ".join(f"{k} ({int(codes[k])}%): {st.loc[k, 'description']}" for k in diag),
               "other_codes": "; ".join(f"{k}: {st.loc[k, 'description'] if k in st.index else ''}" for k in other),
               "age": int(r["age"]), "sex": "male" if r["sex"] == 0 else "female",
               "report_text": str(r["report"]).strip(),
               "set_A_image": f"data/kit_photos/SetA_Clean/clean_ptbxl{i}.png",
               "set_B_photos": "; ".join(sorted(b.loc[b["ecg_id"] == i, "photo_path"].unique()))}
        for cond in ("tilt", "shadow", "crumple", "crumple rescue", "tilt rescue"):
            ph = c.loc[(c["ecg_id"] == i) & (c["condition"] == cond), "photo"].dropna().unique()
            row[f"set_C_{cond.replace(' ', '_')}"] = "; ".join(sorted(ph))
        rows.append(row)
    d = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT.with_suffix(".csv"), index=False, lineterminator="\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/list_test_sheets.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", "results/device/setC_runs.csv")),
        "params": {"ptbxl": "data/ptb-xl/1.0.3 (ptbxl_database.csv, scp_statements.csv)",
                   "superclass": "diagnostic_class of every code with diagnostic == 1 (example_physionet.py)",
                   "sex": "PTB-XL coding: 0 male, 1 female"},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
