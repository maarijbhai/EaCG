#!/usr/bin/env python3
"""This work's fidelity in the metrics the published digitisers report.

results/report/lit_comparison.csv / .json. Presentation of committed results
only: the per-lead scores of results/device/sheet_scores_all_sets.csv with
python/test_set_all.csv (Sets A, B and C; the rescue re-runs are left out, as
in the report), loaded with make_report_figures' own loader. No signal is
re-scored. The twelve leads only; the rhythm strip is left out, since the
published figures are per lead.

The report quotes medians; the published works quote means, each under its
own rule. Each column below is this work's per-lead scores aggregated the way
one source aggregates its own (definitions from the sources, checked
2026-10-25):

  * mean_pcc, mean_snr_db, mean_rmse_mv -- ECGLight (Natraj et al. 2026):
    Pearson r, SNR and RMSE with no time shift, averaged. Unrecovered leads
    are left out, as ECGLight reports its failures separately.
  * mean_shifted_snr_db -- Stenhede et al. 2026: SNR at the best horizontal
    shift (theirs up to 100 ms, here 50 ms), both signals centred, averaged.
    Unrecovered leads are left out.
  * challenge_snr_db -- PhysioNet Challenge 2024 (Krones et al. 2024,
    VinDigitizer): each lead's aligned SNR scaled by the fraction of the
    reference recovered (here the shifted SNR times coverage), an unrecovered
    lead 0 dB, then the mean over all leads. The Challenge aligns within
    0.5 s by cross-correlating rasterised signals; this work's shift search
    is 50 ms, so the value is an approximation of the Challenge score, not
    the Challenge's code run on these signals.

    python make_lit_comparison.py          (from python/)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import make_report_figures as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORES = REPO / "results" / "device" / "sheet_scores_all_sets.csv"
MANIFEST = HERE / "test_set_all.csv"
OUT = REPO / "results" / "report" / "lit_comparison"


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def aggregate(d: pd.DataFrame) -> dict:
    ok = d[d["available"]]
    challenge = (d["shifted_snr_db_asis"] * d["coverage"]).where(d["available"], 0.0)
    return {
        "runs": d["export"].nunique(), "leads": len(d), "unrecovered": int((~d["available"]).sum()),
        "mean_pcc": ok["pcc_asis"].mean(),
        "mean_snr_db": ok["snr_db_asis"].mean(),
        "mean_rmse_mv": ok["rmse_mv_asis"].mean(),
        "mean_shifted_snr_db": ok["shifted_snr_db_asis"].mean(),
        "challenge_snr_db": challenge.mean(),
        "median_pcc": ok["pcc_asis"].median(),
        "median_shifted_snr_db": ok["shifted_snr_db_asis"].median(),
    }


def main() -> int:
    df, report = R.load(SCORES, MANIFEST)
    df = R.leads12(df)
    rows = [{"set": s, "condition": c, "tracer": t, **aggregate(g)}
            for (s, c, t), g in df.groupby(["set", "condition", "tracer_kind"], sort=False)]
    out = pd.DataFrame(rows).round(4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT.with_suffix(".csv"), index=False)
    srcs = ["python/make_lit_comparison.py", "python/make_report_figures.py"]
    OUT.with_suffix(".json").write_text(json.dumps(R.clean_json({
        "generated_by": "python/make_lit_comparison.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", *srcs)),
        "sources": srcs,
        "scores": str(SCORES.relative_to(REPO)),
        "scores_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "params": {"leads": "12 (rhythm strip left out)", "shift_search_ms": 50,
                   "challenge_snr": "shifted SNR x coverage, unrecovered lead = 0 dB, mean over leads"},
        "skipped": report,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }), indent=2) + "\n", encoding="utf-8")
    print(out.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
