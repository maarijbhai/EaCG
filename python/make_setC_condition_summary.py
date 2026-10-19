#!/usr/bin/env python3
"""Set C, one row per condition: what ran, what the quality gate said, what failed and why.

results/report_set_C/setC_condition_summary.csv / .tex / .json. Presentation of
committed results only: the per-run facts come from results/device/setC_runs.csv,
the lead scores from results/device/sheet_scores_all_sets.csv with
python/test_set_all.csv (the inputs of report_set_C/compare_sets), and every
statistic from make_report_figures' own functions (the Sets A/B standard).
The rescue re-runs are left out, as in the report (in-sample).

DEFINITIONS (fixed before looking at the table, from §43 and the app)
  * run: one learned export (the learned tracer is the app's method; the
    classical run of the same photograph is its control).
  * reliable: the learned run's verdict is ok or check (not unreliable). The
    classical median uses the same photographs; the script checks that each
    classical run's own verdict selects the same ones and says so if not.
  * failure: learned median lead PCC < 0.5 (as in §43).
  * cause: the FIRST stage that went wrong, in pipeline order -- no pulse
    (NO_PULSE: time zero from the rhythm locator), else few ticks (FEW_TICKS or
    NO_TICK_SCALE), else scale disagree (SCALE_DISAGREE). One cause per failure.
  * median PCC: over the 12 leads of the reliable runs, an empty PCC a failure,
    95% CI by cluster bootstrap over photographs (2000 resamples, seed 20261012).
  * change vs Set B: each reliable lead minus the median of the same record's
    Set B photographs (same lead, tracer); median with cluster-bootstrap CI.
  * gate row (learned runs, all conditions): failures caught = failures marked
    unreliable; false alarms = good runs (not failures) marked unreliable.

    python make_setC_condition_summary.py          (from python/)
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
RUNS = REPO / "results" / "device" / "setC_runs.csv"
SCORES = REPO / "results" / "device" / "sheet_scores_all_sets.csv"
MANIFEST = HERE / "test_set_all.csv"
OUT = REPO / "results" / "report_set_C" / "setC_condition_summary"
CONDS = ("tilt", "shadow", "crumple")
FAIL_PCC = 0.5
CAUSES = (("no pulse", ("NO_PULSE",)), ("few ticks", ("FEW_TICKS", "NO_TICK_SCALE")),
          ("scale disagree", ("SCALE_DISAGREE",)))


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def cause(warnings) -> str:
    w = str(warnings) if isinstance(warnings, str) else ""
    codes = {x.strip().split("/")[0] for x in w.split(";") if x.strip()}
    for name, keys in CAUSES:
        if codes & set(keys):
            return name
    return "other"


def med_ci(d: pd.DataFrame, col: str) -> tuple:
    sm = R.summary(d[col])
    lo, hi = R.cluster_ci(d, col) if len(d.dropna(subset=[col])) else (np.nan, np.nan)
    return sm["median"], lo, hi, sm["n"]


def main() -> int:
    runs = pd.read_csv(RUNS)
    runs = runs[runs["condition"].isin(CONDS)].copy()
    df, _ = R.load(SCORES, MANIFEST)
    df = df[df["set"].isin(["ideal", "robustness"])]
    learned = runs[runs["tracer"] == "learned"].copy()
    learned["reliable"] = learned["quality"].isin(["ok", "check"])
    learned["failed"] = learned["median_pcc"] < FAIL_PCC
    learned["cause"] = np.where(learned["failed"], learned["warnings"].map(cause), "")
    reliable_photos = set(learned.loc[learned["reliable"], "photo"])

    classical = runs[runs["tracer"] == "classical"]
    own = set(classical.loc[classical["quality"].isin(["ok", "check"]), "photo"])
    same_selection = own == (reliable_photos & set(classical["photo"]))

    pairs = R.robustness_pairs(df)
    rows = []
    for c in CONDS:
        L = learned[learned["condition"] == c]
        rel_exports = set(runs.loc[(runs["condition"] == c) & runs["photo"].isin(reliable_photos), "export"])
        row = {"condition": c, "photos": int(runs.loc[runs["condition"] == c, "photo"].nunique()),
               "learned_runs": int(len(L)), "reliable_runs": int(L["reliable"].sum()),
               "failures": int(L["failed"].sum())}
        for name, _ in CAUSES:
            row[f"failures_{name.replace(' ', '_')}"] = int((L["cause"] == name).sum())
        row["failures_other"] = int((L["cause"] == "other").sum())
        for tk in R.TRACERS:
            d = R.leads12(df[(df["set"] == "robustness") & (df["condition"] == c) & (df["tracer_kind"] == tk)
                             & df["export"].isin(rel_exports)])
            m, lo, hi, n = med_ci(d, "pcc_asis")
            row.update({f"{tk}_median_pcc": m, f"{tk}_ci_lo": lo, f"{tk}_ci_hi": hi, f"{tk}_n_leads": n})
            p = pairs[(pairs["condition"] == c) & (pairs["tracer_kind"] == tk) & pairs["export"].isin(rel_exports)]
            p = p.dropna(subset=["diff"])
            lo, hi = R.cluster_ci(p, "diff") if len(p) else (np.nan, np.nan)
            row.update({f"{tk}_diff_vs_B": float(np.median(p["diff"])) if len(p) else np.nan,
                        f"{tk}_diff_ci_lo": lo, f"{tk}_diff_ci_hi": hi, f"{tk}_n_paired": int(len(p))})
        rows.append(row)
    t = pd.DataFrame(rows)

    fails, good = learned[learned["failed"]], learned[~learned["failed"]]
    gate = {"failures": int(len(fails)), "failures_caught": int((fails["quality"] == "unreliable").sum()),
            "good_runs": int(len(good)), "false_alarms": int((good["quality"] == "unreliable").sum()),
            "false_alarm_runs": sorted(good.loc[good["quality"] == "unreliable", "export"]),
            "missed_failure_runs": sorted(fails.loc[fails["quality"] != "unreliable", "export"])}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    gate_row = {"condition": "gate (all conditions)", "learned_runs": int(len(learned)),
                "failures": gate["failures"], "failures_caught": gate["failures_caught"],
                "good_runs": gate["good_runs"], "false_alarms": gate["false_alarms"]}
    table = pd.concat([t, pd.DataFrame([gate_row])], ignore_index=True)
    for col in table.columns:
        if col in ("photos", "learned_runs", "reliable_runs", "failures", "failures_caught", "good_runs",
                   "false_alarms") or col.startswith("failures_") or col.endswith(("_n_leads", "_n_paired")):
            table[col] = table[col].astype("Int64")
    table.to_csv(OUT.with_suffix(".csv"), index=False, lineterminator="\n")

    def ci(m, lo, hi, signed=False):
        f = "{:+.3f}" if signed else "{:.3f}"
        if not np.isfinite(m):
            return "--"
        return f.format(m) + (f" [{f.format(lo)}, {f.format(hi)}]" if np.isfinite(lo) else "")

    tex = ["% caption: Set C by condition. Reliable = the learned run's verdict ok or check; failure = learned",
           "% median lead PCC < 0.5, cause = first stage that went wrong; PCC and change vs the same records'",
           "% Set B photographs over the reliable runs' 12 leads, median [95% CI, cluster bootstrap over photographs;",
           "% none with a single reliable photograph]. Rescue re-runs not included (in-sample).",
           r"\begin{tabular}{@{}lrrrlllll@{}}", r"\toprule",
           r"\textbf{Condition} & \textbf{Photos} & \textbf{Runs} & \textbf{Reliable} & \textbf{Failures (cause)} & "
           r"\textbf{Learned PCC} & \textbf{$\Delta$ vs B} & \textbf{Classical PCC} & \textbf{$\Delta$ vs B}\\",
           r"\midrule"]
    for r in rows:
        cs = ", ".join(f"{r['failures_' + n.replace(' ', '_')]} {n}" for n, _ in CAUSES
                       if r["failures_" + n.replace(" ", "_")])
        if r["failures_other"]:
            cs += f", {r['failures_other']} other"
        tex.append(f"{r['condition']} & {r['photos']} & {r['learned_runs']} & {r['reliable_runs']} & "
                   f"{r['failures']}" + (f" ({cs})" if cs else "") + " & "
                   f"{ci(r['learned_median_pcc'], r['learned_ci_lo'], r['learned_ci_hi'])} & "
                   f"{ci(r['learned_diff_vs_B'], r['learned_diff_ci_lo'], r['learned_diff_ci_hi'], True)} & "
                   f"{ci(r['classical_median_pcc'], r['classical_ci_lo'], r['classical_ci_hi'])} & "
                   f"{ci(r['classical_diff_vs_B'], r['classical_diff_ci_lo'], r['classical_diff_ci_hi'], True)}\\\\")
    tex += [r"\midrule",
            rf"\multicolumn{{9}}{{@{{}}l}}{{Quality gate, {len(learned)} learned runs: "
            rf"{gate['failures_caught']} of {gate['failures']} failures marked unreliable; "
            rf"{gate['false_alarms']} of {gate['good_runs']} good runs marked unreliable (false alarms)}}\\",
            r"\bottomrule", r"\end{tabular}", ""]
    OUT.with_suffix(".tex").write_text("\n".join(tex), encoding="utf-8")

    srcs = ["results/device/setC_runs.csv", "results/device/sheet_scores_all_sets.csv", "python/test_set_all.csv"]
    OUT.with_suffix(".json").write_text(json.dumps(R.clean_json({
        "generated_by": "python/make_setC_condition_summary.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", *srcs)),
        "sources": srcs, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "params": {"fail_pcc": FAIL_PCC, "reliable": ["ok", "check"], "n_boot": R.N_BOOT, "seed": R.SEED,
                   "cause_order": [n for n, _ in CAUSES], "rescues": "left out (in-sample)"},
        "classical_own_verdict_selects_same_photos": same_selection,
        "gate": gate,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }), indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(OUT.with_suffix(".csv")).to_string())
    print("gate", gate, "same selection", same_selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
