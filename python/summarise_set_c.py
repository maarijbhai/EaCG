#!/usr/bin/env python3
"""Set C (robustness) for the report: every run, every condition, against Sets A and B.

Reads (never rewrites) the committed scores:
  results/device/sheet_scores_setC_tilt.csv          Set C Tilt      (phone tilted ~25-35 deg)
  results/device/sheet_scores_setC_shadow.csv        Set C Shadow    (hard shadow, one warm lamp)
  results/device/sheet_scores_setC_crumple.csv       Set C Crumple   (crumpled sheets)
  results/device/sheet_scores_setC_crumple_rescue.csv  the failed Crumple photographs re-run
                                                     with the guarded t0 fallback (§41-42)
  results/device/sheet_scores_setC_tilt_rescue.csv   the failed Tilt PTB-XL 9 photograph, same
and, for reference, results/device/sheet_scores.csv x python/test_set.csv (Set A clean, Set B
ideal). The lead is the unit (12 per run; the rhythm strip apart), as in
make_report_figures.py; an empty PCC counts as a failure (in n, not >= 0.9). A run
"failed" if its median lead PCC is below 0.5. The rescue rows are IN-SAMPLE (the guard was
designed on these failures) and are not pooled with the conditions.

    python summarise_set_c.py          (from python/)

Writes results/device/setC_runs.csv, setC_summary.csv (+ .json each) and stats_set_C.xlsx.
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

#: condition -> (scores file, export folder, what it is, pooled with the conditions?)
SOURCES = {
    "tilt": ("sheet_scores_setC_tilt.csv", "Set C Tilt", "phone tilted ~25-35 deg from parallel", True),
    "shadow": ("sheet_scores_setC_shadow.csv", "Set C Shadow", "hard shadow from a single warm lamp", True),
    "crumple": ("sheet_scores_setC_crumple.csv", "Set C Crumple", "crumpled sheets", True),
    "crumple rescue": ("sheet_scores_setC_crumple_rescue.csv", "Crumple ResQ",
                       "the failed crumple photographs re-run with the t0 guard (in-sample)", False),
    "tilt rescue": ("sheet_scores_setC_tilt_rescue.csv", "Tilt ResQ",
                    "the failed tilt PTB-XL 9 photograph re-run with the t0 guard (in-sample)", False),
}
#: The committed files this reads (the dirty check covers these and python/, not its own outputs).
INPUTS = [f"results/device/{v[0]}" for v in SOURCES.values()] + [
    "results/device/sheet_scores.csv", "results/device/setC_crumple_rescue_before_after.csv"]
MEASURES = ["pcc_asis", "pcc_aligned", "snr_db_asis", "shifted_snr_db_asis", "rmse_mv_asis", "gain_ratio", "r_ratio"]
FAIL_BELOW = 0.5


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


def photo_table(folder: str) -> dict:
    """pair_id or run time -> photograph, from the folder's README."""
    out = {}
    for line in (EXPORTS / folder / "README.txt").read_text(encoding="utf-8").splitlines():
        ph = re.search(r"(eacg_\d{8}_\d{6}(?:_\d+)?\.jpg)", line)
        if not ph:
            continue
        for t in re.findall(r"(?<![\d])(\d{8}_\d{6}|\d{6})(?![\d])", line.split(ph.group(1))[0]):
            out[t] = ph.group(1)
    return out


def tracer(t) -> str:
    return "classical" if str(t).startswith("classical") else "learned"


def summarise(leads: pd.DataFrame, group: str, runs: pd.DataFrame | None = None) -> list[dict]:
    out = []
    for tr, g in leads.groupby(leads["tracer"].map(tracer)):
        row = {"group": group, "tracer": tr, "runs": g["export"].nunique(), "leads": len(g)}
        if runs is not None:
            r = runs[runs["tracer"] == tr]
            row["runs_failed"] = int((r["median_pcc"].fillna(-1) < FAIL_BELOW).sum())
            row["verdict_ok_check_unreliable"] = "/".join(str(int((r["quality"] == q).sum()))
                                                          for q in ("ok", "check", "unreliable"))
        for m in MEASURES:
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
            row.update({f"{m}_median": round(q[1], 3), f"{m}_q1": round(q[0], 3), f"{m}_q3": round(q[2], 3)})
        pcc = pd.to_numeric(g["pcc_asis"], errors="coerce")
        row["leads_pcc_ge_0.9"] = int((pcc >= 0.9).sum())
        row["fraction_pcc_ge_0.9"] = round(float((pcc >= 0.9).sum() / len(g)), 3)
        out.append(row)
    return out


def write_csv(rows: list[dict], name: str, params: dict) -> None:
    p = DEV / name
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with p.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    p.with_suffix(".json").write_text(json.dumps(sidecar(params), indent=2) + "\n", encoding="utf-8")
    print(p.with_suffix(".csv"))


def sidecar(params: dict) -> dict:
    return {"generated_by": "python/summarise_set_c.py",
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "python", *INPUTS)),
            "params": params,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}


def main() -> int:
    runs, leads = [], []
    for cond, (fname, folder, what, _) in SOURCES.items():
        s = pd.read_csv(DEV / fname)
        s.insert(0, "condition", cond)
        leads.append(s)
        photos = photo_table(folder)
        for exp, g in s.groupby("export", sort=False):
            h = header(next(p for p in (EXPORTS / folder).rglob(exp)))
            cells = g[~g["rhythm"].astype(bool)]
            rh = g[g["rhythm"].astype(bool)]
            num = lambda m: pd.to_numeric(cells[m], errors="coerce")
            geo = h.get("geometry", "")
            inl = re.search(r"inliers ([\d.]+)", geo)
            gp = re.search(r"grid \(.*?([\d.]+) px/mm", geo)
            tp = re.match(r"ticks ([\d.]+) px/mm", h.get("calibration", ""))
            pid = h.get("pair_id", "")
            t = h.get("local_time", "")
            runs.append({
                "condition": cond, "pair_id": pid, "export": exp,
                "photo": photos.get(pid) or photos.get(t) or photos.get(t[9:]) or "",
                "record": g["target"].iloc[0].split("_")[0], "tracer": tracer(h.get("tracer")),
                "header_target": h.get("target", "").split("_")[0],
                "photo_rotation": h.get("photo_rotation", ""), "corners": h.get("corners", ""),
                "quality": h.get("quality"), "warnings": h.get("warnings"),
                "ticks_found": g["ticks_found"].iloc[0], "t0_source": g["t0_source"].iloc[0],
                "t0_guard": h.get("t0_guard", ""),
                "tick_vs_grid_pct": round(100 * (float(tp.group(1)) / float(gp.group(1)) - 1), 1) if tp and gp else "",
                "grid_inliers": float(inl.group(1)) if inl else "",
                **{f"median_{m.replace('_asis', '')}": round(float(num(m).median()), 3) for m in MEASURES},
                "leads_pcc_ge_0.9": int((num("pcc_asis") >= 0.9).sum()),
                "rhythm_pcc": rh["pcc_asis"].iloc[0] if len(rh) else "",
                "trace_ms": (re.search(r"trace=(\d+)", h.get("timings_ms", "")) or [None, ""])[1],
                "orient_ms": h.get("orient_ms", "")})
    R = pd.DataFrame(runs).rename(columns={"median_pcc": "median_pcc"})
    R = R.rename(columns={"median_pcc_aligned": "median_pcc_aligned"})
    write_csv(R.to_dict("records"), "setC_runs", {"sources": {k: v[0] for k, v in SOURCES.items()},
                                                  "failed": f"median lead PCC < {FAIL_BELOW}"})

    L = pd.concat(leads, ignore_index=True)
    cells = L[~L["rhythm"].astype(bool)]
    rows = []
    pooled = [c for c, v in SOURCES.items() if v[3]]
    for cond in SOURCES:
        rows += summarise(cells[cells["condition"] == cond], f"C {cond}", R[R["condition"] == cond])
    rows += summarise(cells[cells["condition"].isin(pooled)], "C all conditions (tilt + shadow + crumple)",
                      R[R["condition"].isin(pooled)])
    m = pd.read_csv(HERE / "test_set.csv")
    a = pd.read_csv(DEV / "sheet_scores.csv")
    a = a[~a["rhythm"].astype(bool)]
    rows += summarise(a[a["export"].isin(set(m.loc[m["set"] == "ideal", "export"]))], "B ideal (reference)")
    rows += summarise(a[a["export"].isin(set(m.loc[m["set"] == "clean", "export"]))], "A clean (reference)")
    write_csv(rows, "setC_summary", {"unit": "lead (12 per run; rhythm strip excluded)",
                                     "pooled": pooled, "failed": f"median lead PCC < {FAIL_BELOW}",
                                     "reference": "results/device/sheet_scores.csv x python/test_set.csv"})

    ba = pd.read_csv(DEV / "setC_crumple_rescue_before_after.csv")
    readme = pd.DataFrame({"item": [
        "What", "Conditions", "Unit", "Failed run", "Rescue rows", "Sheets",
        "Sources", "Made by", "Commit"], "value": [
        "Set C (robustness): photographs of the printed sheets under tilt, shadow and crumpling, "
        "digitised on the Galaxy A30s, both tracers, scored against the PTB-XL truth.",
        "; ".join(f"{k}: {v[2]}" for k, v in SOURCES.items()),
        "The lead (12 per run; the rhythm strip separately in 'runs'). Medians [IQR] over leads.",
        f"Median lead PCC below {FAIL_BELOW}.",
        "IN-SAMPLE: the t0 guard was designed after seeing these failures (progress_log §41-42); "
        "not pooled with the conditions.",
        "summary = per condition and tracer, with Set B and Set A for reference; runs = one row per "
        "export; leads = every lead scored; rescue_before_after = the four crumple failures before "
        "and after the guard.",
        "; ".join(v[0] for v in SOURCES.values()) + "; sheet_scores.csv x test_set.csv (A, B); "
        "setC_crumple_rescue_before_after.csv",
        "python/summarise_set_c.py (reads the committed scores; scores by score_device_sheets via "
        "score_device_set.py)", git("rev-parse", "--short", "HEAD")]})
    xl = DEV / "stats_set_C.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as wr:
        readme.to_excel(wr, sheet_name="README", index=False)
        pd.DataFrame(rows).to_excel(wr, sheet_name="summary", index=False)
        R.to_excel(wr, sheet_name="runs", index=False)
        L.to_excel(wr, sheet_name="leads", index=False)
        ba.to_excel(wr, sheet_name="rescue_before_after", index=False)
        for ws in wr.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or ""))
                                                                                       for c in col[:50]) + 2))
    (DEV / "stats_set_C.json").write_text(json.dumps(sidecar({"workbook": "stats_set_C.xlsx"}), indent=2) + "\n",
                                          encoding="utf-8")
    print(xl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
