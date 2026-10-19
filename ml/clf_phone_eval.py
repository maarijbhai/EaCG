#!/usr/bin/env python3
"""The diagnosis classifier on every phone export of Sets A, B and C (read-only).

Per export (one phone run, one tracer):
  * phone      the export's own `# diagnosis:` line (what the app showed);
  * offline    the deployed classifier (assets/ml/clf_superclass_fp16.tflite, the
               app's thresholds) run here on the export's signal, through
               ecgproto.clf_input.from_columns -- the phone's input path;
  * truth      the same classifier on the record's PTB-XL signal (the printed view
               of the 100 Hz record, as in clf_degradation.py / clf_sheets);
  * labels     the record's PTB-XL superclass labels (clf_data cache).
Compared: phone vs offline (port check: max |dp|, decisions identical), offline
vs truth (the digitiser's effect on the decision), offline vs labels (exact set /
every label among the decisions). CD flagged without a CD label is a "CD false
flag"; when the truth signal flags it too it is the known CD-threshold false flag
(threshold 0.05, progress log §32-33), not a digitiser error.

Which exports: python/test_set_all.csv (Set A = clean, Set B = ideal, Set C =
robustness tilt / shadow / crumple), and the rescue re-runs (Crumple ResQ, Tilt
ResQ; condition from results/device/setC_runs.csv), reported apart as IN-SAMPLE.
Record id and quality verdict from the committed score files (so the two
target corrections, python/target_corrections.csv, apply). Nothing is re-scored.

    wsl -d Ubuntu -- bash /mnt/c/dev/ECG-App/ml/wsl_run.sh python ml/clf_phone_eval.py

Writes results/ml/clf_phone_eval.csv (per export) + .json,
clf_phone_eval_summary.csv (per set / condition / tracer / verdict) and
clf_phone_eval.tex.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import clf_data as D  # noqa: E402
import train_classifier as TC  # noqa: E402
from clf_degradation import Deployed  # noqa: E402
from ecgproto import clf_input as CI  # noqa: E402
import score_device_sheets as SDS  # noqa: E402

RESULTS = REPO / "results" / "ml"
EXPORTS = REPO / "data" / "device_exports"
DEV = REPO / "results" / "device"
MANIFEST = REPO / "python" / "test_set_all.csv"
SCORES = [DEV / "sheet_scores_all_sets.csv", DEV / "sheet_scores_setC_crumple_rescue.csv",
          DEV / "sheet_scores_setC_tilt_rescue.csv"]
RUNS = DEV / "setC_runs.csv"
SET_NAME = {"clean": "A", "ideal": "B", "robustness": "C"}
RESCUE_FOLDERS = ("Crumple ResQ", "Tilt ResQ")


def export_paths() -> dict:
    """export file name -> path, every folder except _excluded and the holding folder."""
    out = {}
    for p in EXPORTS.rglob("eacg_sheet_*.csv"):
        if any(part.startswith("_") for part in p.relative_to(EXPORTS).parts[:-1]):
            continue
        if p.name in out:
            raise SystemExit(f"two copies of {p.name}: {out[p.name]} and {p}")
        out[p.name] = p
    return out


def says(p, thr, classes) -> str:
    return " ".join(c for j, c in enumerate(classes) if p[j] >= thr[j]) or "none"


def main() -> int:
    man = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    man["set_name"] = man["set"].map(SET_NAME)
    runs = pd.read_csv(RUNS)
    paths = export_paths()
    resc = sorted(n for n, p in paths.items() if p.parent.name in RESCUE_FOLDERS)
    cond_of = dict(zip(runs["export"], runs["condition"]))
    todo = [(r.export, r.set_name, r.condition) for r in man.itertuples()]
    todo += [(e, "C rescue", cond_of.get(e, "rescue")) for e in resc]

    sc = pd.concat([pd.read_csv(f) for f in SCORES], ignore_index=True).drop_duplicates("export")
    sc = sc.set_index("export")

    data = D.load(D.CACHE)
    pos = {int(e): k for k, e in enumerate(data["ecg_id"])}
    model = Deployed()
    classes, thr = model.classes, model.thr
    truth_cache = {}

    rows = []
    for exp, set_name, cond in todo:
        if exp not in paths:
            raise SystemExit(f"{exp}: not found under {EXPORTS}")
        if exp not in sc.index:
            raise SystemExit(f"{exp}: not in the committed score files")
        s = sc.loc[exp]
        eid = int(s["ecg_id"])
        k = pos[eid]
        meta, cols = SDS.read_export(paths[exp])
        if eid not in truth_cache:
            truth_cache[eid] = model(CI.printed_view(data["X"][k].astype(np.float64))[None])[0]
        p_true = truth_cache[eid]
        p_off = model(CI.from_columns(cols)[None])[0]
        got = dict(re.findall(r"(\w+)=([0-9.]+)", meta.get("diagnosis", "")))
        have_phone = all(c in got for c in classes)
        p_ph = np.array([float(got[c]) for c in classes]) if have_phone else np.full(len(classes), np.nan)
        labels = [c for c, y in zip(classes, data["y_super"][k]) if y]
        d_off, d_true = p_off >= thr, p_true >= thr
        off_set = {c for j, c in enumerate(classes) if d_off[j]}
        cd = classes.index("CD")
        r = {"export": exp, "set": set_name, "condition": cond, "ecg_id": eid,
             "tracer": "learned" if str(s["tracer"]).startswith("model") else "classical",
             "quality": s["quality"] if isinstance(s["quality"], str) and s["quality"] else "not recorded",
             "labels": " ".join(labels) or "none",
             "phone_says": says(p_ph, thr, classes) if have_phone else "",
             "offline_says": says(p_off, thr, classes), "truth_says": says(p_true, thr, classes),
             "phone_line_present": have_phone,
             "phone_max_abs_dp": float(np.max(np.abs(p_ph - p_off))) if have_phone else np.nan,
             "phone_eq_offline": bool(np.all((p_ph >= thr) == d_off)) if have_phone else None,
             "offline_eq_truth": bool(np.all(d_off == d_true)),
             "offline_mean_abs_dp_truth": float(np.abs(p_off - p_true).mean()),
             "label_exact": off_set == set(labels),
             "label_contained": set(labels) <= off_set,
             "truth_label_exact": {c for j, c in enumerate(classes) if d_true[j]} == set(labels),
             "cd_false_flag": bool(d_off[cd] and "CD" not in labels),
             "cd_false_flag_in_truth_too": bool(d_off[cd] and "CD" not in labels and d_true[cd])}
        for j, c in enumerate(classes):
            r[f"p_phone_{c}"] = p_ph[j]
            r[f"p_offline_{c}"] = float(p_off[j])
            r[f"p_truth_{c}"] = float(p_true[j])
        rows.append(r)
        print(exp, set_name, cond, r["tracer"], r["quality"], "| labels", r["labels"], "| phone", r["phone_says"],
              "| offline", r["offline_says"], "| truth", r["truth_says"], flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(RESULTS / "clf_phone_eval.csv", index=False, lineterminator="\n")

    def agg(g: pd.DataFrame) -> dict:
        ph = g[g["phone_line_present"]]
        return {"exports": int(len(g)), "phone_line": int(len(ph)),
                "phone_eq_offline": int(ph["phone_eq_offline"].astype(bool).sum()),
                "phone_max_abs_dp": float(ph["phone_max_abs_dp"].max()) if len(ph) else np.nan,
                "offline_eq_truth": int(g["offline_eq_truth"].sum()),
                "label_exact": int(g["label_exact"].sum()), "label_contained": int(g["label_contained"].sum()),
                "truth_label_exact": int(g["truth_label_exact"].sum()),
                "cd_false_flags": int(g["cd_false_flag"].sum()),
                "cd_false_flags_in_truth_too": int(g["cd_false_flag_in_truth_too"].sum())}

    order = {"A": 0, "B": 1, "C": 2, "C rescue": 3}
    corder = {"clean": 0, "ideal": 1, "tilt": 2, "shadow": 3, "crumple": 4, "crumple rescue": 5, "tilt rescue": 6}
    vorder = {"all": 0, "ok": 1, "check": 2, "unreliable": 3, "not recorded": 4}
    summ = []
    for (st, cond, tk), g in df.groupby(["set", "condition", "tracer"]):
        summ.append({"set": st, "condition": cond, "tracer": tk, "verdict": "all", **agg(g)})
        for v, gv in g.groupby("quality"):
            summ.append({"set": st, "condition": cond, "tracer": tk, "verdict": v, **agg(gv)})
    for tk, g in df[df["set"] == "C"].groupby("tracer"):
        summ.append({"set": "C", "condition": "all", "tracer": tk, "verdict": "all", **agg(g)})
        for v, gv in g.groupby("quality"):
            summ.append({"set": "C", "condition": "all", "tracer": tk, "verdict": v, **agg(gv)})
    S = pd.DataFrame(summ)
    S["_o"] = (S["set"].map(order) * 1000 + S["condition"].map(lambda c: corder.get(c, 7)) * 100
               + S["tracer"].map({"learned": 0, "classical": 1}) * 10 + S["verdict"].map(vorder))
    S = S.sort_values("_o").drop(columns="_o")
    S.round(5).to_csv(RESULTS / "clf_phone_eval_summary.csv", index=False, lineterminator="\n")

    tex = ["% caption: the deployed classifier on every phone export. Phone = offline: the app's decisions equal",
           "% the deployed model run offline on the exported signal; same as truth: offline decisions equal the",
           "% model's on the record's PTB-XL signal; exact / contains: offline decisions equal / include the PTB-XL",
           "% superclass labels; CD false flags (of which also on the truth signal: the CD threshold 0.05).",
           "% Rescue re-runs are in-sample (the t0 guard was designed on them).",
           r"\begin{tabular}{@{}llrrrrrl@{}}", r"\toprule",
           r"\textbf{Set / condition} & \textbf{Verdict} & \textbf{Exports} & \textbf{Phone = offline} & "
           r"\textbf{Same as truth} & \textbf{Exact label} & \textbf{Contains label} & \textbf{CD false flags}\\"]
    for tk in ("learned", "classical"):
        tex += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{tk.capitalize()} tracer}}}}\\"]
        for _, r in S[S["tracer"] == tk].iterrows():
            lab = f"{r['set']} {r['condition']}" if r["set"] in ("C", "C rescue") else r["set"]
            lab = lab.replace("C rescue crumple rescue", "C crumple rescue").replace("C rescue tilt rescue", "C tilt rescue")
            n = r["exports"]
            pe = f"{r['phone_eq_offline']}/{r['phone_line']}" if r["phone_line"] else "--"
            tex.append(f"{lab if r['verdict'] == 'all' else ''} & {r['verdict']} & {n} & {pe} & "
                       f"{r['offline_eq_truth']}/{n} & {r['label_exact']}/{n} & {r['label_contained']}/{n} & "
                       f"{r['cd_false_flags']} ({r['cd_false_flags_in_truth_too']})\\\\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    (RESULTS / "clf_phone_eval.tex").write_text("\n".join(tex), encoding="utf-8")

    srcs = ["python/test_set_all.csv", *[str(f.relative_to(REPO)).replace("\\", "/") for f in SCORES],
            "results/device/setC_runs.csv"]
    truth_rows = [{"ecg_id": e, "labels": " ".join(c for c, y in zip(classes, data["y_super"][pos[e]]) if y),
                   "truth_says": says(p, thr, classes), **{f"p_{c}": float(p[j]) for j, c in enumerate(classes)}}
                  for e, p in sorted(truth_cache.items())]
    (RESULTS / "clf_phone_eval.json").write_text(json.dumps({
        "generated_by": "ml/clf_phone_eval.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": TC.git("rev-parse", "HEAD"),
        "git_dirty": bool(TC.git("status", "--porcelain", "--", "ml", "python", *srcs)),
        "files": ["clf_phone_eval.csv", "clf_phone_eval_summary.csv", "clf_phone_eval.tex"],
        "sources": srcs, "exports_dir": "data/device_exports (not _excluded, not _from_phone_*)",
        "n_exports": int(len(df)), "n_rescue_in_sample": int((df["set"] == "C rescue").sum()),
        "model": model.meta, "truth_per_record": truth_rows,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(S.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
