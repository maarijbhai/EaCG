#!/usr/bin/env python3
"""Does the classifier survive the digitiser? The degradation curve.

The deployed classifier (`assets/ml/clf_superclass_fp16.tflite`, from
`export_classifier.py`) was trained on true PTB-XL signals. The phone feeds
it digitised ones. Three measurements, all on the TEST fold (10), which no
model or threshold was chosen on:

1. RESAMPLE. The phone's input path on TRUE signals: each test record at
   500 Hz, cut to its printed cells exactly as SheetPipeline.csv lays them
   out, through `ecgproto.clf_input.from_columns` (the block-mean 500 ->
   100 Hz the phone uses), against the training view from PTB-XL's own
   100 Hz files. Isolates what the phone's resampling costs before any
   digitiser error.

2. LADDER. One digitiser error at a time, at increasing size, applied to
   the true 100 Hz signals window by window (each 2.5 s cell and the rhythm
   strip independently, as the digitiser reads them): amplitude gain, a time
   shift per window, a time stretch per window, additive noise, gaps. Then
   "measured": all of them at once at the sizes the phone actually showed
   (`results/device/sheet_scores.csv`, learned tracer, 3x4 leads): median
   gain, the spread of the per-lead shift, the median missing coverage;
   noise 0.02 mV is an assumption (no direct measurement) and is labelled
   so. Fixed seed per step.

3. SHEETS. Every phone export in data/device_exports/ with a known record:
   the classifier on the digitised sheet against the classifier on the
   same record's true signals -- and, when the export carries the phone's
   own `# diagnosis:` line, the phone's probabilities against the desktop's
   on the same leads (port check).

Per step: macro and per-class AUROC, mean |change in probability| against
the clean reference, and the fraction of records whose five yes/no
decisions (the app's thresholds) all stay the same.

    bash ml/wsl_run.sh python ml/clf_degradation.py

Writes results/ml/clf_degradation.csv + .json and clf_sheets.csv + .json.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import clf_data as D  # noqa: E402
import export_classifier as EX  # noqa: E402
import train_classifier as TC  # noqa: E402
from ecgproto import clf_input as CI  # noqa: E402
from ecgproto import ptbxl  # noqa: E402

RESULTS = REPO / "results" / "ml"
EXPORTS = REPO / "data" / "device_exports"
SEED = 20261012
FS = 100

LADDER = {
    "gain": [1.0, 0.9, 0.83, 0.7, 0.6, 0.5],
    "shift_ms": [0, 5, 10, 20, 40, 80],        # per window, uniform in +-level
    "stretch_pct": [0, 0.5, 1, 2, 5],          # per window, uniform in +-level
    "noise_mv": [0, 0.01, 0.02, 0.05, 0.1],
    "gap_pct": [0, 2, 5, 10, 25],              # one contiguous run per window
}


# ----------------------------------------------------------------- the model

class Deployed:
    """The .tflite the app ships, run one record at a time (as on the phone)."""

    def __init__(self):
        import tensorflow as tf
        self.meta = json.loads((EX.ASSETS / EX.ASSET_META).read_text(encoding="utf-8"))
        self.classes = self.meta["classes"]
        self.thr = np.array([self.meta["thresholds"][c] for c in self.classes])
        it = tf.lite.Interpreter(model_path=str(EX.ASSETS / EX.ASSET_MODEL), num_threads=8)
        self.run = it.get_signature_runner()
        self.name = list(self.run.get_input_details())[0]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self.run(**{self.name: np.asarray(x, np.float32)[None]})["prob"][0]
                         for x in X]).astype(np.float64)


def step_metrics(Y, P, Pref, thr, classes) -> dict:
    per = {c: TC.auroc(Y[:, k], P[:, k]) for k, c in enumerate(classes)}
    return {"n": int(len(Y)), "macro_auroc": float(np.nanmean(list(per.values()))),
            **{f"auroc_{c}": v for c, v in per.items()},
            "mean_abs_dp": float(np.abs(P - Pref).mean()),
            "decisions_same": float(np.all((P >= thr) == (Pref >= thr), axis=1).mean())}


# ----------------------------------------------------------------- degradations

def degrade(x: np.ndarray, rng, gain=1.0, shift_ms=0.0, stretch_pct=0.0, noise_mv=0.0,
            gap_pct=0.0) -> tuple[np.ndarray, np.ndarray]:
    """True (1000, 12) at 100 Hz -> (windows (1000, 12), rhythm (1000,)) as
    a digitiser might read them: every window independently shifted and
    stretched (reading a neighbouring piece of the true signal), scaled,
    noised and holed. Outside its window a lead is NaN (unused)."""
    t = np.arange(D.N, dtype=np.float64)
    out = np.full((D.N, 12), np.nan)

    def read(sig, a, b):
        n = b - a
        sh = rng.uniform(-shift_ms, shift_ms) * FS / 1000.0
        st = 1.0 + rng.uniform(-stretch_pct, stretch_pct) / 100.0
        v = np.interp(a + np.arange(n) * st + sh, t, sig) * gain
        if noise_mv:
            v = v + rng.normal(0.0, noise_mv, n)
        g = int(round(n * gap_pct / 100.0))
        if g:
            s = int(rng.integers(0, n - g + 1))
            v[s:s + g] = np.nan
        return v

    for lead, col in CI.PRINTED_COLUMN.items():
        a, b = col * CI.CELL, (col + 1) * CI.CELL
        out[a:b, lead] = read(x[:, lead], a, b)
    rhythm = read(x[:, CI.LEAD_II], 0, D.N)
    return out, rhythm


def measured_errors() -> dict:
    """The phone's own error sizes, from the committed scores (learned, 3x4 leads)."""
    import pandas as pd
    s = pd.read_csv(REPO / "results" / "device" / "sheet_scores.csv")
    s = s[s["tracer"].astype(str).str.startswith("model") & ~s["rhythm"].astype(bool)]
    # Shifts the scorer itself marks unreliable (a lost lead: the 3x3-kernel
    # run's +-160 ms) are not timing errors and would set the spread alone.
    shift = pd.to_numeric(s.loc[s["shift_status"] == "ok", "shift_ms"], errors="coerce").dropna()
    cov = pd.to_numeric(s["coverage"], errors="coerce").dropna()
    gain = pd.to_numeric(s["gain_ratio"], errors="coerce").dropna()
    # Uniform in +-L has sd L/sqrt(3): L chosen so the sd matches the measured one.
    return {"gain": round(float(gain.median()), 3),
            "shift_ms": round(float(shift.std() * np.sqrt(3)), 1),
            "stretch_pct": 0.0,
            "noise_mv": 0.02,
            "gap_pct": round(float(max(0.0, 1.0 - cov.median()) * 100), 2),
            "source": "results/device/sheet_scores.csv (learned, 3x4 leads, shift_status ok; noise assumed)",
            "n_shifts": int(len(shift)),
            "n_leads": int(len(s))}


# ----------------------------------------------------------------- main

def write(rows, name, extra):
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    out.with_suffix(".json").write_text(json.dumps({
        "generated_by": "ml/clf_degradation.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": TC.git("rev-parse", "HEAD"),
        "git_dirty": bool(TC.git("status", "--porcelain", "--", "ml", "python")),
        **extra,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=D.CACHE)
    ap.add_argument("--n500", type=int, default=0, help="first N test records for RESAMPLE (0 = all)")
    args = ap.parse_args(argv)

    data = D.load(args.cache)
    keep = (data["y_super"].sum(1) > 0) & (data["fold"] == 10)
    idx = np.flatnonzero(keep)
    X = data["X"][idx].astype(np.float64)
    Y = data["y_super"][idx].astype(np.float32)
    ids = data["ecg_id"][idx]
    model = Deployed()
    classes, thr = model.classes, model.thr
    print(f"test {len(idx)} records; model {model.meta['model']} ({model.meta['run']})")

    t0 = time.time()
    ref = np.stack([CI.printed_view(x) for x in X])
    Pref = model(ref)
    rows = [{"step": "reference", "level": "", **step_metrics(Y, Pref, Pref, thr, classes)}]
    print(rows[-1], f"{time.time() - t0:.0f} s", flush=True)

    # 1. RESAMPLE: the phone's input path on true 500 Hz signals.
    df = ptbxl.load_metadata()
    sel = np.arange(len(ids)) if not args.n500 else np.arange(min(args.n500, len(ids)))
    V = []
    for i in sel:
        sig, _ = ptbxl.load_record(int(ids[i]), sampling_rate=500, df=df)
        cols = {}
        for lead, col in CI.PRINTED_COLUMN.items():
            c = np.full(5000, np.nan)
            c[col * 1250:(col + 1) * 1250] = sig[col * 1250:(col + 1) * 1250, lead]
            cols[CI.LEADS[lead]] = c
        cols[CI.RHYTHM] = sig[:, CI.LEAD_II]
        V.append(CI.from_columns(cols))
    P = model(np.stack(V))
    rows.append({"step": "resample", "level": "500->100 block mean",
                 **step_metrics(Y[sel], P, Pref[sel], thr, classes)})
    print(rows[-1], f"{time.time() - t0:.0f} s", flush=True)

    # 2. LADDER: one error at a time, then the measured combination.
    def run_step(name, level, **kw):
        rng = np.random.default_rng(SEED)
        Vs = np.stack([CI.printed_view(*degrade(x, rng, **kw)) for x in X])
        r = {"step": name, "level": level, **step_metrics(Y, model(Vs), Pref, thr, classes)}
        rows.append(r)
        print(r["step"], r["level"], round(r["macro_auroc"], 4), round(r["mean_abs_dp"], 4),
              round(r["decisions_same"], 3), f"{time.time() - t0:.0f} s", flush=True)

    for kind, levels in LADDER.items():
        for lv in levels:
            run_step(kind, lv, **{kind: lv})
    meas = measured_errors()
    run_step("measured", "phone", **{k: meas[k] for k in LADDER})
    write(rows, "clf_degradation", {"params": {"seed": SEED, "ladder": LADDER, "measured": meas,
                                               "n500": int(len(sel)), "model": model.meta}})

    # 3. SHEETS: phone exports of printed test sheets.
    import score_device_sheets as SDS
    pos = {int(e): k for k, e in enumerate(data["ecg_id"])}
    srows = []
    for path in sorted(EXPORTS.glob("eacg_sheet_*.csv")):
        meta, cols = SDS.read_export(path)
        m = re.match(r"ptbxl(\d+)_", meta.get("target", ""))
        if not m or int(m.group(1)) not in pos:
            continue
        eid = int(m.group(1))
        k = pos[eid]
        p_true = model(CI.printed_view(data["X"][k].astype(np.float64))[None])[0]
        p_dig = model(CI.from_columns(cols)[None])[0]
        r = {"export": path.name, "ecg_id": eid, "fold": int(data["fold"][k]),
             "tracer": "learned" if meta.get("tracer", "").startswith("model") else "classical",
             "labels": " ".join(c for c, y in zip(classes, data["y_super"][k]) if y) or "none"}
        for j, c in enumerate(classes):
            r[f"p_true_{c}"] = float(p_true[j])
            r[f"p_dig_{c}"] = float(p_dig[j])
        r["true_says"] = " ".join(c for j, c in enumerate(classes) if p_true[j] >= thr[j]) or "none"
        r["dig_says"] = " ".join(c for j, c in enumerate(classes) if p_dig[j] >= thr[j]) or "none"
        r["decisions_same"] = bool(np.all((p_true >= thr) == (p_dig >= thr)))
        r["mean_abs_dp"] = float(np.abs(p_true - p_dig).mean())
        phone = meta.get("diagnosis", "")
        got = dict(re.findall(r"(\w+)=([0-9.]+)", phone))
        r["phone_max_abs_dp"] = (float(max(abs(float(got[c]) - p_dig[j]) for j, c in enumerate(classes)))
                                 if all(c in got for c in classes) else "")
        srows.append(r)
        print(r["export"], r["labels"], "| true:", r["true_says"], "| digitised:", r["dig_says"], flush=True)
    if srows:
        write(srows, "clf_sheets", {"params": {"exports": str(EXPORTS.relative_to(REPO)), "model": model.meta}})
    else:
        print("no phone exports of known records found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
