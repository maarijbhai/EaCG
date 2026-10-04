#!/usr/bin/env python3
"""Is there a time-scale error inside each 3x4 row, and does a per-row time
map remove it?

The phone's full-sheet path on the desktop (ecgproto.sheet.digitise_sheet:
3,200 px decode, automatic corners), both geometries, both tracers
(classical and the app's b3 FP16 LiteRT file). Each sheet is traced ONCE per
geometry and tracer; the same row traces are then cut under every time map
in ecgproto.timemap.MODES (sheet.retime), so only the time map varies.

Per lead: best shift against the truth (timemap.best_shift: +-200 ms, a shift
at the window edge is FAILED, one below PCC 0.5 UNRELIABLE; both excluded
from the fits), plus score_lead's PCC, SNR and shifted SNR. The rhythm strip
is cut into four 2.5 s windows for the same shift-vs-position fit.

Per row: shift = intercept + slope * column, per photo, and pooled over the
photos with each photo's own intercept (within-photo slope, 95% CI).

    bash ml/wsl_run.sh python ml/run_timemap.py

Writes results/kitphoto/timemap_leads.csv, timemap_rows.csv (+ .json).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "python"))

import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
import score_12lead as S12  # noqa: E402
from ecgproto import segment, sheet, timemap  # noqa: E402

ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "ml"
OUT = REPO / "results" / "kitphoto"
FS = S12.FS
WIN = int(segment.CELL_S * FS)


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def lead_rows(res, truth, head) -> list[dict]:
    out = []
    for cell in res.layout.cells:
        rhythm = cell.row == 3
        sig = res.leads[(cell.name, rhythm)]
        tr = truth.leads[(cell.name, rhythm)]["mv"]
        avail = cell.row not in res.unavailable
        base = {**head, "lead": cell.name, "rhythm": rhythm, "row": cell.row,
                "available": avail, "unavailable_reason": res.unavailable.get(cell.row, "")}
        if not avail:
            out.append({**base, "col": cell.col, "shift_status": "UNAVAILABLE"})
            continue
        sc = S12.score_lead(sig, tr, segment.RECORDING_S if rhythm else segment.CELL_S)
        keep = {k: sc.get(k, "") for k in ("coverage", "pcc_asis", "snr_db_asis",
                                           "shifted_snr_db_asis", "pcc_aligned")}
        windows = range(4) if rhythm else [None]
        for w in windows:
            if w is None:
                col, t_, s_ = cell.col, tr, sig
            else:
                col, t_, s_ = w, tr[w * WIN:(w + 1) * WIN], sig[w * WIN:(w + 1) * WIN]
            sh = timemap.best_shift(t_, s_, FS)
            out.append({**base, **(keep if w in (None, 0) else {}),
                        "col": col, "window": "" if w is None else w,
                        "shift_ms": round(sh.shift_ms, 1) if np.isfinite(sh.shift_ms) else "",
                        "shift_pcc": round(sh.pcc, 4) if np.isfinite(sh.pcc) else "",
                        "shift_status": sh.status})
    return out


def row_fits(leads: list[dict]) -> list[dict]:
    """Per photo-row fits, and pooled within-photo slopes per row."""
    keyf = lambda r: (r["geometry"], r["tracer"], r["time_map"])
    groups = {}
    for r in leads:
        if r.get("shift_status") is None:
            continue
        groups.setdefault(keyf(r) + (r["row"],), []).append(r)
    out = []
    for (geo, tr, tm, row), rs in sorted(groups.items()):
        pooled_c, pooled_s = [], []
        for photo in sorted({r["photo"] for r in rs}):
            pts = [r for r in rs if r["photo"] == photo]
            ok = [r for r in pts if r["shift_status"] == "ok"]
            f = timemap.fit_row([r["col"] for r in ok], [r["shift_ms"] for r in ok])
            out.append({"geometry": geo, "tracer": tr, "time_map": tm, "row": row,
                        "photo": photo, "n_leads": len(pts), "n_used": f.n,
                        "n_failed": sum(r["shift_status"] == "FAILED" for r in pts),
                        "n_unreliable": sum(r["shift_status"] == "UNRELIABLE" for r in pts),
                        "n_unavailable": sum(r["shift_status"] == "UNAVAILABLE" for r in pts),
                        **_fmt(f)})
            if f.n >= timemap.MIN_FIT_POINTS:
                c = np.array([r["col"] for r in ok], float)
                s = np.array([r["shift_ms"] for r in ok], float)
                pooled_c += list(c - c.mean())
                pooled_s += list(s - s.mean())
        out.append({"geometry": geo, "tracer": tr, "time_map": tm, "row": row,
                    "photo": "POOLED", **_pooled(np.array(pooled_c), np.array(pooled_s))})
    return out


def _fmt(f: timemap.RowFit) -> dict:
    r = lambda v, d: round(v, d) if np.isfinite(v) else ""
    return {"slope_ms_per_cell": r(f.slope_ms_per_cell, 2), "intercept_ms": r(f.intercept_ms, 1),
            "r2": r(f.r2, 3), "scale_error_pct": r(f.scale_error_pct, 3)}


def _pooled(c: np.ndarray, s: np.ndarray) -> dict:
    """Slope through demeaned points (each photo its own intercept), with a
    95% CI from the residual scatter."""
    if len(c) < 3 or (c ** 2).sum() == 0:
        return {"n_used": len(c)}
    b = float((c * s).sum() / (c ** 2).sum())
    res = s - b * c
    dof = max(1, len(c) - 1)
    se = float(np.sqrt((res ** 2).sum() / dof / (c ** 2).sum()))
    tot = float((s ** 2).sum())
    return {"n_used": len(c), "slope_ms_per_cell": round(b, 2),
            "slope_ci95": round(1.96 * se, 2),
            "r2": round(1 - float((res ** 2).sum()) / tot, 3) if tot > 0 else "",
            "scale_error_pct": round(100 * b / (1000 * segment.CELL_S), 3)}


def write(rows: list[dict], name: str, params: dict) -> Path:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    p = OUT / f"{name}.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    p.with_suffix(".json").write_text(json.dumps({
        "generated_by": "ml/run_timemap.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "python", "ml")),
        "params": params,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    return p


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="b3_512x1024_fp16")
    p.add_argument("--decode-px", type=int, default=3200)
    args = p.parse_args(argv)
    from export_tflite import TfliteTracer
    tfl = TfliteTracer(ASSETS / f"{args.model}.tflite")
    height = int(tfl.run.get_input_details()[tfl.in_name]["shape"][1])
    tracers = {"classical": sheet.classical_row_tracer(),
               "b3": sheet.model_row_tracer(tfl, height)}
    leads = []
    for stem, e in sorted(skp.load_registry()["photos"].items()):
        t = skp.Target(e["target"])
        truth = S12.SheetTruth(t)
        photo = rp.read_oriented(skp.PHOTOS / e.get("file", f"{stem}.jpg"))
        sc = args.decode_px / max(photo.shape[:2])
        if sc < 1:
            photo = cv2.resize(photo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        for geo in ("corners", "grid"):
            for tname, tr in tracers.items():
                res0 = sheet.digitise_sheet(photo, tr, None, geo)
                for mode in timemap.MODES:
                    res = sheet.retime(res0, mode)
                    head = {"photo": stem, "geometry": res.geometry, "tracer": tname,
                            "time_map": mode, "calibration": res.calibration,
                            "ticks_found": res.ticks_found}
                    rows = lead_rows(res, truth, head)
                    leads += rows
                    cells = [r for r in rows if not r["rhythm"] and r.get("pcc_asis") not in (None, "")]
                    med = lambda k: np.median([float(r[k]) for r in cells]) if cells else float("nan")
                    print(f"{stem} {geo:7s} {tname:9s} {mode:12s} unavailable "
                          f"{sorted(res.unavailable) or '-'}  PCC {med('pcc_asis'):.3f}  "
                          f"SNR {med('snr_db_asis'):.1f} / {med('shifted_snr_db_asis'):.1f} dB",
                          flush=True)
    params = {"model": args.model, "decode_px": args.decode_px, "corners": "auto",
              "modes": list(timemap.MODES), "max_shift_s": timemap.MAX_SHIFT_S,
              "min_pcc": timemap.MIN_PCC, "min_fit_points": timemap.MIN_FIT_POINTS}
    print(write(leads, "timemap_leads", params))
    print(write(row_fits(leads), "timemap_rows", params))
    return 0


if __name__ == "__main__":
    import os
    code = main()
    sys.stdout.flush()
    os._exit(code)
