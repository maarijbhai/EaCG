#!/usr/bin/env python3
"""Learned tracer vs classical tracer on the REAL photographs, matched.

`docs/ml_plan.md` §4 decision rule 1 is judged here: on real photographs,
against the classical control "at the same corners and the same
calibration". So both tracers get exactly the same input and are scored by
exactly the same function:

    photo -> registered corners (python/kit_photos.json, as placed; the same
    corners the classical `known` rows of results/kitphoto/score.csv used)
    -> warp onto the printed sheet's own pixel grid (score_kit_photo.
    warp_to_sheet) -> the sheet's known calibration (300 dpi, t0 from the
    kit). Then:

    classical  the shipped FR5/FR6: binarise_trace (midpoint) + isolate_trace
               + column centroid over the known rhythm-strip box. Run with
               the phone's 2x2 opening AND the desktop's 3x3, and the better
               of the two by aligned PCC is kept per photo -- the control is
               given every benefit of the doubt.
    model B    strips.to_canonical on the warped sheet, centred on the
               classical trace's median row (what inference has; no truth),
               starting 4 mm before t0; the row head; soft-argmax; mapped
               back to sheet pixels.

Both are then scored only over the known box's columns:

    trace_rmse_mm   each sample's drawn row vs the tracer's row at that
                    sample's column (image space, no calibration)
    r_ratio_img     R-peak ratio in image space (the validation metric)
    pcc_asis, r_ratio, gain   vs the PTB-XL truth in mV at the known
                    calibration, sample 0 at the kit's t0
    pcc_aligned     after run_photo.best_alignment's scale/lag search,
                    reported only if the overlap is >= 8.5 s (the repo rule)

Corner error moves every one of these for both tracers alike; it is the
DIFFERENCE between the rows that is the tracer comparison.

    bash ml/wsl_run.sh python ml/eval_photos.py --run b0
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

import strips as S  # noqa: E402
import run_photo as rp  # noqa: E402
import score_kit_photo as skp  # noqa: E402
from ecgproto import digitise as dig, evaluate, segment  # noqa: E402

RESULTS = REPO / "results" / "ml"
DIAG = REPO / "out" / "ml_photo"
LEAD_IN_MM = 4.0


def classical_rows(strip: np.ndarray, open_kernel: int) -> np.ndarray:
    mask = segment.isolate_trace(dig.binarise_trace(strip, open_kernel=open_kernel))
    return dig.trace_positions(mask)


def model_rows(model, sheet: np.ndarray, mm: float, t0_col: float,
               center_row: float, head: str = "row"):
    """A model on the warped sheet -> (sheet x, sheet row) per output column.

    Model B: soft-argmax per output column (2 per input column). Model A:
    the predicted ink mask read by the classical column centroid, one per
    input column -- the same reading the classical tracer applies to its
    own threshold mask."""
    import train_tracer as T
    canon, cm = S.to_canonical(sheet, mm, t0_col - LEAD_IN_MM * mm, center_row)
    out = model(canon[..., ::-1][None].copy(), training=False)
    out = {k: np.asarray(v)[0] for k, v in out.items()}
    if head == "mask":
        row = T.decode_mask(out["mask"])
        c = np.arange(len(row), dtype=np.float64)
        x = cm.column_to_source_x(c)
        r = (row + cm.y0 + 0.5) / cm.k - 0.5
        return x, r, canon, cm
    row = T.decode_rows(out["rows"], out["presence"])
    j = np.arange(len(row))
    c = (j + 0.5) / S.OUT_UPSAMPLE - 0.5                 # canonical input col
    x = cm.column_to_source_x(c)
    r = (row + cm.y0 + 0.5) / cm.k - 0.5
    return x, r, canon, cm


def score(x: np.ndarray, r: np.ndarray, target, x0: int, x1: int) -> dict:
    """One scoring path for both tracers: (x, row) in sheet pixels."""
    keep = np.isfinite(r) & (x >= x0) & (x < x1)
    x, r = x[keep], r[keep]
    mm = target.px_per_mm
    # Coverage counts sheet columns holding at least one traced point, so a
    # tracer with two outputs per column is not credited twice.
    covered = np.unique(np.floor(x + 0.5).astype(int))
    out = {"cols_traced": int(len(covered)),
           "coverage": round(len(covered) / max(1, (x1 - x0)), 4)}
    if len(x) < 50:
        return out
    o = np.argsort(x)
    x, r = x[o], r[o]
    inside = (target.cols >= x[0]) & (target.cols <= x[-1])
    at = np.interp(target.cols[inside], x, r)
    err = (at - target.rows[inside]) / mm
    out["trace_rmse_mm"] = round(float(np.sqrt(np.mean(err ** 2))), 4)
    out["trace_bias_mm"] = round(float(np.mean(err)), 4)
    out["trace_p95_abs_mm"] = round(float(np.percentile(np.abs(err), 95)), 4)
    ref_img, test_img = -target.rows[inside], -at
    out["pcc_img"] = round(evaluate.pcc(ref_img, test_img), 4)
    out["r_ratio_img"] = round(evaluate.r_peak_ratio(ref_img, test_img, target.fs)[0], 4)

    # The waveform at the known calibration: 500 Hz from t0, mV up to offset.
    t = (x - target.t0_col) / (mm * dig.MM_PER_S)
    n = int(np.floor(t[-1] * target.fs)) + 1
    k0 = int(np.ceil(max(0.0, t[0]) * target.fs))
    ts = np.arange(k0, n) / target.fs
    mv = -(np.interp(ts, t, r) - np.median(r)) / (mm * dig.MM_PER_MV)
    truth = target.lead_signal[k0:n]
    m = evaluate.evaluate_all(truth, mv, target.fs)
    out["pcc_asis"] = round(float(m.pcc), 4)
    out["gain_ratio"] = round(float(m.gain_ratio), 4)
    out["r_ratio"] = round(evaluate.r_peak_ratio(truth, mv, target.fs)[0], 4)
    pcc_a, scale, lag, nn, at_bound = rp.best_alignment(truth, mv)
    overlap = nn / target.fs
    out["align_overlap_s"] = round(overlap, 3)
    out["align_scale_pct"] = round(100 * (scale - 1), 3)
    out["pcc_aligned"] = (round(float(pcc_a), 4) if overlap >= rp.MIN_OVERLAP_S
                          else "")
    # R ratio after alignment too, since a time-base error moves R peaks
    # outside the 25 ms window and reads as lost height.
    if overlap >= rp.MIN_OVERLAP_S:
        res = np.interp(np.arange(int(round(len(mv) / scale))) * scale,
                        np.arange(len(mv)), mv)
        if lag >= 0:
            a_ref, a_test = truth[lag:], res
        else:
            a_ref, a_test = truth, res[-lag:]
        kk = min(len(a_ref), len(a_test))
        out["r_ratio_aligned"] = round(evaluate.r_peak_ratio(
            a_ref[:kk], a_test[:kk], target.fs)[0], 4)
    else:
        out["r_ratio_aligned"] = ""
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="b0")
    p.add_argument("--weights", default="best")
    p.add_argument("--only", default="",
                   help="score only photos whose stem starts with this")
    args = p.parse_args(argv)

    import tensorflow as tf
    import keras
    import train_tracer as T
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    # Same policy the model was trained and validated under. In float32 a
    # full-width 384 x 2048 strip does not fit the ~1.8 GB WSL leaves usable
    # (measured: OOM in the dilated block). INT8/FP32 on the phone is a
    # separate measurement (§5).
    keras.mixed_precision.set_global_policy("mixed_float16")
    # The head comes from the run's own training record, not a flag, so a
    # run cannot be scored as the wrong kind of model.
    head = json.loads((RESULTS / f"train_{args.run}.json").read_text())["params"]["head"]
    model = T.build_model(head)
    wpath = REPO / "out" / "ml_runs" / args.run / f"{args.weights}.weights.h5"
    model.load_weights(wpath)

    reg = skp.load_registry()["photos"]
    DIAG.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, entry in sorted(reg.items()):
        if args.only and not stem.startswith(args.only):
            continue
        target = skp.Target(entry["target"])
        photo = rp.read_oriented(skp.PHOTOS / entry.get("file", f"{stem}.jpg"))
        corners = np.asarray(entry["corners_px"], np.float32)
        sheet = skp.warp_to_sheet(photo, corners, target)
        x0, y0, x1, y1 = target.known_box()
        strip = sheet[y0:y1, x0:x1]
        mm = target.px_per_mm
        info = {"photo": stem, "target": target.name, "ecg_id": target.ecg_id,
                "strat_fold": target.manifest["source"]["strat_fold"],
                "note": entry.get("note", "")[:80]}

        cls = {}
        for ok in (2, 3):
            cr = classical_rows(strip, ok) + y0
            xs = np.arange(x0, x1, dtype=np.float64)
            cls[ok] = (cr, score(xs, cr, target, x0, x1))
        def key(sc):
            v = sc.get("pcc_aligned")
            return v if v != "" and v is not None else -2.0
        best_ok = max(cls, key=lambda k: key(cls[k][1]))
        c_rows, c_score = cls[best_ok]
        rows.append({**info, "tracer": "classical", "open_kernel": best_ok,
                     **c_score})

        center = float(np.nanmedian(c_rows))
        mx, mr, canon, cm = model_rows(model, sheet, mm, target.t0_col, center,
                                       head)
        m_score = score(mx, mr, target, x0, x1)
        label = "model_A" if head == "mask" else "model_B"
        rows.append({**info, "tracer": f"{label}_{args.run}", "open_kernel": "",
                     **m_score})

        np.savez(DIAG / f"{stem}_traces.npz",
                 classical_x=np.arange(x0, x1, dtype=np.float64), classical_r=c_rows,
                 model_x=mx, model_r=mr, truth_x=target.cols, truth_r=target.rows,
                 truth_mv=target.lead_signal, t0_col=target.t0_col, px_per_mm=mm,
                 box=np.array([x0, y0, x1, y1]))
        # Overlay: truth (green), classical (red), model (blue), on the strip.
        view = strip.copy()
        for xx, rr in zip(target.cols, target.rows):
            if x0 <= xx < x1:
                cv2.circle(view, (int(xx - x0), int(rr - y0)), 1, (0, 170, 0), -1)
        for xx in range(x1 - x0):
            if np.isfinite(c_rows[xx]):
                view[int(np.clip(c_rows[xx] - y0, 0, view.shape[0] - 1)), xx] = (0, 0, 255)
        for xx, rr in zip(mx, mr):
            if np.isfinite(rr) and x0 <= xx < x1:
                yy = int(np.clip(rr - y0, 0, view.shape[0] - 1))
                view[yy, int(xx - x0)] = (255, 80, 0)
        cv2.imwrite(str(DIAG / f"{stem}_overlay.png"), view)
        print(f"{stem}: classical(k{best_ok}) "
              f"rmse {c_score.get('trace_rmse_mm')} mm  R {c_score.get('r_ratio_aligned')}"
              f"  PCCa {c_score.get('pcc_aligned')}  |  model "
              f"rmse {m_score.get('trace_rmse_mm')} mm  R {m_score.get('r_ratio_aligned')}"
              f"  PCCa {m_score.get('pcc_aligned')}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out = RESULTS / f"photo_compare_{args.run}{'_' + args.only if args.only else ''}.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip()
    out.with_suffix(".json").write_text(json.dumps({
        "generated_by": "ml/eval_photos.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "ml", "python")),
        "params": {"run": args.run, "head": head, "only": args.only,
                   "weights": str(wpath.relative_to(REPO)),
                   "lead_in_mm": LEAD_IN_MM, "corners": "python/kit_photos.json as placed",
                   "calibration": "known (sheet frame, 300 dpi)",
                   "classical": "binarise_trace midpoint + isolate_trace + centroid; "
                                "better of open_kernel 2 and 3 per photo by pcc_aligned",
                   "min_overlap_s": rp.MIN_OVERLAP_S},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__, "tensorflow": tf.__version__,
                     "keras": keras.__version__},
    }, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
