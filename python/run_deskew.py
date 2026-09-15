#!/usr/bin/env python3
"""Measurement driver for geometric rectification (deskew).

The code in `ecgproto/deskew.py` is not the deliverable; these numbers are.
Each experiment writes a CSV plus a JSON sidecar carrying the git commit,
the parameters and the library versions, because the audit found that no
result in this project was persisted anywhere and every reported figure had
to be regenerated to be checked.

Results land in `results/deskew/` and are **version controlled**, unlike
`out/`, which is generated-and-disposable. A measurement that is not in the
repository cannot be cited in a thesis six months from now.

Experiments::

    angle      angle recovery accuracy over a sweep of true rotations
    measures   which sharpness objective discriminates better
    endtoend   full pipeline: unrotated vs rotated vs rotated-then-deskewed
    interp     nearest/bilinear/bicubic/lanczos, on pitch, contrast and PCC
    failure    where it breaks: angle, blur, JPEG, faint grid, partial grid
    cost       wall-clock, against estimate_grid_pitch as the reference stage
    guards     B1 calibration guards, and confidence vs spacing_cv

Usage::

    python python/run_deskew.py --experiment all
    python python/run_deskew.py --experiment angle endtoend --n 12
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import scipy

from ecgproto import deskew as dsk
from ecgproto import digitise as dig
from ecgproto import evaluate, ptbxl, synth

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "deskew"

#: The three bundled fixtures, with the px/mm each was rendered at. Exact by
#: construction (dpi/25.4) and confirmed by measuring the major grid rules
#: back out of the pixels to within 0.004%. See ReferenceImages.kt.
FIXTURES = {
    "gt_norm_150dpi": (REPO / "android/app/src/main/assets/reference/gt_norm_150dpi.png", 150 / 25.4),
    "gt_mi_200dpi": (REPO / "android/app/src/main/assets/reference/gt_mi_200dpi.png", 200 / 25.4),
    "gt_sttc_300dpi": (REPO / "android/app/src/main/assets/reference/gt_sttc_300dpi.png", 300 / 25.4),
}

#: True rotations swept for angle recovery. Dense below one degree because
#: that is where the damage to PCC is already severe and where sub-step
#: interpolation of the objective has to earn its place; sparse above three
#: degrees where the question is only whether it still locks on at all.
ANGLE_SWEEP = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0,
               -0.25, -0.75, -1.5, -3.0, -5.0, -8.0)


def git_commit() -> tuple[str, bool]:
    """Current commit and whether the tree is dirty."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def provenance(experiment: str, params: dict) -> dict:
    sha, dirty = git_commit()
    return {
        "experiment": experiment,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": sha,
        "git_dirty": dirty,
        "params": params,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }


def persist(experiment: str, rows: list[dict], params: dict) -> Path:
    """Write rows to CSV and provenance to a JSON sidecar."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"{experiment}.csv"
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    meta = provenance(experiment, params)
    meta["n_rows"] = len(rows)
    (RESULTS / f"{experiment}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return csv_path


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def minor_grid_contrast(
    image: np.ndarray, px_per_mm: float, crop_px: int = 1000
) -> float:
    """Modulation depth of the 1 mm rules, relative to mean ink.

    The projection of the grid response onto x is a comb at the minor
    pitch. The magnitude of its Fourier component at exactly that frequency,
    normalised by the mean of the profile, is proportional to how *visible*
    the 1 mm rules still are and is insensitive to overall exposure.
    Interpolation blurs the rules together and drives it down; that is the
    cost this function exists to price.

    **Measured on a fixed centred crop.** `rotate_image(expand=True)` grows
    the canvas and pads it with white, which contributes zero grid response
    and so drags the normalising mean down -- an image can then score
    *higher* contrast after being degraded, purely because it acquired
    blank margins. A fixed crop makes the before and after numbers
    statistics of the same amount of paper.
    """
    h, w = image.shape[:2]
    side = min(crop_px, h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    patch = image[y0:y0 + side, x0:x0 + side]

    prof = dig.redness(patch).sum(axis=0).astype(np.float64)
    denom = float(prof.mean())
    if len(prof) < 16 or px_per_mm <= 0 or denom <= 0:
        return float("nan")
    centred = prof - denom
    freqs = np.fft.rfftfreq(len(centred), d=1.0)
    mag = np.abs(np.fft.rfft(centred * np.hanning(len(centred))))
    k = int(np.argmin(np.abs(freqs - 1.0 / px_per_mm)))
    lo, hi = max(1, k - 2), min(len(mag), k + 3)
    return float(mag[lo:hi].max() / denom)


def strip_for(signal: np.ndarray, fs: float, px_per_mm: float):
    return synth.render_printout(signal, fs=fs, px_per_mm=px_per_mm,
                                 on_overflow="expand")


def safe_digitise(image, fs_out, expected_duration_s=None):
    """Digitise, returning (result, error_string). Guards are non-fatal here.

    `strict=False` on purpose: these experiments are *about* the conditions
    under which calibration fails, so a rejected estimate has to be scored
    and reported rather than raised past.
    """
    try:
        res = dig.digitise(image, fs_out=fs_out, strict=False,
                           expected_duration_s=expected_duration_s)
        return res, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_records(n: int, rate: int, root: Path):
    df = ptbxl.load_metadata(root)
    _, _, test = ptbxl.split_folds(df)
    out = []
    for ecg_id in test.index:
        if len(out) >= n:
            break
        try:
            sig, meta = ptbxl.load_record(ecg_id, root=root, sampling_rate=rate, df=df)
        except FileNotFoundError:
            continue
        out.append((int(ecg_id), sig, meta))
    return out


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def exp_angle(args) -> list[dict]:
    """Angle recovery accuracy on the three known-scale fixtures."""
    rows = []
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for truth in ANGLE_SWEEP:
            rot = dsk.rotate_image(img, truth, interpolation=args.sim_interp)
            t0 = time.perf_counter()
            est = dsk.estimate_skew(rot, measure=args.measure)
            ms = (time.perf_counter() - t0) * 1000
            # Residual after correction: deskew the rotated image, then ask
            # the estimator again. A good correction leaves ~0.
            corrected = dsk.rotate_image(rot, -est["angle_deg"],
                                         interpolation=args.interp)
            residual = dsk.estimate_skew(corrected, measure=args.measure)["angle_deg"]
            rows.append({
                "fixture": name,
                "true_px_per_mm": round(true_px, 5),
                "true_angle_deg": truth,
                "recovered_deg": round(est["angle_deg"], 4),
                "abs_error_deg": round(abs(est["angle_deg"] - truth), 4),
                "residual_deg": round(residual, 4),
                "margin": round(est["margin"], 2),
                "axis_disagreement_deg": round(est["axis_disagreement_deg"], 4),
                "coarse_deg": est["coarse_angle_deg"],
                "elapsed_ms": round(ms, 1),
            })
    return rows


def exp_measures(args) -> list[dict]:
    """Which sharpness objective discriminates better."""
    rows = []
    img = cv2.imread(str(FIXTURES["gt_mi_200dpi"][0]), cv2.IMREAD_COLOR)
    for measure in dsk.MEASURES:
        errs, margins, times = [], [], []
        for truth in ANGLE_SWEEP:
            rot = dsk.rotate_image(img, truth, interpolation=args.sim_interp)
            try:
                t0 = time.perf_counter()
                est = dsk.estimate_skew(rot, measure=measure)
                times.append((time.perf_counter() - t0) * 1000)
                errs.append(abs(est["angle_deg"] - truth))
                margins.append(est["margin"])
            except Exception:
                errs.append(float("nan"))
                margins.append(float("nan"))
        rows.append({
            "measure": measure,
            "max_abs_error_deg": round(float(np.nanmax(errs)), 4),
            "median_abs_error_deg": round(float(np.nanmedian(errs)), 5),
            "median_margin": round(float(np.nanmedian(margins)), 2),
            "min_margin": round(float(np.nanmin(margins)), 2),
            "median_ms": round(float(np.median(times)), 1) if times else float("nan"),
        })
    return rows


def exp_endtoend(args) -> list[dict]:
    """Unrotated vs rotated vs rotated-then-deskewed, through full digitisation."""
    rows = []
    records = load_records(args.n, args.rate, args.root)
    lead_i = ptbxl.LEAD_NAMES.index(args.lead)

    for ecg_id, sig, meta in records:
        s = sig[:, lead_i]
        image, spec = strip_for(s, meta.fs, args.px_per_mm)
        duration = len(s) / meta.fs

        def score(img, condition, angle_applied, recovered=float("nan"),
                  interp=args.interp):
            res, err = safe_digitise(img, meta.fs, expected_duration_s=duration)
            row = {
                "ecg_id": ecg_id, "lead": args.lead, "condition": condition,
                "applied_deg": angle_applied, "recovered_deg": recovered,
                "interp": interp, "error": err,
            }
            if res is None:
                row.update(dict.fromkeys(
                    ("snr_db", "shifted_snr_db", "rmse_mv", "pcc", "gain_ratio",
                     "px_per_mm_x", "px_per_mm_y", "spacing_cv", "confidence",
                     "calibration_ok"), float("nan")))
                return row
            m = evaluate.evaluate_all(s, res.signal_mv, fs=meta.fs)
            row.update({
                "snr_db": round(m.snr_db, 3),
                "shifted_snr_db": round(m.shifted_snr_db, 3),
                "rmse_mv": round(m.rmse_mv, 5),
                "pcc": round(m.pcc, 4),
                "gain_ratio": round(m.gain_ratio, 4),
                "px_per_mm_x": round(res.grid.px_per_mm_x, 4),
                "px_per_mm_y": round(res.grid.px_per_mm_y, 4),
                "spacing_cv": round(res.grid.spacing_cv, 4),
                "confidence": round(res.grid.confidence, 4),
                "calibration_ok": bool(res.calibration.ok) if res.calibration else None,
            })
            return row

        rows.append(score(image, "unrotated", 0.0, 0.0))

        for theta in args.angles:
            rot = dsk.rotate_image(image, theta, interpolation=args.sim_interp)
            rows.append(score(rot, "rotated", theta))

            d = dsk.deskew(rot, measure=args.measure, interpolation=args.interp)
            rows.append(score(d.image, "deskewed", theta, round(d.angle_deg, 4)))

            # Control: rotate and unrotate by the *known* angle. Two warps,
            # zero estimation error. Separates what interpolation costs from
            # what the angle estimate costs -- without it, any shortfall in
            # the deskewed condition is unattributable.
            back = dsk.rotate_image(rot, -theta, interpolation=args.interp)
            rows.append(score(back, "roundtrip_control", theta, theta))
    return rows


def exp_interp(args) -> list[dict]:
    """Interpolation method against pitch accuracy, grid contrast and PCC."""
    rows = []
    records = load_records(args.n_interp, args.rate, args.root)
    lead_i = ptbxl.LEAD_NAMES.index(args.lead)
    theta = args.angles[-1]

    # Part 1: grid contrast and pitch, on the fixtures (known true px/mm).
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        base_contrast = minor_grid_contrast(img, true_px)
        base_grid = dig.estimate_grid_pitch(img)
        rot = dsk.rotate_image(img, theta, interpolation=args.sim_interp)
        for interp in dsk.INTERPOLATIONS:
            d = dsk.deskew(rot, measure=args.measure, interpolation=interp)
            g = dig.estimate_grid_pitch(d.image)
            # These fixtures make `estimate_grid_pitch` report the 5 mm rule
            # rather than the 1 mm one (see the deskew report); the pitch
            # error against `true_px` is therefore ~400% in every row and
            # measures that bug, not this warp. Scoring against the pitch
            # the *unrotated* image returns isolates what interpolation
            # costs, which is the question here.
            contrast = minor_grid_contrast(d.image, true_px)
            rows.append({
                "part": "fixture",
                "fixture": name, "interp": interp, "applied_deg": theta,
                "true_px_per_mm": round(true_px, 5),
                "base_px_per_mm": round(base_grid.px_per_mm, 4),
                "deskewed_px_per_mm": round(g.px_per_mm, 4),
                "pitch_shift_vs_base_pct": round(
                    (g.px_per_mm - base_grid.px_per_mm) / base_grid.px_per_mm * 100, 3
                ) if np.isfinite(base_grid.px_per_mm) else float("nan"),
                "pitch_error_vs_true_pct": round(
                    (g.px_per_mm - true_px) / true_px * 100, 1),
                "base_contrast": round(base_contrast, 4),
                "deskewed_contrast": round(contrast, 4),
                "contrast_retained": round(contrast / base_contrast, 4)
                if base_contrast else float("nan"),
            })

    # Part 2: end-to-end PCC and gain, on rendered strips.
    for ecg_id, sig, meta in records:
        s = sig[:, lead_i]
        image, _ = strip_for(s, meta.fs, args.px_per_mm)
        duration = len(s) / meta.fs
        rot = dsk.rotate_image(image, theta, interpolation=args.sim_interp)
        for interp in dsk.INTERPOLATIONS:
            d = dsk.deskew(rot, measure=args.measure, interpolation=interp)
            res, err = safe_digitise(d.image, meta.fs, expected_duration_s=duration)
            row = {"part": "endtoend", "ecg_id": ecg_id, "interp": interp,
                   "applied_deg": theta, "error": err}
            if res is not None:
                m = evaluate.evaluate_all(s, res.signal_mv, fs=meta.fs)
                row.update({"pcc": round(m.pcc, 4),
                            "gain_ratio": round(m.gain_ratio, 4),
                            "snr_db": round(m.snr_db, 3),
                            "rmse_mv": round(m.rmse_mv, 5)})
            rows.append(row)
    return rows


def exp_failure(args) -> list[dict]:
    """Where it breaks. Each perturbation is applied to a known 2 degree skew."""
    rows = []
    img = cv2.imread(str(FIXTURES["gt_mi_200dpi"][0]), cv2.IMREAD_COLOR)
    true_px = FIXTURES["gt_mi_200dpi"][1]
    theta = 2.0

    def record(kind, level, image, truth=theta, **kw):
        try:
            est = dsk.estimate_skew(image, measure=args.measure, **kw)
            g = dig.estimate_grid_pitch(image)
            rows.append({
                "kind": kind, "level": level, "true_angle_deg": truth,
                "recovered_deg": round(est["angle_deg"], 4),
                "abs_error_deg": round(abs(est["angle_deg"] - truth), 4),
                "margin": round(est["margin"], 2),
                "axis_disagreement_deg": round(est["axis_disagreement_deg"], 4),
                "px_per_mm": round(g.px_per_mm, 4),
                "pitch_error_pct": round((g.px_per_mm - true_px) / true_px * 100, 2)
                if np.isfinite(g.px_per_mm) else float("nan"),
                "spacing_cv": round(g.spacing_cv, 4),
                "confidence": round(g.confidence, 4),
                "error": "",
            })
        except Exception as exc:
            rows.append({"kind": kind, "level": level, "true_angle_deg": truth,
                         "error": f"{type(exc).__name__}: {exc}"})

    # 1. maximum recoverable angle -- search widened so the limit measured is
    #    the objective's, not the sweep range's.
    for a in (10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 44.0, 46.0, 50.0):
        rot = dsk.rotate_image(img, a, interpolation=args.sim_interp)
        record("angle", a, rot, truth=a, coarse_half_range_deg=50.0,
               coarse_step_deg=0.5)

    rot2 = dsk.rotate_image(img, theta, interpolation=args.sim_interp)

    # 2. blur
    for sigma in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        k = int(2 * round(3 * sigma) + 1)
        record("blur_sigma", sigma, cv2.GaussianBlur(rot2, (k, k), sigma))

    # 3. JPEG
    for q in (95, 80, 60, 40, 20, 10, 5):
        ok, buf = cv2.imencode(".jpg", rot2, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        record("jpeg_quality", q, cv2.imdecode(buf, cv2.IMREAD_COLOR))

    # 4. faint / desaturated grid, toward the black-and-white photocopy case
    for frac in (0.75, 0.5, 0.25, 0.1, 0.0):
        grey = cv2.cvtColor(cv2.cvtColor(rot2, cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR)
        faded = cv2.addWeighted(rot2, frac, grey, 1 - frac, 0)
        record("grid_saturation", frac, faded)

    # 4b. The greyscale case again, with a response that does not key on
    #     red. `redness` is silent on a photocopy by construction, so the
    #     question worth answering is not whether the default fails -- it
    #     must -- but whether the *method* survives a different front end.
    def grey_response(image: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Ink is dark; subtract a local mean so that page shading and the
        # broad dark band of a trace do not swamp the thin rules.
        return np.clip(cv2.blur(g, (31, 31)) - g, 0.0, None)

    grey = cv2.cvtColor(cv2.cvtColor(rot2, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    record("grid_saturation_greyfallback", 0.0, grey, response_fn=grey_response)

    # 5. grid occupying only part of the frame; the rest is blank paper
    for frac in (0.75, 0.5, 0.3, 0.15):
        partial = np.full_like(rot2, 255)
        h, w = rot2.shape[:2]
        ch, cw = int(h * frac), int(w * frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        partial[y0:y0 + ch, x0:x0 + cw] = rot2[y0:y0 + ch, x0:x0 + cw]
        record("grid_coverage", frac, partial)

    return rows


def exp_cost(args) -> list[dict]:
    """Wall clock, against estimate_grid_pitch as the reference stage."""
    rows = []
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        for angle in (0.0, 3.0):
            rot = dsk.rotate_image(img, angle, interpolation=args.sim_interp)
            reps = 3
            sk = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dsk.estimate_skew(rot, measure=args.measure)
                sk.append((time.perf_counter() - t0) * 1000)
            gp = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dig.estimate_grid_pitch(rot)
                gp.append((time.perf_counter() - t0) * 1000)
            wp = []
            for _ in range(reps):
                t0 = time.perf_counter()
                dsk.rotate_image(rot, -angle, interpolation=args.interp)
                wp.append((time.perf_counter() - t0) * 1000)
            rows.append({
                "fixture": name, "megapixels": round(img.size / 3 / 1e6, 2),
                "applied_deg": angle,
                "estimate_skew_ms": round(float(np.median(sk)), 1),
                "estimate_grid_pitch_ms": round(float(np.median(gp)), 1),
                "single_warp_ms": round(float(np.median(wp)), 1),
                "ratio_skew_over_pitch": round(
                    float(np.median(sk)) / float(np.median(gp)), 2),
            })
    return rows


def exp_guards(args) -> list[dict]:
    """B1 guards, and whether confidence or spacing_cv discriminates."""
    rows = []
    img = cv2.imread(str(FIXTURES["gt_mi_200dpi"][0]), cv2.IMREAD_COLOR)

    conditions = [("clean", img)]
    for a in (1.0, 3.0, 6.0):
        conditions.append((f"rotated_{a}deg",
                           dsk.rotate_image(img, a, interpolation=args.sim_interp)))
    for sigma in (2.0, 5.0):
        k = int(2 * round(3 * sigma) + 1)
        conditions.append((f"blur_{sigma}", cv2.GaussianBlur(img, (k, k), sigma)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    conditions.append(("jpeg_q10", cv2.imdecode(buf, cv2.IMREAD_COLOR)))

    # A 3x4 fixture's full width is the 10 s rhythm strip, so 10 s is the
    # right expected duration and the paper-speed check is live for these.
    true_px = FIXTURES["gt_mi_200dpi"][1]
    for name, im in conditions:
        g = dig.estimate_grid_pitch(im)
        chk = dig.check_calibration(
            g, image_width_px=im.shape[1], expected_duration_s=10.0
        )
        rows.append({
            "condition": name,
            "px_per_mm_x": round(g.px_per_mm_x, 4),
            "px_per_mm_y": round(g.px_per_mm_y, 4),
            "true_px_per_mm": round(true_px, 4),
            "pitch_error_x_pct": round((g.px_per_mm_x - true_px) / true_px * 100, 1)
            if np.isfinite(g.px_per_mm_x) else float("nan"),
            "anisotropy": round(g.anisotropy, 4),
            "confidence": round(g.confidence, 4),
            "spacing_cv": round(g.spacing_cv, 4),
            "implied_mm_per_s": round(chk.implied_mm_per_s, 2),
            "calibration_ok": chk.ok,
            "reasons": " | ".join(chk.reasons),
            "warnings": " | ".join(chk.warnings),
        })

    # The 5x harmonic failure, induced by rendering below the resolution at
    # which 1 mm rules separate. This is the case the paper-speed check
    # exists for, and the only one nothing else catches.
    records = load_records(1, args.rate, args.root)
    if records:
        _, sig, meta = records[0]
        s = sig[:, ptbxl.LEAD_NAMES.index(args.lead)]
        duration = len(s) / meta.fs
        for px_mm in (8.0, 4.0, 2.5, 2.0):
            image, _ = strip_for(s, meta.fs, px_mm)
            g = dig.estimate_grid_pitch(image)
            chk = dig.check_calibration(
                g, image_width_px=image.shape[1], expected_duration_s=duration
            )
            rows.append({
                "condition": f"render_{px_mm}px_per_mm",
                "px_per_mm_x": round(g.px_per_mm_x, 4),
                "px_per_mm_y": round(g.px_per_mm_y, 4),
                "true_px_per_mm": px_mm,
                "pitch_error_x_pct": round(
                    (g.px_per_mm_x - px_mm) / px_mm * 100, 1)
                if np.isfinite(g.px_per_mm_x) else float("nan"),
                "anisotropy": round(g.anisotropy, 4),
                "confidence": round(g.confidence, 4),
                "spacing_cv": round(g.spacing_cv, 4),
                "implied_mm_per_s": round(chk.implied_mm_per_s, 2),
                "calibration_ok": chk.ok,
                "reasons": " | ".join(chk.reasons),
                "warnings": " | ".join(chk.warnings),
            })
    return rows


EXPERIMENTS = {
    "angle": exp_angle,
    "measures": exp_measures,
    "endtoend": exp_endtoend,
    "interp": exp_interp,
    "failure": exp_failure,
    "cost": exp_cost,
    "guards": exp_guards,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiment", nargs="+", default=["all"],
                   choices=["all", *EXPERIMENTS])
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--n", type=int, default=10, help="records for endtoend")
    p.add_argument("--n-interp", type=int, default=4, help="records for interp")
    p.add_argument("--rate", type=int, choices=(100, 500), default=100)
    p.add_argument("--lead", default="II")
    p.add_argument("--px-per-mm", type=float, default=8.0)
    p.add_argument("--angles", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--measure", default="ssd", choices=dsk.MEASURES)
    p.add_argument("--interp", default=dsk.DEFAULT_INTERPOLATION,
                   choices=list(dsk.INTERPOLATIONS),
                   help="interpolation used to APPLY the correction")
    p.add_argument("--sim-interp", default="lanczos",
                   choices=list(dsk.INTERPOLATIONS),
                   help="interpolation used to SIMULATE the test rotation; "
                        "held fixed so that varying --interp varies one thing")
    args = p.parse_args(argv)

    chosen = list(EXPERIMENTS) if "all" in args.experiment else args.experiment
    params = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items() if k != "experiment"}

    for name in chosen:
        t0 = time.perf_counter()
        print(f"\n=== {name} ===", flush=True)
        rows = EXPERIMENTS[name](args)
        path = persist(name, rows, params)
        print(f"{len(rows)} rows -> {path.relative_to(REPO)} "
              f"({time.perf_counter() - t0:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
