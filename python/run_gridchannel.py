#!/usr/bin/env python3
"""Measurement driver: which channel and which threshold recover the grid pitch.

Two implementations of grid pitch estimation disagree and had never been
reconciled. `digitise.estimate_grid_pitch` returns 39.26 px/mm against a
true 7.874 on the ecg-image-kit fixtures at its default `peak_fraction` of
0.20 -- a fivefold error, the 5 mm rule reported as the 1 mm rule -- and
needs `peak_fraction=0.03` to return the right answer. `GridEstimator.kt`,
the Android port, uses a threshold of 0.25 and reportedly gets it right on
the same images. The visible difference is the channel: the Kotlin reads
green, the Python a redness response.

The question is not which number is prettier. The Kotlin port has to match
the Python exactly or the thesis is characterising two different pipelines,
so whichever configuration is correct has to be the same in both. This
script separates the three things that differ between them -- **channel**,
**detrend**, and **what the threshold is a fraction of** -- and measures
each on the ecg-image-kit fixtures *and* on this repository's own synthetic
renders, because those two sources are known to differ and that difference
is the whole point.

It does not change any peak selection logic. `digitise._pitch_1d` is called
unmodified; only the profile handed to it varies.

Usage::

    python python/run_gridchannel.py
    python python/run_gridchannel.py --experiment contrast
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from ecgproto import digitise as dig
from ecgproto import ptbxl, synth
from ecgproto import grid_reference
from run_deskew import FIXTURES, provenance

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "grid"

#: Thresholds compared. 0.20 is the shipped Python default, 0.25 the shipped
#: Kotlin one, 0.03 the value CLAUDE.md records as necessary to get the
#: right answer out of the Python on the kit fixtures.
THRESHOLDS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01)

#: Synthetic render scales, bracketing the three fixtures' 5.91, 7.87 and
#: 11.81 px/mm so that source and scale are not confounded.
SYNTH_PX_PER_MM = (6.0, 8.0, 12.0, 16.0)

#: Moving-average window used by `GridEstimator.kt` for its high-pass
#: detrend. Reproduced here rather than imported so that the Kotlin constant
#: and the Python one cannot silently drift apart.
DETREND_WINDOW = grid_reference.DETREND_WINDOW


# --------------------------------------------------------------------------
# the four channel responses under test
# --------------------------------------------------------------------------

def resp_redness(img: np.ndarray) -> np.ndarray:
    """What `digitise.redness` computes: red saturation. Silent on the trace."""
    return dig.redness(img)


def resp_green(img: np.ndarray) -> np.ndarray:
    """What `GridEstimator.kt` reads: 255 - G. High on the grid *and* the trace."""
    return 255.0 - img[:, :, 1].astype(np.float64)


def resp_blue(img: np.ndarray) -> np.ndarray:
    """255 - B. Included because a red grid darkens B as much as G."""
    return 255.0 - img[:, :, 0].astype(np.float64)


def resp_luma(img: np.ndarray) -> np.ndarray:
    """255 - luminance. The channel-agnostic fallback, for a photocopy."""
    return 255.0 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


RESPONSES = {
    "redness": resp_redness,
    "green": resp_green,
    "blue": resp_blue,
    "luma": resp_luma,
}


def profile_x(img: np.ndarray, response: str, detrend: str) -> np.ndarray:
    """Column projection profile of `img`, for one channel and detrend.

    Projected onto x, so the periodicity measured is that of the *vertical*
    rules. `sum` versus `mean` over rows is a constant factor and vanishes
    when the autocorrelation is normalised by its zero lag, so the two
    implementations' disagreement on that point cannot matter and is not
    swept.
    """
    prof = RESPONSES[response](img).sum(axis=0).astype(np.float64)
    if detrend == "mean":
        return prof - prof.mean()
    if detrend == "movavg":
        # Same box high-pass as GridEstimator.kt, edges shortened the same
        # way (the window is clipped at the image border rather than padded).
        n = len(prof)
        cs = np.concatenate(([0.0], np.cumsum(prof)))
        half = DETREND_WINDOW // 2
        a = np.maximum(0, np.arange(n) - half)
        b = np.minimum(n, np.arange(n) + half + 1)
        local = (cs[b] - cs[a]) / (b - a)
        return prof - local
    raise ValueError(detrend)


def acf(profile: np.ndarray) -> np.ndarray:
    """Zero-lag-normalised autocorrelation, as `digitise._pitch_1d` computes it."""
    p = profile - profile.mean()
    n = 1 << int(np.ceil(np.log2(len(p) * 2)))
    spec = np.fft.rfft(p, n=n)
    a = np.fft.irfft(spec * np.conj(spec), n=n)[: len(p)]
    return a / a[0] if a[0] > 0 else a


def peak_height(a: np.ndarray, lag: float) -> float:
    """Height of the autocorrelation at the peak nearest `lag`."""
    k = int(round(lag))
    lo, hi = max(1, k - 2), min(len(a) - 1, k + 3)
    return float(a[lo:hi].max())


# --------------------------------------------------------------------------
# image sources
# --------------------------------------------------------------------------

def sources(args):
    """(name, kind, image, true_px_per_mm) for both image families."""
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            yield name, "ecg-image-kit", img, true_px

    df = ptbxl.load_metadata(args.root)
    _, _, test = ptbxl.split_folds(df)
    ecg_id = int(test.index[0])
    sig, meta = ptbxl.load_record(ecg_id, root=args.root,
                                  sampling_rate=args.rate, df=df)
    s = sig[:, ptbxl.LEAD_NAMES.index(args.lead)]
    for px in SYNTH_PX_PER_MM:
        img, _ = synth.render_printout(s, fs=meta.fs, px_per_mm=px,
                                       on_overflow="expand")
        yield f"synth_{px:g}px_per_mm", "synth", img, px


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def exp_contrast(args) -> list[dict]:
    """Minor-to-major contrast, per channel, per detrend, per image family.

    Two readings of "contrast", because they answer different questions:

    * **profile**: the ratio of the Fourier magnitudes of the projection
      profile at the 1 mm and 5 mm frequencies. This is how visible the
      minor rules are in the ink, and it is a property of the printed
      image.
    * **acf**: the ratio of the autocorrelation heights at the 1 mm and
      5 mm lags. This is the quantity peak selection actually thresholds
      on, and it is what decides whether the estimate is right or five
      times wrong.

    Also recorded is `required_fraction`: the largest threshold at which
    the four minor peaks below the first major one all still survive. That
    is the number a default has to sit under, and comparing it across
    channels answers the question directly -- does a different channel
    remove the need for a low threshold, or only move where it has to sit?
    """
    rows = []
    for name, kind, img, true_px in sources(args):
        h, w = img.shape[:2]
        for response in RESPONSES:
            for detrend in ("mean", "movavg"):
                prof = profile_x(img, response, detrend)
                if not np.any(prof):
                    continue
                a = acf(prof)

                minor_lag, major_lag = true_px, 5.0 * true_px
                h_minor = peak_height(a, minor_lag)
                h_major = peak_height(a, major_lag)

                # The four minor peaks strictly inside the first major one.
                inner = [peak_height(a, k * true_px) for k in (1, 2, 3, 4)]
                required = float(min(inner))

                # Fourier view of the same two components.
                win = prof * np.hanning(len(prof))
                mag = np.abs(np.fft.rfft(win))
                freqs = np.fft.rfftfreq(len(prof), d=1.0)

                def band(f):
                    k = int(np.argmin(np.abs(freqs - f)))
                    lo, hi = max(1, k - 2), min(len(mag), k + 3)
                    return float(mag[lo:hi].max())

                f_minor = band(1.0 / true_px)
                f_major = band(1.0 / (5.0 * true_px))

                rows.append({
                    "source": name, "kind": kind,
                    "rows_px": h, "cols_px": w,
                    "true_px_per_mm": round(true_px, 5),
                    "response": response, "detrend": detrend,
                    "profile_mag_minor": round(f_minor, 1),
                    "profile_mag_major": round(f_major, 1),
                    "profile_minor_over_major": round(f_minor / f_major, 4)
                    if f_major > 0 else float("nan"),
                    "acf_h_minor": round(h_minor, 4),
                    "acf_h_major": round(h_major, 4),
                    "acf_minor_over_major": round(h_minor / h_major, 4)
                    if h_major > 0 else float("nan"),
                    "required_fraction": round(required, 4),
                })
    return rows


def exp_threshold(args) -> list[dict]:
    """What pitch comes back, per channel, per detrend, per threshold.

    The contrast numbers predict where the threshold has to sit; this is
    the check that it does. `digitise._pitch_1d` is called unmodified, so
    the peak selection under test is the shipped one and the only thing
    varying is the profile handed to it and the threshold.
    """
    rows = []
    for name, kind, img, true_px in sources(args):
        for response in RESPONSES:
            for detrend in ("mean", "movavg"):
                prof = profile_x(img, response, detrend)
                if not np.any(prof):
                    continue
                for frac in THRESHOLDS:
                    pitch, n_peaks, conf, cv_ = dig._pitch_1d(
                        prof, 3.0, 80.0, frac)
                    err = ((pitch - true_px) / true_px * 100
                           if np.isfinite(pitch) else float("nan"))
                    rows.append({
                        "source": name, "kind": kind,
                        "true_px_per_mm": round(true_px, 5),
                        "response": response, "detrend": detrend,
                        "peak_fraction": frac,
                        "pitch_px_per_mm": round(pitch, 4)
                        if np.isfinite(pitch) else float("nan"),
                        "error_pct": round(err, 2) if np.isfinite(err) else
                        float("nan"),
                        "correct": bool(np.isfinite(err) and abs(err) < 5.0),
                        "n_peaks": n_peaks,
                        "spacing_cv": round(cv_, 4) if np.isfinite(cv_)
                        else float("nan"),
                    })
    return rows


def exp_kotlin(args) -> list[dict]:
    """The Kotlin's own rule, on the same images, against the Python's.

    Reproduces `GridEstimator.kt` exactly by calling `grid_reference`, which
    exists to be its line-for-line Python twin -- but with the projection
    vectorised, since the reference's pixel loop takes minutes on an 8 MP
    image and produces the same profile.

    The two rules differ in more than the number: the Kotlin thresholds at
    a fraction of the tallest peak in the *grid lag window*, whereas the
    Python thresholds at a fraction of ``ac[:hi].max()``, and because the
    autocorrelation is normalised by its zero lag that maximum is always
    exactly 1.0. So `peak_fraction` is not a fraction of anything about the
    grid; it is an absolute threshold on normalised autocorrelation height.
    Same-looking knob, different meaning, which is why 0.20 and 0.25 were
    never comparable.
    """
    rows = []
    for name, kind, img, true_px in sources(args):
        # Verify the vectorised profile against the reference's own loop on
        # a crop, so the equivalence claim is measured rather than asserted.
        crop = img[:120, :400]
        ref = np.array(grid_reference.projection_profile(crop))
        h, w = crop.shape[:2]
        prof_full = (255.0 - crop[:, :, 1].astype(np.float64)).mean(axis=0)
        n = len(prof_full)
        cs = np.concatenate(([0.0], np.cumsum(prof_full)))
        half = DETREND_WINDOW // 2
        a_i = np.maximum(0, np.arange(n) - half)
        b_i = np.minimum(n, np.arange(n) + half + 1)
        mine = prof_full - (cs[b_i] - cs[a_i]) / (b_i - a_i)
        max_dev = float(np.abs(ref - mine).max())

        # Kotlin rule, full image.
        prof = profile_x(img, "green", "movavg")
        a = acf(prof)
        k_pitch, k_peaks, k_obj = grid_reference.pitch_from_acf(list(a))
        if not (grid_reference.MIN_PX_PER_MM <= k_pitch
                <= grid_reference.MAX_PX_PER_MM):
            k_pitch = float("nan")

        # Python rule, its own channel and default, same axis.
        p_prof = profile_x(img, "redness", "mean")
        p_pitch, p_peaks, _, _ = dig._pitch_1d(p_prof, 3.0, 80.0, 0.20)

        rows.append({
            "source": name, "kind": kind,
            "true_px_per_mm": round(true_px, 5),
            "reference_profile_max_dev": round(max_dev, 9),
            "kotlin_green_movavg_0.25": round(k_pitch, 4)
            if np.isfinite(k_pitch) else float("nan"),
            "kotlin_error_pct": round((k_pitch - true_px) / true_px * 100, 2)
            if np.isfinite(k_pitch) else float("nan"),
            "kotlin_n_peaks": k_peaks,
            "kotlin_window_max": round(k_obj, 4),
            "python_redness_mean_0.20": round(p_pitch, 4)
            if np.isfinite(p_pitch) else float("nan"),
            "python_error_pct": round((p_pitch - true_px) / true_px * 100, 2)
            if np.isfinite(p_pitch) else float("nan"),
            "python_n_peaks": p_peaks,
        })
    return rows


def _estimate_both_axes(img, response, detrend, frac):
    """`estimate_grid_pitch` with the channel, detrend and threshold opened up.

    Mirrors the shipped function rather than replacing it: the peak
    selection is still `digitise._pitch_1d`, unmodified, and the only
    difference is that the profile and the threshold are arguments. Nothing
    here is imported by the pipeline.
    """
    prof_x = profile_x(img, response, detrend)
    prof_y = profile_x(np.swapaxes(img, 0, 1), response, detrend)
    px, nx, cx, cvx = dig._pitch_1d(prof_x, 3.0, 80.0, frac)
    py, ny, cy, cvy = dig._pitch_1d(prof_y, 3.0, 80.0, frac)
    worst = float(np.nanmax([cvx, cvy])) if np.any(
        np.isfinite([cvx, cvy])) else float("nan")
    return dig.GridEstimate(
        px_per_mm_x=px, px_per_mm_y=py,
        confidence=float(np.nanmean([cx, cy])),
        n_peaks_x=nx, n_peaks_y=ny, spacing_cv=worst)


def exp_robust(args) -> list[dict]:
    """Does a low threshold survive degradation, and does it invent answers?

    Lowering the threshold is only a defensible recommendation if it fails
    in the two directions that matter. It must still recover the pitch when
    the image is blurred or JPEG-crushed, and it must **not** start
    returning a confident 1 mm pitch on images where the 1 mm rules are
    genuinely below the resolution at which they separate -- at 2.0 to 3.0
    px/mm the correct behaviour is the fivefold error being caught by
    `check_calibration`, not a lucky number produced by admitting noise
    peaks. Both directions are recorded here, with the guard's verdict.
    """
    rows = []
    cases = []
    for name, (path, true_px) in FIXTURES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cases.append((name, "clean", "ecg-image-kit", img, true_px, 10.0))
        for sigma in (1.0, 2.0, 3.0):
            k = int(2 * round(3 * sigma) + 1)
            cases.append((name, f"blur_{sigma}", "ecg-image-kit",
                          cv2.GaussianBlur(img, (k, k), sigma), true_px, 10.0))
        for q in (20, 10, 5):
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            cases.append((name, f"jpeg_q{q}", "ecg-image-kit",
                          cv2.imdecode(buf, cv2.IMREAD_COLOR), true_px, 10.0))

    df = ptbxl.load_metadata(args.root)
    _, _, test = ptbxl.split_folds(df)
    ecg_id = int(test.index[0])
    sig, meta = ptbxl.load_record(ecg_id, root=args.root,
                                  sampling_rate=args.rate, df=df)
    s = sig[:, ptbxl.LEAD_NAMES.index(args.lead)]
    duration = len(s) / meta.fs
    for px in (2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        img, _ = synth.render_printout(s, fs=meta.fs, px_per_mm=px,
                                       on_overflow="expand")
        cases.append((f"synth_{px:g}", "clean", "synth", img, px, duration))

    for source, condition, kind, img, true_px, dur in cases:
        for response in ("redness", "blue", "green", "luma"):
            for frac in (0.20, 0.15, 0.10, 0.05, 0.03, 0.01):
                g = _estimate_both_axes(img, response, "mean", frac)
                chk = dig.check_calibration(
                    g, image_width_px=img.shape[1], expected_duration_s=dur)
                err = ((g.px_per_mm_x - true_px) / true_px * 100
                       if np.isfinite(g.px_per_mm_x) else float("nan"))
                # 1 mm rules stop separating below about 3 px/mm, so at the
                # smallest render scales the honest outcome is a rejected
                # estimate, not a recovered one.
                resolvable = true_px >= 3.0
                rows.append({
                    "source": source, "condition": condition, "kind": kind,
                    "true_px_per_mm": round(true_px, 5),
                    "minor_resolvable": resolvable,
                    "response": response, "peak_fraction": frac,
                    "px_per_mm_x": round(g.px_per_mm_x, 4)
                    if np.isfinite(g.px_per_mm_x) else float("nan"),
                    "error_pct": round(err, 2) if np.isfinite(err)
                    else float("nan"),
                    "correct": bool(np.isfinite(err) and abs(err) < 5.0),
                    "anisotropy": round(g.anisotropy, 4)
                    if np.isfinite(g.anisotropy) else float("nan"),
                    "spacing_cv": round(g.spacing_cv, 4)
                    if np.isfinite(g.spacing_cv) else float("nan"),
                    "implied_mm_per_s": round(chk.implied_mm_per_s, 2)
                    if np.isfinite(chk.implied_mm_per_s) else float("nan"),
                    "calibration_ok": chk.ok,
                    "reasons": " | ".join(chk.reasons),
                })
    return rows


EXPERIMENTS = {
    "contrast": exp_contrast,
    "threshold": exp_threshold,
    "kotlin": exp_kotlin,
    "robust": exp_robust,
}


def persist(experiment: str, rows: list[dict], params: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{experiment}.csv"
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    meta = provenance(experiment, params)
    meta["n_rows"] = len(rows)
    (RESULTS / f"{experiment}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--experiment", nargs="+", default=["all"],
                   choices=["all", *EXPERIMENTS])
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--rate", type=int, choices=(100, 500), default=100)
    p.add_argument("--lead", default="II")
    args = p.parse_args(argv)

    chosen = list(EXPERIMENTS) if "all" in args.experiment else args.experiment
    params = {k: (str(v) if isinstance(v, Path) else v)
              for k, v in vars(args).items() if k != "experiment"}
    params["thresholds"] = list(THRESHOLDS)
    params["synth_px_per_mm"] = list(SYNTH_PX_PER_MM)
    params["detrend_window"] = DETREND_WINDOW

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
