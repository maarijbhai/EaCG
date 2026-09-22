"""Real photographs of a real printout: what survives the camera.

Every number this repository has about digitisation fidelity so far was
measured on an image this repository drew. `make_strip_fixtures.py` says so
itself: the renderer and the decoder share their constants, so the pitch
comes back nearly exact and the strip starts where the renderer said it
starts. A camera grants neither. This script is the first measurement on
the other side of that loop.

Five experiments, and they answer different questions:

  channels   which per-pixel response actually carries the grid, measured
             per response and per axis. The shipped `redness` was chosen
             against renders where the grid is drawn (0,0,255); on a
             photographed pale-pink grid it is the WORST of four, because
             redness is a chroma quantity and JPEG stores chroma at half
             resolution. The 1 mm rules survive in luma and die in redness.

  pitch      what FR3 recovers, and what it costs. The shipped
             `estimate_grid_pitch` on the whole frame (which fails), the
             5 mm comb on trace-free paper (which does not), the same comb
             where the trace actually is (~10% different, because the page
             is tilted), the spread across the page, the disagreement
             between two photographs one second apart -- and a duration
             anchor. The rhythm strip of a 3x4 printout is the whole 10 s
             recording, so inked width / (px_per_mm * 25) must equal 10,
             and that says which pitch, if any, was right. On these frames
             the answer is neither: the paper band overestimates duration
             by ~3% and the trace region underestimates it by ~6%.

  otsu       the binarisation threshold's stability, which is worse than
             frame-to-frame noise: Otsu is DISCONTINUOUS IN THE CROP. On
             one frame, widening the strip ROI by four pixel rows moves
             the threshold from 125 to 109 and the ink fraction from 3.7%
             to 10.8%, because the trace mode is small next to
             paper-plus-grid and the between-class variance maximum jumps
             between two basins. A pinned percentile tracks smoothly over
             the same sweep and holds ink at 3.4%. This is reported as a
             sweep over ROI height, not as a single number per frame, and
             that is the point: a threshold that depends on how much blank
             paper you happened to include is not a property of the ink.

  fidelity   PCC, RMSE, SNR and gain against the PTB-XL record that was
             printed -- the first true fidelity figure this repository has
             from paper. Reported twice per row: `asis`, which is the
             honest end-to-end number, and `aligned`, after the best lag
             and time-rescale, which separates a calibration error from a
             tracing error. The gap between them IS the calibration cost.

  repeat     agreement between two photographs of the same paper. Read it
             with `source` beside it and never alone -- see below.

  source     the photograph against the clean digital image it was printed
             from. This is the experiment that matters, because the page
             turned out to be a print of ecg-image-kit's sample
             `ecg00001.png`, a public-domain scanned ECG with **no
             waveform ground truth anywhere**. The clean PNG is not a
             waveform, but it is the same page acquired without a camera,
             so digitising both through the identical pipeline isolates
             exactly what the camera costs -- perspective, lighting, JPEG,
             blur, skew -- from what the tracer costs.

             **And it shows that `repeat` overstates accuracy by about
             two.** Two photographs of this page agree at PCC 0.62-0.85;
             either one against the clean original tops out at 0.47. The
             dominant errors -- the hand crop, the page tilt, the
             centroid's amplitude bias -- are common to both photographs
             and cancel between them. Repeatability is a lower bound on
             error, not a stand-in for accuracy, and a deployment number
             quoted from `repeat` alone would be roughly twice too
             flattering.

WHAT THIS IS NOT. FR1 (page detection, perspective rectification) and FR4
(lead segmentation) do not exist, and this script stands in for both with
hand-measured boxes from `photo_rois.json`. So no row here is an
end-to-end result -- the two hardest stages were done by a human. Read
every number as "what the downstream chain does on real ink, given the
crop it would otherwise have had to find".

    python run_photo.py --experiment channels pitch otsu repeat source
    python run_photo.py --experiment fidelity --ecg-id 9

`source` needs `--reference`, the clean image the page was printed from;
it defaults to the ecg-image-kit sample that this page came from and is
recorded in the sidecar. `fidelity` needs `--ecg-id` and a PTB-XL record,
which **this page does not have** -- it is a scanned public-domain image,
not a render of any waveform. Keep `fidelity` for a page you printed
yourself from PTB-XL with a provenance sidecar.

`--ecg-id` is required for `fidelity` and is the PTB-XL record that was
printed. The page carries no printed identifier, so it cannot be recovered
from the image; it has to be supplied by whoever printed it. Use
`--rate 500` (the default): the strip's native tracer rate at ~22 px/mm
and 25 mm/s is about 550 Hz, so 500 Hz truth needs no upsampling, and
limitation 4's "fewer samples than pixels" flattery does not apply.

Writes results/photo/*.csv plus a JSON sidecar per experiment.
"""

from __future__ import annotations

import argparse
import csv
import os
import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
import scipy.signal as sps

from ecgproto import deskew, digitise as dig, evaluate, ptbxl, quality

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "photo"
PHOTOS = REPO / "data" / "a30_pics"
ROIS = Path(__file__).resolve().parent / "photo_rois.json"

#: Percentile used when the binarisation threshold is pinned instead of
#: left to Otsu. 96.5 because the printed trace occupies about 3.5% of a
#: tightly cropped strip, which is where Otsu lands on the crops it gets
#: right; pinning it makes the ink fraction an input rather than an
#: outcome, so every other measurement in this file varies one thing.
#:
#: It is a DIAGNOSTIC, not a proposed replacement, and it would be a bad
#: one: it assumes the ink fraction, which is exactly what changes when a
#: strip is cropped loosely or a lead is low-amplitude. A real fix reads
#: the threshold from the trace-to-paper separation, which on these
#: photographs is about 120 levels wide (trace 255-R ~205, paper ~79,
#: grid ~85) and therefore not hard to find -- Otsu simply is not looking
#: for it.
PINNED_PERCENTILE = 96.5

#: Sub-band count for the perspective measurement. Three is the fewest
#: that can show a gradient rather than merely a difference.
PERSPECTIVE_SLICES = 3

#: Largest comb irregularity accepted from a perspective slice. A third of
#: a page carries few enough 5 mm rules that the comb sometimes locks onto
#: a harmonic, which reports 50-70% of "perspective" that is nothing of the
#: kind. Set at four times the cv the full-width combs actually show
#: (0.007 to 0.012), so it admits a genuinely noisier slice and rejects a
#: harmonic lock. A frame with fewer than PERSPECTIVE_SLICES surviving
#: slices reports no spread rather than a spread over what is left.
SLICE_MAX_CV = 0.05

#: The clean digital image this repository's photographs were printed from.
#: Outside the repo (a vendored third-party sample), so it is a default
#: rather than a committed asset, and the path is recorded in every
#: `source` sidecar. There is no waveform behind it: ecg-image-kit's
#: sample-data README calls these "deidentified ECG images, sourced from
#: the public domain", and its `ecg-time-series/` folder is empty.
DEFAULT_REFERENCE = (
    r"C:\dev\vendor\ecg-image-kit\sample-data\ecg-images\ecg00001.png")

#: Smallest overlap, in seconds, a scored alignment may use. Without this a
#: scale search shrinks one signal until only a few well-matched seconds
#: remain and reports the PCC of those: the same page scored 0.85 over
#: 4 s and 0.41 over 9.7 s. Any agreement figure has to state the span it
#: covers, and this forces one.
MIN_OVERLAP_S = 8.5

#: Moving-average width, in pixels, subtracted from a projection before its
#: autocorrelation. A photograph's projection carries a broad illumination
#: envelope that the mean subtraction in `_pitch_1d` does not remove, and
#: which flattens every comb peak below the acceptance threshold. 61 px is
#: about 3 mm at these scales: wide enough to leave the 5 mm comb intact,
#: narrow enough to remove the envelope. NOTE this is a property of THIS
#: SCRIPT, not of `digitise`, and it is why the 5 mm comb is recoverable
#: here and NaN through the shipped call.
DETREND_PX = 61


# --------------------------------------------------------------------------
# responses under test
# --------------------------------------------------------------------------

def resp_redness(image: np.ndarray) -> np.ndarray:
    """The shipped response. `digitise.redness`, re-exported for the table."""
    return dig.redness(image)


def resp_blue(image: np.ndarray) -> np.ndarray:
    return 255.0 - image[:, :, 0].astype(np.float32)


def resp_green(image: np.ndarray) -> np.ndarray:
    return 255.0 - image[:, :, 1].astype(np.float32)


def resp_luma(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image.astype(np.float32))
    return np.clip(255.0 - (0.114 * b + 0.587 * g + 0.299 * r), 0.0, None)


RESPONSES = {
    "redness": resp_redness,
    "blue": resp_blue,
    "green": resp_green,
    "luma": resp_luma,
}


# --------------------------------------------------------------------------
# provenance, to the same contract as run_gate.py
# --------------------------------------------------------------------------

def git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, check=True
                             ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                    capture_output=True, text=True, check=True
                                    ).stdout.strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def module_defaults() -> dict:
    return {
        "digitise.DEFAULT_PEAK_FRACTION": dig.DEFAULT_PEAK_FRACTION,
        "digitise.MAX_ANISOTROPY": dig.MAX_ANISOTROPY,
        "digitise.MAX_SPACING_CV": dig.MAX_SPACING_CV,
        "digitise.PAPER_SPEED_RANGE_MM_S": list(dig.PAPER_SPEED_RANGE_MM_S),
        "digitise.MM_PER_S": dig.MM_PER_S,
        "digitise.MM_PER_MV": dig.MM_PER_MV,
        "run_photo.PINNED_PERCENTILE": PINNED_PERCENTILE,
        "run_photo.DETREND_PX": DETREND_PX,
        "run_photo.PERSPECTIVE_SLICES": PERSPECTIVE_SLICES,
        "run_photo.SLICE_MAX_CV": SLICE_MAX_CV,
        "run_photo.ALIGN_HALF_RANGE": ALIGN_HALF_RANGE,
        "run_photo.MIN_OVERLAP_S": MIN_OVERLAP_S,
    }


def provenance(experiment: str, params: dict) -> dict:
    sha, dirty = git_commit()
    return {
        "experiment": experiment,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": sha,
        "git_dirty": dirty,
        "params": params,
        "module_defaults": module_defaults(),
        "caveat": (
            "FR1 and FR4 do not exist. The page crop and the lead regions came "
            "from photo_rois.json, measured by hand. No row is an end-to-end "
            "result. The source photographs are in data/a30_pics/ and are not "
            "committed, because data/ is gitignored."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
    }


def persist(experiment: str, rows: list[dict], params: dict) -> Path:
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
        json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


# --------------------------------------------------------------------------
# image plumbing
# --------------------------------------------------------------------------

def load_rois() -> dict:
    return json.loads(ROIS.read_text(encoding="utf-8"))


def load_oriented(name: str) -> np.ndarray:
    """Read a photograph and apply EXIF orientation only. No deskew.

    `cv2.imread` ignores the EXIF orientation tag, and three of the six
    frames carry a non-identity one, so without this the ROI fractions land
    on a rotated page. Kept separate from the deskew because any
    measurement OF the skew estimator has to see the frame before the
    estimator's own output has been applied to it -- measuring skew on an
    already-straightened image returns ~0 for every response and compares
    nothing.
    """
    path = PHOTOS / f"{name}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read {path}")
    try:
        from PIL import Image
        o = Image.open(path).getexif().get(274, 1)
    except Exception:
        o = 1
    if o == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif o == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif o == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def load_deskewed(name: str) -> tuple[np.ndarray, float, dict]:
    """Read a photograph and straighten it with the SHIPPED deskew."""
    image = load_oriented(name)
    sk = deskew.estimate_skew(image, response_fn=dig.redness)
    rot = deskew.rotate_image(image, -sk["angle_deg"])
    if isinstance(rot, tuple):
        rot = rot[0]
    return rot, float(sk["angle_deg"]), sk


def crop(image: np.ndarray, box) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    return image[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def detrended_projection(response: np.ndarray, axis: int) -> np.ndarray:
    prof = response.sum(axis=axis).astype(np.float64)
    if DETREND_PX > 1:
        k = np.ones(DETREND_PX) / DETREND_PX
        prof = prof - np.convolve(prof, k, mode="same")
    return prof


def comb(profile: np.ndarray, min_lag: int = 20,
         max_lag: int = 600) -> tuple[float, float, int]:
    """Peak-spacing pitch of a 1-D comb. Returns (pitch_px, cv, n_peaks).

    Deliberately the same rule as `digitise._pitch_1d` -- FFT
    autocorrelation, `DEFAULT_PEAK_FRACTION` height, median peak spacing,
    parabolic refinement -- so that a difference between this and the
    shipped call is the detrending and the lag floor, nothing else.

    `min_lag` of 20 excludes the 1 mm comb on purpose: on these
    photographs it is not there to be found, and admitting short lags only
    lets noise set the median. That makes the returned pitch the 5 mm
    spacing, so callers divide by 5.
    """
    prof = profile - profile.mean()
    if not np.any(prof):
        return float("nan"), float("nan"), 0
    n = 1 << int(np.ceil(np.log2(len(prof) * 2)))
    spec = np.fft.rfft(prof, n=n)
    ac = np.fft.irfft(spec * np.conj(spec), n=n)[: len(prof)]
    if ac[0] <= 0:
        return float("nan"), float("nan"), 0
    ac = ac / ac[0]
    hi = min(max_lag, len(ac) - 2)
    if hi <= min_lag + 2:
        return float("nan"), float("nan"), 0
    peaks, _ = sps.find_peaks(
        ac[:hi], height=dig.DEFAULT_PEAK_FRACTION * ac[:hi].max())
    peaks = peaks[peaks >= min_lag]
    if len(peaks) < 2:
        return float("nan"), float("nan"), int(len(peaks))
    refined = np.array([dig._refine(ac, p) for p in peaks], dtype=np.float64)
    spacings = np.diff(np.concatenate(([0.0], refined)))
    pitch = float(np.median(spacings))
    cv = float(np.std(spacings) / pitch) if pitch > 0 else float("nan")
    return pitch, cv, int(len(peaks))


def band_power(profile: np.ndarray, period_px: float) -> float:
    """Welch power at the fundamental of `period_px`, peak of three bins."""
    f, p = sps.welch(profile, nperseg=min(1024, len(profile)))
    i = int(np.argmin(np.abs(f - 1.0 / period_px)))
    return float(p[max(0, i - 1):i + 2].max())


def pinned_mask(strip: np.ndarray) -> tuple[np.ndarray, int]:
    inv = (255 - cv2.split(strip)[2]).astype(np.uint8)
    t = int(np.percentile(inv, PINNED_PERCENTILE))
    return dig.binarise_trace(strip, threshold=t), t


def trace_to_mv(strip: np.ndarray, px_per_mm: float, fs_out: float,
                pinned: bool = True) -> tuple[np.ndarray, dict]:
    """FR5 to FR7 by hand, so the threshold can be pinned.

    `digitise()` does not expose the binarisation threshold, and the `otsu`
    experiment exists precisely because that threshold is the unstable
    part, so this reproduces the same stages with it under control. The
    arithmetic is `digitise`'s, call for call.
    """
    if pinned:
        mask, thr = pinned_mask(strip)
    else:
        mask, thr = dig.binarise_trace(strip), -1
    y = dig.trace_positions(mask)
    on = np.where(~np.isnan(y))[0]
    if len(on) < 2:
        raise ValueError("fewer than two inked columns")
    y = y[on[0]:on[-1] + 1]
    coverage = float(np.mean(~np.isnan(y)))
    y = dig._interpolate_gaps(y, 12)
    baseline = dig.baseline_from_trace(y)
    mv = (baseline - y) / (px_per_mm * dig.MM_PER_MV)
    px_per_s = px_per_mm * dig.MM_PER_S
    n = int(np.floor(len(mv) / px_per_s * fs_out))
    if n < 2:
        raise ValueError("strip shorter than two output samples")
    sig = np.interp(np.arange(n) / fs_out, np.arange(len(mv)) / px_per_s, mv)
    return sig, {
        "threshold": thr,
        "ink_pct": round(100.0 * float((mask > 0).mean()), 4),
        "coverage": round(coverage, 4),
        "baseline_px": round(float(baseline), 2),
        "px_per_s": round(float(px_per_s), 3),
        "nan_pct": round(100.0 * float(np.mean(np.isnan(sig))), 4),
    }


def fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    i = np.arange(len(x))
    x[~ok] = np.interp(i[~ok], i[ok], x[ok])
    return x


#: Scale-search half-range for `best_alignment`. Wide enough to cover the
#: pitch errors actually measured on these photographs (up to ~5% between
#: frames, ~10% between page regions), narrow enough that the search cannot
#: reach a harmonic. An optimum landing ON this bound is not an optimum --
#: `align_at_bound` marks those rows and their `pcc_aligned` must not be read
#: as "what this would have scored with the calibration fixed".
ALIGN_HALF_RANGE = 0.10
ALIGN_STEP = 0.0025


def best_alignment(reference: np.ndarray, test: np.ndarray,
                   scales=None) -> tuple[float, float, int, int, bool]:
    """Best (pcc, scale, lag, n) over a time-rescale and lag search.

    The scale search is the diagnostic that separates a calibration fault
    from a tracing fault. A pitch error is a time-base error, so a wrong
    px/mm shows up here as a scale that is not 1.0 -- and the height of the
    peak tells you what the trace would have scored had the calibration
    been right.
    """
    if scales is None:
        scales = np.arange(1.0 - ALIGN_HALF_RANGE,
                           1.0 + ALIGN_HALF_RANGE + ALIGN_STEP / 2,
                           ALIGN_STEP)
    scales = np.asarray(scales, dtype=np.float64)
    reference = fill_nan(reference)
    test = fill_nan(test)
    best = (-1.0, 1.0, 0, 0)
    for s in scales:
        n = int(round(len(test) / s))
        if n < 100:
            continue
        resampled = np.interp(np.arange(n) * s, np.arange(len(test)), test)
        m = min(len(reference), len(resampled))
        if m < 200:
            continue
        a = reference[:m] - reference[:m].mean()
        b = resampled[:m] - resampled[:m].mean()
        xc = sps.correlate(a, b, mode="full", method="fft")
        lag = int(np.argmax(xc)) - (m - 1)
        if lag >= 0:
            x, y = reference[lag:], resampled[:len(reference) - lag]
        else:
            x, y = reference[:len(resampled) + lag], resampled[-lag:]
        k = min(len(x), len(y))
        if k < 200:
            continue
        sd = np.std(x[:k]) * np.std(y[:k])
        if sd <= 0:
            continue
        v = float(np.corrcoef(x[:k], y[:k])[0, 1])
        if v > best[0]:
            best = (v, float(s), lag, k)
    at_bound = bool(len(scales) > 1 and (
        abs(best[1] - float(scales[0])) < ALIGN_STEP / 2
        or abs(best[1] - float(scales[-1])) < ALIGN_STEP / 2))
    return best + (at_bound,)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def experiment_channels(rois: dict) -> list[dict]:
    """Which response carries the grid, per response and per axis."""
    rows = []
    for name, spec in rois["frames"].items():
        if spec.get("paper") is None:
            continue
        rot, angle, _ = load_deskewed(name)
        paper = crop(rot, spec["paper"])
        # Skew is estimated on the ORIGINAL frame -- the one the estimator
        # would really be handed. The comb measurements below are on the
        # deskewed frame, because a projection over a rotated page smears
        # the comb away and would measure the rotation rather than the
        # response. The two halves of this table therefore look at
        # different images on purpose.
        original = load_oriented(name)
        for resp_name, fn in RESPONSES.items():
            sk = deskew.estimate_skew(original, response_fn=fn)
            response = fn(paper)
            prof = detrended_projection(response, axis=0)
            p5, cv, npk = comb(prof)
            px_per_mm = p5 / 5.0 if np.isfinite(p5) else float("nan")
            if np.isfinite(px_per_mm) and px_per_mm > 0:
                ratio = band_power(prof, px_per_mm) / band_power(prof, p5)
            else:
                ratio = float("nan")
            rows.append({
                "frame": name,
                "usable": spec.get("usable"),
                "response": resp_name,
                "shipped": resp_name == "redness",
                "skew_deg": round(float(sk["angle_deg"]), 4),
                "skew_h_deg": round(float(sk["angle_h_deg"]), 4),
                "skew_v_deg": round(float(sk["angle_v_deg"]), 4),
                "axis_disagreement_deg": round(
                    float(sk["axis_disagreement_deg"]), 4),
                "comb_5mm_px": round(float(p5), 4) if np.isfinite(p5) else "",
                "comb_cv": round(float(cv), 4) if np.isfinite(cv) else "",
                "n_peaks": npk,
                "px_per_mm": (round(float(px_per_mm), 4)
                              if np.isfinite(px_per_mm) else ""),
                "power_1mm_over_5mm": (round(float(ratio), 4)
                                       if np.isfinite(ratio) else ""),
            })
    return rows


def experiment_pitch(rois: dict) -> list[dict]:
    """What FR3 recovers on a photograph, and how far it varies."""
    rows = []
    for name, spec in rois["frames"].items():
        rot, angle, _ = load_deskewed(name)
        h, w = rot.shape[:2]
        row = {
            "frame": name,
            "usable": spec.get("usable"),
            "megapixels": round(h * w / 1e6, 2),
            "skew_deg": round(angle, 4),
        }
        # the SHIPPED call, on the whole frame, exactly as a caller would
        try:
            g = dig.estimate_grid_pitch(rot)
            row.update({
                "shipped_px_per_mm_x": ("" if not np.isfinite(g.px_per_mm_x)
                                        else round(float(g.px_per_mm_x), 4)),
                "shipped_px_per_mm_y": ("" if not np.isfinite(g.px_per_mm_y)
                                        else round(float(g.px_per_mm_y), 4)),
                "shipped_n_peaks_x": g.n_peaks_x,
                "shipped_n_peaks_y": g.n_peaks_y,
                "shipped_failed": not np.isfinite(g.px_per_mm_x),
            })
        except Exception as exc:
            row.update({"shipped_failed": True,
                        "shipped_error": f"{type(exc).__name__}: {exc}"})

        if spec.get("paper") is not None:
            paper = crop(rot, spec["paper"])
            prof = detrended_projection(dig.redness(paper), axis=0)
            p5, cv, npk = comb(prof)
            row.update({
                "paper_comb_5mm_px": ("" if not np.isfinite(p5)
                                      else round(float(p5), 4)),
                "paper_comb_cv": ("" if not np.isfinite(cv)
                                  else round(float(cv), 4)),
                "paper_px_per_mm": ("" if not np.isfinite(p5)
                                    else round(float(p5) / 5.0, 4)),
                "paper_n_peaks": npk,
            })
            # The same comb measured where the TRACE is, not where the
            # paper is. These disagree by about 10% on every frame that
            # yields both, which is vertical perspective: the page is
            # tilted, so a millimetre at the top of the sheet is not a
            # millimetre at the bottom. Calibrating a strip at the bottom
            # with a grid measured at the top therefore costs ~10% of time
            # base before any estimator error.
            #
            # Neither figure is right, and the duration anchor below is how
            # we know. The rhythm-region comb is contaminated: redness of
            # trace ink (+21.6) is close to redness of grid ink (+26.2) on
            # these photographs, so a region full of trace biases the comb.
            if spec.get("rhythm") is not None:
                rp, rcv, _ = comb(detrended_projection(
                    dig.redness(crop(rot, spec["rhythm"])), axis=0))
                if np.isfinite(rp):
                    row["rhythm_px_per_mm"] = round(float(rp) / 5.0, 4)
                    row["rhythm_comb_cv"] = round(float(rcv), 4)
                    if np.isfinite(p5):
                        row["vertical_gradient_pct"] = round(
                            100.0 * (rp / p5 - 1.0), 3)
                # The one external anchor available on this page: the
                # rhythm strip of a 3x4 printout is the whole recording,
                # 10 s. So the inked width divided by (px_per_mm * 25)
                # must come to 10, and whichever pitch gets closest is the
                # one that was right. This is the only check here that does
                # not rely on the pipeline's own self-consistency.
                strip = crop(rot, spec["rhythm"])
                mask, _ = pinned_mask(strip)
                on = np.where((mask > 0).any(axis=0))[0]
                if len(on) > 1 and np.isfinite(p5):
                    inked_px = int(on[-1] - on[0] + 1)
                    row["inked_width_px"] = inked_px
                    for tag, pitch in (("paper", p5 / 5.0),
                                       ("rhythm", rp / 5.0 if np.isfinite(rp)
                                        else float("nan"))):
                        if not np.isfinite(pitch) or pitch <= 0:
                            continue
                        d = inked_px / (pitch * dig.MM_PER_S)
                        row[f"implied_duration_{tag}_s"] = round(d, 4)
                        row[f"duration_error_{tag}_pct"] = round(
                            100.0 * (d / 10.0 - 1.0), 3)
                    # and the pitch that WOULD have given exactly 10 s
                    row["px_per_mm_implied_by_10s"] = round(
                        inked_px / (10.0 * dig.MM_PER_S), 4)

            # perspective: the same comb in slices across the page
            x0, y0, x1, y1 = spec["paper"]
            slices = []
            span = (x1 - x0) / PERSPECTIVE_SLICES
            for i in range(PERSPECTIVE_SLICES):
                sub = crop(rot, (x0 + i * span, y0, x0 + (i + 1) * span, y1))
                v, vcv, _ = comb(detrended_projection(dig.redness(sub),
                                                      axis=0))
                # A slice is a third of the page wide, which is often too
                # few 5 mm rules for a clean comb, and a slice that locked
                # onto a harmonic reports a spread of tens of percent that
                # is estimator failure and not perspective. Require the
                # same regularity the full-width comb shows.
                if np.isfinite(v) and np.isfinite(vcv) and vcv < SLICE_MAX_CV:
                    slices.append(v / 5.0)
            row["perspective_slices_ok"] = len(slices)
            if len(slices) == PERSPECTIVE_SLICES:
                spread = (max(slices) - min(slices)) / float(np.mean(slices))
                row["perspective_spread_pct"] = round(100.0 * spread, 3)
                row["slice_px_per_mm"] = ";".join(f"{v:.4f}" for v in slices)
            else:
                row["perspective_spread_pct"] = ""
        rows.append(row)

    # between-frame reproducibility of the recovered pitch
    usable = [r for r in rows if r.get("paper_px_per_mm") not in (None, "")]
    for a, b in itertools.combinations(usable, 2):
        pa, pb = float(a["paper_px_per_mm"]), float(b["paper_px_per_mm"])
        rows.append({
            "frame": f"{a['frame']} vs {b['frame']}",
            "usable": bool(a.get("usable") and b.get("usable")),
            "pair_px_per_mm_a": pa,
            "pair_px_per_mm_b": pb,
            "pair_disagreement_pct": round(100.0 * (pb / pa - 1.0), 3),
        })
    return rows


#: Vertical insets applied to the rhythm ROI for the Otsu sweep, as a
#: fraction of image height taken off the top and added to the bottom. The
#: crop stays centred on the same trace throughout; all that changes is how
#: much blank gridded paper surrounds it. A threshold that is a property of
#: the ink would not move across this sweep.
OTSU_INSETS = (0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030)


def experiment_otsu(rois: dict) -> list[dict]:
    """Is the binarisation threshold a property of the ink? Measured: no.

    Swept over ROI height rather than reported once per frame. A single
    number per frame conflates two effects -- the frame and the crop -- and
    the crop turns out to dominate, which a per-frame table would hide.
    """
    rows = []
    for name, spec in rois["frames"].items():
        if spec.get("rhythm") is None:
            continue
        rot, _, _ = load_deskewed(name)
        x0, y0, x1, y1 = spec["rhythm"]
        for inset in OTSU_INSETS:
            box = (x0, y0 + inset, x1, y1 - inset)
            if box[3] - box[1] <= 0.02:
                continue
            strip = crop(rot, box)
            inv = (255 - cv2.split(strip)[2]).astype(np.uint8)
            otsu_t, _ = cv2.threshold(inv, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m_otsu = dig.binarise_trace(strip)
            m_pin, pin_t = pinned_mask(strip)
            rows.append({
                "frame": name,
                "usable": spec.get("usable"),
                "roi_inset": inset,
                "roi_rows": int(strip.shape[0]),
                "otsu_threshold": int(otsu_t),
                "otsu_ink_pct": round(100.0 * float((m_otsu > 0).mean()), 4),
                "pinned_threshold": pin_t,
                "pinned_ink_pct": round(100.0 * float((m_pin > 0).mean()), 4),
                # the histogram, to show how little actually changes
                "inv_p5": int(np.percentile(inv, 5)),
                "inv_p50": int(np.percentile(inv, 50)),
                "inv_p95": int(np.percentile(inv, 95)),
                "inv_p995": int(np.percentile(inv, 99.5)),
                "mean_bgr": ";".join(
                    f"{v:.1f}" for v in strip.reshape(-1, 3).mean(0)),
            })
    return rows


def experiment_fidelity(rois: dict, ecg_id: int, rate: int,
                        root: Path) -> list[dict]:
    """PCC, RMSE, SNR and gain against the record that was printed."""
    signal, meta = ptbxl.load_record(ecg_id, root=root, sampling_rate=rate)
    fs = float(meta.fs)
    rows = []
    for name, spec in rois["frames"].items():
        if spec.get("rhythm") is None:
            continue
        rot, angle, _ = load_deskewed(name)
        paper = crop(rot, spec["paper"])
        p5, cv, _ = comb(detrended_projection(dig.redness(paper), axis=0))
        if not np.isfinite(p5):
            rows.append({"frame": name, "region": "rhythm",
                         "error": "no grid comb recovered"})
            continue
        px_per_mm = p5 / 5.0

        regions = [("rhythm", spec["rhythm"], "II", 0.0, 10.0)]
        for lead, box in (spec.get("cells") or {}).items():
            regions.append((f"cell_{lead}", box, lead, 0.0, 2.5))

        for region, box, lead, t0, t1 in regions:
            strip = crop(rot, box)
            try:
                sig, info = trace_to_mv(strip, px_per_mm, fs)
            except Exception as exc:
                rows.append({"frame": name, "region": region, "lead": lead,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            idx = ptbxl.LEAD_NAMES.index(lead.upper())
            truth = signal[int(t0 * fs):int(t1 * fs), idx]
            m = evaluate.evaluate_all(truth, fill_nan(sig), fs)
            pcc_a, scale, lag, n, at_bound = best_alignment(truth, sig)
            row = {
                "frame": name,
                "usable": spec.get("usable"),
                "region": region,
                "lead": lead,
                "ecg_id": ecg_id,
                "fs_hz": fs,
                "px_per_mm": round(px_per_mm, 4),
                "comb_cv": round(float(cv), 4) if np.isfinite(cv) else "",
                "n_samples": len(sig),
                "duration_s": round(len(sig) / fs, 4),
                "truth_duration_s": round(len(truth) / fs, 4),
                # the honest end-to-end figures
                "pcc_asis": round(float(m.pcc), 4),
                "rmse_mv_asis": round(float(m.rmse_mv), 4),
                "snr_db_asis": round(float(m.snr_db), 3),
                "shifted_snr_db_asis": round(float(m.shifted_snr_db), 3),
                "gain_ratio": round(float(m.gain_ratio), 4),
                "n_compared": int(m.n_compared),
                # the diagnostic split
                "pcc_aligned": round(float(pcc_a), 4),
                "align_scale": round(float(scale), 4),
                "align_scale_pct": round(100.0 * (scale - 1.0), 3),
                "align_lag_samples": lag,
                "pcc_gain_from_alignment": round(
                    float(pcc_a) - float(m.pcc), 4),
                # True means the scale search stopped at its own boundary,
                # so pcc_aligned is a floor, not an optimum. On a row like
                # that the honest reading is "no scale in +-10% aligns
                # these", which usually means the truth is the wrong record.
                "align_at_bound": at_bound,
            }
            row.update(info)
            # the gate, on the same waveform
            try:
                g = dig.GridEstimate(px_per_mm, px_per_mm, 0.5, 0, 0,
                                     float(cv))
                res = dig.digitise(strip, fs_out=fs, grid=g, strict=False)
                rep = quality.assess(res, mask=dig.binarise_trace(strip))
                row["gate_verdict"] = rep.verdict.name
                row["gate_rejections"] = ";".join(rep.rejections or [])
                row["gate_flags"] = ";".join(rep.flags or [])
            except Exception as exc:
                row["gate_verdict"] = f"error: {type(exc).__name__}"
            rows.append(row)
    return rows


def experiment_repeat(rois: dict, rate: int) -> list[dict]:
    """Agreement between two photographs of the same paper."""
    fs = float(rate)
    digitised = {}
    for name, spec in rois["frames"].items():
        if spec.get("rhythm") is None or spec.get("paper") is None:
            continue
        rot, _, _ = load_deskewed(name)
        p5, cv, _ = comb(detrended_projection(
            dig.redness(crop(rot, spec["paper"])), axis=0))
        if not np.isfinite(p5):
            continue
        try:
            sig, info = trace_to_mv(crop(rot, spec["rhythm"]), p5 / 5.0, fs)
        except Exception:
            continue
        digitised[name] = (sig, p5 / 5.0, spec.get("usable"), info)

    rows = []
    for a, b in itertools.combinations(sorted(digitised), 2):
        sa, pa, ua, ia = digitised[a]
        sb, pb, ub, ib = digitised[b]
        asis, _, lag0, n0, _ = best_alignment(sa, sb, scales=[1.0])
        aligned, scale, lag, n, at_bound = best_alignment(sa, sb)
        rows.append({
            "frame_a": a,
            "frame_b": b,
            "both_usable": bool(ua and ub),
            "px_per_mm_a": round(pa, 4),
            "px_per_mm_b": round(pb, 4),
            "pitch_disagreement_pct": round(100.0 * (pb / pa - 1.0), 3),
            "duration_a_s": round(len(sa) / fs, 4),
            "duration_b_s": round(len(sb) / fs, 4),
            "coverage_a": ia["coverage"],
            "coverage_b": ib["coverage"],
            "pcc_asis": round(float(asis), 4),
            "pcc_aligned": round(float(aligned), 4),
            "align_scale_pct": round(100.0 * (scale - 1.0), 3),
            "align_at_bound": at_bound,
            "align_lag_samples": lag,
            "overlap_s": round(n / fs, 3),
            # the whole point: the scale that actually maximises agreement
            # against the scale the grid claimed
            "scale_error_vs_grid_pct": round(
                100.0 * ((scale - 1.0) - (pb / pa - 1.0)), 3),
        })
    return rows


def reference_waveform(path: str, fs: float) -> tuple[np.ndarray, float, dict]:
    """Digitise the rhythm strip of the clean original, same pipeline.

    The calibration pulse is excluded explicitly. It is a tall narrow block
    at the far left of every row, it is inked, and counting it as trace
    both corrupts the centroid and inflates the inked width that the
    duration anchor divides by -- 12.644 px/mm with it, 11.704 without,
    an 8% calibration error from one printed rectangle.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read reference {path}")
    h, w = image.shape[:2]
    mask_all = dig.binarise_trace(image)
    rows = np.convolve(mask_all.sum(1).astype(float),
                       np.ones(21) / 21, mode="same")
    peaks, _ = sps.find_peaks(rows, distance=int(h * 0.10))
    peaks = np.sort(peaks[np.argsort(rows[peaks])[::-1]][:4])
    if len(peaks) < 4:
        raise RuntimeError(f"found {len(peaks)} trace rows, expected 4")
    lo = peaks[-2] + int(np.argmin(rows[peaks[-2]:peaks[-1]]))
    strip = image[lo:min(h, peaks[-1] + (peaks[-1] - lo))]
    mask, _ = pinned_mask(strip)
    col = (mask > 0).sum(0)
    nz = np.where(col > 0)[0]
    tall = np.where(col > np.percentile(col[col > 0], 99) * 0.5)[0]
    left = tall[tall < strip.shape[1] // 10]
    x0 = int(left.max()) + 8 if len(left) else int(nz[0])
    x1 = int(nz[-1]) + 1
    px_per_mm = (x1 - x0) / (10.0 * dig.MM_PER_S)
    sig, info = trace_to_mv(strip[:, x0:x1], px_per_mm, fs)
    info.update({"x0": x0, "x1": x1, "rows": f"{lo}-{lo+strip.shape[0]}"})
    return fill_nan(sig), px_per_mm, info


def experiment_source(rois: dict, reference: str, rate: int) -> list[dict]:
    """The photograph against the clean image it was printed from."""
    fs = float(rate)
    ref, ref_px, ref_info = reference_waveform(reference, fs)
    rows = [{
        "frame": "REFERENCE",
        "usable": True,
        "reference": os.path.basename(reference),
        "px_per_mm": round(ref_px, 4),
        "duration_s": round(len(ref) / fs, 4),
        "coverage": ref_info["coverage"],
        "amplitude_pp_mv": round(float(np.nanmax(ref) - np.nanmin(ref)), 4),
        "note": "the clean original; every row below is scored against this",
    }]
    for name, spec in rois["frames"].items():
        if spec.get("rhythm") is None or not spec.get("usable"):
            continue
        rot, _, _ = load_deskewed(name)
        strip = crop(rot, spec["rhythm"])
        mask, _ = pinned_mask(strip)
        on = np.where((mask > 0).any(axis=0))[0]
        anchor = (on[-1] - on[0] + 1) / (10.0 * dig.MM_PER_S)
        sig, info = trace_to_mv(strip, anchor, fs)
        sig = fill_nan(sig)
        m = evaluate.evaluate_all(ref, sig, fs)

        # best lag at the anchor pitch, then best (pitch, lag) jointly --
        # both constrained to a real overlap, so neither can buy agreement
        # by throwing the strip away.
        min_ovl = int(MIN_OVERLAP_S * fs)

        def scan(candidate_px):
            best = (-1.0, 0, 0)
            s2 = fill_nan(trace_to_mv(strip, candidate_px, fs)[0])
            for lag in range(-400, 401, 10):
                if lag >= 0:
                    x, y = ref[lag:], s2[:len(ref) - lag]
                else:
                    x, y = ref[:len(s2) + lag], s2[-lag:]
                k = min(len(x), len(y))
                if k < min_ovl:
                    continue
                sd = np.std(x[:k]) * np.std(y[:k])
                if sd <= 0:
                    continue
                v = float(np.corrcoef(x[:k], y[:k])[0, 1])
                if v > best[0]:
                    best = (v, lag, k)
            return best

        pcc_lag, lag_a, ovl_a = scan(anchor)
        best = (-1.0, anchor, 0, 0)
        for px in np.arange(anchor * 0.92, anchor * 1.12, anchor * 0.0015):
            v, lag, k = scan(px)
            if v > best[0]:
                best = (v, float(px), lag, k)
        pcc_best, px_best, lag_b, ovl_b = best

        # Amplitude, scored at the best alignment. `gain_ratio` above comes
        # from evaluate_all on the unaligned pair, where a time-base error
        # drags it toward zero and it says nothing about amplitude at all --
        # 0.13 unaligned against 0.57 aligned on the same frame. Reported
        # separately rather than replaced, so the CSV shows what a
        # misaligned gain figure looks like.
        s_best = fill_nan(trace_to_mv(strip, px_best, fs)[0])
        if lag_b >= 0:
            xb, yb = ref[lag_b:], s_best[:len(ref) - lag_b]
        else:
            xb, yb = ref[:len(s_best) + lag_b], s_best[-lag_b:]
        kb = min(len(xb), len(yb))
        m_best = evaluate.evaluate_all(xb[:kb], yb[:kb], fs)

        rows.append({
            "frame": name,
            "usable": spec.get("usable"),
            "reference": os.path.basename(reference),
            "px_per_mm_anchor": round(anchor, 4),
            "duration_s": round(len(sig) / fs, 4),
            "coverage": info["coverage"],
            "amplitude_pp_mv": round(float(np.nanmax(sig) - np.nanmin(sig)), 4),
            # sample-aligned from t=0: no alignment help at all
            "pcc_asis": round(float(m.pcc), 4),
            "rmse_mv_asis": round(float(m.rmse_mv), 4),
            "snr_db_asis": round(float(m.snr_db), 3),
            "gain_ratio": round(float(m.gain_ratio), 4),
            # lag corrected, anchor pitch kept
            "pcc_lag_only": round(float(pcc_lag), 4),
            "lag_samples": lag_a,
            "overlap_lag_s": round(ovl_a / fs, 3),
            # pitch and lag both optimised: the CEILING for this frame
            "pcc_best": round(float(pcc_best), 4),
            "px_per_mm_best": round(px_best, 4),
            "anchor_error_pct": round(100.0 * (anchor / px_best - 1.0), 3),
            "overlap_best_s": round(ovl_b / fs, 3),
            "rmse_mv_best": round(float(m_best.rmse_mv), 4),
            "snr_db_best": round(float(m_best.snr_db), 3),
            "gain_ratio_best": round(float(m_best.gain_ratio), 4),
        })
    return rows


# --------------------------------------------------------------------------

EXPERIMENTS = ("channels", "pitch", "otsu", "fidelity", "repeat", "source")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", nargs="+", default=list(EXPERIMENTS),
                   choices=EXPERIMENTS)
    p.add_argument("--ecg-id", type=int, default=None,
                   help="the PTB-XL record that was printed. Required for "
                        "`fidelity`; the page carries no identifier, so it "
                        "cannot be recovered from the image.")
    p.add_argument("--rate", type=int, choices=(100, 500), default=500,
                   help="truth sampling rate. 500 is the default and the "
                        "right choice: the strip's native tracer rate is "
                        "about 550 Hz, so 500 Hz truth needs no upsampling.")
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--reference", default=DEFAULT_REFERENCE,
                   help="clean digital image the page was printed from, "
                        "for `source`. Defaults to the ecg-image-kit sample "
                        "this repository's photographs came from.")
    args = p.parse_args(argv)

    if not PHOTOS.exists():
        print(f"no photographs at {PHOTOS}; data/ is gitignored and these "
              f"images are not committed")
        return 2

    rois = load_rois()
    for name in args.experiment:
        params = {"rate": args.rate, "rois": str(ROIS.name)}
        if name == "channels":
            rows = experiment_channels(rois)
        elif name == "pitch":
            rows = experiment_pitch(rois)
        elif name == "otsu":
            rows = experiment_otsu(rois)
        elif name == "repeat":
            rows = experiment_repeat(rois, args.rate)
        elif name == "source":
            if not Path(args.reference).exists():
                print(f"source needs --reference; {args.reference} not found. "
                      f"Skipping.")
                continue
            params["reference"] = str(args.reference)
            rows = experiment_source(rois, args.reference, args.rate)
        elif name == "fidelity":
            if args.ecg_id is None:
                print("fidelity needs --ecg-id: the printed record is not "
                      "recoverable from the page. Skipping.")
                continue
            params["ecg_id"] = args.ecg_id
            rows = experiment_fidelity(rois, args.ecg_id, args.rate,
                                       args.root)
        else:
            continue
        path = persist(name, rows, params)
        print(f"{name:10s} {len(rows):4d} rows -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
