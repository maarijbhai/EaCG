"""Page geometry from the printed grid itself, not just its four corners.

The corner finder fixes a homography from four points, so every millimetre
of corner error becomes a scale or keystone error spread across the whole
page -- and on a full sheet a scale error is a time-base error that grows
with each later lead (CLAUDE.md, limitation 14: half a percent of pitch
costs a third of PCC). Registering the photograph to the printed image
("truth corners", `corner_truth`) lifted the median lead PCC from 0.913 to
0.969 on the PTB-XL 40 photographs, so geometry is the largest lever left.
The phone holds no printed image, but it does hold a ruler printed across
the entire page: the 5 mm grid. Hundreds of line samples constrain the
mapping where four corners cannot.

**How.** On the corner-rectified page:

1. Response: redness, r - (g+b)/2. It is the strongest 5 mm comb on all six
   PTB-XL 40 photographs (spectral peak 157-325x the noise floor, against
   71-165x for luma or green) and the black trace and labels are ~0 in it,
   so they do not pull the line positions. Only the 5 mm family is used:
   four of the six photographs are of a print with no 1 mm rules.
2. Vertical rules: the page is cut into bands 10 mm tall; each band's
   column profile is the MEDIAN over its rows (a rule runs the full band;
   trace ink crosses only a few rows), detrended, and its peaks located to
   sub-pixel by a parabola. Peaks are indexed along the band from their
   gaps (a gap of ~2 pitches is a missed rule), and bands are chained to
   their neighbours by matching positions, outward from the middle band, so
   one rule keeps one index down the page. Horizontal rules likewise.
3. A homography from the rectified page to an ideal lattice -- vertical
   rule i at u = i * P, horizontal rule j at v = j * P -- fitted to every
   sample at once (each sample constrains one coordinate), linear first,
   then iteratively reweighted (Huber) to shed mis-indexed samples.

The lattice's absolute phase is arbitrary (which rule is "0"), and does
not matter: time zero comes from the calibration pulses and the cells from
the separator ticks, both located afterwards on the regularised page. What
the fit removes is the projective distortion -- scale varying across the
page -- which neither of those can see.

The fit's own residual is reported (`rms_mm`): what a homography cannot
explain (lens distortion, paper curl) shows up there. It is a diagnostic,
not an accuracy figure; accuracy is measured against registration
(`python/run_geometry.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal as sps

#: Rule family fitted, mm. Every print carries it; the 1 mm family is
#: missing from prints made before 2026-09-28.
PITCH_MM = 5.0
#: Band thickness across the rules, mm, and the step between bands.
BAND_MM = 10.0
BAND_STEP_MM = 5.0
#: Pages edges are skipped: the corner warp's border replication and the
#: grid's outer frame live there.
EDGE_MARGIN_MM = 3.0
#: A peak counts as a rule if its prominence is at least this fraction of
#: the band's 80th-percentile prominence.
MIN_REL_PROMINENCE = 0.3
#: Two bands' samples are the same rule if within this fraction of a pitch.
MATCH_FRACTION = 0.3
#: Huber threshold for the reweighted fit, px of the output lattice.
HUBER_PX = 1.5
#: The fit is refused below this many rules in either direction.
MIN_RULES = 8


@dataclass(frozen=True)
class GridFit:
    H: np.ndarray                 # rectified page -> regular page, 3x3
    out_size: tuple               # (w, h) of the regular page
    px_per_mm: float              # of the regular page, exactly by construction
    rms_mm: float                 # fit residual, over inlying samples
    n_vertical: int               # rules found
    n_horizontal: int
    n_samples: int
    inlier_fraction: float


def redness(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return np.maximum(f[..., 2] - 0.5 * (f[..., 0] + f[..., 1]), 0.0)


def _band_peaks(profile: np.ndarray, pitch_px: float) -> np.ndarray:
    """Sub-pixel rule positions in one band's profile."""
    k = int(round(pitch_px)) | 1
    p = profile - cv2.blur(profile.reshape(1, -1).astype(np.float32), (k, 1))[0]
    peaks = _peaks_by_distance(p, max(2, int(0.6 * pitch_px)))
    if len(peaks) < 3:
        return np.empty(0)
    prom = sps.peak_prominences(p, peaks)[0]
    keep = prom >= MIN_REL_PROMINENCE * np.percentile(prom, 80)
    peaks = peaks[keep]
    peaks = peaks[(peaks > 0) & (peaks < len(p) - 1)]
    a, b, c = p[peaks - 1], p[peaks], p[peaks + 1]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return peaks + np.clip(off, -0.5, 0.5)


def _peaks_by_distance(x: np.ndarray, distance: int) -> np.ndarray:
    """scipy.signal.find_peaks(x, distance=d), with ties broken stably.

    SciPy ranks peaks for the distance rule with NumPy's default (unstable)
    argsort, so between two equal maxima closer than `distance` the survivor
    is an accident of the sort -- measured on ptbxl9_20261005_01: two maxima
    of 12.005102 at 283 and 285, SciPy keeping 283. No port can reproduce
    that, so the taller-first rule here ranks with a stable sort (the later
    of two equal peaks wins), exactly as StripLocator.findPeaks does.
    """
    peaks, _ = sps.find_peaks(x)
    if distance <= 1 or len(peaks) < 2:
        return peaks
    keep = np.ones(len(peaks), bool)
    for j in np.argsort(x[peaks], kind="stable")[::-1]:
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < len(peaks) and peaks[k] - peaks[j] < distance:
            keep[k] = False
            k += 1
    return peaks[keep]


#: A peak is kept only if a gap to a neighbour is within this fraction of a
#: whole number of pitches.
ON_PITCH_FRACTION = 0.2


def _index_band(xs: np.ndarray, pitch_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(kept positions, integer rule index) along one band, from the gaps.

    Sequential counting is only as good as its worst gap: one spurious peak
    between two rules (a reddish label, the pulse) adds a step and shifts
    every later rule by one -- measured, on ptbxl40_20260929_05, as whole
    groups of bands one rule apart and a fit with 60% inliers. So peaks off
    the local pitch on BOTH sides are dropped first, using the band's own
    median gap (the page's scale varies by a percent or more across it).
    """
    if len(xs) < 3:
        return np.empty(0), np.empty(0, np.int64)
    g = np.diff(xs)
    near = g[(g > 0.8 * pitch_px) & (g < 1.2 * pitch_px)]
    p = float(np.median(near)) if len(near) >= 3 else pitch_px
    for _ in range(3):
        g = np.diff(xs)
        f = g / p
        ok = (np.abs(f - np.rint(f)) < ON_PITCH_FRACTION) & (np.rint(f) >= 1)
        keep = np.concatenate([[False], ok]) | np.concatenate([ok, [False]])
        if keep.all() or keep.sum() < 3:
            break
        xs = xs[keep]
    steps = np.maximum(np.rint(np.diff(xs) / p).astype(np.int64), 1)
    return xs, np.concatenate([[0], np.cumsum(steps)])


def band_profiles(resp: np.ndarray, pitch_px: float, axis: int):
    """(band centres, per-band median profiles, margin) for one rule family.

    The only image-sized step; everything after it works on these profiles
    (which is what the Kotlin port's golden vectors ship).
    """
    r = resp if axis == 0 else resp.T
    h, w = r.shape
    px_mm = pitch_px / PITCH_MM
    band, step = int(round(BAND_MM * px_mm)), int(round(BAND_STEP_MM * px_mm))
    margin = int(round(EDGE_MARGIN_MM * px_mm))
    starts = list(range(margin, h - margin - band + 1, step))
    return ([y0 + band / 2.0 for y0 in starts],
            [np.median(r[y0:y0 + band, margin:w - margin], axis=0) for y0 in starts],
            margin)


def _rule_samples(resp: np.ndarray, pitch_px: float, axis: int):
    """(along, across, index) samples of one rule family.

    axis=0: vertical rules -- bands are horizontal slabs, profile over x.
    axis=1: horizontal rules -- transpose and reuse.
    """
    centres, profiles, margin = band_profiles(resp, pitch_px, axis)
    if not centres:
        return np.empty((0, 3))
    bands = []
    for yc, prof in zip(centres, profiles):
        xs, idx = _index_band(_band_peaks(prof, pitch_px) + margin, pitch_px)
        bands.append((yc, xs, idx))
    # Chain bands outward from the middle so one rule keeps one index.
    mid = len(bands) // 2
    offsets = {mid: 0}
    for order in (range(mid + 1, len(bands)), range(mid - 1, -1, -1)):
        for b in order:
            prev = b - 1 if b > mid else b + 1
            off = _chain(bands[prev], bands[b], offsets[prev], pitch_px)
            if off is None:
                break
            offsets[b] = off
    out = []
    for b, off in offsets.items():
        yc, xs, idx = bands[b]
        out += [(x, yc, i + off) for x, i in zip(xs, idx)]
    out = np.asarray(out, np.float64).reshape(-1, 3)
    # Transposed samples are (y, x, j); hand back (x, y, j).
    return out if axis == 0 else out[:, [1, 0, 2]]


def _chain(prev, cur, prev_off: int, pitch_px: float):
    """Index offset that puts `cur`'s rules on `prev`'s numbering."""
    _, px, pi = prev
    _, cx, ci = cur
    if len(px) == 0 or len(cx) == 0:
        return None
    j = np.searchsorted(px, cx).clip(1, len(px) - 1)
    near = np.where(np.abs(px[j - 1] - cx) < np.abs(px[j] - cx), j - 1, j)
    ok = np.abs(px[near] - cx) < MATCH_FRACTION * pitch_px
    if ok.sum() < 3:
        return None
    d = (pi[near] + prev_off - ci)[ok]
    vals, counts = np.unique(d, return_counts=True)
    return int(vals[np.argmax(counts)])


def _fit_homography(xy_u: np.ndarray, u: np.ndarray, xy_v: np.ndarray, v: np.ndarray,
                    w_u: np.ndarray, w_v: np.ndarray) -> np.ndarray:
    """H (h33 = 1) with H(x,y) -> u for xy_u samples and -> v for xy_v samples.

    Linear in the eight unknowns after multiplying out the denominator:
        h1 x + h2 y + h3 - u h7 x - u h8 y = u
        h4 x + h5 y + h6 - v h7 x - v h8 y = v
    """
    x, y = xy_u.T
    au = np.column_stack([x, y, np.ones_like(x), 0 * x, 0 * x, 0 * x, -u * x, -u * y])
    x, y = xy_v.T
    av = np.column_stack([0 * x, 0 * x, 0 * x, x, y, np.ones_like(x), -v * x, -v * y])
    A = np.vstack([au * w_u[:, None], av * w_v[:, None]])
    b = np.concatenate([u * w_u, v * w_v])
    # Normalise columns for conditioning.
    s = np.abs(A).max(axis=0)
    s[s == 0] = 1
    h, *_ = np.linalg.lstsq(A / s, b, rcond=None)
    h = h / s
    return np.append(h, 1.0).reshape(3, 3)


def _apply(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def fit_grid(rect: np.ndarray, px_per_mm_guess: float, out_px_per_mm: float | None = None
             ) -> GridFit:
    """Homography that makes the 5 mm grid of `rect` exactly regular.

    `px_per_mm_guess` (the duration anchor or the ticks) only has to pick the
    5 mm family: within ~20% is enough. `out_px_per_mm` sets the regular
    page's scale (default: the measured median pitch, so the page keeps its
    size).
    """
    resp = redness(rect)
    pitch = PITCH_MM * px_per_mm_guess
    # Refine the pitch from the median gap so indexing is robust to a guess
    # several percent out.
    prof = np.median(resp[resp.shape[0] // 3: 2 * resp.shape[0] // 3], axis=0)
    pk = _band_peaks(prof, pitch)
    g = np.diff(pk)
    g = g[(g > 0.7 * pitch) & (g < 1.3 * pitch)]
    if len(g) >= 5:
        pitch = float(np.median(g))
    sv = _rule_samples(resp, pitch, 0)          # (x, y, i)
    sh_t = _rule_samples(resp, pitch, 1)        # (x, y, j) after the transpose
    if len(sv) == 0 or len(sh_t) == 0:
        raise ValueError("grid fit: no rules found")
    nv, nh = len(np.unique(sv[:, 2])), len(np.unique(sh_t[:, 2]))
    if nv < MIN_RULES or nh < MIN_RULES:
        raise ValueError(f"grid fit: {nv} vertical / {nh} horizontal rules, "
                         f"need {MIN_RULES}")
    out_mm = out_px_per_mm if out_px_per_mm else pitch / PITCH_MM
    P = PITCH_MM * out_mm
    xy_u, u = sv[:, :2], sv[:, 2] * P
    xy_v, v = sh_t[:, :2], sh_t[:, 2] * P
    w_u, w_v = np.ones(len(u)), np.ones(len(v))
    for _ in range(8):
        H = _fit_homography(xy_u, u, xy_v, v, w_u, w_v)
        ru = _apply(H, xy_u)[:, 0] - u
        rv = _apply(H, xy_v)[:, 1] - v
        w_u = np.minimum(1.0, HUBER_PX / np.maximum(np.abs(ru), 1e-9))
        w_v = np.minimum(1.0, HUBER_PX / np.maximum(np.abs(rv), 1e-9))
    res = np.concatenate([ru, rv])
    inl = np.abs(res) < 3 * HUBER_PX
    # Place the regular page so the rectified page's corners land inside it.
    h, w = rect.shape[:2]
    quad = _apply(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    lo = quad.min(axis=0)
    T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
    H = T @ H
    size = tuple(int(np.ceil(v)) + 1 for v in quad.max(axis=0) - lo)
    return GridFit(H, size, float(out_mm),
                   float(np.sqrt(np.mean(res[inl] ** 2)) / out_mm),
                   nv, nh, len(res), float(inl.mean()))


def regularise(photo: np.ndarray, corners: np.ndarray, rect_shape: tuple, fit: GridFit
               ) -> tuple[np.ndarray, np.ndarray]:
    """Warp the PHOTOGRAPH straight onto the regular page (one resampling).

    Returns (page, M) with M the photo -> regular page homography.
    """
    h, w = rect_shape[:2]
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    P = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst).astype(np.float64)
    M = fit.H @ P
    page = cv2.warpPerspective(photo, M, fit.out_size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return page, M
