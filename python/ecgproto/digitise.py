"""Classical (non-ML) image-to-waveform digitisation.

This is the **control condition** for the whole project. Every learned
component added later has to beat these numbers to justify its latency and
memory cost on a phone, so this path should be kept working rather than
improved away once a network exists.

Scope, stated up front because it is easy to misuse:

**`digitise` handles a single continuous strip only.** Handed a 3x4
printout it will trace straight across cell boundaries -- at each vertical
cell edge it will jump to whichever lead's ink happens to be nearest and
produce a waveform that is a concatenation of fragments of different leads
at different times. It does not detect this and cannot warn about it. Lead
segmentation is the next stage of the project and does not exist yet; until
it does, feed this function one strip at a time.

**Calibration cannot come from image DPI.** For a photograph of paper there
is no reliable correspondence between an image pixel and a physical inch --
the camera distance is unknown. The printed grid is the only ruler in the
frame: 1 mm minor divisions, 5 mm major, conventionally 25 mm/s and
10 mm/mV. `estimate_grid_pitch` recovers pixels-per-millimetre from the
grid itself, and everything downstream is expressed through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal as sps

from .synth import MM_PER_MV, MM_PER_S


@dataclass(frozen=True)
class GridEstimate:
    """Recovered image scale, plus enough diagnostics to distrust it."""

    px_per_mm_x: float
    px_per_mm_y: float
    confidence: float          # mean autocorrelation height at accepted peaks
    n_peaks_x: int
    n_peaks_y: int

    @property
    def px_per_mm(self) -> float:
        """Isotropic estimate. Use `anisotropy` before trusting it."""
        return 0.5 * (self.px_per_mm_x + self.px_per_mm_y)

    @property
    def anisotropy(self) -> float:
        """|x - y| / mean. Large values mean perspective, or a failed axis."""
        mean = self.px_per_mm
        return abs(self.px_per_mm_x - self.px_per_mm_y) / mean if mean else float("inf")


@dataclass(frozen=True)
class DigitiseResult:
    """A recovered waveform and the calibration used to produce it."""

    signal_mv: np.ndarray      # (n_samples,) float64, millivolts
    fs: float                  # Hz
    grid: GridEstimate
    baseline_px: float
    coverage: float            # fraction of columns that actually had ink


def redness(image: np.ndarray) -> np.ndarray:
    """Per-pixel grid response, float32, high where the paper grid is.

    The grid is printed red and the trace in near-neutral ink, so red
    *saturation* separates them: grid pixels keep a high red channel while
    dropping green and blue; trace pixels drop all three; background keeps
    all three.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    return np.clip(r - 0.5 * (g + b), 0.0, None)


def estimate_grid_pitch(
    image: np.ndarray,
    min_pitch_px: float = 3.0,
    max_pitch_px: float = 80.0,
    peak_fraction: float = 0.20,
) -> GridEstimate:
    """Recover pixels-per-millimetre from the printed grid, per axis.

    Method: project the grid response onto each axis, autocorrelate the
    mean-subtracted profile, and take the pitch as the **median spacing
    between successive autocorrelation peaks**.

    Why the median spacing rather than the tallest peak. A regular comb of
    lines autocorrelates to peaks at every integer multiple of its pitch,
    and on ECG paper the peak at 5 mm is *taller* than the one at 1 mm --
    at a 5 mm lag the heavy major rules align with each other as well as
    the minor rules aligning with each other, whereas at a 1 mm lag major
    rules must align with minor ones. An earlier version of this function
    took the global maximum and consequently locked onto that harmonic,
    reporting a pitch five times too large and every amplitude five times
    too small. Peak *spacing* is immune to which harmonic happens to be
    tallest, because the peaks at 1, 2, 3, 4 mm are all still present.

    Where it still fails, in rough order of likelihood:

    * **Minor rules unresolved.** Below roughly 3 px/mm, or under blur or
      JPEG chroma subsampling, the 1 mm rules stop producing separate peaks
      and only the 5 mm ones survive. The peak spacing is then a clean 5 mm
      and the function returns a confident answer that is 5x wrong. Nothing
      in the autocorrelation distinguishes this case -- guard it outside, by
      checking the implied paper speed against a plausible range.
    * **Rotation.** The projection smears line energy across columns once
      the grid is more than about a degree off-axis, flattening the peaks.
      Deskew first.
    * **Perspective.** Pitch genuinely varies across the frame, so a single
      number is wrong everywhere but the centre. `anisotropy` catches the
      gross cases; rectify first (ML Kit Document Scanner is the intended
      capture-time fix).
    * **Non-red grids.** `redness` returns near-zero for a black-and-white
      photocopy and the profile is then mostly trace, not grid.
    """
    response = redness(image)
    px_x, peaks_x, conf_x = _pitch_1d(
        response.sum(axis=0), min_pitch_px, max_pitch_px, peak_fraction
    )
    px_y, peaks_y, conf_y = _pitch_1d(
        response.sum(axis=1), min_pitch_px, max_pitch_px, peak_fraction
    )
    return GridEstimate(
        px_per_mm_x=px_x,
        px_per_mm_y=px_y,
        confidence=float(np.nanmean([conf_x, conf_y])),
        n_peaks_x=peaks_x,
        n_peaks_y=peaks_y,
    )


def _pitch_1d(
    profile: np.ndarray,
    min_pitch_px: float,
    max_pitch_px: float,
    peak_fraction: float,
) -> tuple[float, int, float]:
    """Fundamental period of a 1-D comb, by autocorrelation peak spacing."""
    profile = np.asarray(profile, dtype=np.float64)
    profile = profile - profile.mean()
    if not np.any(profile):
        return float("nan"), 0, 0.0

    # FFT autocorrelation, zero-padded to avoid circular wraparound.
    n = 1 << int(np.ceil(np.log2(len(profile) * 2)))
    spectrum = np.fft.rfft(profile, n=n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[: len(profile)]
    if ac[0] <= 0:
        return float("nan"), 0, 0.0
    ac = ac / ac[0]

    hi = min(int(max_pitch_px * 6), len(ac) - 1)
    if hi <= int(min_pitch_px) + 2:
        return float("nan"), 0, 0.0

    peaks, props = sps.find_peaks(ac[:hi], height=peak_fraction * ac[:hi].max())
    peaks = peaks[peaks >= min_pitch_px]
    if len(peaks) < 2:
        # One peak (or none) is not a comb; fall back to its lag if plausible.
        if len(peaks) == 1 and min_pitch_px <= peaks[0] <= max_pitch_px:
            return float(_refine(ac, peaks[0])), 1, float(ac[peaks[0]])
        return float("nan"), int(len(peaks)), 0.0

    refined = np.array([_refine(ac, p) for p in peaks], dtype=np.float64)
    # Include the origin: the lag of the first peak is itself one period.
    spacings = np.diff(np.concatenate(([0.0], refined)))
    pitch = float(np.median(spacings))

    if not (min_pitch_px <= pitch <= max_pitch_px):
        return float("nan"), int(len(peaks)), 0.0

    heights = props["peak_heights"][peaks >= min_pitch_px] if len(props) else ac[peaks]
    return pitch, int(len(peaks)), float(np.mean(heights))


def _refine(ac: np.ndarray, lag: int) -> float:
    """Parabolic sub-sample refinement of an autocorrelation peak."""
    if lag <= 0 or lag >= len(ac) - 1:
        return float(lag)
    y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(lag)
    return float(lag + 0.5 * (y0 - y2) / denom)


def binarise_trace(
    image: np.ndarray,
    threshold: int | None = None,
) -> np.ndarray:
    """Boolean mask of trace ink, shape (H, W).

    **Keys on the red channel.** Both the white paper and the red grid have
    a saturated red channel, while the neutral dark trace does not, so
    ``255 - R`` is large only on ink. That is a property of *this renderer*,
    which draws the grid in red. Real ECG paper varies -- faded thermal
    prints, blue-grey grids, black-and-white photocopies -- and on a
    photocopy the grid is neutral and dark and this function will return it
    as trace. Replacing this stage is expected work for real images.

    Threshold defaults to Otsu on the inverted red channel.
    """
    red = image[:, :, 2] if image.ndim == 3 else image
    inverted = (255 - red).astype(np.uint8)

    if threshold is None:
        _, mask = cv2.threshold(
            inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, mask = cv2.threshold(inverted, int(threshold), 255, cv2.THRESH_BINARY)

    # A single 3x3 opening removes isolated speckle without eating the
    # one-or-two-pixel-wide trace itself.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask.astype(bool)


def trace_positions(mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Row position of the trace per image column, shape (W,), float64.

    Columns with no ink are NaN rather than zero, so that "no data" cannot
    be mistaken for "0 mV at the top of the image". Callers must handle
    them; `digitise` interpolates across short gaps.

    The position is the ink centroid of the column. On a steep QRS upstroke
    a single column contains a tall vertical run of ink and its centroid
    sits at the run's midpoint, which systematically *clips the peak*: the
    recovered R amplitude is biased low by roughly half the within-column
    excursion. This is the dominant amplitude error of the classical path
    and the main thing a learned tracer would be expected to beat.
    """
    if weights is None:
        weights = mask.astype(np.float64)
    else:
        weights = np.where(mask, weights.astype(np.float64), 0.0)

    rows = np.arange(mask.shape[0], dtype=np.float64)[:, None]
    total = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = (weights * rows).sum(axis=0) / total
    return np.where(total > 0, centroid, np.nan)


def baseline_from_trace(y_px: np.ndarray) -> float:
    """Isoelectric row of the trace, as the median inked position.

    The median is used rather than the mean because the QRS complex is a
    large, brief, one-sided excursion: it drags a mean well off the
    isoelectric line but barely moves a median, since it occupies only a
    few percent of the columns.

    It is not unconditionally safe. The median assumes the trace spends
    more than half its columns near baseline, which fails for a lead with
    sustained large deflections -- deep S waves in V1-V2, giant T waves,
    ventricular tachycardia, or a heavily elevated ST segment at high heart
    rate. In those cases the estimate drifts toward the deflection and the
    whole recovered signal acquires a DC offset. The offset is invisible to
    PCC (which is offset-invariant) and shows up only in RMSE, which is
    exactly why the metrics have to be read together.
    """
    finite = y_px[np.isfinite(y_px)]
    if finite.size == 0:
        raise ValueError("no inked columns: nothing to estimate a baseline from")
    return float(np.median(finite))


def pixels_to_mv(
    y_px: np.ndarray,
    baseline_px: float,
    px_per_mv: float,
) -> np.ndarray:
    """Convert row positions to millivolts.

    ``mv = (baseline_px - y_px) / px_per_mv``

    The subtraction is the right way round on purpose. Image rows grow
    **downward** and millivolts grow **upward**, so a trace *above* the
    baseline has a *smaller* row index and must yield a *positive* voltage.
    Writing ``(y_px - baseline_px)`` instead produces a waveform that is
    correct in every respect except that it is upside down -- which SNR and
    RMSE report as a large error, but PCC reports as a tidy -1.0 rather
    than as a failure.
    """
    if px_per_mv <= 0 or not np.isfinite(px_per_mv):
        raise ValueError(f"px_per_mv must be finite and positive, got {px_per_mv}")
    return (baseline_px - np.asarray(y_px, dtype=np.float64)) / px_per_mv


def digitise(
    image: np.ndarray,
    fs_out: float = 100.0,
    mm_per_s: float = MM_PER_S,
    mm_per_mv: float = MM_PER_MV,
    grid: GridEstimate | None = None,
    max_gap_px: int = 12,
) -> DigitiseResult:
    """Trace a **single continuous strip** and return it in millivolts.

    See the module docstring: on a 3x4 printout this silently produces
    nonsense. One strip at a time.

    `mm_per_s` and `mm_per_mv` are the *paper* calibration, which is a
    property of the machine that printed the ECG and is not recoverable
    from the grid alone -- the grid gives millimetres, the print settings
    give what a millimetre means. 25 mm/s and 10 mm/mV are conventional;
    50 mm/s and 5 mm/mV occur and are usually annotated on the printout.
    Passing the wrong value rescales time or amplitude by exactly 2.
    """
    if grid is None:
        grid = estimate_grid_pitch(image)
    px_per_mm = grid.px_per_mm
    if not np.isfinite(px_per_mm) or px_per_mm <= 0:
        raise ValueError(
            "grid pitch estimation failed; cannot calibrate. "
            f"({grid.px_per_mm_x=}, {grid.px_per_mm_y=})"
        )

    mask = binarise_trace(image)
    y_px = trace_positions(mask)

    inked = np.flatnonzero(np.isfinite(y_px))
    if inked.size < 2:
        raise ValueError("fewer than two inked columns: no trace found")
    y_px = y_px[inked[0]: inked[-1] + 1]
    coverage = float(np.isfinite(y_px).mean())

    y_px = _interpolate_gaps(y_px, max_gap_px)
    baseline_px = baseline_from_trace(y_px)

    # Vertical scale uses the vertical pitch, horizontal uses the horizontal
    # one; on a perspective-distorted image these genuinely differ.
    px_per_mv = grid.px_per_mm_y * mm_per_mv
    px_per_s = grid.px_per_mm_x * mm_per_s
    mv = pixels_to_mv(y_px, baseline_px, px_per_mv)

    # One pixel column is one time sample at px_per_s; resample onto fs_out.
    # Linear interpolation rather than a Fourier method: the QRS is a step
    # edge at this resolution and sinc resampling rings visibly around it.
    duration_s = len(mv) / px_per_s
    n_out = int(np.floor(duration_s * fs_out))
    if n_out < 1:
        raise ValueError(f"strip is shorter than one output sample ({duration_s=:.4f})")
    t_in = np.arange(len(mv)) / px_per_s
    t_out = np.arange(n_out) / fs_out
    resampled = np.interp(t_out, t_in, mv)

    return DigitiseResult(
        signal_mv=resampled,
        fs=float(fs_out),
        grid=grid,
        baseline_px=baseline_px,
        coverage=coverage,
    )


def _interpolate_gaps(y_px: np.ndarray, max_gap_px: int) -> np.ndarray:
    """Fill short NaN runs by linear interpolation; leave long ones NaN.

    Short gaps are dropout in thin ink and are safe to bridge. Long gaps
    usually mean the trace genuinely left the strip (clipping) or that
    binarisation failed over a region, and bridging those would invent
    plausible-looking signal that was never on the paper.
    """
    y = y_px.copy()
    isnan = np.isnan(y)
    if not isnan.any():
        return y

    idx = np.arange(len(y))
    valid = ~isnan
    filled = np.interp(idx, idx[valid], y[valid])

    # Re-blank any run longer than max_gap_px.
    starts = np.flatnonzero(isnan & ~np.r_[False, isnan[:-1]])
    for start in starts:
        end = start
        while end < len(y) and isnan[end]:
            end += 1
        if end - start > max_gap_px:
            filled[start:end] = np.nan
    return filled
