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
    spacing_cv: float = float("nan")   # see below; the usable discriminator
    #: 1, or the factor `_resolve_harmonic` divided a harmonic lock by.
    harmonic_x: int = 1
    harmonic_y: int = 1

    @property
    def px_per_mm(self) -> float:
        """Isotropic estimate. Use `anisotropy` before trusting it."""
        return 0.5 * (self.px_per_mm_x + self.px_per_mm_y)

    @property
    def anisotropy(self) -> float:
        """|x - y| / mean. Large values mean perspective, or a failed axis."""
        mean = self.px_per_mm
        return abs(self.px_per_mm_x - self.px_per_mm_y) / mean if mean else float("inf")

    # `confidence` is the mean autocorrelation height at the accepted peaks,
    # and it does not work as a trust signal. Measured across every condition
    # tested for the deskew report -- clean, rotated, blurred, JPEG-crushed,
    # and outright failed estimates alike -- it stayed inside 0.45 to 0.64.
    # The reason is structural rather than a tuning problem: peaks are
    # *selected* by `peak_fraction` as a fraction of the maximum, so whatever
    # survives selection is tall by construction and the mean height of the
    # survivors is nearly a constant of the algorithm. It is kept because
    # removing a field from a frozen dataclass breaks callers, and because a
    # diagnostic known to be uninformative is worth more than a silently
    # misleading one.
    #
    # `spacing_cv` is the coefficient of variation of the spacings between
    # successive autocorrelation peaks. A clean comb has near-identical
    # spacings, so it is small; a profile picking up unrelated structure has
    # erratic ones. It is **advisory, not a rejection criterion**, because
    # measurement showed it does not separate the cases that matter:
    #
    #   render at 4.0 px/mm   pitch correct to -0.0%   spacing_cv 0.412
    #   fixture at 200 dpi    pitch wrong by +399%     spacing_cv 0.184
    #
    # The high value on the correct estimate is quantisation -- at a 4 px
    # pitch the sub-sample refinement has little to work with, and the
    # spacing scatter is a fixed fraction of a pixel divided by a small
    # pitch. The low value on the wrong one is worse news: an unresolved
    # minor grid produces a *genuinely regular* comb of 5 mm rules, and
    # every internal statistic agrees it is excellent.
    #
    # A ratio of autocorrelation height at 5x the pitch against the pitch
    # itself was also tried, on the theory that major rules are darker and
    # would betray which family had been found. It does not separate either
    # (1.21 and 1.31 on correct estimates; 1.48 on one that was 4x wrong).
    #
    # The conclusion is the one `estimate_grid_pitch` already documents:
    # nothing inside the autocorrelation distinguishes a 1 mm grid from a
    # 5 mm grid misread as 1 mm, because the two are the same picture at
    # different distances. Only `check_calibration`'s paper-speed test,
    # which brings in one fact from outside the image, catches it.


#: Plausible ECG paper speeds, mm/s. 25 is conventional and 50 is the
#: common alternative; the bounds are set wide enough to admit both plus
#: honest estimation error, and narrow enough that the 5x harmonic failure
#: (25 -> 5 mm/s, or 125) falls outside on either side.
PAPER_SPEED_RANGE_MM_S = (15.0, 60.0)

#: Largest |x - y| / mean accepted between the two axis pitches. Mild
#: perspective moves them a few percent apart; a harmonic lock on one axis
#: alone moves them by 60% or more.
MAX_ANISOTROPY = 0.15

#: Largest accepted coefficient of variation of autocorrelation peak
#: spacings. A resolved grid sits near zero.
MAX_SPACING_CV = 0.25


@dataclass(frozen=True)
class CalibrationCheck:
    """Whether a `GridEstimate` may be trusted, and if not, why not.

    These are the guards the `GridEstimate` docstrings promised and nothing
    consulted. Audit finding B1: `estimate_grid_pitch` can lock onto a
    harmonic and return a confident, self-consistent, wrong pitch, and the
    resulting waveform is plausible in shape and wrong in amplitude by an
    integer factor. Nothing downstream can detect that from the waveform,
    so it has to be caught here or not at all.
    """

    ok: bool
    reasons: tuple[str, ...]              # any of these means: do not use
    warnings: tuple[str, ...] = ()        # worth logging, not worth refusing
    implied_mm_per_s: float = float("nan")

    def __bool__(self) -> bool:
        return self.ok


def check_calibration(
    grid: GridEstimate,
    image_width_px: int | None = None,
    mm_per_s: float = MM_PER_S,
    expected_duration_s: float | None = None,
    max_anisotropy: float = MAX_ANISOTROPY,
    max_spacing_cv: float = MAX_SPACING_CV,
    paper_speed_range: tuple[float, float] = PAPER_SPEED_RANGE_MM_S,
) -> CalibrationCheck:
    """Sanity-check a recovered pitch before anything is calibrated with it.

    The paper-speed test is the only one of these that can catch the 5x
    harmonic failure, and it needs one fact from outside the image:
    roughly how long the strip is meant to be. Grid pitch alone cannot
    distinguish a correctly-read 1 mm grid from a 5 mm grid misread as
    1 mm, because both are internally consistent -- the second is simply a
    picture of a smaller region of paper. Supplying `expected_duration_s`
    turns the strip width into an implied paper speed, and 25 mm/s misread
    by 5x implies either 5 or 125 mm/s, both far outside anything a
    machine prints. Without it the check is skipped rather than faked, and
    the returned reasons say so.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if not np.isfinite(grid.px_per_mm_x) or not np.isfinite(grid.px_per_mm_y):
        reasons.append(
            f"an axis failed to resolve (x={grid.px_per_mm_x}, y={grid.px_per_mm_y})"
        )
    elif grid.anisotropy > max_anisotropy:
        reasons.append(
            f"anisotropy {grid.anisotropy:.3f} exceeds {max_anisotropy:.3f} "
            f"(x={grid.px_per_mm_x:.3f}, y={grid.px_per_mm_y:.3f} px/mm) -- "
            "perspective, or one axis locked onto a harmonic"
        )

    # Advisory only. See `GridEstimate`: this fires on correct estimates at
    # low px/mm and stays quiet on the 5x harmonic failure, so rejecting on
    # it would cost good measurements without buying the one it cannot see.
    if np.isfinite(grid.spacing_cv) and grid.spacing_cv > max_spacing_cv:
        warnings.append(
            f"autocorrelation peak spacings are irregular "
            f"(cv {grid.spacing_cv:.3f} > {max_spacing_cv:.3f})"
        )

    implied = float("nan")
    if expected_duration_s and image_width_px and np.isfinite(grid.px_per_mm_x):
        implied = image_width_px / (grid.px_per_mm_x * expected_duration_s)
        lo, hi = paper_speed_range
        if not (lo <= implied <= hi):
            reasons.append(
                f"implied paper speed {implied:.1f} mm/s outside "
                f"[{lo:.0f}, {hi:.0f}] -- pitch is probably wrong by an "
                f"integer factor (nominal {mm_per_s:.0f} mm/s)"
            )

    if expected_duration_s is None:
        warnings.append(
            "paper-speed check skipped: no expected_duration_s supplied, so "
            "an integer-factor pitch error cannot be detected"
        )

    return CalibrationCheck(
        ok=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        implied_mm_per_s=implied,
    )


@dataclass(frozen=True)
class DigitiseResult:
    """A recovered waveform and the calibration used to produce it."""

    signal_mv: np.ndarray      # (n_samples,) float64, millivolts
    fs: float                  # Hz
    grid: GridEstimate
    baseline_px: float
    coverage: float            # fraction of columns that actually had ink
    calibration: CalibrationCheck | None = None


def redness(image: np.ndarray) -> np.ndarray:
    """Per-pixel grid response, float32, high where the paper grid is.

    The grid is printed red and the trace in near-neutral ink, so red
    *saturation* separates them: grid pixels keep a high red channel while
    dropping green and blue; trace pixels drop all three; background keeps
    all three.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    return np.clip(r - 0.5 * (g + b), 0.0, None)


#: Minimum autocorrelation height for a lag to be accepted as a grid peak.
#:
#: **Read the name with care: this is not a fraction of anything about the
#: grid.** `_pitch_1d` thresholds at ``peak_fraction * ac[:hi].max()``, and
#: because the autocorrelation is normalised by its zero lag that maximum is
#: always exactly 1.0. The knob is an absolute threshold on normalised
#: autocorrelation height, and the 0.20 it used to hold was an absolute 0.20.
#:
#: The value was lowered from 0.20 to 0.05 on 2026-09-17 after measuring how
#: far apart the 1 mm and 5 mm rules sit in the autocorrelation. On the
#: ecg-image-kit fixtures the minor rules stand at 0.11 to 0.19 against 0.72
#: to 0.95 for the majors, so a threshold of 0.20 deletes every minor peak
#: and the median spacing collapses onto the 5 mm comb -- a confident,
#: self-consistent, fivefold error. On this repository's own `synth`
#: renders the same peaks stand at 0.78 to 0.83, which is why the fault
#: never appeared there. See `results/grid/`.
#:
#: 0.05 is a factor of two below the lowest value measured on any fixture
#: (0.1007) and sixteen times below the synthetic renders, and it is the
#: highest threshold at which *every* channel tested recovers *every* clean
#: source correctly. Lower still buys nothing measurable: 0.01 returns the
#: same pitch to four decimal places everywhere.
#:
#: What the change costs, stated because it is a real loss. At 0.20 a
#: one-degree rotation made grid estimation fail outright and `digitise`
#: refused; at 0.05 it succeeds and returns a waveform at PCC 0.49 to 0.77,
#: and `check_calibration` accepts 10 of 12 such records because the pitch
#: it recovers really is right. The two it rejects are caught by anisotropy,
#: the y-axis having locked onto a harmonic. Rotation was never a tolerable
#: input condition -- deskew runs first by design -- but the accidental
#: safety net that used to sit here is gone, and a rotated frame now fails
#: on PCC downstream rather than on calibration here.
DEFAULT_PEAK_FRACTION = 0.05


def estimate_grid_pitch(
    image: np.ndarray,
    min_pitch_px: float = 3.0,
    max_pitch_px: float = 80.0,
    peak_fraction: float = DEFAULT_PEAK_FRACTION,
    resolve_harmonics: bool = True,
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
      checking the implied paper speed against a plausible range. Measured
      on rendered strips, the floor is sharp: 3.0 px/mm recovers exactly,
      2.5 returns a threefold error, 2.0 returns NaN.
    * **Minor rules resolved but faint.** Distinct from the above and more
      dangerous, because the information is present and gets thrown away.
      ecg-image-kit draws the 1 mm rules about seven times lighter than the
      5 mm ones, so their autocorrelation peaks stand at 0.11 to 0.19 where
      the majors stand at 0.72 to 0.95, and any threshold above about 0.10
      removes them. This is what `DEFAULT_PEAK_FRACTION` was lowered for.
      The margin is source-dependent by a factor of eight -- the synthetic
      renders here put both families at 0.78 to 0.83 -- so no fixed height
      threshold is safe for every printer. `_resolve_harmonic` now tests
      each candidate fundamental against its own sub-multiples and walks a
      2x or 5x lock down to the 1 mm pitch when the finer comb is evidenced
      (`resolve_harmonics=True`, the default, as in `GridEstimator.kt`).
      Pass False for the original height-threshold rule, which is kept as
      the control. It cannot help when the 1 mm rules are genuinely absent
      from the image; that case still needs an outside anchor.
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
    px_x, peaks_x, conf_x, cv_x, div_x = _pitch_1d_full(
        detrend(response.mean(axis=0)), min_pitch_px, max_pitch_px,
        peak_fraction, resolve_harmonics
    )
    px_y, peaks_y, conf_y, cv_y, div_y = _pitch_1d_full(
        detrend(response.mean(axis=1)), min_pitch_px, max_pitch_px,
        peak_fraction, resolve_harmonics
    )
    # The worse axis, not the mean: one bad axis is enough to poison
    # `px_per_mm`, and averaging would hide it behind a good one.
    worst_cv = float(np.nanmax([cv_x, cv_y])) if np.any(
        np.isfinite([cv_x, cv_y])
    ) else float("nan")
    return GridEstimate(
        px_per_mm_x=px_x,
        px_per_mm_y=px_y,
        confidence=float(np.nanmean([conf_x, conf_y])),
        n_peaks_x=peaks_x,
        n_peaks_y=peaks_y,
        spacing_cv=worst_cv,
        harmonic_x=div_x,
        harmonic_y=div_y,
    )


#: Moving-average window, in samples, subtracted from a grid projection
#: before its autocorrelation. The same 41 as `GridEstimator.kt`'s
#: DETREND_WINDOW, and the same shrinking window at the ends, so the two
#: ports agree -- this was the last structural difference between them.
#:
#: Why it is here at all, measured rather than argued. A photograph's
#: projection carries a broad illumination envelope that subtracting the
#: mean does not remove, and on all six 2026-09-21 frames it flattened every
#: comb peak below DEFAULT_PEAK_FRACTION: the shipped call returned NaN on
#: both axes (`results/photo/pitch.csv`, `shipped_failed`), while
#: `run_photo.py` recovered the 5 mm comb immediately once it detrended.
#: On renders it is close to inert -- the Kotlin docstring records under
#: 0.15 percentage points on every fixture.
DETREND_WINDOW = 41


def detrend(profile: np.ndarray, window: int = DETREND_WINDOW) -> np.ndarray:
    """Subtract a centred moving average; the window shrinks at the ends.

    Line for line `GridEstimator.projectionProfile`'s high-pass: element x
    loses the mean of profile[max(0, x-h) : min(n, x+h+1)], h = window//2.
    """
    p = np.asarray(profile, dtype=np.float64)
    n = len(p)
    if window <= 1 or n == 0:
        return p.copy()
    h = window // 2
    c = np.concatenate(([0.0], np.cumsum(p)))
    idx = np.arange(n)
    a = np.maximum(0, idx - h)
    b = np.minimum(n, idx + h + 1)
    return p - (c[b] - c[a]) / (b - a)


def _pitch_1d(
    profile: np.ndarray,
    min_pitch_px: float,
    max_pitch_px: float,
    peak_fraction: float,
) -> tuple[float, int, float, float]:
    """Fundamental period of a 1-D comb, by autocorrelation peak spacing.

    Returns ``(pitch_px, n_peaks, mean_peak_height, spacing_cv)``. The last
    is the coefficient of variation of the spacings between successive
    peaks -- see `GridEstimate` for why it, and not the peak height, is the
    value worth thresholding on.

    This is the original height-threshold rule, with no harmonic
    resolution: the control, and the "spacing" the harmonic test starts
    from. `estimate_grid_pitch` calls `_pitch_1d_full`.
    """
    return _pitch_1d_full(profile, min_pitch_px, max_pitch_px, peak_fraction,
                          resolve_harmonics=False)[:4]


def _pitch_1d_full(
    profile: np.ndarray,
    min_pitch_px: float,
    max_pitch_px: float,
    peak_fraction: float,
    resolve_harmonics: bool,
) -> tuple[float, int, float, float, int]:
    """`_pitch_1d`, optionally with `_resolve_harmonic`, plus its divisor.

    The harmonic is resolved on the raw median spacing *before* the range
    check, as `GridEstimator.estimate` does. The order matters on a
    photograph: at ~20 px/mm a 5 mm lock is a 100 px spacing, outside
    `max_pitch_px`, and checking first would return NaN for a lock the test
    can undo.
    """
    profile = np.asarray(profile, dtype=np.float64)
    profile = profile - profile.mean()
    if not np.any(profile):
        return float("nan"), 0, 0.0, float("nan"), 1

    ac = _acf(profile)
    if ac[0] <= 0:
        return float("nan"), 0, 0.0, float("nan"), 1

    hi = min(int(max_pitch_px * 6), len(ac) - 1)
    if hi <= int(min_pitch_px) + 2:
        return float("nan"), 0, 0.0, float("nan"), 1

    peaks, props = sps.find_peaks(ac[:hi], height=peak_fraction * ac[:hi].max())
    # `keep` is held rather than recomputed further down: after `peaks` is
    # filtered, `peaks >= min_pitch_px` is an all-true mask of the *filtered*
    # length, and indexing the unfiltered `props["peak_heights"]` with it
    # raises as soon as anything is actually dropped. Nothing was ever
    # dropped at the default threshold -- a peak below a lag of 3 needs a
    # normalised autocorrelation above 0.20 at lag 1 or 2 -- so the fault sat
    # latent until the threshold sweep went low enough to admit one. It
    # affects `confidence` only; `pitch` is computed from `peaks` and is
    # unchanged.
    keep = peaks >= min_pitch_px
    peaks = peaks[keep]
    if len(peaks) < 2:
        # One peak (or none) is not a comb; fall back to its lag if plausible.
        if len(peaks) == 1 and min_pitch_px <= peaks[0] <= max_pitch_px:
            # A single peak has no spacings to be regular about.
            return (float(_refine(ac, peaks[0])), 1, float(ac[peaks[0]]),
                    float("nan"), 1)
        return float("nan"), int(len(peaks)), 0.0, float("nan"), 1

    refined = np.array([_refine(ac, p) for p in peaks], dtype=np.float64)
    # Include the origin: the lag of the first peak is itself one period.
    spacings = np.diff(np.concatenate(([0.0], refined)))
    pitch = float(np.median(spacings))
    spacing_cv = (
        float(np.std(spacings) / pitch) if pitch > 0 and len(spacings) > 1
        else float("nan")
    )

    divisor = 1
    if resolve_harmonics:
        pitch, divisor = _resolve_harmonic_acf(ac, pitch, min_pitch_px)

    if not (min_pitch_px <= pitch <= max_pitch_px):
        return float("nan"), int(len(peaks)), 0.0, spacing_cv, divisor

    heights = props["peak_heights"][keep] if len(props) else ac[peaks]
    return pitch, int(len(peaks)), float(np.mean(heights)), spacing_cv, divisor


# --------------------------------------------------------------------------
# Harmonic resolution. `GridEstimator.kt` ports this line for line, and
# `make_photo_port_fixtures.py` holds the two to the same answer.
# --------------------------------------------------------------------------

#: Smallest "novelty" -- see `_novelty` -- at which a sub-multiple of the
#: measured spacing is accepted as the true fundamental. Measured over
#: synthetic renders at 3-16 px/mm, the three kit fixtures and simulated
#: photographs of the four kit sheets (2026-09-24): wherever the spacing was
#: already right, every sub-multiple scored at most 0.044; where it had
#: locked onto twice the pitch, the half scored 0.198 to 0.409. 0.10 sits
#: between with room on both sides.
HARMONIC_NOVELTY = 0.10

#: Sub-multiples tried. 2 and 5 are the ones ECG paper produces -- a comb
#: read at every other rule, and the 5 mm majors read as the pitch -- and 3
#: and 4 cost nothing to test and would catch a comb read at every third or
#: fourth rule.
HARMONIC_DIVISORS = (2, 3, 4, 5)

#: How many periods of the measured spacing the novelty is averaged over.
HARMONIC_PERIODS = 5


def _acf(profile: np.ndarray) -> np.ndarray:
    """Autocorrelation of the mean-subtracted profile, normalised by lag 0.

    FFT, zero-padded to avoid circular wraparound. Returned unnormalised if
    lag 0 is not positive, so a caller can tell a flat profile apart.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    n = 1 << int(np.ceil(np.log2(len(p) * 2)))
    s = np.fft.rfft(p, n=n)
    ac = np.fft.irfft(s * np.conj(s), n=n)[: len(p)]
    return ac / ac[0] if ac[0] > 0 else ac


def _acf_at(ac: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Linear interpolation of the autocorrelation at fractional lags.

    Lags whose right neighbour is past the end are skipped, not clamped.
    """
    lags = np.asarray(lags, dtype=np.float64)
    i = np.floor(lags).astype(int)
    f = lags - i
    ok = i + 1 < len(ac)
    return ac[i[ok]] * (1.0 - f[ok]) + ac[i[ok] + 1] * f[ok]


def _novelty(ac: np.ndarray, pitch: float, m: int) -> float:
    """Evidence that pitch/m, not pitch, is the fundamental.

    The autocorrelation is sampled at the multiples of q = pitch/m that are
    NOT multiples of pitch -- the lags a finer comb would add -- and compared
    with the mid-points between multiples of q. A genuine finer comb has
    peaks at the new lags and troughs between; a spurious one has a trough
    or nothing at the new lags. Only the new lags are used because at the
    old ones the peaks are there regardless, and counting them would make
    every sub-multiple look half-right.
    """
    q = pitch / m
    span = min(len(ac) - 2, HARMONIC_PERIODS * pitch)
    k = np.arange(1, int(span / q) + 1)
    new = k[k % m != 0]
    if new.size == 0:
        return float("nan")
    fresh = _acf_at(ac, new * q)
    troughs = _acf_at(ac, (k - 0.5) * q)
    if fresh.size == 0 or troughs.size == 0:
        return float("nan")
    return float(np.mean(fresh) - np.mean(troughs))


def _refine_fundamental(ac: np.ndarray, q: float) -> float:
    """Re-measure a corrected fundamental from many peaks, not one division.

    Dividing a median spacing by the divisor leaves that spacing's error
    divided by less than it could be. The peak nearest each multiple j*q
    (searched within a quarter period) is refined parabolically and the
    pitch taken as the least-squares slope of lag against j through the
    origin, which weights the long lags where a fixed lag error is the
    smallest fraction of a period.
    """
    span = min(len(ac) - 2, HARMONIC_PERIODS * 5 * q)
    js, lags = [], []
    for j in range(1, int(span / q) + 1):
        lo = max(1, int(np.floor(j * q - q / 4)))
        hi = min(len(ac) - 2, int(np.ceil(j * q + q / 4)))
        if hi <= lo:
            break
        i = lo + int(np.argmax(ac[lo:hi + 1]))
        if i in (lo, hi):
            continue                          # no interior maximum
        js.append(j)
        lags.append(_refine(ac, i))
    if len(js) < 2:
        return q
    js, lags = np.asarray(js, float), np.asarray(lags, float)
    return float(np.dot(js, lags) / np.dot(js, js))


def _resolve_harmonic(profile: np.ndarray, pitch: float,
                      min_pitch_px: float) -> tuple[float, int]:
    """`_resolve_harmonic_acf` on a projection profile."""
    return _resolve_harmonic_acf(_acf(profile), pitch, min_pitch_px)


def _resolve_harmonic_acf(ac: np.ndarray, pitch: float,
                          min_pitch_px: float) -> tuple[float, int]:
    """Replace a harmonic lock with the fundamental. Returns (pitch, divisor).

    Repeatedly tries each divisor in HARMONIC_DIVISORS and moves to the one
    with the largest novelty above HARMONIC_NOVELTY, so a spacing locked at
    10x (two corrections, 2 then 5) is walked all the way down. Leaves the
    pitch alone -- divisor 1 -- when no finer comb is evidenced, which is
    also what it must do when the 1 mm rules are genuinely not in the image:
    it cannot invent a comb that is not there, and that case still needs an
    outside anchor (limitation 15).
    """
    if not np.isfinite(pitch):
        return pitch, 1
    total = 1
    while True:
        best, best_n = None, HARMONIC_NOVELTY
        for m in HARMONIC_DIVISORS:
            if pitch / m < min_pitch_px:
                continue
            n = _novelty(ac, pitch, m)
            if np.isfinite(n) and n > best_n:
                best, best_n = m, n
        if best is None:
            break
        pitch /= best
        total *= best
    if total > 1:
        pitch = _refine_fundamental(ac, pitch)
    return float(pitch), total


def _refine(ac: np.ndarray, lag: int) -> float:
    """Parabolic sub-sample refinement of an autocorrelation peak."""
    if lag <= 0 or lag >= len(ac) - 1:
        return float(lag)
    y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(lag)
    return float(lag + 0.5 * (y0 - y2) / denom)


#: Quantile of the inverted red channel taken as the ink level by
#: `midpoint_threshold`. `TraceExtractor.INK_QUANTILE`.
INK_QUANTILE = 0.999

#: Smallest gap, in grey levels, between the paper level and the threshold.
#: On a blank or near-blank crop the ink quantile sits on paper noise and the
#: midpoint would land inside it. `TraceExtractor.MIN_INK_CONTRAST`.
MIN_INK_CONTRAST = 16


def midpoint_threshold(inverted: np.ndarray) -> int:
    """Threshold halfway between paper and ink on the inverted red channel.

    Paper is the median level and ink the `INK_QUANTILE` level, both by
    nearest rank over the 256-bin histogram -- the smallest level whose
    cumulative count reaches the fraction -- so this and
    `TraceExtractor.midpointThreshold` agree exactly, with no interpolation
    to differ over.

    Why not Otsu, measured: on the 2026-09-21 photographs Otsu's threshold
    jumps between two basins as the crop changes -- 109 -> 125, ink 12.09%
    -> 4.47%, between rhythm-strip crops 44 rows apart -- because the trace
    mode is small beside paper-plus-grid (`results/photo/otsu.csv`). The
    paper median and the ink tail barely move with how much blank paper is
    included, so neither does their midpoint.
    """
    v = np.asarray(inverted)
    if v.dtype != np.uint8:
        raise ValueError(f"need uint8 levels, got {v.dtype}")
    hist = np.bincount(v.ravel(), minlength=256)
    n = int(hist.sum())
    if n == 0:
        raise ValueError("empty image")
    cum = np.cumsum(hist)

    def level(fraction: float) -> int:
        return int(np.searchsorted(cum, fraction * n, side="left"))

    paper = level(0.5)
    ink = level(INK_QUANTILE)
    return max(paper + MIN_INK_CONTRAST, (paper + ink) // 2)


def binarise_trace(
    image: np.ndarray,
    threshold: int | None = None,
    method: str = "midpoint",
    open_kernel: int = 3,
) -> np.ndarray:
    """Boolean mask of trace ink, shape (H, W).

    **Keys on the red channel.** Both the white paper and the red grid have
    a saturated red channel, while the neutral dark trace does not, so
    ``255 - R`` is large only on ink. That is a property of *this renderer*,
    which draws the grid in red. Real ECG paper varies -- faded thermal
    prints, blue-grey grids, black-and-white photocopies -- and on a
    photocopy the grid is neutral and dark and this function will return it
    as trace. Replacing this stage is expected work for real images.

    With no explicit `threshold`, `method` picks it on the inverted red
    channel: "midpoint" (`midpoint_threshold`, the default, as in
    `TraceExtractor.binarise`) or "otsu", the original rule, kept as the
    control.
    """
    red = image[:, :, 2] if image.ndim == 3 else image
    inverted = (255 - red).astype(np.uint8)

    if threshold is None and method == "otsu":
        _, mask = cv2.threshold(
            inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif threshold is None:
        if method != "midpoint":
            raise ValueError(f"unknown threshold method {method!r}")
        t = midpoint_threshold(inverted)
        _, mask = cv2.threshold(inverted, t, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(inverted, int(threshold), 255, cv2.THRESH_BINARY)

    # An open_kernel x open_kernel opening removes speckle and short grid
    # fragments -- and everything else narrower than the kernel, trace
    # included. 3 suits renders and full-resolution photographs; a photograph
    # decoded to ~10 px/mm has a 2-3 px trace, and 3x3 erased its thin
    # stretches: duration anchor -35.6% on the first A30s sheet at 3200 px,
    # +5.8% with 2x2 (`TraceExtractor.binarise` has the measurements). An
    # area filter in place of the opening was worse: it keeps the grid rules.
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
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
    expected_duration_s: float | None = None,
    strict: bool = True,
    mask: np.ndarray | None = None,
) -> DigitiseResult:
    """Trace a **single continuous strip** and return it in millivolts.

    See the module docstring: on a 3x4 printout this silently produces
    nonsense. One strip at a time.

    `mask` is an already-computed trace mask the same shape as `image`. It
    defaults to `binarise_trace(image)`; pass one when the strip carries ink
    that is not trace -- a lead name, "25mm/s" -- and it has been removed,
    as `segment.isolate_trace` does.

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

    # B1 guards. `strict` defaults to raising because the failure being
    # guarded against produces a waveform that looks entirely reasonable and
    # is wrong by an integer factor -- returning it with a warning attached
    # invites it into a results table. Pass strict=False to score the
    # failures deliberately, as the deskew experiments do.
    calibration = check_calibration(
        grid,
        image_width_px=image.shape[1],
        mm_per_s=mm_per_s,
        expected_duration_s=expected_duration_s,
    )
    if strict and not calibration.ok:
        raise ValueError(
            "calibration rejected: " + "; ".join(calibration.reasons)
        )

    mask = binarise_trace(image) if mask is None else np.asarray(mask, bool)
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
        calibration=calibration,
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
