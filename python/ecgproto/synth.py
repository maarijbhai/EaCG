"""Synthetic test signals and single-strip printout rendering.

Two independent jobs live here, and it matters that they are not confused
with the real thing.

`synthetic_ecg` is a **sum of Gaussians, not a physiological model**. It
has no conduction system, no respiratory baseline wander, no beat-to-beat
variability. It is a test fixture, used when PTB-XL is unavailable and in
unit tests where a signal with exactly known properties is worth more than
a realistic one. No thesis result should rest on it.

`render_printout` produces a **synthetic image of a real signal**. The
waveform may be genuine PTB-XL, but the paper, grid, ink and camera are
not: there is no print bleed, no thermal fade, no shadow, no perspective
and no sensor noise. ECG-Image-Kit is the intended replacement. Results
measured against these renders are an upper bound on real-image
performance, never an estimate of it.

Two rendering choices are load-bearing for the rest of the pipeline:

* The grid is drawn in **red**, which is what lets `digitise.binarise_trace`
  separate grid from ink by channel arithmetic. Real ECG paper is not
  reliably red -- faded thermal prints, blue-grey grids and black-and-white
  photocopies all occur -- so that shortcut is expected to need replacing.
* At 100 Hz and 25 mm/s there are **fewer samples than pixel columns**, so
  the trace is drawn by interpolating between samples. The rendered curve
  is therefore smoother than a real printout at the same scale, which
  flatters the digitiser. Render at 500 Hz to remove this effect.

**The strip must be tall enough for the signal, and this is a measurement
concern rather than a cosmetic one.** `cv2.polylines` clips silently at the
canvas edge: a trace that leaves the strip does not disappear, it flattens
against the boundary. Column coverage still reports ~100% and nothing
raises, so the reconstruction loses amplitude and every symptom points at
the tracer instead of at the paper. That matters here more than it would
elsewhere, because the project's headline finding is that the classical
tracer *under-reads amplitude*; canvas clipping biases the gain ratio in
exactly the same direction and would be indistinguishable from it.

The old 40 mm default spans only +/-2.0 mV at 10 mm/mV, which PTB-XL
exceeds far more often than a glance at a few lead-II strips suggests.
Measured over the 2,198-record test fold at 100 Hz, all 12 leads:

    lead-traces needing more than +/-2.0 mV      1,971 / 26,376   7.5 %
    records with at least one such lead            892 /  2,198  40.6 %
    worst leads          V4 20.2 %   V2 17.7 %   V5 16.3 %   V3 16.1 %
    per-lead half-extent   p50 0.93 mV   p99 3.28 mV   max 12.97 mV

`DEFAULT_HEIGHT_MM` is therefore set from the 99th percentile of that
distribution, and `on_overflow` defaults to raising so the residual 1 %
leaves the sample loudly and countably instead of quietly biased. Height
costs blank paper and nothing else -- it does not change `px_per_mv`, so
amplitude resolution is unaffected and only the pixel count grows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Conventional clinical calibration. 50 mm/s and 5 mm/mV also occur in the
# wild, which is exactly why the digitiser must recover scale from the image
# rather than assume these values.
MM_PER_S = 25.0
MM_PER_MV = 10.0

# BGR, because OpenCV. The grid must stay dominant in the red channel and
# the trace must stay neutral for `binarise_trace` to separate them.
COLOUR_BACKGROUND = (255, 255, 255)
COLOUR_GRID_MINOR = (190, 190, 255)
COLOUR_GRID_MAJOR = (120, 120, 255)
COLOUR_TRACE = (32, 32, 32)

#: Sub-pixel bits for cv2.polylines. See `draw_trace`.
SUBPIXEL_BITS = 4

#: Vertical gap kept between the trace extremes and the strip edge. Two
#: millimetres is two minor grid squares -- enough that a trace which fits is
#: visibly inside the paper rather than grazing it, which also keeps the
#: outermost ink away from the image border where `trace_positions` would
#: compute a centroid from a truncated column.
CLEARANCE_MM = 2.0

#: Strip height in millimetres. At 10 mm/mV this is +/-3.5 mV of paper and
#: +/-3.3 mV of usable range after `CLEARANCE_MM`, which covers the 99th
#: percentile of per-lead amplitude in the PTB-XL test fold (see the module
#: docstring). The 100th percentile would need 263 mm for the sake of a
#: handful of records; those are better raised on than rendered.
DEFAULT_HEIGHT_MM = 70.0

#: How `render_printout` reacts to a signal taller than the strip.
OVERFLOW_MODES = ("raise", "expand", "clip")


@dataclass(frozen=True)
class RenderSpec:
    """Ground-truth geometry of a rendered image.

    This is the answer key. `digitise` must **never** be handed one: the
    whole question the thesis asks is whether scale can be recovered from
    image content, and passing the true pitch in would beg it. Use it to
    score `estimate_grid_pitch`, and to place leads when rendering.

    The four amplitude fields exist so that clipping is detectable *after*
    the fact. `render_printout` refuses to clip by default, but a caller
    that asked for `on_overflow="clip"` -- or that built a spec by hand, as
    `layout` does -- can still produce a flattened trace, and a run that
    silently lost amplitude is worth strictly less than one that says so.
    They default to zero so that callers constructing a spec for geometry
    alone need not supply them; `clipped=False` then means "not checked"
    rather than "checked and fine", which is why `render_printout` always
    fills them in.
    """

    px_per_mm: float
    mm_per_s: float
    mm_per_mv: float
    baseline_px: float      # image row of 0 mV
    x0_px: float            # image column of t = 0
    fs: float               # Hz of the source signal
    width_px: int
    height_px: int
    v_min_mv: float = 0.0           # most negative sample drawn
    v_max_mv: float = 0.0           # most positive sample drawn
    required_height_mm: float = 0.0  # strip height the signal needed
    clipped: bool = False           # True if ink fell outside the canvas

    @property
    def px_per_s(self) -> float:
        return self.px_per_mm * self.mm_per_s

    @property
    def px_per_mv(self) -> float:
        return self.px_per_mm * self.mm_per_mv

    @property
    def height_mm(self) -> float:
        """Strip height actually rendered, which `on_overflow="expand"` grows."""
        return self.height_px / self.px_per_mm

    @property
    def headroom_mm(self) -> float:
        """Millimetres of unused vertical paper. Negative means clipped."""
        return self.height_mm - self.required_height_mm


def synthetic_ecg(
    duration_s: float = 10.0,
    fs: float = 500.0,
    heart_rate_bpm: float = 60.0,
    noise_mv: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """A crude periodic ECG-shaped waveform, shape (n_samples,) in mV.

    Each beat is a sum of five Gaussians standing in for the P wave, the
    QRS complex and the T wave. Amplitudes are roughly lead-II-like (R
    around 1.2 mV) so that rendering at 10 mm/mV produces a trace of
    plausible physical size. Beats are exactly periodic, which no real
    heart is.
    """
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    period = 60.0 / heart_rate_bpm

    # (offset within beat [s], amplitude [mV], sigma [s])
    waves = (
        (-0.200, 0.15, 0.0250),   # P
        (-0.030, -0.10, 0.0080),  # Q
        (0.000, 1.20, 0.0100),    # R
        (0.035, -0.25, 0.0100),   # S
        (0.200, 0.30, 0.0450),    # T
    )

    signal = np.zeros(n, dtype=np.float64)
    # Start one period in so the first P wave is not clipped by t = 0.
    beat_times = np.arange(period, duration_s + period, period)
    for beat_t in beat_times:
        for offset, amplitude, sigma in waves:
            centre = beat_t + offset
            signal += amplitude * np.exp(-0.5 * ((t - centre) / sigma) ** 2)

    if noise_mv > 0:
        rng = np.random.default_rng(seed)
        signal += rng.normal(0.0, noise_mv, size=n)
    return signal


def required_height_mm(
    signal: np.ndarray,
    mm_per_mv: float = MM_PER_MV,
    clearance_mm: float = CLEARANCE_MM,
) -> float:
    """Strip height in millimetres that would hold `signal` without clipping.

    The extent is taken from the signal's **actual minimum and maximum**,
    not from a symmetric assumption about its peak. An ECG is not centred
    on zero and is frequently not even close: across the PTB-XL test fold,
    |min| exceeds |max| on 94.7 % of AVR traces and 93.2 % of V1, because
    those leads are dominated by a downward deflection. Sizing from
    `signal.max()` alone would under-size exactly the leads that overflow
    most often.

    The baseline sits at the middle of the strip, so the binding constraint
    is the *larger* of the two extents and the returned height is symmetric
    about zero. Keeping the baseline centred is what makes the grid rows
    land on whole millimetres either side of it, as they do on real paper.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        return 2.0 * clearance_mm
    half_extent_mv = float(max(abs(np.nanmin(signal)), abs(np.nanmax(signal))))
    return 2.0 * (half_extent_mv * mm_per_mv + clearance_mm)


def render_printout(
    signal: np.ndarray,
    fs: float,
    px_per_mm: float = 8.0,
    mm_per_s: float = MM_PER_S,
    mm_per_mv: float = MM_PER_MV,
    margin_mm: float = 5.0,
    height_mm: float = DEFAULT_HEIGHT_MM,
    trace_thickness: int = 2,
    on_overflow: str = "raise",
    clearance_mm: float = CLEARANCE_MM,
) -> tuple[np.ndarray, RenderSpec]:
    """Render one continuous lead as a gridded strip.

    Returns (BGR uint8 image, RenderSpec).

    The vertical mapping is ``y = baseline - v * px_per_mv``, with a
    **minus** because image rows grow downward while millivolts grow
    upward. Every inverse of this in `digitise` carries the same sign flip;
    getting it wrong produces a perfectly shaped, perfectly inverted trace
    that scores near -1.0 on PCC rather than failing loudly.

    `on_overflow` decides what happens when the signal is taller than
    `height_mm`. See the module docstring for why this is checked at all.

    ``"raise"``   the default. A signal that does not fit is a measurement
                  that cannot be trusted, and an experiment is better off
                  skipping it visibly than recording a quietly flattened
                  one. The message names the amplitude and the height that
                  would have worked.
    ``"expand"``  grow the strip to fit, rounded up to a whole millimetre so
                  the grid still lands on integer rules. Use when every
                  record must be scored and the image size does not matter.
    ``"clip"``    the pre-2026-09-14 behaviour, kept so that earlier runs can
                  be reproduced exactly. It sets ``spec.clipped``, which is
                  the only difference from the old silent version.
    """
    if on_overflow not in OVERFLOW_MODES:
        raise ValueError(
            f"on_overflow must be one of {OVERFLOW_MODES}, got {on_overflow!r}"
        )

    signal = np.asarray(signal, dtype=np.float64).ravel()
    duration_s = len(signal) / fs

    needed_mm = required_height_mm(signal, mm_per_mv, clearance_mm)
    v_min = float(np.nanmin(signal)) if signal.size else 0.0
    v_max = float(np.nanmax(signal)) if signal.size else 0.0
    clipped = False

    if needed_mm > height_mm:
        if on_overflow == "raise":
            raise ValueError(
                f"signal spans {v_min:+.3f} .. {v_max:+.3f} mV and needs a "
                f"{needed_mm:.1f} mm strip at {mm_per_mv:g} mm/mV, but "
                f"height_mm={height_mm:g} was given. Raising rather than "
                f"clipping: a clipped trace flattens against the canvas edge "
                f"and depresses the recovered gain exactly as a tracing fault "
                f"would. Pass height_mm={np.ceil(needed_mm):.0f}, or "
                f'on_overflow="expand".'
            )
        if on_overflow == "expand":
            height_mm = float(np.ceil(needed_mm))
        else:
            clipped = True

    width_px = int(round((duration_s * mm_per_s + 2 * margin_mm) * px_per_mm))
    height_px = int(round(height_mm * px_per_mm))
    x0_px = margin_mm * px_per_mm
    baseline_px = height_px / 2.0

    image = np.full((height_px, width_px, 3), COLOUR_BACKGROUND, dtype=np.uint8)
    draw_grid(image, px_per_mm)

    spec = RenderSpec(
        px_per_mm=px_per_mm,
        mm_per_s=mm_per_s,
        mm_per_mv=mm_per_mv,
        baseline_px=baseline_px,
        x0_px=x0_px,
        fs=fs,
        width_px=width_px,
        height_px=height_px,
        v_min_mv=v_min,
        v_max_mv=v_max,
        required_height_mm=needed_mm,
        clipped=clipped,
    )
    draw_trace(image, signal, spec, thickness=trace_thickness)
    return image, spec


def draw_trace(
    image: np.ndarray,
    signal: np.ndarray,
    spec: RenderSpec,
    x0_px: float | None = None,
    baseline_px: float | None = None,
    thickness: int = 2,
) -> None:
    """Draw one waveform onto `image` in place, in trace ink.

    `x0_px` and `baseline_px` override the spec so that `layout` can place
    the same lead into any cell of a 3x4 printout without rebuilding it.

    Polyline vertices are sub-pixel (`shift=SUBPIXEL_BITS`) because at
    100 Hz and 25 mm/s consecutive samples land ~3 px apart; rounding each
    vertex to an integer row would quantise amplitude to 1/px_per_mv mV and
    put a visible staircase on low-slope segments such as the T wave.
    """
    x0 = spec.x0_px if x0_px is None else x0_px
    base = spec.baseline_px if baseline_px is None else baseline_px

    signal = np.asarray(signal, dtype=np.float64).ravel()
    t = np.arange(len(signal)) / spec.fs
    x = x0 + t * spec.px_per_s
    # Minus: rows grow downward, millivolts grow upward.
    y = base - signal * spec.px_per_mv

    scale = 1 << SUBPIXEL_BITS
    pts = np.stack([x * scale, y * scale], axis=1).round().astype(np.int32)
    cv2.polylines(
        image,
        [pts.reshape(-1, 1, 2)],
        isClosed=False,
        color=COLOUR_TRACE,
        thickness=thickness,
        lineType=cv2.LINE_AA,
        shift=SUBPIXEL_BITS,
    )


def draw_grid(image: np.ndarray, px_per_mm: float) -> None:
    """Draw 1 mm minor and 5 mm major rules in red, in place.

    The grid is the only physical ruler in the image. Camera capture breaks
    any correspondence between image DPI and physical inches, so this is
    what `estimate_grid_pitch` must key on to recover scale.
    """
    height, width = image.shape[:2]

    # Minor first, then major over the top, so major wins where they
    # coincide -- every fifth line is both.
    for step_mm, colour in ((1.0, COLOUR_GRID_MINOR), (5.0, COLOUR_GRID_MAJOR)):
        step_px = step_mm * px_per_mm
        for i in range(int(width / step_px) + 1):
            x = int(round(i * step_px))
            if x < width:
                image[:, x] = colour
        for i in range(int(height / step_px) + 1):
            y = int(round(i * step_px))
            if y < height:
                image[y, :] = colour
