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


@dataclass(frozen=True)
class RenderSpec:
    """Ground-truth geometry of a rendered image.

    This is the answer key. `digitise` must **never** be handed one: the
    whole question the thesis asks is whether scale can be recovered from
    image content, and passing the true pitch in would beg it. Use it to
    score `estimate_grid_pitch`, and to place leads when rendering.
    """

    px_per_mm: float
    mm_per_s: float
    mm_per_mv: float
    baseline_px: float      # image row of 0 mV
    x0_px: float            # image column of t = 0
    fs: float               # Hz of the source signal
    width_px: int
    height_px: int

    @property
    def px_per_s(self) -> float:
        return self.px_per_mm * self.mm_per_s

    @property
    def px_per_mv(self) -> float:
        return self.px_per_mm * self.mm_per_mv


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


def render_printout(
    signal: np.ndarray,
    fs: float,
    px_per_mm: float = 8.0,
    mm_per_s: float = MM_PER_S,
    mm_per_mv: float = MM_PER_MV,
    margin_mm: float = 5.0,
    height_mm: float = 40.0,
    trace_thickness: int = 2,
) -> tuple[np.ndarray, RenderSpec]:
    """Render one continuous lead as a gridded strip.

    Returns (BGR uint8 image, RenderSpec).

    The vertical mapping is ``y = baseline - v * px_per_mv``, with a
    **minus** because image rows grow downward while millivolts grow
    upward. Every inverse of this in `digitise` carries the same sign flip;
    getting it wrong produces a perfectly shaped, perfectly inverted trace
    that scores near -1.0 on PCC rather than failing loudly.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    duration_s = len(signal) / fs

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
