"""Geometric rectification: recover and remove in-plane rotation.

Pipeline stage 2, between image acquisition and lead region extraction.

**Why this stage is not optional.** Measured in this repository, a 3 degree
rotation takes end-to-end SNR from +7 dB to -7 dB, and 1 degree takes PCC
from 0.981 to 0.734 -- while `estimate_grid_pitch` continues to report an
apparently perfect pitch. The metric that looks healthiest is the one that
tells you least. Worse, rotation makes grid estimation fail *dangerously*:
at 3 degrees the x-axis locked onto the 2x harmonic and returned 16.19
px/mm against a true 8.0, and the run only failed at all because the other
axis happened to return NaN. Had both locked, the pipeline would have
returned a plausible-looking waveform at half true amplitude, silently.

Method: **projection profile sharpness maximisation.**

The printed grid is a periodic lattice of straight lines covering the whole
image. When it is axis-aligned, every pixel of a given gridline falls in
one image row, so the row projection profile is a comb of sharp,
well-separated peaks. Rotate the image and each line smears across several
rows, adjacent lines start to overlap, and the profile flattens toward its
mean. Sharpness of the projection is therefore a function of angle with a
maximum at the true skew, and deskew reduces to a 1-D optimisation.

The grid is used rather than the page boundary deliberately. A photographed
printout may be cropped, or the page edges may lie outside the frame, and a
single boundary line is a weak and easily-occluded estimator. The grid
integrates evidence from hundreds of lines spanning the entire image.

**The search never warps the image.** Rotating once per candidate angle
would be O(n) full warps, and -- more importantly -- every warp interpolates,
which is itself a low-pass filter that attenuates the very fine grid detail
the objective measures. Instead the projection is taken *along* a candidate
direction directly: binning pixels by ``y - x*tan(a)`` collects any line of
slope ``a`` into a single bin. Because rounding that bin index makes the
shift constant over runs of adjacent columns, the whole projection reduces
to a grouped column sum followed by a few hundred shifted vector adds, and
costs one pass over the image per angle with no interpolation at all. At
``a = 0`` this is exactly ``response.sum(axis=0)`` -- the projection
`estimate_grid_pitch` already takes -- so the two stages measure the same
quantity and this module is the angled generalisation of it.

**The rotation is applied exactly once.** Every interpolation attenuates
the grid, so a pipeline that deskews and later rectifies perspective must
compose the two transforms and warp once. `estimate_skew` returns the angle
without touching the image precisely so that a caller can do that;
`deskew` is the convenience path for when rotation is the only correction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .digitise import redness

#: Interpolation methods compared in the write-up. Names are the public API;
#: callers should not have to import cv2 to choose one.
INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

#: Sharpness measures available as the objective. See `_sharpness`.
MEASURES = ("variance", "ssd", "kurtosis", "entropy")

#: Interpolation used to apply the correction, chosen by measurement rather
#: than by reputation. A fixture is rotated 3 degrees (always by Lanczos, so
#: that the simulated capture is held fixed) and then deskewed back by each
#: method in turn:
#:
#:     interp     pitch shift   1 mm contrast     PCC     gain    SNR dB
#:     lanczos        0.064%        0.919       0.9697   0.8779    8.768
#:     nearest        0.070%        0.949       0.9628   0.8727    8.761
#:     bicubic        0.075%        0.949       0.9592   0.8693    8.539
#:     bilinear       0.090%        0.926       0.9537   0.8647    8.469
#:
#: Lanczos is best on the waveform metrics, which are what the pipeline
#: exists to produce, so it is the default. Nearest and bicubic preserve
#: more 1 mm grid contrast -- nearest never mixes adjacent rows, so the
#: rules stay separated -- but that does not convert into a better
#: reconstruction here.
#:
#: The spread is small (PCC 0.954 to 0.970) and the one robust finding is
#: negative: **bilinear, which is OpenCV's default, is the worst of the four
#: on every axis measured.** Taking the library default would have cost
#: about 1.6 points of PCC and 1.3 points of gain for nothing.
#:
#: An earlier version of this comparison varied the simulating warp and the
#: correcting warp together and concluded nearest won everything. It was
#: measuring two things at once: an image simulated with nearest arrives at
#: the correction step less blurred, which flattered nearest. The numbers
#: above come from the corrected design.
DEFAULT_INTERPOLATION = "lanczos"

#: Coarse sweep half-range, degrees. Handheld capture against a flat surface
#: is rarely worse than this, and ML Kit Document Scanner is the intended
#: capture-time fix for gross rotation; this stage cleans up the residual.
#: The cost of widening it is linear and small, so the limit is set by what
#: the objective can still resolve rather than by compute.
COARSE_HALF_RANGE_DEG = 10.0

#: Coarse step. Half a degree is fine enough that the true optimum is always
#: within one step of a sampled point, given the objective's main lobe is
#: several degrees wide at the scales used here.
COARSE_STEP_DEG = 0.5

#: Fine sweep half-range about the coarse optimum. One and a half coarse
#: steps, so the true peak cannot fall outside it.
FINE_HALF_RANGE_DEG = 0.75

#: Fine step. 0.05 degrees is already below the accuracy the measurements
#: show is achievable, and parabolic interpolation of the objective refines
#: below it again.
FINE_STEP_DEG = 0.05

#: Longest image dimension used for the coarse pass. Downsampling costs
#: grid resolution -- at 2x the 1 mm rules of a 200 dpi render sit near the
#: 3 px/mm floor below which they stop resolving -- but the coarse pass only
#: needs to find the right half-degree, and the 5 mm rules survive far more
#: aggressive reduction than the 1 mm ones do.
COARSE_MAX_DIM = 900


@dataclass(frozen=True)
class DeskewResult:
    """A corrected image, the angle removed, and how much to trust it.

    Downstream stages need to know how far the image was disturbed, not
    just that it was: every interpolation is a low-pass filter, so a large
    `angle_deg` means the grid reaching `estimate_grid_pitch` has measurably
    less contrast than the original had.
    """

    image: np.ndarray
    angle_deg: float            # rotation removed; + is anticlockwise correction
    objective: float            # sharpness at the optimum
    margin: float               # optimum / objective one degree away; >1 is a peak
    measure: str
    interpolation: str
    angle_h_deg: float          # optimum from horizontal rules alone
    angle_v_deg: float          # optimum from vertical rules alone
    axis_disagreement_deg: float
    coarse_angle_deg: float
    n_evaluations: int
    elapsed_ms: float

    def as_dict(self) -> dict:
        """Everything except the image, for logging to CSV or JSON."""
        d = asdict(self)
        d.pop("image")
        return d


def _sheared_projection(
    mat: np.ndarray,
    tan_a: float,
    fixed_len: int | None = None,
) -> np.ndarray | None:
    """Project `mat` along lines of slope `tan_a`; return the covered core.

    Bins each pixel by ``row - col * tan_a``, which is constant along any
    line of that slope. The bin index is rounded, so the vertical shift
    applied to a column is a step function of the column index -- constant
    over runs of adjacent columns. Summing each run first with
    `np.add.reduceat` turns what looks like a per-pixel scatter into one
    reduction over the image plus roughly ``cols * |tan_a|`` shifted adds.

    Only the *fully covered* bins are returned. A bin near either end of
    the accumulator receives contributions from some columns and not
    others, so its height reflects how many columns reached it rather than
    how much ink they carried, and including it would make the objective a
    function of the sweep geometry instead of the image. `fixed_len` crops
    every angle's profile to a common centred length so that profiles from
    different candidate angles are comparable statistics of the same size.
    """
    rows, cols = mat.shape
    shifts = np.rint(-np.arange(cols, dtype=np.float64) * tan_a).astype(np.int64)
    lo = int(shifts.min())
    span = int(shifts.max()) - lo
    if span >= rows - 8:
        return None

    # shifts is monotonic in the column index, so equal values are
    # contiguous and the run boundaries are just where it changes.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(shifts)) + 1))
    groups = np.add.reduceat(mat, starts, axis=1)
    offsets = shifts[starts] - lo

    acc = np.zeros(rows + span, dtype=np.float64)
    for j in range(groups.shape[1]):
        o = int(offsets[j])
        acc[o: o + rows] += groups[:, j]

    core = acc[span:rows]
    if fixed_len is None:
        return core
    if len(core) < fixed_len:
        return None
    start = (len(core) - fixed_len) // 2
    return core[start: start + fixed_len]


def _sharpness(profile: np.ndarray, measure: str) -> float:
    """How comb-like a projection profile is. Larger is sharper.

    Every measure is normalised by the profile mean so that it is invariant
    to overall ink quantity -- otherwise the objective would partly be
    measuring how much of the image the sweep geometry admitted, and would
    drift with angle for reasons that have nothing to do with alignment.

    ``variance`` is the squared coefficient of variation. It responds to
    how far the profile departs from flat, but not to how *ordered* that
    departure is: the same histogram shuffled into noise scores identically.

    ``ssd`` is the mean squared successive difference. Shuffling destroys
    it, so it is sensitive to the spatial arrangement and not just the
    distribution -- which is the property that actually distinguishes a comb
    of resolved gridlines from a smeared one.
    """
    m = float(profile.mean())
    if not np.isfinite(m) or m <= 0:
        return 0.0
    if measure == "variance":
        return float(profile.var() / (m * m))
    if measure == "ssd":
        d = np.diff(profile)
        return float((d * d).mean() / (m * m))
    if measure == "kurtosis":
        s = float(profile.std())
        if s <= 0:
            return 0.0
        z = (profile - m) / s
        return float((z ** 4).mean())
    if measure == "entropy":
        total = float(profile.sum())
        if total <= 0:
            return 0.0
        p = profile / total
        p = p[p > 0]
        # Negated so that, like the others, larger means sharper.
        return float((p * np.log(p)).sum())
    raise ValueError(f"unknown measure {measure!r}; expected one of {MEASURES}")


def _objective(
    response: np.ndarray,
    angle_deg: float,
    measure: str,
    len_h: int,
    len_v: int,
) -> tuple[float, float]:
    """Sharpness from horizontal and from vertical rules, at one angle.

    Both families of the lattice are used because they are independent
    evidence for the same angle, and because their *disagreement* is a
    free diagnostic: a pure rotation moves them together, while perspective
    or a shear moves them apart.

    `angle_deg` is in the *external* convention -- the rotation to be
    removed. The internal binning slope is its negation because OpenCV's
    rotation matrix carries a horizontal line to slope ``-tan(angle)`` in
    image coordinates, rows growing downward. Sweeping in the external
    convention directly keeps every angle in this module, and in
    `DeskewResult`, meaning the same thing.
    """
    t = float(np.tan(np.deg2rad(-angle_deg)))
    # Horizontal rules: bin by y - x*tan(a).
    ph = _sheared_projection(response, t, len_h)
    # Vertical rules: bin by x + y*tan(a), i.e. the same operation on the
    # transpose with the opposite slope.
    pv = _sheared_projection(response.T, -t, len_v)
    sh = _sharpness(ph, measure) if ph is not None else 0.0
    sv = _sharpness(pv, measure) if pv is not None else 0.0
    return sh, sv


def _core_lengths(shape: tuple[int, int], half_range_deg: float) -> tuple[int, int]:
    """Common profile length for every angle in a sweep of this half-range."""
    rows, cols = shape[:2]
    t = abs(np.tan(np.deg2rad(half_range_deg)))
    len_h = rows - int(np.ceil(cols * t)) - 1
    len_v = cols - int(np.ceil(rows * t)) - 1
    return max(len_h, 8), max(len_v, 8)


def _sweep(
    response: np.ndarray,
    angles: np.ndarray,
    measure: str,
    len_h: int,
    len_v: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.empty(len(angles))
    v = np.empty(len(angles))
    for i, a in enumerate(angles):
        h[i], v[i] = _objective(response, float(a), measure, len_h, len_v)
    return h, v


def _parabolic_peak(angles: np.ndarray, values: np.ndarray) -> float:
    """Sub-step optimum, by fitting a parabola to the best sample and its neighbours."""
    k = int(np.argmax(values))
    if k == 0 or k == len(values) - 1:
        return float(angles[k])
    y0, y1, y2 = values[k - 1], values[k], values[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(angles[k])
    step = float(angles[1] - angles[0])
    return float(angles[k] + 0.5 * step * (y0 - y2) / denom)


def estimate_skew(
    image: np.ndarray,
    measure: str = "ssd",
    coarse_half_range_deg: float = COARSE_HALF_RANGE_DEG,
    coarse_step_deg: float = COARSE_STEP_DEG,
    fine_half_range_deg: float = FINE_HALF_RANGE_DEG,
    fine_step_deg: float = FINE_STEP_DEG,
    coarse_max_dim: int = COARSE_MAX_DIM,
    response_fn=redness,
) -> dict:
    """Recover the in-plane skew angle **without modifying the image**.

    The returned angle is the rotation that must be *removed*: a printout
    photographed 2 degrees anticlockwise returns ``+2.0``, and rotating the
    image by ``-angle`` straightens it.

    Separate from `deskew` so that a caller adding perspective correction
    later can compose both transforms and interpolate exactly once.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, got {measure!r}")

    started = time.perf_counter()
    response = np.ascontiguousarray(response_fn(image), dtype=np.float64)
    if not np.any(response):
        raise ValueError(
            "grid response is identically zero: nothing to align. The default "
            "response keys on red saturation, which a black-and-white or "
            "blue-grey grid does not produce -- pass response_fn."
        )

    # --- coarse pass, on a reduced copy -------------------------------------
    scale = 1.0
    coarse_resp = response
    longest = max(response.shape)
    if coarse_max_dim and longest > coarse_max_dim:
        scale = coarse_max_dim / longest
        coarse_resp = cv2.resize(
            response, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    # Angle is scale-invariant, so the reduction costs resolution of the
    # grid but not correctness of the geometry.
    coarse_angles = np.arange(
        -coarse_half_range_deg,
        coarse_half_range_deg + 0.5 * coarse_step_deg,
        coarse_step_deg,
    )
    lh, lv = _core_lengths(coarse_resp.shape, coarse_half_range_deg)
    ch, cvv = _sweep(coarse_resp, coarse_angles, measure, lh, lv)
    coarse_total = ch + cvv
    coarse_angle = float(coarse_angles[int(np.argmax(coarse_total))])

    # --- fine pass, at full resolution --------------------------------------
    fine_angles = np.arange(
        coarse_angle - fine_half_range_deg,
        coarse_angle + fine_half_range_deg + 0.5 * fine_step_deg,
        fine_step_deg,
    )
    lh, lv = _core_lengths(response.shape, abs(coarse_angle) + fine_half_range_deg)
    fh, fv = _sweep(response, fine_angles, measure, lh, lv)
    fine_total = fh + fv

    angle = _parabolic_peak(fine_angles, fine_total)
    angle_h = _parabolic_peak(fine_angles, fh)
    angle_v = _parabolic_peak(fine_angles, fv)

    # Confidence as peak prominence: the objective at the optimum against
    # the objective a degree away on each side. A real lattice gives a sharp
    # maximum; a featureless or non-red image gives a nearly flat curve
    # near 1.0. Both probes and the optimum are evaluated with core lengths
    # sized to admit the full +/-1 degree, because a ratio taken between two
    # different croppings would measure the cropping.
    mh, mv = _core_lengths(response.shape, abs(angle) + 1.0)
    at_peak = sum(_objective(response, angle, measure, mh, mv))
    off = max(
        sum(_objective(response, angle + 1.0, measure, mh, mv)),
        sum(_objective(response, angle - 1.0, measure, mh, mv)),
    )
    peak = float(np.max(fine_total))
    margin = float(at_peak / off) if off > 0 else float("inf")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "angle_deg": angle,
        "objective": peak,
        "margin": margin,
        "measure": measure,
        "angle_h_deg": angle_h,
        "angle_v_deg": angle_v,
        "axis_disagreement_deg": abs(angle_h - angle_v),
        "coarse_angle_deg": coarse_angle,
        "n_evaluations": int(len(coarse_angles) + len(fine_angles) + 1),
        "elapsed_ms": elapsed_ms,
        "coarse_scale": scale,
        "coarse_angles": coarse_angles,
        "coarse_objective": coarse_total,
        "fine_angles": fine_angles,
        "fine_objective": fine_total,
    }


def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    interpolation: str = DEFAULT_INTERPOLATION,
    expand: bool = True,
    border_value: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Rotate anticlockwise by `angle_deg`. One warp, one interpolation.

    The border is filled white rather than black because the grid response
    keys on red saturation, for which white is silent and black is not: a
    black margin would contribute a hard straight edge to exactly the
    statistic the angle search maximises. White also matches what a
    photograph of paper on a desk actually looks like at the margins.

    `expand` grows the canvas so that no ink is pushed outside it. For a
    strip whose trace runs to both ends, cropping the corners would remove
    signal and make the subsequent digitisation score a shorter waveform.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(
            f"interpolation must be one of {tuple(INTERPOLATIONS)}, "
            f"got {interpolation!r}"
        )
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)

    if expand:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        new_w = int(np.ceil(h * sin + w * cos))
        new_h = int(np.ceil(h * cos + w * sin))
        m[0, 2] += new_w / 2.0 - centre[0]
        m[1, 2] += new_h / 2.0 - centre[1]
        size = (new_w, new_h)
    else:
        size = (w, h)

    return cv2.warpAffine(
        image,
        m,
        size,
        flags=INTERPOLATIONS[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def deskew(
    image: np.ndarray,
    measure: str = "ssd",
    interpolation: str = DEFAULT_INTERPOLATION,
    expand: bool = True,
    **kwargs,
) -> DeskewResult:
    """Estimate the skew and remove it with a single warp."""
    est = estimate_skew(image, measure=measure, **kwargs)
    corrected = rotate_image(
        image, -est["angle_deg"], interpolation=interpolation, expand=expand
    )
    return DeskewResult(
        image=corrected,
        angle_deg=est["angle_deg"],
        objective=est["objective"],
        margin=est["margin"],
        measure=measure,
        interpolation=interpolation,
        angle_h_deg=est["angle_h_deg"],
        angle_v_deg=est["angle_v_deg"],
        axis_disagreement_deg=est["axis_disagreement_deg"],
        coarse_angle_deg=est["coarse_angle_deg"],
        n_evaluations=est["n_evaluations"],
        elapsed_ms=est["elapsed_ms"],
    )
