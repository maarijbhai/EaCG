#!/usr/bin/env python3
"""Generate golden vectors so the Kotlin port can be checked against the Python.

**The problem this solves.** `ecgproto/deskew.py` and
`android/.../SkewEstimator.kt` are two implementations of one algorithm, in
two languages, run on two processors. Nothing forces them to agree. They were
checked once, by hand, by transcribing the Kotlin back into Python and
comparing -- which established agreement on the day but protects nothing
afterwards, and which has a hole in it: a transcription is not the Kotlin, so
a mistake made identically in both directions survives the check.

This closes the hole from the other side. The Python computes a set of
reference values and commits them; the Kotlin unit tests load those values
and assert that the **actually compiled Kotlin** reproduces them. A porting
bug then fails a test run rather than waiting to be noticed in a waveform.

**Why the input is a formula and not an image.** A realistic response plane
is roughly 900x700 doubles, which is far too large to commit and would make
the fixture unreadable. Instead both sides generate identical input from a
linear congruential generator specified below in integer arithmetic, so the
values are bit-identical in Python and on the JVM with nothing to store. A
pseudo-random field has no grid in it and the objective over it is flat --
which does not matter here, because what these vectors check is that the
*arithmetic* agrees, one candidate angle at a time. Whether the search finds
a real angle is a separate question, tested on the Kotlin side against a
synthetic lattice it builds itself, where the right answer is known by
construction and no cross-language transfer is involved.

Run after any change to the numerics on either side::

    python python/make_port_fixtures.py

Writes android/app/src/test/resources/port_fixtures.txt, which is committed.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import scipy

from ecgproto import deskew as dsk
from ecgproto import digitise as dig
from ecgproto import quality as qual
from ecgproto import evaluate

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "android" / "app" / "src" / "test" / "resources" / "port_fixtures.txt"

#: LCG parameters. glibc's constants, masked to 31 bits so that every
#: intermediate fits a signed 32-bit integer and Python's arbitrary-precision
#: integers and Kotlin's Long produce identical values with no overflow
#: subtleties on either side.
LCG_SEED = 20260918
LCG_MULT = 1103515245
LCG_ADD = 12345
LCG_MASK = 0x7FFFFFFF

#: Deliberately not square, and deliberately not a multiple of anything. A
#: square plane would let a transposed index pass as an untransposed one, and
#: round dimensions hide off-by-ones in the run-boundary arithmetic.
ROWS, COLS = 67, 103

#: Angles the sharpness is sampled at: the ends of the coarse sweep, several
#: interior points, and exact zero. These cover the geometry the sweep
#: actually walks.
#:
#: What they do NOT cover is rounding. None of them puts a shift on an exact
#: half-integer -- measured, not assumed -- because `tan(radians(x))` is never
#: an exact dyadic rational. See SHEAR_TANS, which exists for that.
ANGLES = [-10.0, -8.0, -5.0, -3.0, -1.5, -0.5, -0.05, 0.0, 0.05, 0.5,
          1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 0.25, -0.25, 7.5, -7.5]

#: Slopes fed to the projection *directly*, bypassing the angle.
#:
#: `np.rint` rounds half to even and Java's `Math.round` rounds half up, so a
#: port that reaches for the obvious Kotlin spelling is wrong — but only on
#: shifts that land exactly on an odd half-integer. Measured: none of the
#: twenty angles above produces one, because `tan(radians(x))` is never an
#: exact dyadic rational and so `col * tan` never lands exactly on .5. A
#: mutation check confirmed it: swapping `rint` for `round` passed every
#: angle-based test.
#:
#: These values are exactly representable, so `col * tan_a` is exact too and
#: lands on .5 for real. At 0.5 the two roundings first disagree at column 3
#: (-1.5 goes to -2 or -1), at 0.25 by column 6. 1.5 is included to exercise
#: the `span >= rows - 8` bail-out, and 0.0 the degenerate no-shear path
#: where the projection collapses to a plain column sum.
SHEAR_TANS = [0.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, 1.5]


#: Shape of the trace mask. Prime, unequal, and not a multiple of the two
#: moduli below, so that a transposed index or an off-by-one at the last
#: column cannot pass unnoticed.
TRACE_ROWS, TRACE_COLS = 41, 97

#: Columns where `c % TRACE_GAP_MOD` is 0 or 1 carry no ink at all. These are
#: the dropout the gap interpolation exists for, and they also put the first
#: inked column away from zero so the span trim has something to trim.
#:
#: **Two columns wide, not one, and that width is the point.** A one-column
#: gap is bridged by every `max_gap_px` down to and including 1, so the
#: re-blanking half of `_interpolate_gaps` -- the half that refuses to invent
#: signal across a long gap -- is never reached and a port could omit it
#: entirely and pass. At two wide, `max_gap_px=1` must leave them NaN.
TRACE_GAP_MOD = 23
TRACE_GAP_WIDTH = 2

#: The band's row wanders as `8 + (c * c) % TRACE_BAND_MOD`. A quadratic in
#: the column index rather than a sine: it is exactly representable in
#: integer arithmetic on both sides, and it moves by several pixels between
#: adjacent columns, which a linear ramp would not.
TRACE_BAND_MOD = 19
TRACE_THICKNESS = 2

#: Two gap widths. The first bridges this mask's runs, the second is narrower
#: than them and must leave them NaN.
TRACE_MAX_GAP_PX = 12
TRACE_MAX_GAP_TIGHT = 1

#: Scale the conversion is exercised at, and the output rates. 8 px/mm is the
#: repository's working scale; 100 Hz is PTB-XL's lower rate and 37.0 is
#: deliberately not a divisor of anything, so the interpolation is doing real
#: work rather than selecting samples.
TRACE_PX_PER_MM = 8.0
TRACE_FS_OUT = (100.0, 37.0)

#: The stacked mask: four bands, as a 3x4 sheet's column carries.
BANDS_ROWS, BANDS_COLS = 400, 91
BANDS_ROWS_AT = (40, 140, 240, 340)

#: Two bands 12 px apart, for the millimetre-versus-pixel semantics. See the
#: comment at `bands.close_*`: 1.5 mm at 8 px/mm, 0.375 mm at 32 px/mm.
BANDS_CLOSE_AT = (100, 112)


def trace_mask(rows: int, cols: int) -> np.ndarray:
    """A single wandering ink band with periodic dropout.

    Must match `PortFixtures.traceMask` exactly. Integer arithmetic only,
    for the same reason `lcg_plane` is: so that neither side can diverge
    through a floating-point detail the other does not share.
    """
    mask = np.zeros((rows, cols), dtype=bool)
    for c in range(cols):
        if c % TRACE_GAP_MOD < TRACE_GAP_WIDTH:
            continue
        top = 8 + (c * c) % TRACE_BAND_MOD
        for k in range(TRACE_THICKNESS):
            r = top + k
            if 0 <= r < rows:
                mask[r, c] = True
    return mask


def stacked_mask(rows: int, cols: int, at, thickness: int) -> np.ndarray:
    """Several straight bands, one per entry in `at`. A 3x4 sheet's column."""
    mask = np.zeros((rows, cols), dtype=bool)
    for top in at:
        mask[top: top + thickness, :] = True
    return mask


def lcg_plane(rows: int, cols: int) -> np.ndarray:
    """The shared input. Must match `SkewEstimator.lcgPlane` exactly."""
    state = LCG_SEED & LCG_MASK
    out = np.empty(rows * cols, dtype=np.float64)
    for i in range(rows * cols):
        state = (state * LCG_MULT + LCG_ADD) & LCG_MASK
        out[i] = float(state % 256)
    return out.reshape(rows, cols)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    lines: list[str] = []

    def emit(key: str, *values) -> None:
        lines.append(key + " " + " ".join(_fmt(v) for v in values))

    def _fmt(v) -> str:
        # Labels pass through. The format is `key token token ...`, so a token
        # may be a bare word provided it carries no space; the loader reads
        # numeric rows by index and never parses these.
        if isinstance(v, str):
            assert " " not in v, f"label {v!r} would break the row format"
            return v
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        # repr-round-trip precision: these are compared at 1e-12 relative, so
        # the text must not be the thing that loses the agreement.
        #
        # Non-finite values are spelled the way java.lang.Double.parseDouble
        # spells them, not the way repr() does: Python emits 'nan' and 'inf',
        # and parseDouble accepts only 'NaN' and 'Infinity'. A trace with
        # dropout is full of NaN, so this is not an edge case here.
        f = float(v)
        if f != f:
            return "NaN"
        if f == float("inf"):
            return "Infinity"
        if f == float("-inf"):
            return "-Infinity"
        return repr(f)

    lines.append("# Golden vectors for the Kotlin port of ecgproto/deskew.py.")
    lines.append("# GENERATED by python/make_port_fixtures.py -- do not hand-edit.")
    lines.append("# Regenerate after any change to the numerics on either side.")
    lines.append(f"# utc {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"# git {git_commit()}")
    lines.append(f"# python {platform.python_version()} numpy {np.__version__} "
                 f"scipy {scipy.__version__}")
    lines.append("")

    # ---- the shared input ------------------------------------------------
    emit("lcg.seed", LCG_SEED)
    emit("lcg.mult", LCG_MULT)
    emit("lcg.add", LCG_ADD)
    emit("lcg.mask", LCG_MASK)
    emit("lcg.rows", ROWS)
    emit("lcg.cols", COLS)

    plane = lcg_plane(ROWS, COLS)
    # A checksum, so that a divergence in the INPUT is reported as such
    # rather than surfacing as a confusing disagreement in every output.
    emit("lcg.sum", float(plane.sum()))
    emit("lcg.first8", *plane.ravel()[:8])
    lines.append("")

    # ---- sharpness, one angle at a time ----------------------------------
    # Core lengths fixed across the sweep, exactly as a real sweep does it,
    # so the centred crop is exercised rather than bypassed.
    len_h, len_v = dsk._core_lengths(plane.shape, dsk.COARSE_HALF_RANGE_DEG)
    emit("sharp.core_len_h", len_h)
    emit("sharp.core_len_v", len_v)
    emit("sharp.n", len(ANGLES))
    for i, a in enumerate(ANGLES):
        t = float(np.tan(np.deg2rad(-a)))
        ph = dsk._sheared_projection(plane, t, len_h)
        pv = dsk._sheared_projection(plane.T, -t, len_v)
        sh = dsk._sharpness(ph, "ssd") if ph is not None else 0.0
        sv = dsk._sharpness(pv, "ssd") if pv is not None else 0.0
        emit(f"sharp.{i}", a, sh, sv)
    lines.append("")

    # ---- sharpness at exact dyadic slopes --------------------------------
    # See SHEAR_TANS: this is the only part of the fixture that can catch a
    # round-half-to-even error, because it is the only part where the shift
    # lands exactly on a half-integer.
    emit("shear.n", len(SHEAR_TANS))
    for i, t in enumerate(SHEAR_TANS):
        ph = dsk._sheared_projection(plane, t, len_h)
        pv = dsk._sheared_projection(plane.T, -t, len_v)
        sh = dsk._sharpness(ph, "ssd") if ph is not None else 0.0
        sv = dsk._sharpness(pv, "ssd") if pv is not None else 0.0
        emit(f"shear.{i}", t, sh, sv)
    lines.append("")

    # ---- core lengths ----------------------------------------------------
    core_cases = [
        (708, 900, 10.0), (742, 900, 10.0), (695, 900, 10.0),
        (ROWS, COLS, 10.0), (COLS, ROWS, 10.0),
        (708, 900, 0.5), (708, 900, 3.5), (708, 900, 11.0),
        # Degenerate: a half-range wide enough that the formula goes
        # negative and the floor of 8 has to hold.
        (40, 900, 10.0), (12, 900, 44.0),
    ]
    emit("core.n", len(core_cases))
    for i, (r, c, hr) in enumerate(core_cases):
        lh, lv = dsk._core_lengths((r, c), hr)
        emit(f"core.{i}", r, c, hr, lh, lv)
    lines.append("")

    # ---- arange ----------------------------------------------------------
    # The sweep lengths come out of exactly this formula, and an off-by-one
    # at the end of a sweep changes which sample the parabola is fitted
    # around. Includes the two the shipped configuration actually uses.
    arange_cases = [
        (-dsk.COARSE_HALF_RANGE_DEG,
         dsk.COARSE_HALF_RANGE_DEG + 0.5 * dsk.COARSE_STEP_DEG,
         dsk.COARSE_STEP_DEG),
        (-0.5, 0.5 + 0.5 * dsk.FINE_STEP_DEG, dsk.FINE_STEP_DEG),
        (2.0 - 0.5, 2.0 + 0.5 + 0.5 * 0.05, 0.05),
        (-7.5 - 0.5, -7.5 + 0.5 + 0.025, 0.05),
        (0.0, 1.0, 0.25),
    ]
    emit("arange.n", len(arange_cases))
    for i, (start, stop, step) in enumerate(arange_cases):
        v = np.arange(start, stop, step)
        emit(f"arange.{i}.in", start, stop, step)
        emit(f"arange.{i}.out", len(v), v[0], v[-1])
    lines.append("")

    # ---- parabolic peak ---------------------------------------------------
    peak_cases = [
        # (angles, values) -- an interior maximum, the two edge cases that
        # must return the sample unrefined, and a flat triple whose
        # denominator is zero.
        ([0.0, 0.5, 1.0], [1.0, 3.0, 2.0]),
        ([0.0, 0.5, 1.0], [3.0, 1.0, 2.0]),
        ([0.0, 0.5, 1.0], [1.0, 2.0, 3.0]),
        ([0.0, 0.5, 1.0], [2.0, 2.0, 2.0]),
        ([-0.1, 0.0, 0.1, 0.2, 0.3], [1.0, 2.0, 5.0, 2.0, 1.0]),
        ([1.5, 1.55, 1.6, 1.65], [0.2, 0.9, 0.95, 0.1]),
    ]
    emit("peak.n", len(peak_cases))
    for i, (angles, values) in enumerate(peak_cases):
        got = dsk._parabolic_peak(np.array(angles), np.array(values))
        emit(f"peak.{i}.angles", *angles)
        emit(f"peak.{i}.values", *values)
        emit(f"peak.{i}.out", got)
    lines.append("")

    # ---- redness ----------------------------------------------------------
    # The channel-order trap: OpenCV on the desktop reads files as BGR, an
    # Android bitmap arrives as ARGB/RGBA. Same arithmetic, opposite order.
    # These cases are chosen so that a swapped r and b gives a DIFFERENT
    # answer, which a symmetric colour would hide.
    redness_cases = [
        (255, 255, 255),   # paper
        (254, 223, 219),   # ecg-image-kit minor rule
        (255, 0, 0),       # ecg-image-kit major rule
        (32, 32, 32),      # trace ink: clipped to zero
        (0, 0, 0),
        (200, 10, 90),     # asymmetric: catches a red/blue swap
        (90, 10, 200),
    ]
    emit("redness.n", len(redness_cases))
    for i, (r, g, b) in enumerate(redness_cases):
        v = max(0.0, r - 0.5 * (g + b))
        emit(f"redness.{i}", r, g, b, v)
    lines.append("")

    # ---- the tracer: FR5 to FR7 ------------------------------------------
    # Input is a mask built from the integer formula in `trace_mask`, so both
    # sides construct bit-identical pixels with nothing stored. Emitting the
    # arrays in full rather than a checksum is deliberate: when a value
    # moves, `git diff` should say which column, because an off-by-one in the
    # inked-span trim and a sign error in the conversion look identical
    # through a checksum.
    mask = trace_mask(TRACE_ROWS, TRACE_COLS)
    emit("trace.rows", TRACE_ROWS)
    emit("trace.cols", TRACE_COLS)
    emit("trace.gap_mod", TRACE_GAP_MOD)
    emit("trace.gap_width", TRACE_GAP_WIDTH)
    emit("trace.band_mod", TRACE_BAND_MOD)
    emit("trace.thickness", TRACE_THICKNESS)
    emit("trace.ink", int(mask.sum()))

    centroids = dig.trace_positions(mask)
    emit("trace.centroids", *centroids)

    # `digitise` trims to the first and last inked column before measuring
    # coverage, so that leading and trailing blank paper is not counted as
    # a tracing failure. The trim has to be ported too, and it is the kind
    # of boundary that survives a careless port by being nearly right.
    inked = np.flatnonzero(np.isfinite(centroids))
    trimmed = centroids[inked[0]: inked[-1] + 1]
    emit("trace.trim_lo", int(inked[0]))
    emit("trace.trim_hi", int(inked[-1]))
    emit("trace.coverage", float(np.isfinite(trimmed).mean()))

    filled = dig._interpolate_gaps(trimmed, TRACE_MAX_GAP_PX)
    emit("trace.max_gap_px", TRACE_MAX_GAP_PX)
    emit("trace.filled", *filled)
    # A second gap width, small enough that the runs this mask contains are
    # left NaN instead of bridged. Without this the "leave long gaps alone"
    # half of the function is never exercised.
    emit("trace.max_gap_px_tight", TRACE_MAX_GAP_TIGHT)
    emit("trace.filled_tight", *dig._interpolate_gaps(trimmed, TRACE_MAX_GAP_TIGHT))

    emit("trace.baseline", dig.baseline_from_trace(filled))

    # The sign convention. Rows grow down and millivolts grow up, so this is
    # the one line in the pipeline whose error produces a perfectly shaped,
    # perfectly inverted waveform -- PCC -1.0 rather than a visible failure.
    px_per_mv = TRACE_PX_PER_MM * dig.MM_PER_MV
    baseline = dig.baseline_from_trace(filled)
    mv = dig.pixels_to_mv(filled, baseline, px_per_mv)
    emit("trace.px_per_mm", TRACE_PX_PER_MM)
    emit("trace.px_per_mv", px_per_mv)
    emit("trace.mv", *mv)

    # Resampling onto a fixed output rate. One pixel column is one sample at
    # px_per_s, and n_out floors, so the last partial sample is dropped
    # rather than extrapolated.
    px_per_s = TRACE_PX_PER_MM * dig.MM_PER_S
    for j, fs_out in enumerate(TRACE_FS_OUT):
        duration_s = len(mv) / px_per_s
        n_out = int(np.floor(duration_s * fs_out))
        t_in = np.arange(len(mv)) / px_per_s
        t_out = np.arange(n_out) / fs_out
        emit(f"resample.{j}.in", px_per_s, fs_out, len(mv))
        emit(f"resample.{j}.out", *np.interp(t_out, t_in, mv))
    emit("resample.n", len(TRACE_FS_OUT))
    lines.append("")

    # ---- the FR8 layout check --------------------------------------------
    # Counts ink bands down a column. One band is a strip; four is a 3x4
    # sheet, which is the failure the app's own bundled fixtures would walk
    # into. Two masks: the single-band trace mask above, and an explicitly
    # stacked one.
    emit("bands.trace_mask", qual.traces_per_column(mask, TRACE_PX_PER_MM))
    stacked = stacked_mask(BANDS_ROWS, BANDS_COLS, BANDS_ROWS_AT, TRACE_THICKNESS)
    emit("bands.rows", BANDS_ROWS)
    emit("bands.cols", BANDS_COLS)
    emit("bands.at", *BANDS_ROWS_AT)
    emit("bands.stacked", qual.traces_per_column(stacked, TRACE_PX_PER_MM))
    # Same pixels, two recovered pitches. The separation is specified in
    # millimetres and converted with the pitch, so a FINER pitch -- more
    # pixels per millimetre -- raises the pixel threshold and merges bands
    # that a coarser one resolves. Note the direction: it is the opposite
    # way round from the first guess, and a check written in pixels would
    # return the same answer for both while changing meaning with camera
    # distance.
    #
    # The four-band mask above has its bands 100 px apart, which no
    # plausible pitch can merge, so the semantics are exercised on a
    # closely-spaced pair instead: 12 px apart is 1.5 mm at 8 px/mm (two
    # bands) and 0.375 mm at 32 px/mm (one).
    close = stacked_mask(BANDS_ROWS, BANDS_COLS, BANDS_CLOSE_AT, 1)
    emit("bands.close_at", *BANDS_CLOSE_AT)
    emit("bands.close_fine_pitch", 32.0)
    emit("bands.close_coarse_pitch", TRACE_PX_PER_MM)
    emit("bands.close_coarse", qual.traces_per_column(close, TRACE_PX_PER_MM))
    emit("bands.close_fine", qual.traces_per_column(close, 32.0))
    emit("bands.separation_mm", qual.BAND_SEPARATION_MM)
    emit("bands.column_stride", 7)
    lines.append("")

    # ---- reconstruction fidelity -----------------------------------------
    # The numbers the handset will quote for itself. Cases chosen so that the
    # three metrics DISAGREE: a scaled copy is perfect on PCC and wrong on
    # gain, an inverted one is -1.0 on PCC, and an offset one is invisible to
    # both PCC and gain while RMSE sees it. A fixture where all three agree
    # would not detect a metric wired to the wrong formula.
    ref = np.array([0.0, 0.2, 1.0, -0.3, 0.1, 0.4, -0.1, 0.0, 0.6, 0.15])
    fid_cases = [
        ("identical", ref.copy()),
        ("scaled_0.87", ref * 0.87),
        ("inverted", -ref),
        ("offset_0.5", ref + 0.5),
        ("offset_and_scaled", ref * 0.87 + 0.5),
        ("noisy", ref + np.array([0.01, -0.02, 0.03, 0.0, -0.01,
                                  0.02, -0.03, 0.01, 0.0, -0.02])),
        ("flat", np.zeros_like(ref)),
        # Unequal lengths, so the truncation is exercised. `align_length`
        # truncates rather than resampling on purpose: resampling would undo
        # a time-axis calibration error before any metric could see it.
        ("longer", np.concatenate([ref * 0.9, [0.3, 0.4, 0.5]])),
    ]
    emit("fid.ref", *ref)
    emit("fid.n", len(fid_cases))
    for i, (name, test) in enumerate(fid_cases):
        emit(f"fid.{i}.name", name)
        emit(f"fid.{i}.test", *test)
        emit(f"fid.{i}.out",
             evaluate.pcc(ref, test),
             evaluate.rmse_mv(ref, test),
             evaluate.gain_ratio(ref, test),
             len(evaluate.align_length(ref, test)[0]))
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}  ({len(lines)} lines)")
    print(f"  plane {ROWS}x{COLS}, sum {plane.sum():.1f}")
    print(f"  {len(ANGLES)} sharpness angles, {len(SHEAR_TANS)} dyadic slopes, "
          f"{len(core_cases)} core-length cases, "
          f"{len(arange_cases)} arange, {len(peak_cases)} peak, "
          f"{len(redness_cases)} redness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
