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
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        # repr-round-trip precision: these are compared at 1e-12 relative, so
        # the text must not be the thing that loses the agreement.
        return repr(float(v))

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
