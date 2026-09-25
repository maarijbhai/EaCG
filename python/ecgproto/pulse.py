"""Measure the printed 1 mV calibration pulse: a gain anchor from the paper.

Limitation 12 lists the implied-gain check as impossible to build, because
the amplitude scale is *assumed* -- 10 mm/mV -- and an implied gain computed
from an assumption only hands the assumption back. The pulse breaks that
circle. Every clinical printer draws, at the start of each row, a rectangle
exactly 1 mV tall at the gain it printed with; its height in pixels is the
pixels-per-millivolt of that print, measured from something whose physical
meaning is known rather than from the grid the rest of the calibration uses.

What it is good for, in order of certainty:

1. **Catching a wrong mm/mV.** A 5 mm/mV print read as 10 halves every
   amplitude and nothing else in the pipeline can see it. The pulse reads
   5 mm there, not 10 -- a factor of two, far outside any measurement error.
2. **Catching a wrong vertical pitch.** The pulse is 10 mm tall at the
   conventional gain, so height_px / px_per_mm_y should come out at 10; a
   vertical harmonic lock makes it 2 or 50.
3. **Calibrating amplitude directly**, bypassing the vertical grid pitch.
   Measured to about 1% (below); whether that beats the grid is for the
   before-and-after to say, not this docstring.

METHOD. The pulse is found where `segment.locate_rhythm_strip` found it: a
rising edge that is the first ink of the row, a flat top, and a falling edge
at `t0_x`. Its height is the mean vertical ink extent of the RISING edge's
columns minus the mean ink thickness of the flat top's columns -- an edge
column spans the pulse from the baseline line's bottom to the top line's
top, one line thickness more than the pulse. Only the rising edge: the
falling edge shares columns with the trace's first deflection, which on
PTB-XL 9 adds 5 px and reads the pulse 4% tall.

MEASURED (2026-09-24, not yet committed as a CSV; the gate check that uses
this carries its own before-and-after): 9.99 mm on three of the four clean
kit sheets at 300 dpi and 9.86-9.96 mm on simulated photographs of all four
-- a bias of up to -1.4%, from binarising a blurred edge at half its level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segment import StripLocation

#: How far left of the recording start the pulse is searched for, as a
#: fraction of the recording width. The pulse is 5 mm of ~250, 2%.
SEARCH_FRACTION = 0.04


@dataclass(frozen=True)
class Pulse:
    height_px: float          # 1 mV, in pixels, at the print's own gain
    line_px: float            # ink thickness of the pulse's flat top
    edge_columns: int         # columns averaged on the rising edge
    top_columns: int          # columns averaged on the flat top

    def implied_mm_per_mv(self, px_per_mm_y: float) -> float:
        """What mm/mV the print used, given a vertical pitch."""
        return self.height_px / px_per_mm_y


def measure_pulse(mask: np.ndarray, loc: StripLocation) -> Pulse | None:
    """The pulse ahead of the rhythm strip `loc`, or None if none is found.

    `mask` is the full page's trace mask -- the same one the strip was
    located on -- because the pulse lies LEFT of the strip's crop. None is
    returned, not a guess, when the locator cut no pulse or the region does
    not show a rise, a top and a fall: a missing anchor must read as
    unavailable, never as a measurement.
    """
    if loc.pulse_x1 is None or not np.isfinite(loc.t0_x):
        return None
    width = float(loc.x1) - loc.t0_x
    x_lo = max(0, int(loc.t0_x - SEARCH_FRACTION * width))
    x_hi = min(mask.shape[1], int(np.ceil(loc.t0_x)) + 2)
    reg = np.asarray(mask[loc.y0:loc.y1, x_lo:x_hi], dtype=bool)
    counts = reg.sum(axis=0)
    if counts.max() == 0:
        return None
    tall = np.flatnonzero(counts >= 0.5 * counts.max())
    clusters = np.split(tall, np.flatnonzero(np.diff(tall) > 1) + 1)
    if len(clusters) < 2:
        return None
    rise, fall = clusters[0], clusters[1]
    top = np.arange(rise[-1] + 1, fall[0])
    top = top[counts[top] > 0]
    if top.size == 0:
        return None

    extents = []
    for j in rise:
        rows = np.flatnonzero(reg[:, j])
        extents.append(rows[-1] - rows[0] + 1)
    line = float(np.mean(counts[top]))
    return Pulse(height_px=float(np.mean(extents)) - line, line_px=line,
                 edge_columns=len(rise), top_columns=len(top))
