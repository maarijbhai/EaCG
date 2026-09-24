"""Locate the rhythm strip on a 3x4 printout: the smallest useful piece of FR4.

A full lead segmenter finds all thirteen regions of a clinical sheet. This
module finds one of them -- the full-width rhythm strip at the bottom -- and
that one is worth having on its own. It is the only region of a 3x4 printout
that is a single continuous 10 s trace, so it is the only region `digitise`
can be handed as-is, and the only one against which a 10 s duration anchor
holds. Everything this repository measures on photographs goes through it.

**Input contract: a page, not a photograph.** The image must already be
cropped to the printed sheet -- by the four-corner rectification in
`score_kit_photo.py` today, by FR1 (ML Kit Document Scanner) later. Handed a
raw photograph, the row-ink profile also sees the table, the shadow and the
background, and the four tallest bands are no longer the four trace rows.
That was measured, not assumed: on the three usable 2026-09-21 frames,
deskewed but not cropped, the box misses the hand-measured rhythm strip in
all three. The function does not detect this; the caller owns the crop.

Method, in order, and why each step is the way it is:

1. **Row ink density.** The trace rows of a printout are where the ink is,
   so the row sum of a trace mask has one broad hump per printed row. The
   top and bottom 5% are zeroed, because a deskewed image carries a padded
   border and a rectified one carries a replicated edge, both of which can
   ink.
2. **The four tallest well-separated humps, and the bottom one.** A 3x4
   sheet with a rhythm strip has four trace rows. Tallest-four rather than
   all peaks, because the calibration pulses and any printed text make small
   humps of their own. The rhythm strip is the bottom row by the layout
   convention of every clinical printer this project has seen, including
   ecg-image-kit's `full_mode`.
3. **Bounds at the valley.** The top edge is the minimum of the profile
   between the rhythm row and the row above it; the bottom edge is the same
   distance below the hump. A fixed height would either clip a tall QRS or
   take in the row above, and the valley is where neither happens.
4. **The longest inked run of columns**, after closing gaps shorter than
   3.5% of the width. The closing bridges the flat stretches where a thin
   trace momentarily loses ink; the longest run discards stray marks.
5. **Printed text is removed and the box shrunk to the trace.** Lead names
   and "25mm/s 10mm/mV" sit inside the band and ink like trace; they are
   dropped by `isolate_trace` on shape, and the box is cut to the rows and
   columns the remaining trace occupies. Callers should trace the strip
   with `isolate_trace` applied too, since text can still share columns
   with the trace inside the final box.
6. **The calibration pulse is cut off the left.** It is inked, it is tall,
   and it sits flush against the start of the trace. Counting it as trace
   corrupts the centroid and -- worse -- inflates the inked width that the
   10 s duration anchor divides by: on ecg-image-kit's `ecg00001` that is
   12.644 px/mm with the pulse against 11.704 without, an 8% calibration
   error from one rectangle. A pulse is declared only when the run *starts*
   with a tall column (its rising edge); its falling edge is then the last
   column in the first 3% at least half as tall, and the trace starts just
   after it. Known miss: a trace that begins mid-QRS with no pulse printed
   looks like a pulse and loses up to 3% of its width.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import signal as sps

from . import digitise as dig

#: Trace rows on a 3x4 printout with a rhythm strip.
N_ROWS = 4

#: Fraction of the image height ignored at the top and at the bottom.
EDGE_FRACTION = 0.05

#: Row-profile smoothing window, as a fraction of image height. About 4 mm
#: on an A4 sheet: wide enough to merge a trace row's QRS spikes into one
#: hump, narrow enough not to merge two rows.
SMOOTH_FRACTION = 0.02

#: Minimum spacing between trace rows, as a fraction of image height. A 3x4
#: sheet puts rows about a quarter of the height apart; this is well below
#: that and well above the hump width.
MIN_ROW_SPACING_FRACTION = 0.07

#: Gap closing along the strip, as a fraction of image width.
CLOSE_FRACTION = 0.035

#: Region at the left of the strip searched for the calibration pulse's
#: falling edge, as a fraction of the strip width. The pulse is 5 mm of
#: about 255 mm, 2%; 3% leaves room for a pulse printed a little wide.
PULSE_SEARCH_FRACTION = 0.03

#: Region at the very left that must hold the pulse's rising edge for a
#: pulse to be declared at all. 0.5% is about 1.3 mm: the rising edge is the
#: first ink of the run when a pulse is printed.
PULSE_START_FRACTION = 0.005

#: A column right of the pulse's falling edge still belongs to the edge
#: while its ink count exceeds this many times the strip's median column
#: count. The edge is anti-aliased over a pixel or two; a trace column at
#: baseline carries a few pixels of ink, an edge remnant tens.
PULSE_TAIL_FACTOR = 3.0


#: Smallest connected ink component kept as trace, as a fraction of the
#: strip width. 3% is about 7.5 mm on a 10 s strip: wider than any printed
#: character (ecg-image-kit's are about 2.5 mm), narrower than any stretch
#: of trace between dropouts on the photographs measured so far.
TRACE_MIN_WIDTH_FRACTION = 0.03

#: Dilation used only to decide connectivity in `isolate_trace`, as a
#: fraction of the strip width: about 0.5 mm on a 10 s strip at any
#: resolution. 5 px rejoins every kit sheet at 11.8 px/mm; 3 px leaves one
#: of four split in two.
BRIDGE_FRACTION = 0.002

#: A candidate component is kept only if less than this fraction of its
#: columns are already covered by wider kept components. Trace pieces meet
#: end to end and share a few columns at most; text lies wholly under or
#: over the trace and shares all of them.
MAX_SHARED_COLUMNS = 0.5

#: Rows kept above and below the trace's own extent, as a fraction of the
#: image height. About 2 mm on an A4 sheet: room for anti-aliasing and
#: blur, not enough to reach the line of text under the strip.
ROW_PAD_FRACTION = 0.01


def isolate_trace(mask: np.ndarray,
                  min_width_fraction: float = TRACE_MIN_WIDTH_FRACTION
                  ) -> np.ndarray:
    """Keep only the ink components wide enough to be trace.

    Printed text -- lead names, "25mm/s", "10mm/mV" -- is inked exactly like
    trace, so no threshold separates it; its shape does. A trace is one
    long connected run across the strip, or a few long ones where the ink
    thins out; a character is a small blob. Components narrower than
    `min_width_fraction` of the strip are removed. The column centroid of a
    column holding a character and the trace would otherwise sit between
    the two, which is how the first version of this module scored PCC 0.30
    on a clean sheet whose trace was otherwise perfectly recovered.

    What it cannot remove: text that touches the trace, which then belongs
    to the trace's component. Not seen on any sheet so far.
    """
    m = np.asarray(mask, dtype=np.uint8)
    if not m.any():
        return m.astype(bool)
    # Label on a slightly dilated copy, keep original pixels. The opening in
    # `binarise_trace` erodes the one-pixel steep slopes of a QRS, so the
    # undilated trace falls apart into one piece per beat -- 12 to 23
    # pieces on the kit sheets -- and the short first and last pieces were
    # then dropped as if they were text, costing up to 43 px of recording.
    # A bridge of about half a millimetre rejoins every sheet into a single
    # component while leaving characters separate.
    k = max(3, int(round(m.shape[1] * BRIDGE_FRACTION)) | 1)
    bridged = cv2.dilate(m, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    min_w = max(2, int(round(m.shape[1] * min_width_fraction)))

    # Greedy column coverage, widest first. The bridge that rejoins the
    # trace also joins the letters of "25mm/s" into a word 15 mm wide, so
    # width alone cannot tell a word from a stretch of trace. Where they
    # sit can: text is printed in columns the trace already occupies, and a
    # genuine piece of trace fills columns nothing else does.
    covered = np.zeros(m.shape[1], dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for i in 1 + np.argsort(-stats[1:, cv2.CC_STAT_WIDTH]):
        x, w = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        if w < min_w:
            break
        span = covered[x:x + w]
        if span.mean() < MAX_SHARED_COLUMNS:
            keep[i] = True
            covered[x:x + w] = True
    return keep[labels] & (m > 0)


@dataclass(frozen=True)
class StripLocation:
    """Where the rhythm strip is, and what the search saw on the way."""

    x0: int
    y0: int
    x1: int
    y1: int
    row_peaks: tuple[int, ...]      # centres of the trace rows found
    pulse_x1: int | None            # right edge of the pulse, if one was cut
    t0_x: float = float("nan")      # column where the recording begins

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def recording_width_px(self) -> float:
        """Width the recording spans, for a duration anchor.

        From `t0_x` -- the pulse's falling edge when a pulse was printed --
        to the last inked column, **not** the crop width: the crop starts a
        few pixels later, clear of the edge's anti-aliasing, and those
        pixels are recording time. Both ends include half a line width of
        ink, so this runs long by about one line width, roughly 0.1% on a
        300 dpi sheet. That is the size of limitation 14's whole budget, so
        it is stated rather than rounded away.
        """
        return float(self.x1) - self.t0_x

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y0:self.y1, self.x0:self.x1]


def locate_rhythm_strip(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    n_rows: int = N_ROWS,
) -> StripLocation:
    """Find the bottom full-width strip of a page-cropped 3x4 printout.

    `mask` defaults to `digitise.binarise_trace(image)`; pass one to locate
    with a different binarisation than the one used to trace. Raises
    `RuntimeError` when fewer than `n_rows` trace rows are found, rather
    than guessing which of the rows it did find is the rhythm strip.
    """
    h, w = image.shape[:2]
    m = dig.binarise_trace(image) if mask is None else np.asarray(mask, bool)

    rows = m.sum(axis=1).astype(np.float64)
    edge = int(h * EDGE_FRACTION)
    rows[:edge] = 0.0
    rows[h - edge:] = 0.0
    win = max(5, int(h * SMOOTH_FRACTION)) | 1
    smooth = np.convolve(rows, np.ones(win) / win, mode="same")

    peaks, _ = sps.find_peaks(smooth, distance=max(1, int(h * MIN_ROW_SPACING_FRACTION)))
    if len(peaks) < n_rows:
        raise RuntimeError(
            f"found {len(peaks)} trace rows, expected {n_rows}; is the image "
            "cropped to the page, and is it a 3x4 sheet with a rhythm strip?")
    peaks = np.sort(peaks[np.argsort(smooth[peaks])[::-1]][:n_rows])

    band, above = int(peaks[-1]), int(peaks[-2])
    y0 = above + int(np.argmin(smooth[above:band]))
    y1 = min(h, band + (band - y0))

    inked = (m[y0:y1].sum(axis=0) > 0).astype(np.uint8)
    k = max(3, int(w * CLOSE_FRACTION))
    closed = cv2.morphologyEx(inked.reshape(1, -1), cv2.MORPH_CLOSE,
                              np.ones((1, k), np.uint8)).ravel()
    idx = np.flatnonzero(closed)
    if idx.size == 0:
        raise RuntimeError("the rhythm row carries no ink")
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    # Trim back to real ink. OpenCV's closing treats the image border as
    # foreground, so a trace starting within half a kernel of the edge gets
    # a run reaching column 0 -- measured on a render with an 8 mm margin.
    real = run[inked[run] > 0]
    x0, x1 = int(real[0]), int(real[-1]) + 1

    # The calibration pulse: a rectangle whose rising edge is the first ink
    # of the run. Its signature is a tall column at the very start, which a
    # trace beginning at baseline does not have. A QRS inside the first
    # PULSE_SEARCH_FRACTION is tall too, which is why "tall columns near the
    # left" alone is not the test -- it would cut the first beat.
    # Drop printed text before anything is measured from the columns or rows.
    # ecg-image-kit prints the lead name under the start of the strip and
    # "25mm/s  10mm/mV" further down; a clinical printer prints the same
    # sort of thing. Both sit inside the band found above, and the bottom
    # line of text is where the first cut of this function put y1.
    trace = isolate_trace(m[y0:y1, x0:x1])
    rows_on = np.flatnonzero(trace.any(axis=1))
    if rows_on.size == 0:
        raise RuntimeError("the rhythm row carries no trace-sized ink")
    pad = max(2, int(h * ROW_PAD_FRACTION))
    y0, y1 = max(y0, y0 + int(rows_on[0]) - pad), min(y1, y0 + int(rows_on[-1]) + 1 + pad)
    cols_on = np.flatnonzero(trace.any(axis=0))
    x0, x1 = x0 + int(cols_on[0]), x0 + int(cols_on[-1]) + 1
    trace = isolate_trace(m[y0:y1, x0:x1])

    counts = trace.sum(axis=0).astype(np.float64)
    width = x1 - x0
    search = counts[:max(2, int(width * PULSE_SEARCH_FRACTION))]
    start = counts[:max(1, int(width * PULSE_START_FRACTION))]
    pulse_x1 = None
    t0_x = float(x0)
    tall_idx = np.flatnonzero(search >= 0.5 * search.max()) if search.max() > 0 \
        else np.array([], dtype=int)
    clusters = np.split(tall_idx, np.flatnonzero(np.diff(tall_idx) > 1) + 1) \
        if tall_idx.size else []
    # Rising edge, flat top, falling edge: the falling edge is the SECOND
    # cluster of tall columns, never simply the last one. A trace that
    # leaves the pulse with a steep deflection makes a third cluster within
    # the search window (PTB-XL 38 does, 3 px after the edge), and taking
    # the last would start the recording at that deflection.
    if len(clusters) >= 2 and start.max() >= 0.5 * search.max():
        first, last = int(clusters[1][0]), int(clusters[1][-1])
        # The recording starts where the pulse falls back to baseline: the
        # centre of the falling edge, not its right-hand side, because the
        # edge is a line of finite width.
        t0_x = x0 + 0.5 * (first + last)
        pulse_x1 = x0 + last + 1
        # The crop starts once the anti-aliased tail of the edge is gone --
        # a partial edge column left in would be traced as a spike.
        base = float(np.median(counts[counts > 0]))
        j = last + 1
        while j < width and counts[j] > PULSE_TAIL_FACTOR * base:
            j += 1
        x0 = x0 + j

    return StripLocation(x0=x0, y0=int(y0), x1=x1, y1=int(y1),
                         row_peaks=tuple(int(p) for p in peaks),
                         pulse_x1=pulse_x1, t0_x=float(t0_x))
