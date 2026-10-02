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

#: How far past the first ink the pulse's rising edge may start, as a
#: fraction of the strip width, provided every column before it is thin
#: (baseline-sized, see PULSE_TAIL_FACTOR). 5% is about 13 mm. On the
#: first A30s photograph of PTB-XL 40 a grid fragment 20 px long sat at the
#: strip's left edge with the rising edge 27 px in -- 1% of the width -- so
#: the pulse was never declared, it was traced as signal, and the duration
#: anchor counted its width: a 0.34 s lag and about +3% on the anchor. With
#: the corners on the sheet edge the lead-in is the whole ~8 mm margin (85
#: px, 3%). A lead-in before a pulse cannot be waveform -- the printer draws
#: the pulse first -- but one before an early QRS can look like it, which
#: is what PULSE_MIN_WIDTH_FRACTION guards.
PULSE_LEADIN_FRACTION = 0.05

#: After a lead-in, the pulse's rising and falling edges must be at least
#: this far apart, as a fraction of the strip width: ~3 mm, against the
#: pulse's 5 mm (0.2 s). The longer lead-in above is what makes this
#: necessary. 1.5% was enough for run 3's corners, which cut the sheet's
#: margin; with corners on the sheet edge the kit's pulse starts ~8 mm in,
#: behind 85 px of thin grid-rule ink, and was missed again (duration
#: anchor -4.5% against the truth). A lead-in that long could precede an
#: early QRS instead of a pulse, and a QRS's two edges sit within about a
#: millimetre of each other; a pulse's are a flat top apart.
PULSE_MIN_WIDTH_FRACTION = 0.012

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

#: A narrow piece continues the trace past an end if it starts within this
#: fraction of the strip width of that end (~5 mm on a 10 s strip)...
END_GAP_FRACTION = 0.02

#: ...and its rows come within this fraction of the strip height of the rows
#: the trace occupies at that end.
END_ROW_TOLERANCE = 0.10

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
    narrow = []
    for i in 1 + np.argsort(-stats[1:, cv2.CC_STAT_WIDTH], kind="stable"):
        x, w = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        if w < min_w:
            narrow.append(i)
            continue
        span = covered[x:x + w]
        if span.mean() < MAX_SHARED_COLUMNS:
            keep[i] = True
            covered[x:x + w] = True

    # A narrow piece that fills a hole INSIDE the kept trace is trace. Text
    # sits in columns the trace already covers, or beyond its ends; it never
    # fills columns the trace leaves empty between two kept stretches. On
    # the A30s photograph of PTB-XL 40 at 3200 px a thin stroke broke before
    # the beat at 9.14 s, the beat and its ST segment became an 80 px piece
    # -- under the 3% (81 px) width floor -- and were dropped as text: a
    # 0.24 s gap with an R wave in it.
    on = np.flatnonzero(covered)
    if on.size:
        first, last = int(on[0]), int(on[-1])
        for i in narrow:
            x, w = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
            if x <= first or x + w - 1 >= last:
                continue
            if covered[x:x + w].mean() < MAX_SHARED_COLUMNS:
                keep[i] = True
                covered[x:x + w] = True

    # ...and so is a narrow piece that CONTINUES the trace past either end:
    # it starts within END_GAP_FRACTION of the end and at the height the
    # trace is at there. Nothing is printed beyond a strip's ends, and the
    # lead label sits in columns the trace covers, so text cannot pass both
    # tests. Repeated, so a chain of pieces is followed to the true end. Phone
    # run 6 of PTB-XL 40 lost its last 0.84 s -- the tenth R wave with it --
    # to pieces 16-80 px wide beyond the last kept stretch.
    if np.any(keep):
        gap = max(2, int(round(m.shape[1] * END_GAP_FRACTION)))
        tol = max(2, int(round(m.shape[0] * END_ROW_TOLERANCE)))
        grew = True
        while grew:
            grew = False
            on = np.flatnonzero(covered)
            first, last = int(on[0]), int(on[-1])
            kept = keep[labels] & (m > 0)
            for i in narrow:
                if keep[i]:
                    continue
                x, w = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
                y, hh = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
                if last < x <= last + gap:
                    ref = kept[:, max(0, last - gap):last + 1]
                elif first - gap <= x + w - 1 < first:
                    ref = kept[:, first:first + gap + 1]
                else:
                    continue
                r = np.flatnonzero(ref.any(axis=1))
                if r.size and y <= r[-1] + tol and y + hh - 1 >= r[0] - tol:
                    keep[i] = True
                    covered[x:x + w] = True
                    grew = True
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


#: Rows added above and below the strip for the local skew estimate, as a
#: fraction of the page height: ~15 mm each side on a kit sheet, enough to
#: take in six or more 5 mm rules.
LEVEL_PAD_FRACTION = 0.08

#: A local tilt is removed only when the estimate is a real peak (margin,
#: optimum over the objective 1 degree away) and small. Larger is not a
#: residual of rectification but a failure of it, and rotating would hide it.
LEVEL_MIN_MARGIN = 1.5
LEVEL_MAX_DEG = 2.0


@dataclass(frozen=True)
class LevelledStrip:
    image: np.ndarray       # the strip crop, tilt removed (or as cropped)
    angle_deg: float        # estimated local tilt; + is anticlockwise
    margin: float
    applied: bool


def level_strip(image: np.ndarray, loc: StripLocation) -> LevelledStrip:
    """Crop the rhythm strip with its OWN residual tilt removed.

    A page rectified from imperfect corners is not uniformly level. On the
    A30s photograph of PTB-XL 40 (phone run 5) the 5 mm rules were level at
    the top of the page and sloped 12 px over 2200 px -- 0.31 degrees -- in
    the rhythm band at the bottom, where the placed corners were 6-9 mm off.
    Whole-page deskew measured 0.10 degrees and could not remove it; the
    trace drifted 1.55 mm against the truth across the strip. So the tilt is
    estimated on a band around the strip itself, from the grid rules in it,
    and removed before the strip is cut out.

    The band is rotated about its own centre with no expansion. Rows move by
    at most (width/2) * tan(angle) -- ~7 px at 0.3 degrees on a 2800 px
    page -- well inside the LEVEL_PAD_FRACTION margin, so the white fill of
    the rotation never reaches the strip. Columns move by under a pixel, so
    the strip's x-coordinates, t0 included, still hold.
    """
    from . import deskew as dsk              # deskew imports digitise too
    h = image.shape[0]
    pad = int(h * LEVEL_PAD_FRACTION)
    b0, b1 = max(0, loc.y0 - pad), min(h, loc.y1 + pad)
    band = image[b0:b1, loc.x0:loc.x1]
    est = dsk.estimate_skew(band)
    angle, margin = float(est["angle_deg"]), float(est["margin"])
    ok = margin >= LEVEL_MIN_MARGIN and abs(angle) <= LEVEL_MAX_DEG
    if not ok:
        return LevelledStrip(loc.crop(image), angle, margin, False)
    rotated = dsk.rotate_image(band, -angle, expand=False)
    return LevelledStrip(rotated[loc.y0 - b0:loc.y1 - b0], angle, margin, True)


def find_peaks_distance(x: np.ndarray, distance: int) -> np.ndarray:
    """`scipy.signal.find_peaks(x, distance=distance)`, with ties made stable.

    scipy visits peaks tallest first and drops any within `distance` of one
    already kept, ordering them with numpy's default quicksort -- which is
    not stable, so between two equal heights the survivor is an accident of
    the sort. A port cannot reproduce that, and `StripLocator.findPeaks`
    does not try: it sorts stably, so among equal heights the rightmost is
    visited first. This does the same, and is identical to scipy wherever
    no two local maxima are tied. Measured on the port fixture's integer-
    valued test vector, scipy kept 20 peaks and the stable rule 22.
    """
    peaks = sps.find_peaks(x)[0]
    if distance <= 1 or peaks.size < 2:
        return peaks
    keep = np.ones(peaks.size, dtype=bool)
    for j in np.argsort(x[peaks], kind="stable")[::-1]:
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < peaks.size and peaks[k] - peaks[j] < distance:
            keep[k] = False
            k += 1
    return peaks[keep]


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
    # Sum, then divide once: on integer row counts the sum is exact, so a
    # flat top stays exactly flat and `StripLocator.convolveSameBox` agrees
    # to the bit. Scaling each term by 1/win first left rounding noise that
    # the two ports resolved into different peaks on the same plateau.
    smooth = np.convolve(rows, np.ones(win), mode="same") / win

    peaks = find_peaks_distance(smooth, max(1, int(h * MIN_ROW_SPACING_FRACTION)))
    if len(peaks) < n_rows:
        raise RuntimeError(
            f"found {len(peaks)} trace rows, expected {n_rows}; is the image "
            "cropped to the page, and is it a 3x4 sheet with a rhythm strip?")
    # Stable, so equal heights keep the upper row -- as `sortedByDescending`
    # does in StripLocator.kt. `argsort(...)[::-1]` reversed the tie order.
    peaks = np.sort(peaks[np.argsort(-smooth[peaks], kind="stable")][:n_rows])

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
    base = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0
    # A thin lead-in (a grid fragment, a pen landing) may precede the
    # rising edge; skip it, then look for the pulse from where it ends.
    lead = 0
    lead_max = int(width * PULSE_LEADIN_FRACTION)
    while lead < min(lead_max, width) and counts[lead] <= PULSE_TAIL_FACTOR * base:
        lead += 1
    if lead >= lead_max:
        lead = 0                          # no tall column early on: no lead-in
    search = counts[lead:lead + max(2, int(width * PULSE_SEARCH_FRACTION))]
    start = counts[lead:lead + max(1, int(width * PULSE_START_FRACTION))]
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
    wide_enough = (lead == 0 or len(clusters) < 2 or
                   clusters[1][0] - clusters[0][-1] >= width * PULSE_MIN_WIDTH_FRACTION)
    if len(clusters) >= 2 and start.max() >= 0.5 * search.max() and wide_enough:
        first, last = lead + int(clusters[1][0]), lead + int(clusters[1][-1])
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


# --------------------------------------------------------------------------
# FR4: lead segmentation of the standard 3x4 + rhythm layout
# --------------------------------------------------------------------------

#: The standard 3x4 layout, row by row; each row is one continuous 10 s
#: trace of four consecutive 2.5 s leads, and the fourth trace row is the
#: lead II rhythm strip. Lead identity comes from POSITION, not from reading
#: the printed labels: a stated scope limit (other layouts are out of scope).
LAYOUT_3X4 = (("I", "aVR", "V1", "V4"),
              ("II", "aVL", "V2", "V5"),
              ("III", "aVF", "V3", "V6"))
CELL_S = 2.5
RECORDING_S = 10.0

#: Half-width of the blanked zone around each internal cell boundary, mm.
#: Printers mark the boundary with a short vertical tick that crosses the
#: trace (ecg-image-kit draws it at 3x the trace width); traced, it is a
#: full-height spike. The samples lost are 2 x 1 mm = 80 ms per boundary,
#: interpolated over like any other gap.
SEPARATOR_GUARD_MM = 1.0


@dataclass(frozen=True)
class LeadCell:
    name: str
    row: int                # 0-2 the 3x4 block, 3 the rhythm strip
    col: int                # 0-3; 0 for the rhythm strip
    t_start_s: float        # where this lead's window starts in the recording
    duration_s: float
    x0: int
    x1: int
    y0: int
    y1: int
    baseline_y: float       # the row's trace centre (row-profile peak)


@dataclass(frozen=True)
class LeadLayout:
    cells: tuple[LeadCell, ...]     # 12 leads then the rhythm strip
    t0_x: float                     # recording time zero, sheet column
    px_per_s: float                 # from the caller's calibration
    row_y: tuple[int, ...]          # the four row centres
    row_bands: tuple[tuple[int, int], ...]

    def cell(self, name: str, rhythm: bool = False) -> LeadCell:
        for c in self.cells:
            if c.name == name and (c.row == 3) == rhythm:
                return c
        raise KeyError(name)


def segment_leads(image: np.ndarray, loc: StripLocation,
                  px_per_mm_x: float) -> LeadLayout:
    """Cut a page-cropped 3x4 + rhythm sheet into its 13 traces.

    Rows come from `loc.row_peaks`, the row-profile peaks the rhythm-strip
    search already found; each band runs halfway to its neighbours. Columns
    come from time: recording zero is the rhythm strip's `t0_x` (the
    calibration pulse's falling edge), and cell k of a 3x4 row covers
    [k * 2.5 s, (k+1) * 2.5 s) at `px_per_mm_x` * 25 mm/s. That ASSUMES every
    row starts at the rhythm strip's t0, which the standard layout does (the
    kit puts all four at 15.0 mm); a row printed with its own offset would
    put every cell of it out by that offset.

    The calibration is the caller's, as for the rhythm strip: the duration
    anchor on a photograph. Nothing here measures scale.
    """
    h, w = image.shape[:2]
    peaks = sorted(int(p) for p in loc.row_peaks)
    if len(peaks) != 4:
        raise RuntimeError(f"{len(peaks)} rows found; the 3x4 + rhythm layout has 4")
    bands = []
    for i, y in enumerate(peaks):
        up = (y - peaks[i - 1]) / 2 if i > 0 else (peaks[1] - peaks[0]) / 2
        dn = (peaks[i + 1] - y) / 2 if i < 3 else (y - peaks[2]) / 2
        bands.append((max(0, int(round(y - up))), min(h, int(round(y + dn)))))
    px_per_s = px_per_mm_x * dig.MM_PER_S
    t0 = float(loc.t0_x)
    cells = []
    for r, names in enumerate(LAYOUT_3X4):
        for k, name in enumerate(names):
            x0 = t0 + k * CELL_S * px_per_s
            cells.append(LeadCell(name, r, k, k * CELL_S, CELL_S,
                                  int(round(x0)), int(round(x0 + CELL_S * px_per_s)),
                                  bands[r][0], bands[r][1], float(peaks[r])))
    cells.append(LeadCell("II", 3, 0, 0.0, RECORDING_S, int(round(t0)),
                          int(round(t0 + RECORDING_S * px_per_s)),
                          bands[3][0], bands[3][1], float(peaks[3])))
    return LeadLayout(tuple(cells), t0, px_per_s, tuple(peaks), tuple(bands))


def trace_row(image: np.ndarray, layout: LeadLayout, row: int,
              open_kernel: int = 3, tracked: bool = True
              ) -> tuple[np.ndarray, int]:
    """Classical FR5/FR6 over one whole row: (row per column, x origin).

    Binarise and isolate over the full 10 s band -- the row's own trace is
    the widest component there, which is what lets `isolate_trace` drop the
    QRS tips of the rows above and below that reach into the band -- then
    take the column centroid, and blank the separator ticks at the cell
    boundaries (SEPARATOR_GUARD_MM either side).
    """
    y0, y1 = layout.row_bands[row]
    x0 = int(round(layout.t0_x))
    x1 = int(round(layout.t0_x + RECORDING_S * layout.px_per_s))
    x1 = min(x1, image.shape[1])
    band = image[y0:y1, x0:x1]
    mask = isolate_trace(dig.binarise_trace(band, open_kernel=open_kernel))
    if not tracked:
        pos = dig.trace_positions(mask) + y0
    else:
        # Track each cell from its own middle: a 3x4 row changes lead at
        # every boundary, and the jump there is the signal's, not a fault.
        px_mm = layout.px_per_s / dig.MM_PER_S
        pos = np.full(mask.shape[1], np.nan)
        n_cells = 4 if row < 3 else 1
        span = (CELL_S if row < 3 else RECORDING_S) * layout.px_per_s
        for k in range(n_cells):
            a = int(round(k * span))
            b = mask.shape[1] if k == n_cells - 1 else int(round((k + 1) * span))
            pos[a:b] = trace_positions_tracked(mask[:, a:b], px_mm)
        pos = pos + y0
    if row < 3:
        guard = SEPARATOR_GUARD_MM * layout.px_per_s / dig.MM_PER_S
        for k in (1, 2, 3):
            xb = k * CELL_S * layout.px_per_s
            lo, hi = int(np.floor(xb - guard)), int(np.ceil(xb + guard))
            pos[max(0, lo):max(0, min(len(pos), hi + 1))] = np.nan
    return pos, x0


#: Furthest the tracked run may move between adjacent columns, mm, plus one
#: pixel per column skipped, up to twice this in all. A steep stroke
#: overlaps the previous run (distance 0), so this only bites when a
#: column's ONLY ink is somewhere else -- measured on the clean PTB-XL 40
#: sheet at the start of V3, V5 and V6, where the kit draws the first
#: millimetres of the trace broken and the label is the sole ink in the gaps,
#: ~4-5 mm below. Uncapped, the allowance grew across a 0.3 s gap until it
#: reached the label and the tracker followed the label instead.
TRACK_MAX_JUMP_MM = 1.5


def trace_positions_tracked(mask: np.ndarray,
                            px_per_mm: float | None = None) -> np.ndarray:
    """Row of the trace per column, following ONE ink run column to column.

    `digitise.trace_positions` takes the centroid of all ink in a column,
    which is right when the trace is the only ink there. In a 3x4 row it is
    not at the start of every cell: the printed lead label ("aVR", "V6")
    sits just under the trace's first ~0.1-0.3 s, survives isolate_trace
    (wider than its 3% cut over a 10 s band), and the all-ink centroid lands
    between label and trace -- measured on the CLEAN printed PTB-XL 40 sheet,
    a 0.4 mV false dip that cost aVR and V6 a third of their PCC.

    Here each column's ink is split into vertical runs and the run nearest
    the previous column's chosen run is kept (interval distance, so a
    steep stroke that overlaps the last run always wins). Tracking starts
    from the widest stretch of single-run columns and goes outward both
    ways; a column with no ink is NaN and does not move the reference.
    Where a column holds one run this equals the all-ink centroid exactly.
    With `px_per_mm`, a column whose nearest run is more than
    TRACK_MAX_JUMP_MM away (plus a pixel per skipped column) is a gap, not
    a jump.
    """
    h, w = mask.shape
    runs = []
    for x in range(w):
        col = mask[:, x]
        if not col.any():
            runs.append([])
            continue
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs.append(list(zip(st.tolist(), en.tolist())))
    out = np.full(w, np.nan)
    single = np.array([len(r) == 1 for r in runs])
    if not single.any():
        return dig.trace_positions(mask)
    # Seed: the middle of the longest stretch of single-run columns.
    idx = np.flatnonzero(single)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    segs = np.split(idx, breaks + 1)
    seed_seg = max(segs, key=len)
    seed = int(seed_seg[len(seed_seg) // 2])

    def pick(rs, ref):
        a, b = ref
        best, bd = None, None
        for s, e in rs:
            dist = max(0, s - b, a - e)     # 0 when the intervals overlap
            if bd is None or dist < bd or (dist == bd and (e - s) > (best[1] - best[0])):
                best, bd = (s, e), dist
        return best

    jump = TRACK_MAX_JUMP_MM * px_per_mm if px_per_mm else np.inf
    for direction in (1, -1):
        ref = runs[seed][0]
        x, skipped = seed, 0
        while 0 <= x < w:
            rs = runs[x]
            if rs:
                s, e = pick(rs, ref)
                if max(0, s - ref[1], ref[0] - e) <= jump + min(skipped, jump):
                    out[x] = 0.5 * (s + e - 1)
                    ref, skipped = (s, e), 0
                else:
                    skipped += 1
            else:
                skipped += 1
            x += direction
    return out


#: Rows of a 3x4 + rhythm sheet are ~33-36 mm apart; peaks of the row ink
#: profile closer than this fraction of the page height (~25 mm on A4
#: landscape) are one row's QRS and text, not two rows. `locate_rhythm_strip`
#: uses 0.07 and keeps the four TALLEST peaks, which on unevenly lit photos
#: (ptbxl40_20260929_01, _03) dropped the faint top row for a phantom row
#: between row 3 and the rhythm strip.
SHEET_ROW_SPACING_FRACTION = 0.11

#: The calibration pulse: 1 mV tall at 10 mm/mV, 0.2 s (5 mm) wide, printed
#: at the start of every row. Its TOP edge, a horizontal segment 10 mm above
#: the row's centre, survives where its thin vertical edges vanish on a soft
#: photo; its right end is the falling edge, recording time zero.
PULSE_TOP_MM = 10.0
PULSE_SLAB_MM = 2.5             # search +-2.5 mm around the expected top
PULSE_WIDTH_MM = (3.0, 7.5)     # accepted length of the top segment
PULSE_GAP_MM = 0.5              # gaps closed inside the segment
PULSE_SEARCH_FRACTION_X = 0.15  # pulses lie in the left 15% of the page


@dataclass(frozen=True)
class SheetLocation:
    rows: tuple[int, ...]           # the four row centres, top to bottom
    t0_x: float                     # recording time zero (pulses' falling edge)
    t0_per_row: tuple[float, ...]   # each row's own pulse, NaN where not found
    x_end: int                      # last inked column of the rhythm strip
    rhythm: StripLocation           # locate_rhythm_strip's own answer
    t0_source: str                  # "pulses" or "rhythm_locator"

    @property
    def anchor_px_per_mm(self) -> float:
        """The duration anchor over the whole recording: 10 s at 25 mm/s."""
        return (self.x_end - self.t0_x) / (RECORDING_S * dig.MM_PER_S)

    def as_strip_location(self) -> StripLocation:
        from dataclasses import replace
        return replace(self.rhythm, row_peaks=self.rows, t0_x=self.t0_x)


def sheet_rows(image: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, ...]:
    """The four trace rows of a page-cropped 3x4 + rhythm sheet, by layout."""
    h = image.shape[0]
    m = dig.binarise_trace(image) if mask is None else np.asarray(mask, bool)
    rows = m.sum(axis=1).astype(np.float64)
    edge = int(h * EDGE_FRACTION)
    rows[:edge] = 0.0
    rows[h - edge:] = 0.0
    win = max(5, int(h * SMOOTH_FRACTION)) | 1
    smooth = np.convolve(rows, np.ones(win), mode="same") / win
    peaks = find_peaks_distance(smooth, max(1, int(h * SHEET_ROW_SPACING_FRACTION)))
    if len(peaks) < 4:
        raise RuntimeError(f"found {len(peaks)} trace rows, expected 4")
    top4 = np.sort(peaks[np.argsort(-smooth[peaks], kind="stable")][:4])
    return tuple(int(p) for p in top4)


def pulse_falling_edge(mask: np.ndarray, row_y: int, px_per_mm: float) -> float:
    """Right end of the pulse's top segment in one row, or NaN."""
    h, w = mask.shape
    yc = row_y - PULSE_TOP_MM * px_per_mm
    a = max(0, int(round(yc - PULSE_SLAB_MM * px_per_mm)))
    b = min(h, int(round(yc + PULSE_SLAB_MM * px_per_mm)))
    if b <= a:
        return float("nan")
    cols = mask[a:b, :int(w * PULSE_SEARCH_FRACTION_X)].any(axis=0)
    idx = np.flatnonzero(cols)
    if idx.size == 0:
        return float("nan")
    gap = max(1, int(round(PULSE_GAP_MM * px_per_mm)))
    breaks = np.flatnonzero(np.diff(idx) > gap)
    first = idx[:breaks[0] + 1] if breaks.size else idx
    length_mm = (first[-1] - first[0] + 1) / px_per_mm
    if not (PULSE_WIDTH_MM[0] <= length_mm <= PULSE_WIDTH_MM[1]):
        return float("nan")
    # The falling edge is a line of finite width whose centre is time zero;
    # the segment's last inked column is its right-hand side.
    line_px = 0.25 * px_per_mm
    return float(first[-1]) - 0.5 * line_px


def locate_sheet(image: np.ndarray) -> SheetLocation:
    """Rows, recording time zero and the duration anchor for a whole sheet.

    Rows from `sheet_rows` (layout spacing). Time zero from the pulses of
    all four rows (median of those found), falling back to the rhythm
    locator's own t0 when none is found. The scale needed to place the
    pulse search comes from the rhythm locator's anchor; a t0 error of the
    size measured (up to 1.5 s, 15%) leaves the search slab on the pulse.
    """
    m = dig.binarise_trace(image)
    rows = sheet_rows(image, m)
    loc = locate_rhythm_strip(image, mask=m)
    approx = loc.recording_width_px / (RECORDING_S * dig.MM_PER_S)
    edges = tuple(pulse_falling_edge(m, y, approx) for y in rows)
    found = [e for e in edges if np.isfinite(e)]
    if found:
        t0, src = float(np.median(found)), "pulses"
    else:
        t0, src = float(loc.t0_x), "rhythm_locator"
    return SheetLocation(rows=rows, t0_x=t0, t0_per_row=edges, x_end=int(loc.x1),
                         rhythm=loc, t0_source=src)
