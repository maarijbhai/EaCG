"""Per-row time maps for the whole sheet, and the diagnosis that tests them.

A 3x4 row is one continuous 10 s trace of four leads, so an error in a row's
time scale shows as a lead-by-lead drift: each later cell's waveform sits
further from where the truth has it. `best_shift` measures each lead's shift
against the truth; `fit_row` fits shift against the cell's column (0-3), so
a slope in ms per cell is a scale error inside the row (implied error =
slope / 2,500 ms) and the intercept is a time-zero error.

`row_knots` builds the column -> time map four ways, so they can be compared
one change at a time on the same row traces:

    ticks         the shipped map (segment.segment_leads): time zero from the
                  median pulse edge of the four rows, every separator tick a
                  knot at 2.5 / 5.0 / 7.5 s (missing ticks borrowed from the
                  column-mates), the end extrapolated at the tick scale
    rowstart      as ticks, but each row's OWN pulse edge is its time zero
    rowstart_end  as rowstart, and the row's last inked column pinned to
                  10.0 s (the per-row duration anchor)
    grid          time zero as ticks, then uniform at 25 mm/s on the grid
                  fit's px/mm: no ticks, no row end (only on a grid page)
    grid_tickzero uniform at the grid's px/mm like `grid`, but time zero from
                  the TICKS instead of the pulse edge: each tick found at
                  k * 2.5 s implies an origin x - k * 2.5 s * px/s, and the
                  median over all ticks found is time zero (no pulse edge,
                  no row end). Measured reason: the pulse edge sits ~10 ms
                  early (results/kitphoto/timemap_*.csv, 5d1d80a).
    grid_tickzero_k2fix  as grid_tickzero, with the ticks moved 1 px left:
                  they are found on a 2x2-opened mask, which cv2 shifts +1 px
                  (row_end_x). A separate mode so the two effects are measured
                  one at a time.

A row whose map cannot be built (no own pulse, no detectable end, no grid
fit) is UNAVAILABLE: its leads are NaN and the reason is reported. Nothing
falls back silently to another map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import digitise as dig, segment

MODES = ("ticks", "rowstart", "rowstart_end", "grid", "grid_tickzero",
         "grid_tickzero_k2fix")
#: Largest shift searched, s. Well beyond any time-zero or scale error that
#: still leaves the waveform recognisable (1% of 10 s is 100 ms), and short of
#: half a beat at up to 150 bpm, so the neighbouring beat cannot win.
MAX_SHIFT_S = 0.2
#: Below this peak correlation an alignment says nothing about timing: a lead
#: the tracer lost still returns SOME shift inside the window.
MIN_PCC = 0.5
#: Fewer than this many usable leads in a row: no fit.
MIN_FIT_POINTS = 3


@dataclass(frozen=True)
class Shift:
    shift_ms: float      # + : the digitised waveform is LATE against the truth
    pcc: float           # correlation at that shift
    status: str          # "ok", "FAILED" (at the window edge), "UNRELIABLE"


def best_shift(truth: np.ndarray, sig: np.ndarray, fs: float,
               max_shift_s: float = MAX_SHIFT_S, min_pcc: float = MIN_PCC) -> Shift:
    """Integer-sample lag maximising PCC of truth[n] against sig[n + lag].

    Medians removed; NaN samples of `sig` excluded pairwise; each lag needs
    85% of the window to overlap.
    """
    t = np.asarray(truth, np.float64)
    s = np.asarray(sig, np.float64)
    n = min(len(t), len(s))
    t, s = t[:n], s[:n]
    m = int(round(max_shift_s * fs))
    best = (-np.inf, 0)
    for lag in range(-m, m + 1):
        a = t[max(0, -lag):n - max(0, lag)]
        b = s[max(0, lag):n - max(0, -lag)]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 0.85 * n:
            continue
        x, y = a[ok], b[ok]
        sd = x.std() * y.std()
        if sd <= 0:
            continue
        c = float(np.mean((x - x.mean()) * (y - y.mean())) / sd)
        if c > best[0]:
            best = (c, lag)
    c, lag = best
    if not np.isfinite(c):
        return Shift(float("nan"), float("nan"), "UNRELIABLE")
    status = "FAILED" if abs(lag) == m else ("ok" if c >= min_pcc else "UNRELIABLE")
    return Shift(1000.0 * lag / fs, c, status)


@dataclass(frozen=True)
class RowFit:
    n: int
    slope_ms_per_cell: float
    intercept_ms: float
    r2: float

    @property
    def scale_error_pct(self) -> float:
        """Implied time-scale error, %: + means the digitised row is
        STRETCHED (each later cell later; too few px per s assumed)."""
        return 100.0 * self.slope_ms_per_cell / (1000.0 * segment.CELL_S)


def fit_row(cols, shifts_ms) -> RowFit:
    """Least-squares shift = intercept + slope * column, over usable leads."""
    c = np.asarray(cols, np.float64)
    s = np.asarray(shifts_ms, np.float64)
    ok = np.isfinite(c) & np.isfinite(s)
    c, s = c[ok], s[ok]
    if len(c) < MIN_FIT_POINTS or np.ptp(c) == 0:
        return RowFit(len(c), float("nan"), float("nan"), float("nan"))
    slope, icpt = np.polyfit(c, s, 1)
    res = s - (icpt + slope * c)
    tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / tot if tot > 0 else float("nan")
    return RowFit(len(c), float(slope), float(icpt), r2)


def row_end_x(image: np.ndarray, layout: segment.LeadLayout, row: int,
              open_kernel: int = 2) -> float:
    """Last inked column of a row's trace, found independently of the time
    map: the row band from time zero to the page edge, binarised and with
    text removed (`isolate_trace` keeps only trace-wide components). NaN if
    the band has no trace."""
    y0, y1 = layout.row_bands[row]
    x0 = max(0, int(round(layout.t0_x)))
    band = image[y0:y1, x0:]
    mask = segment.isolate_trace(dig.binarise_trace(band, open_kernel=open_kernel))
    cols = np.flatnonzero(mask.any(axis=0))
    if not cols.size:
        return float("nan")
    # cv2's opening with an EVEN kernel has an off-centre anchor and moves
    # the mask one pixel right and down (measured: k = 2 and 4 shift +1, 3
    # and 5 do not). Undo it so the end is in the image's own columns, like
    # the pulse edge time zero comes from (a 3x3 opening).
    shift = 1 if open_kernel % 2 == 0 else 0
    return float(x0 + cols[-1] - shift)


def tick_origin(tick_positions, px_per_s: float, offset_px: float = 0.0) -> float:
    """Time zero implied by the ticks at a known scale: median over ticks of
    x - k * 2.5 s * px_per_s. NaN with fewer than segment.TICK_MIN_FOUND."""
    o = [x - offset_px - k * segment.CELL_S * px_per_s for _, k, x in (tick_positions or ())]
    return float(np.median(o)) if len(o) >= segment.TICK_MIN_FOUND else float("nan")


def row_knots(mode: str, layout: segment.LeadLayout, t0_per_row,
              row_ends=None, grid_px_per_mm: float | None = None,
              tick_positions=None):
    """Per-row (xs, ts) knots for `mode`, and {row: reason} for rows that
    cannot have them (knots None)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "ticks":
        return tuple(layout.row_knots), {}
    knots, unavailable = [], {}
    for r in range(4):
        xs0, ts0 = (list(v) for v in layout.row_knots[r])
        inner = [(x, t) for x, t in zip(xs0, ts0) if 0.0 < t < segment.RECORDING_S]
        if mode.startswith("grid"):
            if grid_px_per_mm is None or not np.isfinite(grid_px_per_mm):
                knots.append(None)
                unavailable[r] = "no grid fit"
                continue
            pps = grid_px_per_mm * dig.MM_PER_S
            x0 = xs0[0]
            if mode != "grid":
                x0 = tick_origin(tick_positions, pps,
                                 1.0 if mode == "grid_tickzero_k2fix" else 0.0)
                if not np.isfinite(x0):
                    knots.append(None)
                    unavailable[r] = f"fewer than {segment.TICK_MIN_FOUND} ticks"
                    continue
            knots.append(((x0, x0 + segment.RECORDING_S * pps),
                          (0.0, segment.RECORDING_S)))
            continue
        start = float(t0_per_row[r]) if t0_per_row is not None else float("nan")
        if not np.isfinite(start):
            knots.append(None)
            unavailable[r] = "row's own pulse not found"
            continue
        xs, ts = [start], [0.0]
        for x, t in inner:
            if x > xs[-1]:
                xs.append(x)
                ts.append(t)
        if mode == "rowstart":
            # The end exactly as the shipped map: extrapolated at the page's
            # tick scale (layout.px_per_s), as segment_leads does.
            xs.append(xs[-1] + (segment.RECORDING_S - ts[-1]) * layout.px_per_s)
            ts.append(segment.RECORDING_S)
        else:                                   # rowstart_end
            end = float(row_ends[r]) if row_ends is not None else float("nan")
            if not np.isfinite(end) or end <= xs[-1]:
                knots.append(None)
                unavailable[r] = ("row end not found" if not np.isfinite(end)
                                  else "row end before its last knot")
                continue
            xs.append(end)
            ts.append(segment.RECORDING_S)
        knots.append((tuple(xs), tuple(ts)))
    return tuple(knots), unavailable
