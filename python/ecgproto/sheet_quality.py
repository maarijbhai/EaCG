"""Whole-sheet quality warnings (FR8): is this digitisation to be trusted?

The rhythm-strip gate (`quality.py`) judges one finished trace; a whole
sheet fails earlier and differently -- a wrong time reference silently
mistimes every lead that depends on it, at PCC 0.1, while each lead still
looks like a plausible ECG. So these checks look at the TIME AND GEOMETRY
REFERENCES the sheet was cut with, plus a physiological backstop per lead.

Every check is anchored on something printed or physiological, never on
the pipeline's own self-consistency (CLAUDE.md, Conventions): the 5 mm grid
and the separator ticks are two independent printed rulers, the ticks of
one boundary are printed in one column, and a lead's amplitude is bounded
by cardiac physiology. Thresholds come from the nine registered photographs
of PTB-XL 9, 38, 40 and 57 at the phone's 3,200 px decode, grid geometry
(probe 2026-10-07; to be committed as results/kitphoto/warnings.csv):

  grid vs tick scale     good |diff| <= 0.28%; the 3x3-opening phone
                         failure (eacg_sheet_ptbxl9_..._104102_c38) ~ -33%
  tick vs its boundary   distance from the median of the same tick over
                         the rows (one bad tick cannot move a median of
                         three; against the mean of its mates it dragged
                         the two good ones 2.7 mm too): good <= 0.24 mm;
                         ptbxl40_20260929_02's row-3 7.5 s tick 5.49 mm
                         early, which mistimed V3, V6 and the rhythm strip
                         (PCC 0.10 / 0.10 / 0.51)
  ticks found            8-9 of 9 on every good photo; the failure 3
  pulses found           1-4 of 4 on good photos (PTB-XL 38 scores 0.955
                         from ONE), so only "none" is a warning
  lead coverage (b3)     >= 0.979 on every good lead
  lead peak-to-peak      0.24-4.11 mV on every phone lead; the band is
                         quality.REJECT_PP_MV, the physiological anchor
  row spacing            largest / smallest gap between the four rows
                         1.07-1.13 on every good photo; 2.21 on a photo
                         whose top shadow was taken for a row (leads under
                         the wrong names) -- segment.ROW_SPACING_MAX_RATIO

Two severities: WARN (the result may be off; look at the overlay) and
SEVERE (a time or geometry reference is known wrong; do not use the
affected leads). The sheet's verdict is the worst of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quality import REJECT_PP_MV
from .segment import LAYOUT_3X4, ROW_SPACING_MAX_RATIO, TICK_OUTLIER_MM, row_gap_ratio

#: |tick scale / grid scale - 1| above this: the two printed rulers disagree.
SCALE_DISAGREE_FRACTION = 0.01
#: A tick further than this from the median of the same tick over the rows.
#: The same threshold `segment.tick_scale` rejects outliers with: with all
#: three rows found, an outlier is rejected there and reported here as
#: TICK_REPLACED; with two, nothing says which is wrong, and both are
#: TICK_OUTLIER.
TICK_MATE_MM = TICK_OUTLIER_MM
#: Fewer ticks than this (of 9) found: the time map rests on few markers.
MIN_TICKS = 6
#: A lead with less than this fraction of its samples traced.
MIN_LEAD_COVERAGE = 0.8

WARN = "warn"
SEVERE = "severe"


@dataclass(frozen=True)
class Warning:
    code: str
    severity: str
    message: str
    leads: tuple[str, ...] = ()     # lead names affected; "II_rhythm" for the strip


def verdict(warnings: list[Warning]) -> str:
    """"ok", "check" (any WARN) or "unreliable" (any SEVERE)."""
    if any(w.severity == SEVERE for w in warnings):
        return "unreliable"
    return "check" if warnings else "ok"


def lead_stats(mv: np.ndarray) -> tuple[float, float]:
    """(fraction of samples traced, peak-to-peak mV); pp NaN if none traced."""
    v = np.asarray(mv, np.float64)
    fin = np.isfinite(v)
    if not fin.any():
        return 0.0, float("nan")
    return float(fin.mean()), float(v[fin].max() - v[fin].min())


def _tick_leads(row: int, k: int) -> tuple[str, ...]:
    """The leads timed by tick k (1-3) of row `row`: the two cells it joins,
    and the rhythm strip, whose knots are the three rows' mean."""
    return (LAYOUT_3X4[row][k - 1], LAYOUT_3X4[row][k], "II_rhythm")


def assess(*, grid_requested: bool, grid_used: bool, grid_px_per_mm: float,
           calibration: str, px_per_mm: float, ticks_found: int,
           tick_positions, t0_source: str, leads: dict,
           tick_rejected=(), rows=None, t0_guard_pct: float | None = None) -> list[Warning]:
    """Warnings for one digitised sheet.

    `t0_guard_pct`: None unless the guarded time-zero fallback was USED
    (segment.guard_time_zero, progress_log §41); then the tick-vs-grid
    disagreement (%) that made it fire, NaN if there was no tick scale. It is
    always SEVERE: after the fallback the ticks agree with the grid by
    construction, so SCALE_DISAGREE no longer fires, and a time zero re-read
    from the ticks can still be a whole cell out (62.5 mm apart; measured on
    Set C Tilt) -- the verdict must not read better than before.

    `tick_positions`: (row, k, x) per tick kept (TickScale.positions);
    `tick_rejected`: (row, k, x) per tick rejected as an outlier
    (TickScale.rejected);
    `leads`: name -> mV samples, the rhythm strip as "II_rhythm".
    """
    out: list[Warning] = []
    if rows is not None and len(rows) == 4 and row_gap_ratio(rows) > ROW_SPACING_MAX_RATIO:
        g = np.diff(np.sort(np.asarray(rows, np.float64)))
        gaps = " / ".join(f"{x / px_per_mm:.0f}" for x in g) if np.isfinite(px_per_mm) and px_per_mm > 0 \
            else " / ".join(f"{x:.0f} px" for x in g)
        out.append(Warning("ROW_SPACING", SEVERE,
                           f"The four rows found are not evenly spaced (gaps {gaps} mm); a shadow or a "
                           "mark may have been taken for a row, so leads may be missing or under the "
                           "wrong names."))
    if grid_requested and not grid_used:
        out.append(Warning("GRID_REFUSED", WARN,
                           "The printed grid could not be fitted; the page "
                           "geometry comes from the four corners only."))
    if calibration != "ticks":
        out.append(Warning("NO_TICK_SCALE", SEVERE,
                           f"Only {ticks_found} of 9 lead-separator ticks found; "
                           "the time scale is a fallback estimate."))
    elif ticks_found < MIN_TICKS:
        out.append(Warning("FEW_TICKS", WARN,
                           f"Only {ticks_found} of 9 lead-separator ticks found."))
    if (calibration == "ticks" and grid_used and np.isfinite(grid_px_per_mm)
            and grid_px_per_mm > 0):
        d = px_per_mm / grid_px_per_mm - 1.0
        if abs(d) > SCALE_DISAGREE_FRACTION:
            out.append(Warning("SCALE_DISAGREE", SEVERE,
                               f"The ticks and the grid disagree on the page scale "
                               f"by {100 * d:+.1f}%."))
    if calibration == "ticks" and np.isfinite(px_per_mm) and px_per_mm > 0:
        pos = {(int(r), int(k)): float(x) for r, k, x in tick_positions}
        for (r, k), x in sorted(pos.items()):
            # The median over the rows that found this tick, itself included.
            # With two, it is their mean and both are flagged past 2 mm apart:
            # nothing says which one is wrong. Alone, a tick is not judged.
            same = [pos[(rr, k)] for rr in range(3) if (rr, k) in pos]
            if len(same) < 2:
                continue
            off_mm = (x - float(np.median(same))) / px_per_mm
            if abs(off_mm) > TICK_MATE_MM:
                out.append(Warning("TICK_OUTLIER", SEVERE,
                                   f"Row {r + 1}'s {2.5 * k:.1f} s tick is {off_mm:+.1f} mm "
                                   "from the same tick in the other rows.",
                                   _tick_leads(r, k)))
    if calibration == "ticks" and np.isfinite(px_per_mm) and px_per_mm > 0:
        kept = {(int(r), int(k)): float(x) for r, k, x in tick_positions}
        for r, k, x in sorted((int(r), int(k), float(x)) for r, k, x in tick_rejected):
            mates = [kept[(rr, k)] for rr in range(3) if (rr, k) in kept]
            off = f" by {(x - float(np.mean(mates))) / px_per_mm:+.1f} mm" if mates else ""
            out.append(Warning("TICK_REPLACED", WARN,
                               f"Row {r + 1}'s {2.5 * k:.1f} s tick was out of line{off}; "
                               "the same tick in the other rows was used instead.",
                               _tick_leads(r, k)))
    if t0_guard_pct is not None:
        why = (f"the ticks and the grid disagreed by {t0_guard_pct:+.1f}%" if np.isfinite(t0_guard_pct)
               else "no tick scale was found")
        out.append(Warning("T0_GUARD", SEVERE,
                           f"The calibration pulse was not trusted ({why}); time zero was re-read "
                           "from the lead-separator ticks. Check the timing on the page overlay."))
    if t0_source not in ("pulses", "ticks"):
        out.append(Warning("NO_PULSE", WARN,
                           "No calibration pulse found; time zero is estimated "
                           "from the start of the rhythm strip."))
    for name, mv in leads.items():
        cov, pp = lead_stats(mv)
        if cov < MIN_LEAD_COVERAGE:
            out.append(Warning("LEAD_COVERAGE", WARN,
                               f"{name}: only {100 * cov:.0f}% of the lead was traced.",
                               (name,)))
        elif not (REJECT_PP_MV[0] <= pp <= REJECT_PP_MV[1]):
            out.append(Warning("LEAD_AMPLITUDE", WARN,
                               f"{name}: peak-to-peak {pp:.2f} mV is outside "
                               f"{REJECT_PP_MV[0]}-{REJECT_PP_MV[1]} mV.", (name,)))
    return out


def assess_result(res, grid_requested: bool) -> list[Warning]:
    """`assess` on a `sheet.SheetResult`."""
    used = res.geometry == "grid" and res.grid_fit is not None
    leads = {("II_rhythm" if rhythm else name): mv for (name, rhythm), mv in res.leads.items()}
    return assess(grid_requested=grid_requested, grid_used=used,
                  grid_px_per_mm=res.grid_fit.px_per_mm if used else float("nan"),
                  calibration=res.calibration, px_per_mm=res.px_per_mm,
                  ticks_found=res.ticks_found,
                  tick_positions=res.ticks.positions if res.ticks is not None else (),
                  t0_source=res.t0_source, leads=leads,
                  tick_rejected=res.ticks.rejected if res.ticks is not None else (),
                  rows=res.location.rows if res.location is not None else None,
                  t0_guard_pct=(res.t0_guard.disagree_before_pct
                                if getattr(res, "t0_guard", None) is not None and res.t0_guard.accepted
                                else None))
