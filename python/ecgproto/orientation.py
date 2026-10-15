"""Which way up is the sheet? The orientation rule for camera photographs.

A phone stores which way up a photograph is in its EXIF tag, from the tilt
sensor at the moment of capture. Shooting a sheet on a table from above,
the phone lies flat, gravity says nothing about up, and the tag keeps the
orientation the phone last knew -- usually portrait. A landscape shot then
arrives sideways, or upside down, and the pipeline, which expects an
upright landscape page, fails.

Two questions, answered by two different cues:

1. **Sideways or not** (a quarter turn): the printed sheet is landscape
   (277 x 190 mm), so the corner finder's outline taller than wide means
   sideways. Measured (`python/run_orientation.py`,
   results/kitphoto/orientation.csv): 76 of 76 right over the kit
   photographs, the in-app camera photographs and the clean sheets, each
   turned four ways; upright outlines 1.18-2.11 wide-to-high, sideways
   0.47-0.85.

2. **Up or down** (a half turn), on the page printed: the pipeline's own
   lead markers, read on the GRID-FIT page and on that page turned half
   round (`half_turn_markers`). The calibration pulses stand at the LEFT
   of each row, rising from the baseline, so the pulse search finds them
   only on the upright page; and the separator ticks stand in the top
   three rows only, so on a page turned half round one of the three rows
   the tick search reads is the rhythm strip, and it can find at most
   six of the nine (`MAX_TICKS_TURNED`). `upside_down` weighs the two.

   Replaced (2026-10-15, progress_log §39): turning the way the user LAST
   confirmed, which was wrong whenever the grip changed between shots and
   never caught a sheet upside down. Also rejected earlier (§35): the
   pulse count on the CORNER page (20/36 -- coarse corners displace the
   pulse search) and the pulses' vertical strokes in the 2x2 mask (the
   opening breaks them); and in §39 a vertical ink centroid (fails on
   tightly framed photos, whose corner quad cuts off the empty top margin)
   and matched filters for the pulse shape (the 5 mm grid squares match
   too). The grid-fit page is what makes the markers usable: it places the
   pulse and tick searches at the grid's own scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Width / height below which the sheet's outline is sideways.
SIDEWAYS_BELOW = 1.0

#: The most ticks a page turned half round can show: the tick search reads
#: the top three rows, and on a turned page one of them is the rhythm
#: strip, which carries none (3 rows x 3 ticks, less one row).
MAX_TICKS_TURNED = 6


def outline_aspect(corners: np.ndarray) -> float:
    """Width / height of the corner quad (TL, TR, BR, BL), opposite sides averaged."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
    return float(w / h)


def is_sideways(corners: np.ndarray) -> bool:
    return outline_aspect(corners) < SIDEWAYS_BELOW


@dataclass(frozen=True)
class Markers:
    """What the pipeline's own marker searches find on one candidate page."""
    pulses: int      # rows whose calibration pulse was found (0-4)
    ticks: int       # separator ticks found (0-9)


def upside_down(up: Markers, turned: Markers) -> bool:
    """Is the page turned half round the upright one? Evidence in order:

    1. ticks: more than MAX_TICKS_TURNED on one page only -- a turned page
       cannot show that many, so that page is upright;
    2. pulses: more found on one page (they are at the rows' left);
    3. ticks: more found on one page.
    A tie keeps the page as it is (the Rotate button is the override).
    """
    many_up, many_turned = up.ticks > MAX_TICKS_TURNED, turned.ticks > MAX_TICKS_TURNED
    if many_up != many_turned:
        return many_turned
    if up.pulses != turned.pulses:
        return turned.pulses > up.pulses
    return turned.ticks > up.ticks


def page_markers(page: np.ndarray, scale_hint: float | None) -> Markers:
    """`segment.locate_sheet` + `segment.tick_scale` on a page, as `sheet.digitise_sheet`
    calls them; a page they cannot read at all counts as no markers."""
    from . import segment
    try:
        sl = segment.locate_sheet(page, scale_hint)
        tk = segment.tick_scale(page, sl)
    except (RuntimeError, ValueError):
        return Markers(0, 0)
    return Markers(int(sum(np.isfinite(e) for e in sl.t0_per_row)), int(tk.n_found))


def half_turn_markers(photo: np.ndarray, corners: np.ndarray) -> tuple[Markers, Markers, str]:
    """Markers on the page as photographed and on the same page turned half round.

    The page is made as `sheet.digitise_sheet(geometry="grid")` makes it
    (corner rectification, then the grid fit; the corner page if the fit is
    refused). The grid page turned half round is the page the photograph
    turned half round would give, so one fit serves both candidates.
    Returns (up, turned, geometry used).
    """
    from . import gridfit, segment, sheet
    rect, _ = sheet.rectify(photo, corners)
    hint, used = None, "corners"
    try:
        fit = gridfit.fit_grid(rect, segment.locate_sheet(rect).anchor_px_per_mm)
        rect, _ = gridfit.regularise(photo, corners, rect.shape, fit)
        hint, used = fit.px_per_mm, "grid"
    except (RuntimeError, ValueError):
        pass
    return (page_markers(rect, hint),
            page_markers(np.ascontiguousarray(rect[::-1, ::-1]), hint), used)


def quarter_turns(photo: np.ndarray, find_corners) -> tuple[int, dict]:
    """Clockwise quarter turns that make the photographed sheet upright (the app's rule).

    `find_corners(image) -> corners` is the automatic corner finder. A
    sideways outline is turned a quarter clockwise and its corners found
    again; then the half-turn markers decide up from down. Returns
    (quarters 0-3, what was measured). No corners: 0, nothing decided.
    """
    info: dict = {}
    try:
        c = find_corners(photo)
    except (RuntimeError, ValueError) as e:
        info["error"] = f"corners: {e}"
        return 0, info
    q = 0
    info["aspect"] = round(outline_aspect(c), 3)
    if is_sideways(c):
        q = 1
        photo = np.ascontiguousarray(np.rot90(photo, -1))
        try:
            c = find_corners(photo)
        except (RuntimeError, ValueError) as e:
            info["error"] = f"corners after the quarter turn: {e}"
            return q, info
    up, turned, used = half_turn_markers(photo, c)
    info.update(up=up, turned=turned, geometry=used)
    if upside_down(up, turned):
        q += 2
    return q % 4, info
