"""The whole sheet, photograph to 12 leads: the reference for the phone.

One function, `digitise_sheet`, that does on the desktop exactly what the
phone app is to do, so the Kotlin port (`SheetPipeline.kt`) has a single
reference to be golden-vector tested against and so its accuracy can be
measured before the port exists:

    photograph
    -> corners           page.find_sheet_corners (no clicks), or given
    -> rectify           size from the corners alone: the phone cannot know
                         the sheet's true size (rectify, as score_photo)
    -> regularise        optional (geometry="grid"): gridfit.fit_grid makes
                         the printed 5 mm grid exactly regular, one warp
                         from the photograph
    -> locate            segment.locate_sheet: four rows, time zero
    -> calibrate         segment.tick_scale: the separator ticks, falling
                         back to the duration anchor
    -> segment           segment.segment_leads with the tick knots
    -> trace each row    a row tracer: classical (segment.trace_row) or the
                         learned one (canonical crop, tiled model call)
    -> cut into leads    on_sheet_columns + cell_signal, 500 Hz, mV

Framework-free: the learned tracer enters as a callable on a canonical tile
(`model_row_tracer(call, height)`), so the same code runs a Keras model, a
LiteRT interpreter, or a test double.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import canonical, digitise as dig, gridfit, page, segment

FS = 500.0
#: How far before recording time zero the learned tracer's canonical crop
#: starts, mm: the pulse partly in view, as in training (lead-in -2..14 mm).
LEAD_IN_MM = 4.0


def rectify(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, dict]:
    """Perspective-warp the quad to a rectangle. Returns (image, geometry).

    Output size is the mean length of each pair of opposite edges, measured
    in the photograph. Nothing about the sheet's true dimensions enters
    here; that knowledge is reserved for scoring the result. (Moved from
    score_photo.py, which imports it; PhotoPipeline.kt does the same.)
    """
    tl, tr, br, bl = np.asarray(corners, np.float64)
    edge = lambda a, b: float(np.hypot(*(a - b)))
    top, bottom = edge(tl, tr), edge(bl, br)
    left, right = edge(tl, bl), edge(tr, br)
    out_w = int(round((top + bottom) / 2.0))
    out_h = int(round((left + right) / 2.0))
    if out_w < 16 or out_h < 16:
        raise ValueError(
            f"the four corners enclose only {out_w}x{out_h} px; "
            "they are probably in the wrong order or units")
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    geometry = {
        "edge_top_px": top, "edge_bottom_px": bottom,
        "edge_left_px": left, "edge_right_px": right,
        # Opposite edges of a rectangle photographed square are equal; the
        # ratio is a direct read on how oblique the camera was.
        "keystone_horizontal": max(top, bottom) / min(top, bottom),
        "keystone_vertical": max(left, right) / min(left, right),
        "out_w": out_w, "out_h": out_h,
    }
    return warped, geometry


def on_sheet_columns(xs: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, int]:
    """A trace at arbitrary columns -> one row per whole sheet column.

    Identity for the classical tracer (already one per column). For a model,
    interpolates between its outputs and keeps a sheet column NaN where the
    nearer outputs were NaN, so a presence gap stays a gap.
    """
    xs = np.asarray(xs, np.float64)
    rows = np.asarray(rows, np.float64)
    grid = np.arange(int(np.ceil(xs[0])), int(np.floor(xs[-1])) + 1)
    fin = np.isfinite(rows)
    if fin.sum() < 2:
        return np.full(len(grid), np.nan), int(grid[0])
    out = np.interp(grid, xs[fin], rows[fin], left=np.nan, right=np.nan)
    gap = np.interp(grid, xs, (~fin).astype(np.float64)) > 0.5
    out[gap] = np.nan
    return out, int(grid[0])


def cell_signal(pos: np.ndarray, x_origin: int, cell: segment.LeadCell,
                px_per_mm_x: float, px_per_mm_y: float,
                layout: segment.LeadLayout | None = None) -> np.ndarray:
    """A cell's slice of a row trace -> mV at FS over the cell's window.

    Time from the row's column -> time map when the layout has one (tick
    knots), else from the cell's start at `px_per_mm_x`. mV relative to the
    cell's median.
    """
    x = np.arange(len(pos)) + x_origin
    keep = (x >= cell.x0) & (x < cell.x1) & np.isfinite(pos)
    n = int(round(cell.duration_s * FS))
    if keep.sum() < 10:
        return np.full(n, np.nan)
    xs, rs = x[keep].astype(np.float64), pos[keep]
    if layout is not None and layout.row_knots:
        t = layout.time_at(cell.row, xs) - cell.t_start_s
    else:
        t = (xs - cell.x0) / (px_per_mm_x * dig.MM_PER_S)
    mv = -(rs - np.median(rs)) / (px_per_mm_y * dig.MM_PER_MV)
    ts = np.arange(n) / FS
    return np.interp(ts, t, mv, left=np.nan, right=np.nan)


def classical_row_tracer(open_kernel: int = 2):
    def tracer(img, layout, row):
        pos, x0 = segment.trace_row(img, layout, row, open_kernel)
        return x0 + np.arange(len(pos), dtype=np.float64), pos
    return tracer


def model_row_tracer(call, height: int, lead_in_mm: float = LEAD_IN_MM):
    """A row tracer from a model call on canonical tiles.

    `call(batch, training=False) -> {"row": (1, 2W), "presence": (1, 2W)}`
    on an RGB float/uint8 batch -- the deploy interface (soft-argmax in the
    graph), as the phone's LiteRT model has.
    """
    def tracer(img, layout, row):
        px_mm = layout.px_per_s / dig.MM_PER_S
        canon, cm = canonical.to_canonical(img, px_mm, layout.t0_x - lead_in_mm * px_mm,
                                           float(layout.row_y[row]), height)
        out = canonical.predict_tiled(call, canon[..., ::-1].copy())
        return canonical.row_trace_to_source(out["row"], out["presence"], cm)
    return tracer


@dataclass
class SheetResult:
    leads: dict                     # (name, rhythm) -> mV at FS over the lead's window
    corners: np.ndarray
    corners_refined: bool | None    # None when corners were given
    rect_shape: tuple
    px_per_mm: float
    calibration: str                # "ticks" or "duration"
    ticks_found: int
    t0_source: str
    layout: segment.LeadLayout = field(repr=False)
    geometry: str = "corners"       # what was actually used
    grid_fit: gridfit.GridFit | None = field(default=None, repr=False)


def digitise_sheet(photo: np.ndarray, tracer, corners: np.ndarray | None = None,
                   geometry: str = "corners") -> SheetResult:
    """Photograph of a 3x4 + rhythm printout -> every lead in mV at 500 Hz.

    geometry="grid" regularises the page on its printed 5 mm grid after the
    corner rectification; if the fit is refused the corner page is kept and
    `SheetResult.geometry` says "corners".
    """
    refined = None
    if corners is None:
        fc = page.find_sheet_corners(photo)
        corners, refined = fc.corners, fc.refined
    rect, _ = rectify(photo, corners)
    used, fit = "corners", None
    if geometry == "grid":
        try:
            fit = gridfit.fit_grid(rect, segment.locate_sheet(rect).anchor_px_per_mm)
            rect, _ = gridfit.regularise(photo, corners, rect.shape, fit)
            used = "grid"
        except (RuntimeError, ValueError):
            fit = None
    elif geometry != "corners":
        raise ValueError(f"geometry must be 'corners' or 'grid', got {geometry!r}")
    sl = segment.locate_sheet(rect, fit.px_per_mm if fit is not None and used == "grid"
                              else None)
    ticks = segment.tick_scale(rect, sl)
    if np.isfinite(ticks.px_per_mm):
        px, cal, tk = ticks.px_per_mm, "ticks", ticks
    else:
        px, cal, tk = sl.anchor_px_per_mm, "duration", None
    layout = segment.segment_leads(rect, sl.as_strip_location(), px, tk)
    rows = {r: on_sheet_columns(*tracer(rect, layout, r)) for r in range(4)}
    leads = {}
    for cell in layout.cells:
        pos, x0 = rows[cell.row]
        leads[(cell.name, cell.row == 3)] = cell_signal(pos, x0, cell, px, px, layout)
    return SheetResult(leads, np.asarray(corners, np.float64), refined, rect.shape[:2],
                       float(px), cal, ticks.n_found, sl.t0_source, layout, used, fit)
