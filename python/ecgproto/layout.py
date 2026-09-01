"""Clinical 3x4 + rhythm strip 12-lead printout rendering.

The 3x4 layout is not a cosmetic choice; it destroys most of the recording
and the rest of the project has to account for that.

**Each cell holds only 2.5 s of its lead**, and the four columns are
*consecutive* time windows -- column 0 is t = 0.0-2.5 s, column 1 is
2.5-5.0 s, and so on. Leads in different columns were therefore **not
recorded simultaneously**. Lead I (column 0) and lead V4 (column 3) in the
same printed row are seven and a half seconds apart.

**Only the rhythm strip spans the full 10 s**, conventionally lead II.

The arithmetic that follows from this is worth stating plainly, because it
bounds what any digitiser can possibly recover from such an image::

    printed  = 12 leads * 2.5 s  +  10 s rhythm strip   =  40 lead-seconds
    recorded = 12 leads * 10 s                          = 120 lead-seconds
    fraction = 40 / 120                                 = 33.3 %

Two thirds of the recording is simply not on the paper. A classifier
trained on 12 x 10 s *concurrent* waveforms cannot consume digitiser
output from a 3x4 printout without accounting for both the missing time
and the fact that what remains is not synchronous across columns. The
options are to retrain on 2.5 s windows, to reconstruct only the rhythm
lead, or to accept the mismatch and measure the cost -- but it must be a
decision, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import cv2

from . import synth
from .ptbxl import LEAD_NAMES

#: Column-major lead placement of the conventional 3x4 printout. Index
#: [row][col] into LEAD_NAMES order.
GRID_LAYOUT = (
    ("I", "AVR", "V1", "V4"),
    ("II", "AVL", "V2", "V5"),
    ("III", "AVF", "V3", "V6"),
)

#: The rhythm strip lead. Lead II by convention -- its P waves are usually
#: the clearest, which is why it is the one clinicians read rhythm from.
RHYTHM_LEAD = "II"

N_COLUMNS = 4
COLUMN_SECONDS = 2.5


@dataclass(frozen=True)
class CellSpec:
    """Where one lead segment sits in the rendered image, and what it holds.

    `t_start_s`/`t_end_s` are times **in the original recording**, not in
    the cell. Lead segmentation, when it exists, must reproduce this
    mapping to reassemble anything time-aligned.
    """

    lead: str
    lead_index: int
    row: int
    col: int
    t_start_s: float
    t_end_s: float
    x0_px: float          # image column of t_start_s
    baseline_px: float    # image row of 0 mV for this cell
    x_left_px: float      # cell bounding box
    x_right_px: float
    y_top_px: float
    y_bottom_px: float
    is_rhythm: bool = False


@dataclass(frozen=True)
class LayoutSpec:
    """Geometry of a full 3x4 printout."""

    render: synth.RenderSpec
    source_duration_s: float
    cells: tuple[CellSpec, ...] = field(default_factory=tuple)

    def cell(self, lead: str, rhythm: bool = False) -> CellSpec:
        for c in self.cells:
            if c.lead == lead and c.is_rhythm == rhythm:
                return c
        raise KeyError(f"no {'rhythm' if rhythm else 'grid'} cell for lead {lead!r}")

    @property
    def printed_fraction(self) -> float:
        """Fraction of recorded lead-seconds that reached the paper.

        Counts the source recording, not the image: image width includes
        margins, and charging those to the recording would understate how
        much was thrown away.
        """
        printed = sum(c.t_end_s - c.t_start_s for c in self.cells)
        recorded = len(LEAD_NAMES) * self.source_duration_s
        return printed / recorded if recorded else 0.0


def _draw_label(
    image: np.ndarray,
    text: str,
    x_left_px: float,
    y_top_px: float,
    px_per_mm: float,
) -> None:
    """Write a lead name in the top-left of a cell, in trace ink.

    Real printouts label every cell, so a renderer that omits them is
    easier to digitise than the thing it stands in for. The labels are
    drawn in the *same* neutral ink as the trace, which means
    `binarise_trace` returns them as trace and `trace_positions` will treat
    a label as signal. That is deliberate: rejecting non-trace ink is work
    lead segmentation has to do on real images anyway, and hiding it here
    would only postpone the discovery.
    """
    scale = px_per_mm / 12.0
    cv2.putText(
        image, text,
        (int(x_left_px + 1.5 * px_per_mm), int(y_top_px + 5.0 * px_per_mm)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, synth.COLOUR_TRACE,
        max(1, int(round(scale * 2))), cv2.LINE_AA,
    )


def render_12lead(
    signal: np.ndarray,
    fs: float,
    px_per_mm: float = 8.0,
    mm_per_s: float = synth.MM_PER_S,
    mm_per_mv: float = synth.MM_PER_MV,
    cell_height_mm: float = 28.0,
    margin_mm: float = 8.0,
    trace_thickness: int = 2,
    rhythm_lead: str = RHYTHM_LEAD,
    draw_labels: bool = True,
) -> tuple[np.ndarray, LayoutSpec]:
    """Render a (n_samples, 12) mV signal as a 3x4 printout + rhythm strip.

    Returns (BGR uint8 image, LayoutSpec).

    The input must be the full 10 s recording for all 12 leads; this
    function does the windowing, because the mapping from recording time to
    printed column is exactly the invariant that must not be reinvented at
    each call site.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != len(LEAD_NAMES):
        raise ValueError(
            f"expected (n_samples, 12) in {LEAD_NAMES} order, got {signal.shape}"
        )
    duration_s = signal.shape[0] / fs
    needed_s = N_COLUMNS * COLUMN_SECONDS
    if duration_s + 1e-9 < needed_s:
        raise ValueError(
            f"a {N_COLUMNS}-column printout needs {needed_s} s, got {duration_s:.3f} s"
        )

    lead_index = {name: i for i, name in enumerate(LEAD_NAMES)}
    n_rows = len(GRID_LAYOUT)

    col_width_mm = COLUMN_SECONDS * mm_per_s
    cell_h_px = cell_height_mm * px_per_mm
    col_w_px = col_width_mm * px_per_mm
    margin_px = margin_mm * px_per_mm

    width_px = int(round(2 * margin_px + N_COLUMNS * col_w_px))
    # Three lead rows plus one full-width rhythm row.
    height_px = int(round(2 * margin_px + (n_rows + 1) * cell_h_px))

    image = np.full((height_px, width_px, 3), synth.COLOUR_BACKGROUND, dtype=np.uint8)
    synth.draw_grid(image, px_per_mm)

    render = synth.RenderSpec(
        px_per_mm=px_per_mm,
        mm_per_s=mm_per_s,
        mm_per_mv=mm_per_mv,
        baseline_px=0.0,   # per-cell; overridden on every draw_trace call
        x0_px=margin_px,
        fs=fs,
        width_px=width_px,
        height_px=height_px,
    )

    cells: list[CellSpec] = []

    for row, lead_row in enumerate(GRID_LAYOUT):
        for col, lead in enumerate(lead_row):
            t0 = col * COLUMN_SECONDS
            t1 = t0 + COLUMN_SECONDS
            i0 = int(round(t0 * fs))
            i1 = int(round(t1 * fs))
            segment = signal[i0:i1, lead_index[lead]]

            x_left = margin_px + col * col_w_px
            y_top = margin_px + row * cell_h_px
            baseline = y_top + cell_h_px / 2.0

            synth.draw_trace(
                image, segment, render,
                x0_px=x_left, baseline_px=baseline, thickness=trace_thickness,
            )
            if draw_labels:
                _draw_label(image, lead, x_left, y_top, px_per_mm)
            cells.append(CellSpec(
                lead=lead, lead_index=lead_index[lead], row=row, col=col,
                t_start_s=t0, t_end_s=t1,
                x0_px=x_left, baseline_px=baseline,
                x_left_px=x_left, x_right_px=x_left + col_w_px,
                y_top_px=y_top, y_bottom_px=y_top + cell_h_px,
            ))

    # Rhythm strip: one lead, full width, all N_COLUMNS * COLUMN_SECONDS.
    rhythm_t1 = N_COLUMNS * COLUMN_SECONDS
    rhythm_segment = signal[: int(round(rhythm_t1 * fs)), lead_index[rhythm_lead]]
    y_top = margin_px + n_rows * cell_h_px
    baseline = y_top + cell_h_px / 2.0
    synth.draw_trace(
        image, rhythm_segment, render,
        x0_px=margin_px, baseline_px=baseline, thickness=trace_thickness,
    )
    if draw_labels:
        _draw_label(image, rhythm_lead, margin_px, y_top, px_per_mm)
    cells.append(CellSpec(
        lead=rhythm_lead, lead_index=lead_index[rhythm_lead],
        row=n_rows, col=0, t_start_s=0.0, t_end_s=rhythm_t1,
        x0_px=margin_px, baseline_px=baseline,
        x_left_px=margin_px, x_right_px=margin_px + N_COLUMNS * col_w_px,
        y_top_px=y_top, y_bottom_px=y_top + cell_h_px,
        is_rhythm=True,
    ))

    return image, LayoutSpec(
        render=render, source_duration_s=duration_s, cells=tuple(cells)
    )
