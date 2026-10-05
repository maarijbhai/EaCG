"""Tests for the rhythm-strip locator.

Rendered with `layout.render_12lead`, whose `LayoutSpec` knows where every
cell was drawn, so each assertion is against geometry the renderer placed
rather than against a number read back off the image.

Each positive case is paired with the failure it guards against: the
locator must find the rhythm row *and not* the row above it; it must cut a
calibration pulse *and not* invent one where none was printed; and it must
refuse a page that is not a 3x4 sheet rather than return the bottom of
whatever it was given.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import layout, segment, synth   # noqa: E402

PX_PER_MM = 8.0
FS = 500.0


def twelve_leads(seed: int = 0) -> np.ndarray:
    beats = synth.synthetic_ecg(duration_s=10.0, fs=FS, heart_rate_bpm=72.0,
                                seed=seed)
    gains = np.linspace(0.6, 1.2, 12)
    return beats[:, None] * gains[None, :]


def sheet(pulse: bool = False) -> tuple[np.ndarray, layout.LayoutSpec]:
    image, spec = layout.render_12lead(twelve_leads(), FS, px_per_mm=PX_PER_MM,
                                       draw_labels=False)
    if pulse:
        # A 1 mV x 0.2 s rectangle ending exactly where the rhythm trace
        # starts, drawn the way a printer draws it: rise, top, fall.
        cell = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        x1 = int(round(cell.x0_px))
        x0 = x1 - int(round(5 * PX_PER_MM))
        base = int(round(cell.baseline_px))
        top = base - int(round(10 * PX_PER_MM))
        pts = np.array([[x0 - 8, base], [x0, base], [x0, top], [x1, top],
                        [x1, base]], np.int32)
        cv2.polylines(image, [pts], False, (0, 0, 0), 2)
    return image, spec


class LocateRhythmStrip(unittest.TestCase):

    def test_finds_the_rhythm_row_and_not_the_row_above(self):
        image, spec = sheet()
        loc = segment.locate_rhythm_strip(image)
        rhythm = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        above = [c for c in spec.cells if c.row == rhythm.row - 1]
        self.assertLess(loc.y0, rhythm.baseline_px)
        self.assertGreater(loc.y1, rhythm.baseline_px)
        for c in above:
            self.assertGreater(loc.y0, c.baseline_px)

    def test_spans_the_full_ten_seconds(self):
        image, spec = sheet()
        loc = segment.locate_rhythm_strip(image)
        rhythm = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        width_px = (rhythm.t_end_s - rhythm.t_start_s) * synth.MM_PER_S * PX_PER_MM
        # One line width either side is the most the ink can overhang.
        self.assertAlmostEqual(loc.x0, rhythm.x0_px, delta=3)
        self.assertAlmostEqual(loc.recording_width_px, width_px, delta=4)

    def test_no_pulse_is_invented_when_none_was_printed(self):
        image, _ = sheet(pulse=False)
        loc = segment.locate_rhythm_strip(image)
        self.assertIsNone(loc.pulse_x1)
        self.assertEqual(loc.t0_x, float(loc.x0))

    def test_a_printed_pulse_is_cut_and_the_recording_starts_at_its_fall(self):
        image, spec = sheet(pulse=True)
        loc = segment.locate_rhythm_strip(image)
        rhythm = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        self.assertIsNotNone(loc.pulse_x1)
        # The duration anchor measures from the falling edge ...
        self.assertAlmostEqual(loc.t0_x, rhythm.x0_px, delta=2)
        # ... and the crop starts clear of it, so no edge column is traced.
        self.assertGreater(loc.x0, loc.pulse_x1 - 1)
        strip = loc.crop(image)
        counts = (strip.max(axis=2) < 120).sum(axis=0)
        self.assertLess(counts[:3].max(), 5 * PX_PER_MM)

    def test_a_single_strip_is_refused_rather_than_guessed(self):
        image, _ = synth.render_printout(synth.synthetic_ecg(fs=FS), FS,
                                         px_per_mm=PX_PER_MM)
        with self.assertRaises(RuntimeError):
            segment.locate_rhythm_strip(image)


class LocateSheetOnAThinTrace(unittest.TestCase):
    """A photographed trace binarises 2-3 px wide; a 3x3 opening shreds it.

    Measured on ptbxl9_20261005_01: the rhythm strip "ended" three quarters
    of the way across, the duration anchor read 7.13 px/mm against ~9.9, and
    every pulse and 8 of 9 ticks were missed. The render's ink is thinned by
    one pixel to the same width; under the old 3x3 rule this anchor reads
    7.156 (-10.5%), under 2x2 8.000.
    """

    def test_the_duration_anchor_survives_a_two_pixel_trace(self):
        image, spec = layout.render_12lead(twelve_leads(), FS, px_per_mm=PX_PER_MM,
                                           draw_labels=False, trace_thickness=2)
        thin = cv2.dilate(image, np.ones((2, 2), np.uint8))   # ink one pixel thinner
        sl = segment.locate_sheet(thin)
        self.assertAlmostEqual(sl.anchor_px_per_mm / PX_PER_MM, 1.0, delta=0.005)
        rhythm = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        width_px = (rhythm.t_end_s - rhythm.t_start_s) * synth.MM_PER_S * PX_PER_MM
        self.assertAlmostEqual(sl.x_end - sl.t0_x, width_px, delta=4)


class MaskOffsetIsUndone(unittest.TestCase):
    """A position read off a 2x2-opened mask is one pixel late unless
    corrected (segment.mask_offset_px). The invariant: the same drawing read
    through a 2x2 and a 3x3 opening gives the same pulse edge and the same
    ticks. Under the old rule they differ by exactly one pixel (checked)."""

    def test_the_assumed_offset_is_what_cv2_does(self):
        from ecgproto import digitise as dig
        img = np.full((60, 400, 3), 245, np.uint8)
        img[20:30, 100:300] = 20
        for k in (2, 3, 4, 5):
            cols = np.flatnonzero(dig.binarise_trace(img, open_kernel=k).any(axis=0))
            self.assertEqual(cols[0] - 100, segment.mask_offset_px(k), f"kernel {k}")

    def test_pulse_edge_and_ticks_agree_across_openings(self):
        from ecgproto import digitise as dig
        px = PX_PER_MM
        img = np.full((900, 2400, 3), 245, np.uint8)
        rows = (200, 420, 640, 820)
        t0 = 120.0
        for y in rows[:3]:
            img[y - 1:y + 2, 60:2300] = 20                          # the trace's baseline
            top = y - int(10 * px)
            img[top:top + 4, 80:121] = 20                           # pulse top, 5 mm
            img[top:y, 117:121] = 20                                # its falling edge
            for k in (1, 2, 3):                                     # 0.75 mm bars, 6 mm tall
                xc = int(t0 + k * segment.CELL_S * 25 * px)
                img[y - int(3 * px):y + int(3 * px), xc - 3:xc + 3] = 20
        loc = segment.StripLocation(x0=60, y0=780, x1=2300, y1=860, row_peaks=rows,
                                    pulse_x1=None, t0_x=t0)
        sl = segment.SheetLocation(rows=rows, t0_x=t0, t0_per_row=(t0,) * 4, x_end=2300,
                                   rhythm=loc, t0_source="pulses", scale_hint=px)
        edges = {k: segment.pulse_falling_edge(dig.binarise_trace(img, open_kernel=k), rows[0], px,
                                               segment.mask_offset_px(k)) for k in (2, 3)}
        self.assertEqual(edges[2], edges[3])
        ticks = {k: segment.tick_scale(img, sl, open_kernel=k).positions for k in (2, 3)}
        self.assertEqual(len(ticks[2]), 9)
        self.assertEqual(ticks[2], ticks[3])


KIT = Path(__file__).resolve().parents[2] / "out" / "kit_targets"


@unittest.skipUnless(KIT.exists(), "out/kit_targets not generated")
class OnKitSheets(unittest.TestCase):
    """Regression on ecg-image-kit's own renders, where plotted_pixels give
    the true extent. Skipped when the (gitignored) sheets are absent."""

    def test_recording_width_matches_the_kit_to_a_tenth_of_a_percent(self):
        import json
        for sheet_dir in sorted(KIT.glob("ptbxl*_kit_*")):
            name = sheet_dir.name
            meta = json.loads((sheet_dir / f"{name}.json").read_text())
            kit = json.loads((sheet_dir / f"{name}_kit.json").read_text())
            off = meta["page"]["plotted_pixels_offset_px"]
            rhythm = kit["leads"][-1]
            cols = np.array(rhythm["plotted_pixels"])[:, 1] - off["x"]
            n = int(rhythm["end_sample"]) - int(rhythm["start_sample"])
            # plotted_pixels are sample positions; n samples span n intervals
            # of recording time, the last one ending one step past the last.
            true_w = (cols.max() - cols.min()) * n / (n - 1)
            loc = segment.locate_rhythm_strip(
                cv2.imread(str(sheet_dir / f"{name}.png")))
            with self.subTest(sheet=name):
                # 1.5 px: where the pulse is its own blob (PTB-XL 57) it is
                # dropped with the text and t0 is the trace's first inked
                # column, which leads the plotted centre by half a line
                # width -- 1.17 px measured.
                self.assertAlmostEqual(loc.t0_x, cols.min(), delta=1.5)
                self.assertLess(abs(loc.recording_width_px / true_w - 1), 1e-3)


if __name__ == "__main__":
    unittest.main()
