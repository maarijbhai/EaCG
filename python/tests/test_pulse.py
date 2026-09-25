"""Tests for the calibration-pulse measurement.

Pulses are drawn onto `layout.render_12lead` sheets at a known height, so
the answer is set by the test rather than read back off an image. Paired
as elsewhere: the right height must come back for the conventional pulse,
a half-height pulse (a 5 mm/mV print) must come back half height rather
than rounded to the convention, and a sheet with no pulse must report none
rather than measure the trace's first beat.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import digitise as dig      # noqa: E402
from ecgproto import layout, pulse, segment, synth   # noqa: E402

PX = 10.0
FS = 500.0


def sheet(pulse_mm: float | None):
    beats = synth.synthetic_ecg(duration_s=10.0, fs=FS, heart_rate_bpm=70.0, seed=4)
    sig = beats[:, None] * np.linspace(0.6, 1.2, 12)[None, :]
    image, spec = layout.render_12lead(sig, FS, px_per_mm=PX, draw_labels=False)
    if pulse_mm is not None:
        cell = spec.cell(layout.RHYTHM_LEAD, rhythm=True)
        x1 = int(round(cell.x0_px))
        x0 = x1 - int(round(5 * PX))
        base = int(round(cell.baseline_px))
        top = base - int(round(pulse_mm * PX))
        pts = np.array([[x0, base], [x0, top], [x1, top], [x1, base]], np.int32)
        cv2.polylines(image, [pts], False, (0, 0, 0), 2)
    m = dig.binarise_trace(image)
    return m, segment.locate_rhythm_strip(image, mask=m)


class MeasurePulse(unittest.TestCase):

    def test_conventional_pulse_reads_ten_millimetres(self):
        m, loc = sheet(10.0)
        p = pulse.measure_pulse(m, loc)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.implied_mm_per_mv(PX), 10.0, delta=0.2)

    def test_half_gain_pulse_reads_half_not_the_convention(self):
        m, loc = sheet(5.0)
        p = pulse.measure_pulse(m, loc)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.implied_mm_per_mv(PX), 5.0, delta=0.2)

    def test_no_pulse_is_reported_as_none(self):
        m, loc = sheet(None)
        self.assertIsNone(pulse.measure_pulse(m, loc))


if __name__ == "__main__":
    unittest.main()
