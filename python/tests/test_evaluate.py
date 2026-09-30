"""Tests for `evaluate.r_peak_ratio`, the number decision rule 1 is written in.

Paired as elsewhere: a scaled copy must read back its scale, a copy whose
QRS tips alone are shaved (a column-centroid tracer's fault) must read low
while `gain_ratio` barely moves, and a lead whose QRS points down must be
scored on its downward deflection rather than on its T waves.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import evaluate, synth   # noqa: E402

FS = 500.0


def ecg() -> np.ndarray:
    return synth.synthetic_ecg(duration_s=10.0, fs=FS, heart_rate_bpm=70.0,
                               seed=4)


class RPeakRatio(unittest.TestCase):
    def test_identity_and_scale(self):
        x = ecg()
        r, n = evaluate.r_peak_ratio(x, x, FS)
        self.assertAlmostEqual(r, 1.0, places=6)
        self.assertGreaterEqual(n, 10)
        r, _ = evaluate.r_peak_ratio(x, 0.7 * x + 0.3, FS)
        self.assertAlmostEqual(r, 0.7, places=2)

    def test_shaved_tips_read_low_where_gain_does_not(self):
        x = ecg()
        base = np.median(x)
        top = base + 0.6 * (x.max() - base)
        shaved = np.minimum(x, top)
        r, _ = evaluate.r_peak_ratio(x, shaved, FS)
        self.assertLess(r, 0.65)
        # Gain is a whole-waveform least-squares slope, so it moves far
        # less than the tips themselves did.
        self.assertLess(r, evaluate.gain_ratio(x, shaved) - 0.15)

    def test_small_time_shift_tolerated(self):
        x = ecg()
        r, _ = evaluate.r_peak_ratio(x, np.roll(x, 5), FS)   # 10 ms
        self.assertAlmostEqual(r, 1.0, places=6)

    def test_negative_qrs_scored_on_its_own_polarity(self):
        x = -ecg()
        r, _ = evaluate.r_peak_ratio(x, 0.8 * x, FS)
        self.assertAlmostEqual(r, 0.8, places=2)

    def test_nan_in_test_ignored(self):
        x = ecg()
        y = x.copy()
        y[:200] = np.nan
        r, n = evaluate.r_peak_ratio(x, y, FS)
        # Not exactly 1: the test's median, its baseline, moves slightly
        # when 0.4 s of it is missing.
        self.assertAlmostEqual(r, 1.0, places=2)
        self.assertGreaterEqual(n, 10)


if __name__ == "__main__":
    unittest.main()
