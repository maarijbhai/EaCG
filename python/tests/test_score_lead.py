"""score_12lead.score_lead: RMSE is taken about each signal's median, as SNR is."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import score_12lead as S12  # noqa: E402


class Rmse(unittest.TestCase):

    def setUp(self):
        t = np.arange(1250) / S12.FS
        self.truth = 0.8 * np.exp(-((t % 0.8) - 0.4) ** 2 / 0.0004) + 0.1 * np.sin(2 * np.pi * t)
        self.sig = 0.9 * (self.truth - np.median(self.truth))          # a cell: about its own median, 10% low

    def test_a_baseline_offset_in_the_truth_is_not_error(self):
        a = S12.score_lead(self.sig, self.truth, 2.5)["rmse_mv_asis"]
        b = S12.score_lead(self.sig, self.truth + 0.15, 2.5)["rmse_mv_asis"]
        self.assertAlmostEqual(a, b, places=4)       # failed under the old rule: +0.15 mV of 'error'

    def test_rmse_is_the_median_removed_one(self):
        r = S12.score_lead(self.sig, self.truth, 2.5)["rmse_mv_asis"]
        t0 = self.truth - np.median(self.truth)
        s0 = self.sig - np.median(self.sig)
        self.assertAlmostEqual(r, round(float(np.sqrt(np.mean((t0 - s0) ** 2))), 4), places=4)


if __name__ == "__main__":
    unittest.main()
