"""clf_input: the classifier's input from a digitised sheet (the phone's reference)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ml"))

from ecgproto import clf_input as C  # noqa: E402


class Decimate(unittest.TestCase):

    def test_block_mean(self):
        x = np.arange(5000, dtype=float)
        d = C.decimate(x)
        self.assertEqual(d.shape, (1000,))
        self.assertEqual(d[0], 2.0)                    # mean of 0..4
        self.assertEqual(d[999], 4997.0)

    def test_nan_ignored_and_all_nan_block_stays_nan(self):
        x = np.ones(5000)
        x[0:2] = np.nan                                # block 0: three finite samples
        x[5:10] = np.nan                               # block 1: none
        x[10] = 6.0                                    # block 2: (6 + 1 + 1 + 1 + 1) / 5
        d = C.decimate(x)
        self.assertEqual(d[0], 1.0)
        self.assertTrue(np.isnan(d[1]))
        self.assertAlmostEqual(d[2], 2.0)

    def test_short_input_padded_with_nan(self):
        d = C.decimate(np.ones(4998))
        self.assertEqual(d[998], 1.0)
        self.assertEqual(d[999], 1.0)                  # 3 finite samples of 5
        self.assertTrue(np.isnan(C.decimate(np.ones(10))[5]))


class PrintedView(unittest.TestCase):

    def test_equals_the_training_view(self):
        """The view the classifier was trained on (ml/clf_data.printed_view)."""
        import clf_data as D                          # no TensorFlow at import
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.3, (1000, 12)).astype(np.float32)
        x[100:130, 4] = np.nan                         # a gap
        x[:, 7] = np.nan                               # a lead lost entirely
        a, b = C.printed_view(x), D.printed_view(x)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_zero_outside_each_window_and_median_zero_inside(self):
        rng = np.random.default_rng(2)
        v = C.printed_view(rng.normal(1.0, 0.5, (1000, 12)))
        for lead, col in C.PRINTED_COLUMN.items():
            a, b = col * C.CELL, (col + 1) * C.CELL
            self.assertTrue(np.all(np.delete(v[:, lead], np.arange(a, b)) == 0), lead)
            self.assertAlmostEqual(float(np.median(v[a:b, lead])), 0.0, places=5)


class FromColumns(unittest.TestCase):

    def test_cells_land_in_their_windows(self):
        """A cell at 500 Hz [k*1250, (k+1)*1250) is exactly window k at 100 Hz."""
        cols = {}
        for lead, col in C.PRINTED_COLUMN.items():
            c = np.full(5000, np.nan)
            c[col * 1250:(col + 1) * 1250] = np.linspace(0, 1, 1250) + lead
            cols[C.LEADS[lead]] = c
        cols["II_rhythm"] = np.sin(np.arange(5000) / 50.0)
        v = C.from_columns(cols)
        self.assertEqual(v.shape, (1000, 13))
        self.assertEqual(v.dtype, np.float32)
        for lead, col in C.PRINTED_COLUMN.items():
            w = v[col * 250:(col + 1) * 250, lead]
            self.assertTrue(np.all(np.diff(w) > 0), lead)          # the ramp, no edge sample lost
            self.assertTrue(np.all(np.delete(v[:, lead], np.arange(col * 250, (col + 1) * 250)) == 0))
        self.assertGreater(np.abs(v[:, 12]).max(), 0.5)

    def test_missing_column_is_zeros(self):
        v = C.from_columns({})
        self.assertTrue(np.all(v == 0))


if __name__ == "__main__":
    unittest.main()
