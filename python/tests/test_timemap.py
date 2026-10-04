"""timemap: the shift diagnosis, the row fit, and the four per-row time maps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import segment, sheet, timemap  # noqa: E402

FS = 500.0


def ecg_like(n: int, seed: int = 0) -> np.ndarray:
    """Narrow spikes at irregular ~0.8 s intervals on a slow wave: aperiodic
    enough that only the right lag correlates."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = 0.1 * np.sin(2 * np.pi * 0.3 * t)
    pos = 0.1
    while pos < t[-1]:
        x += np.exp(-0.5 * ((t - pos) / 0.012) ** 2)
        x += 0.25 * np.exp(-0.5 * ((t - pos - 0.25) / 0.05) ** 2)
        pos += 0.8 + rng.uniform(-0.15, 0.15)
    return x


def late(x: np.ndarray, lag: int) -> np.ndarray:
    """x delayed by `lag` samples (NaN where there is no source)."""
    y = np.full_like(x, np.nan)
    if lag >= 0:
        y[lag:] = x[:len(x) - lag]
    else:
        y[:lag] = x[-lag:]
    return y


def layout(knots, px_per_s=250.0, t0=100.0) -> segment.LeadLayout:
    return segment.LeadLayout(cells=(), t0_x=t0, px_per_s=px_per_s,
                              row_y=(100, 200, 300, 400),
                              row_bands=((50, 150), (150, 250), (250, 350), (350, 450)),
                              row_knots=tuple(knots))


#: A tick map: t0 at 100, ticks at 2.5/5/7.5 s, 250 px/s, end extrapolated.
TICKS = ((100.0, 725.0, 1350.0, 1975.0, 2600.0), (0.0, 2.5, 5.0, 7.5, 10.0))


class BestShift(unittest.TestCase):
    def test_recovers_a_known_delay_with_the_sign_late_positive(self):
        x = ecg_like(int(2.5 * FS))
        for lag in (15, -20, 0):
            r = timemap.best_shift(x, late(x, lag), FS)
            self.assertEqual(r.status, "ok")
            self.assertAlmostEqual(r.shift_ms, 1000 * lag / FS)
            self.assertGreater(r.pcc, 0.99)

    def test_a_shift_at_the_window_edge_is_failed_not_a_value(self):
        x = ecg_like(int(2.5 * FS))
        r = timemap.best_shift(x, late(x, 60), FS, max_shift_s=0.1)   # true 120 ms
        self.assertEqual(r.status, "FAILED")

    def test_an_unrelated_trace_is_unreliable(self):
        x = ecg_like(int(2.5 * FS), seed=1)
        y = np.random.default_rng(7).normal(size=len(x))
        self.assertEqual(timemap.best_shift(x, y, FS).status, "UNRELIABLE")

    def test_nan_samples_are_excluded_not_propagated(self):
        x = ecg_like(int(2.5 * FS))
        y = late(x, 10)
        y[300:340] = np.nan
        r = timemap.best_shift(x, y, FS)
        self.assertEqual((r.status, r.shift_ms), ("ok", 20.0))


class FitRow(unittest.TestCase):
    def test_a_linear_drift_gives_slope_intercept_and_scale(self):
        f = timemap.fit_row([0, 1, 2, 3], [5.0, 15.0, 25.0, 35.0])
        self.assertAlmostEqual(f.slope_ms_per_cell, 10.0)
        self.assertAlmostEqual(f.intercept_ms, 5.0)
        self.assertAlmostEqual(f.r2, 1.0)
        self.assertAlmostEqual(f.scale_error_pct, 0.4)    # 10 ms per 2,500 ms

    def test_scatter_lowers_r2(self):
        f = timemap.fit_row([0, 1, 2, 3], [0.0, 10.0, -5.0, 6.0])
        self.assertLess(f.r2, 0.5)

    def test_unusable_leads_are_dropped_and_too_few_give_no_fit(self):
        f = timemap.fit_row([0, 1, 2, 3], [1.0, np.nan, np.nan, 4.0])
        self.assertEqual(f.n, 2)
        self.assertTrue(np.isnan(f.slope_ms_per_cell))


class RowKnots(unittest.TestCase):
    def setUp(self):
        self.lay = layout([TICKS] * 4)

    def test_ticks_mode_is_the_shipped_map_unchanged(self):
        k, u = timemap.row_knots("ticks", self.lay, [100.0] * 4)
        self.assertEqual(k, self.lay.row_knots)
        self.assertEqual(u, {})

    def test_rowstart_moves_only_time_zero(self):
        k, u = timemap.row_knots("rowstart", self.lay, [104.0, 100.0, 96.0, 100.0])
        self.assertEqual(u, {})
        self.assertEqual(k[0][0][0], 104.0)
        self.assertEqual(k[2][0][0], 96.0)
        self.assertEqual(k[0][0][1:4], TICKS[0][1:4])            # ticks kept
        self.assertEqual(k[0][0][-1], 1975.0 + 2.5 * 250.0)      # end as shipped

    def test_rowstart_end_pins_the_row_end_to_ten_seconds(self):
        ends = [2640.0, 2600.0, 2580.0, 2610.0]
        k, u = timemap.row_knots("rowstart_end", self.lay, [100.0] * 4, ends)
        self.assertEqual(u, {})
        for r in range(4):
            self.assertEqual(k[r][0][-1], ends[r])
            self.assertEqual(k[r][1][-1], segment.RECORDING_S)
            self.assertEqual(k[r][1][:4], (0.0, 2.5, 5.0, 7.5))

    def test_a_missing_start_or_end_is_unavailable_never_a_fallback(self):
        k, u = timemap.row_knots("rowstart_end", self.lay, [100.0, np.nan, 100.0, 100.0],
                                 [2600.0, 2600.0, np.nan, 2600.0])
        self.assertIsNone(k[1])
        self.assertIsNone(k[2])
        self.assertIn("pulse", u[1])
        self.assertIn("end", u[2])
        self.assertIsNotNone(k[0])

    def test_an_end_before_the_last_tick_is_refused(self):
        k, u = timemap.row_knots("rowstart_end", self.lay, [100.0] * 4, [1900.0] * 4)
        self.assertTrue(all(v is None for v in k))
        self.assertEqual(len(u), 4)

    def test_grid_mode_is_uniform_at_the_grid_scale_from_the_shared_t0(self):
        # 10.4 px/mm = 260 px/s, deliberately not the tick map's 250.
        k, u = timemap.row_knots("grid", self.lay, [90.0] * 4, grid_px_per_mm=10.4)
        self.assertEqual(u, {})
        self.assertEqual(k[0][1], (0.0, 10.0))
        self.assertAlmostEqual(k[0][0][0], 100.0)
        self.assertAlmostEqual(k[0][0][1], 100.0 + 10 * 260.0)

    def test_grid_mode_without_a_grid_fit_is_unavailable(self):
        k, u = timemap.row_knots("grid", self.lay, [100.0] * 4, grid_px_per_mm=None)
        self.assertTrue(all(v is None for v in k))
        self.assertEqual(set(u), {0, 1, 2, 3})

    def test_tick_origin_is_the_median_origin_the_ticks_imply(self):
        # True origin 110 at 260 px/s; one tick 6 px off.
        ticks = [(0, 1, 110 + 650.0), (0, 2, 110 + 1300.0), (1, 3, 110 + 1950.0 + 6),
                 (2, 1, 110 + 650.0)]
        self.assertAlmostEqual(timemap.tick_origin(ticks, 260.0), 110.0)
        self.assertAlmostEqual(timemap.tick_origin(ticks, 260.0, 1.0), 109.0)
        self.assertTrue(np.isnan(timemap.tick_origin(ticks[:2], 260.0)))

    def test_grid_tickzero_takes_time_zero_from_the_ticks_not_the_pulse(self):
        ticks = [(0, 1, 110 + 650.0), (1, 2, 110 + 1300.0), (2, 3, 110 + 1950.0)]
        k, u = timemap.row_knots("grid_tickzero", self.lay, [100.0] * 4,
                                 grid_px_per_mm=10.4, tick_positions=ticks)
        self.assertEqual(u, {})
        for r in range(4):
            self.assertAlmostEqual(k[r][0][0], 110.0)          # not the pulse's 100
            self.assertAlmostEqual(k[r][0][1], 110.0 + 2600.0)
        k2, _ = timemap.row_knots("grid_tickzero_k2fix", self.lay, [100.0] * 4,
                                  grid_px_per_mm=10.4, tick_positions=ticks)
        self.assertAlmostEqual(k2[0][0][0], 109.0)

    def test_grid_tickzero_with_too_few_ticks_is_unavailable(self):
        k, u = timemap.row_knots("grid_tickzero", self.lay, [100.0] * 4,
                                 grid_px_per_mm=10.4, tick_positions=[(0, 1, 760.0)])
        self.assertTrue(all(v is None for v in k))
        self.assertIn("ticks", u[0])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            timemap.row_knots("rows", self.lay, [100.0] * 4)


class EndToEnd(unittest.TestCase):
    def test_a_stretched_row_is_corrected_by_its_own_end(self):
        """The row truly spans 100..2650 (255 px/s); the tick map assumed 250
        px/s past its last tick. Pinning the end puts 10 s where it is."""
        lay = layout([TICKS] * 4)
        k, _ = timemap.row_knots("rowstart_end", lay, [100.0] * 4, [2650.0] * 4)
        fixed = layout(k)
        self.assertAlmostEqual(float(fixed.time_at(0, 2650.0)), 10.0)
        self.assertAlmostEqual(float(lay.time_at(0, 2650.0)), 10.2)

    def test_segment_leads_cuts_cells_on_the_override_and_keeps_none_rows(self):
        img = np.full((500, 3000, 3), 245, np.uint8)
        loc = segment.StripLocation(x0=100, y0=350, x1=2600, y1=450,
                                    row_peaks=(100, 200, 300, 400), pulse_x1=None,
                                    t0_x=100.0)
        base = segment.segment_leads(img, loc, 10.0)            # 250 px/s
        over = ((100.0, 2700.0), (0.0, 10.0))                   # 260 px/s
        lay = segment.segment_leads(img, loc, 10.0,
                                    knots_override=(over, None, over, over))
        self.assertEqual(lay.row_knots[0], over)
        self.assertIsNone(lay.row_knots[1])
        c0 = lay.cell("aVR")                                    # row 0, column 1
        self.assertEqual((c0.x0, c0.x1), (750, 1400))
        self.assertEqual(lay.cell("aVL").x0, base.cell("aVL").x0)  # None: default

    def test_an_unavailable_row_cuts_to_nan(self):
        lay = layout([None, TICKS, TICKS, TICKS])
        cell = segment.LeadCell("I", 0, 0, 0.0, 2.5, 100, 725, 50, 150, 100.0)
        sig = sheet.cell_signal(np.full(700, 100.0), 90, cell, 10.0, 10.0, lay)
        self.assertTrue(np.isnan(sig).all())

    def test_row_end_is_the_last_inked_column_of_the_trace(self):
        img = np.full((500, 3000, 3), 245, np.uint8)
        img[98:102, 100:2400] = 20                    # row 0's trace
        img[60:70, 2700:2706] = 20                     # a speck: not trace-wide
        lay = layout([TICKS] * 4)
        self.assertAlmostEqual(timemap.row_end_x(img, lay, 0), 2399.0)
        self.assertTrue(np.isnan(timemap.row_end_x(img, lay, 2)))


if __name__ == "__main__":
    unittest.main()
