"""ecgproto.sheet_quality: each warning fires on its failure and not on a
good sheet, with both sides of every threshold pinned."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import sheet_quality as Q  # noqa: E402

PX = 9.5664
T0 = 132.3


def good_ticks():
    """Nine ticks where ptbxl40_20260929_02's good ones sit."""
    xs = {1: 733.0, 2: 1328.0, 3: 1926.0}
    return [(r, k, xs[k] + 0.5 * r) for r in range(3) for k in (1, 2, 3)]


def good_leads():
    t = np.arange(1250) / 500.0
    lead = 0.6 * np.sin(2 * np.pi * 1.2 * t)
    names = [n for row in Q.LAYOUT_3X4 for n in row]
    out = {n: lead.copy() for n in names}
    out["II_rhythm"] = 0.6 * np.sin(2 * np.pi * 1.2 * np.arange(5000) / 500.0)
    return out


def kw(**over):
    base = dict(grid_requested=True, grid_used=True, grid_px_per_mm=PX,
                calibration="ticks", px_per_mm=PX, ticks_found=9,
                tick_positions=good_ticks(), t0_source="pulses", leads=good_leads())
    base.update(over)
    return base


def codes(ws):
    return sorted(w.code for w in ws)


class SheetQualityTest(unittest.TestCase):

    def test_good_sheet_has_no_warnings(self):
        ws = Q.assess(**kw())
        self.assertEqual(ws, [])
        self.assertEqual(Q.verdict(ws), "ok")

    def test_grid_refused(self):
        ws = Q.assess(**kw(grid_used=False, grid_px_per_mm=float("nan")))
        self.assertEqual(codes(ws), ["GRID_REFUSED"])
        self.assertEqual(Q.verdict(ws), "check")
        # Corners chosen on purpose is not a refusal.
        self.assertEqual(Q.assess(**kw(grid_requested=False, grid_used=False,
                                       grid_px_per_mm=float("nan"))), [])

    def test_ticks_thresholds(self):
        self.assertEqual(Q.assess(**kw(ticks_found=6)), [])
        self.assertEqual(codes(Q.assess(**kw(ticks_found=5))), ["FEW_TICKS"])
        ws = Q.assess(**kw(calibration="duration", ticks_found=2, tick_positions=[]))
        self.assertEqual(codes(ws), ["NO_TICK_SCALE"])
        self.assertEqual(Q.verdict(ws), "unreliable")

    def test_scale_disagreement_both_sides(self):
        self.assertEqual(Q.assess(**kw(px_per_mm=PX * 1.0095)), [])
        self.assertEqual(Q.assess(**kw(px_per_mm=PX * 0.9905)), [])
        self.assertIn("SCALE_DISAGREE", codes(Q.assess(**kw(px_per_mm=PX * 1.0105))))
        self.assertIn("SCALE_DISAGREE", codes(Q.assess(**kw(px_per_mm=PX * 0.9895))))
        # The measured phone failure: ticks 6.52 against a grid near 9.8.
        ws = Q.assess(**kw(grid_px_per_mm=9.79, px_per_mm=6.5173))
        self.assertEqual(Q.verdict(ws), "unreliable")

    def test_tick_outlier_is_the_measured_failure(self):
        ticks = good_ticks()
        ticks[-1] = (2, 3, 1873.5)          # ptbxl40_20260929_02, row 3 at 7.5 s
        ws = Q.assess(**kw(tick_positions=ticks))
        self.assertEqual(codes(ws), ["TICK_OUTLIER"])
        self.assertEqual(ws[0].leads, ("V3", "V6", "II_rhythm"))
        self.assertEqual(Q.verdict(ws), "unreliable")

    def test_tick_outlier_threshold_both_sides(self):
        for off_mm, fires in ((0.99, False), (1.01, True), (-1.01, True)):
            ticks = [(r, k, x - 0.5 * r) for r, k, x in good_ticks()]   # rows agree
            r, k, x = ticks[0]
            ticks[0] = (r, k, x + off_mm * PX)       # the median stays at x
            ws = [w for w in Q.assess(**kw(tick_positions=ticks)) if w.code == "TICK_OUTLIER"]
            self.assertEqual(len(ws), 1 if fires else 0, off_mm)

    def test_two_ticks_apart_flag_both(self):
        ticks = [t for t in good_ticks() if not (t[1] == 1 and t[0] == 2)]
        ticks = [(r, k, x + (2.1 * PX if (r, k) == (1, 1) else 0.0)) for r, k, x in ticks]
        ws = [w for w in Q.assess(**kw(tick_positions=ticks, ticks_found=8))
              if w.code == "TICK_OUTLIER"]
        self.assertEqual(sorted(w.leads[0] for w in ws), ["I", "II"])
        ticks = [(r, k, x + (1.9 * PX if (r, k) == (1, 1) else 0.0))
                 for r, k, x in [t for t in good_ticks() if not (t[1] == 1 and t[0] == 2)]]
        self.assertEqual([w for w in Q.assess(**kw(tick_positions=ticks, ticks_found=8))
                          if w.code == "TICK_OUTLIER"], [])

    def test_a_tick_without_mates_is_not_judged(self):
        ticks = [(0, 1, 900.0)]             # wildly off, but nothing to compare with
        self.assertNotIn("TICK_OUTLIER",
                         codes(Q.assess(**kw(tick_positions=ticks, ticks_found=6))))

    def test_tick_replaced_is_a_warning_naming_the_leads(self):
        ticks = [t for t in good_ticks() if (t[0], t[1]) != (2, 3)]
        ws = Q.assess(**kw(tick_positions=ticks, ticks_found=8,
                           tick_rejected=[(2, 3, 1873.5)]))
        self.assertEqual([(w.code, w.leads) for w in ws],
                         [("TICK_REPLACED", ("V3", "V6", "II_rhythm"))])
        self.assertIn("-5.5 mm", ws[0].message)
        self.assertEqual(Q.verdict(ws), "check")

    def test_row_spacing(self):
        # The shadow photo's former rows (gaps 536 / 721 / 327 px): severe.
        ws = Q.assess(**kw(rows=(115, 651, 1372, 1699)))
        self.assertEqual([w.code for w in ws], ["ROW_SPACING"])
        self.assertEqual(Q.verdict(ws), "unreliable")
        # A good photo's rows (gaps 333 / 332 / 304, ratio 1.10): nothing.
        self.assertEqual(Q.assess(**kw(rows=(679, 1012, 1344, 1648))), [])
        # Both sides of 1.3.
        self.assertEqual(Q.assess(**kw(rows=(0, 100, 200, 329))), [])
        self.assertEqual([w.code for w in Q.assess(**kw(rows=(0, 100, 200, 331)))], ["ROW_SPACING"])

    def test_no_pulse(self):
        self.assertEqual(codes(Q.assess(**kw(t0_source="rhythm_locator"))), ["NO_PULSE"])

    def test_lead_coverage_and_amplitude(self):
        leads = good_leads()
        leads["V4"][: int(0.21 * 1250)] = np.nan          # 79% traced
        leads["aVL"] = np.full(1250, 0.01)                # flat
        leads["V5"][:] = np.nan                           # nothing
        ws = Q.assess(**kw(leads=leads))
        by = {w.leads[0]: w.code for w in ws}
        self.assertEqual(by, {"V4": "LEAD_COVERAGE", "aVL": "LEAD_AMPLITUDE",
                              "V5": "LEAD_COVERAGE"})
        self.assertEqual(Q.verdict(ws), "check")
        leads = good_leads()
        leads["V4"][: int(0.19 * 1250)] = np.nan          # 81% traced
        leads["V1"] = 3.05 * leads["V1"] / 0.6            # 6.1 mV pp
        ws = Q.assess(**kw(leads=leads))
        self.assertEqual([(w.code, w.leads) for w in ws], [("LEAD_AMPLITUDE", ("V1",))])

    def test_lead_stats(self):
        cov, pp = Q.lead_stats(np.array([0.0, np.nan, 1.0, -0.5]))
        self.assertAlmostEqual(cov, 0.75)
        self.assertAlmostEqual(pp, 1.5)
        cov, pp = Q.lead_stats(np.full(4, np.nan))
        self.assertEqual(cov, 0.0)
        self.assertTrue(np.isnan(pp))


if __name__ == "__main__":
    unittest.main()
