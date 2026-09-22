"""Tests for the FR8 quality gate.

`unittest` rather than pytest: the repository commits no test dependency
and there is no reason for this suite to be the first. Run with

    python -m unittest discover -s tests -v

from `python/`, which is also what `mutate_quality.py` drives.

What these tests are for is worth stating, because a gate is unusually
easy to test badly. Asserting that a bad input is rejected proves almost
nothing -- a function that rejected everything would pass. Each failure
case here is therefore paired with a good case that must be accepted,
and the boundaries of every band are pinned from both sides.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import digitise as dig      # noqa: E402
from ecgproto import quality              # noqa: E402


def make_grid(px_per_mm: float = 8.0, aniso: float = 0.0) -> dig.GridEstimate:
    return dig.GridEstimate(
        px_per_mm_x=px_per_mm,
        px_per_mm_y=px_per_mm * (1.0 + aniso),
        confidence=0.6, n_peaks_x=20, n_peaks_y=20, spacing_cv=0.01,
    )


def beat_train(fs: float = 100.0, seconds: float = 10.0,
               bpm: float = 60.0, amplitude_mv: float = 1.0) -> np.ndarray:
    """A spike train at an exactly known rate, for the rate check.

    Triangular spikes rather than single samples: `implied_bpm` thresholds
    at a fraction of the 98th percentile, and a train of one-sample
    spikes puts that percentile in the silence between them.
    """
    n = int(fs * seconds)
    x = np.zeros(n)
    period = int(round(fs * 60.0 / bpm))
    half = 4
    for centre in range(period // 2, n, period):
        for k in range(-half, half + 1):
            i = centre + k
            if 0 <= i < n:
                x[i] = amplitude_mv * (1.0 - abs(k) / (half + 1))
    return x


def beat_train_with_t_waves(fs: float = 100.0, seconds: float = 10.0,
                            bpm: float = 60.0, r_mv: float = 1.0,
                            t_mv: float = 0.55,
                            t_delay_s: float = 0.20) -> np.ndarray:
    """A spike train whose every beat carries a tall T wave.

    `beat_train` alone cannot constrain the refractory period: a single
    triangular spike has one local maximum however close together peaks
    are allowed to be, so the 240 ms exclusion never has anything to
    exclude. A real ECG does -- the T wave follows the R peak by about
    200 ms and on some leads approaches it in height -- and counting it
    as a beat doubles the implied rate. That is the behaviour the
    refractory period exists for, so the fixture has to contain it.
    """
    n = int(fs * seconds)
    x = np.zeros(n)
    period = int(round(fs * 60.0 / bpm))

    def bump(centre: int, amplitude: float, half: int) -> None:
        for k in range(-half, half + 1):
            i = centre + k
            if 0 <= i < n:
                x[i] = max(x[i], amplitude * (1.0 - abs(k) / (half + 1)))

    for centre in range(period // 2, n, period):
        bump(centre, r_mv, 4)
        bump(centre + int(t_delay_s * fs), t_mv, 6)
    return x


def make_result(signal_mv=None, coverage=1.0, calibration="ok",
                px_per_mm=8.0, fs=100.0) -> dig.DigitiseResult:
    if signal_mv is None:
        signal_mv = beat_train(fs=fs)
    if calibration == "ok":
        chk = dig.CalibrationCheck(ok=True, reasons=(), warnings=(),
                                   implied_mm_per_s=25.0)
    elif calibration == "warn":
        chk = dig.CalibrationCheck(ok=True, reasons=(),
                                   warnings=("spacings irregular",),
                                   implied_mm_per_s=25.0)
    elif calibration == "bad":
        chk = dig.CalibrationCheck(ok=False, reasons=("implied paper speed 5.0",),
                                   warnings=(), implied_mm_per_s=5.0)
    elif calibration is None:
        chk = None
    else:
        raise ValueError(calibration)
    return dig.DigitiseResult(
        signal_mv=np.asarray(signal_mv, dtype=np.float64), fs=fs,
        grid=make_grid(px_per_mm), baseline_px=100.0,
        coverage=coverage, calibration=chk,
    )


def strip_mask(height=200, width=350, rows=(100,), thickness=2) -> np.ndarray:
    """A mask with one horizontal ink band per entry in `rows`."""
    m = np.zeros((height, width), dtype=bool)
    for r in rows:
        m[r:r + thickness, :] = True
    return m


class TestVerdict(unittest.TestCase):
    def test_severity_order(self):
        self.assertLess(quality.Verdict.ACCEPT, quality.Verdict.FLAG)
        self.assertLess(quality.Verdict.FLAG, quality.Verdict.REJECT)

    def test_report_is_falsey_only_when_rejected(self):
        self.assertTrue(quality.QualityReport(quality.Verdict.ACCEPT))
        self.assertTrue(quality.QualityReport(quality.Verdict.FLAG))
        self.assertFalse(quality.QualityReport(quality.Verdict.REJECT))


class TestBanded(unittest.TestCase):
    """Both sides of both bands, so a loosened bound cannot pass."""

    reject = (0.05, 6.0)
    flag = (0.20, 3.5)

    def band(self, v):
        return quality._banded(v, self.reject, self.flag)

    def test_inside_both_bands_accepts(self):
        self.assertIs(self.band(1.0), quality.Verdict.ACCEPT)
        self.assertIs(self.band(0.20), quality.Verdict.ACCEPT)
        self.assertIs(self.band(3.5), quality.Verdict.ACCEPT)

    def test_between_flag_and_reject_flags(self):
        self.assertIs(self.band(0.19), quality.Verdict.FLAG)
        self.assertIs(self.band(3.51), quality.Verdict.FLAG)
        self.assertIs(self.band(0.06), quality.Verdict.FLAG)
        self.assertIs(self.band(5.9), quality.Verdict.FLAG)

    def test_outside_reject_band_rejects(self):
        self.assertIs(self.band(0.04), quality.Verdict.REJECT)
        self.assertIs(self.band(6.01), quality.Verdict.REJECT)

    def test_non_finite_rejects(self):
        self.assertIs(self.band(float("nan")), quality.Verdict.REJECT)
        self.assertIs(self.band(float("inf")), quality.Verdict.REJECT)


class TestImpliedBpm(unittest.TestCase):
    def test_recovers_a_known_rate(self):
        for bpm in (50.0, 60.0, 75.0, 100.0):
            got = quality.implied_bpm(beat_train(bpm=bpm), 100.0)
            self.assertAlmostEqual(got, bpm, delta=1.0, msg=f"{bpm} bpm")

    def test_scales_with_sampling_rate_not_sample_count(self):
        """Doubling fs at the same beat spacing halves the implied rate."""
        x = beat_train(fs=100.0, bpm=60.0)
        self.assertAlmostEqual(quality.implied_bpm(x, 200.0), 120.0, delta=2.0)

    def test_t_waves_are_not_counted_as_beats(self):
        """The refractory period, which a clean spike train cannot test."""
        x = beat_train_with_t_waves(bpm=60.0)
        self.assertAlmostEqual(quality.implied_bpm(x, 100.0), 60.0, delta=1.0)

    def test_flat_signal_gives_nan(self):
        self.assertTrue(np.isnan(quality.implied_bpm(np.zeros(500), 100.0)))

    def test_too_short_gives_nan(self):
        self.assertTrue(np.isnan(quality.implied_bpm(np.array([1.0]), 100.0)))


class TestTracesPerColumn(unittest.TestCase):
    def test_single_band(self):
        self.assertEqual(
            quality.traces_per_column(strip_mask(rows=(100,)), 8.0), 1.0)

    def test_four_bands_like_a_3x4_sheet(self):
        m = strip_mask(height=900, rows=(100, 300, 500, 700))
        self.assertEqual(quality.traces_per_column(m, 8.0), 4.0)

    def test_two_bands_is_already_too_many(self):
        """The threshold case. Two leads in frame is the failure's onset.

        Tested separately from the four-band sheet because a four-band
        fixture alone leaves the threshold free to sit anywhere between
        two and four without any test noticing.
        """
        m = strip_mask(height=400, rows=(100, 300))
        self.assertEqual(quality.traces_per_column(m, 8.0), 2.0)
        rep = quality.assess(make_result(), mask=m)
        self.assertIs(rep.check("traces_per_column").verdict,
                      quality.Verdict.REJECT)
        self.assertIs(rep.verdict, quality.Verdict.REJECT)

    def test_runs_closer_than_the_separation_are_one_band(self):
        """A 2 px gap at 8 px/mm is a quarter of a millimetre: one band."""
        m = strip_mask(rows=(100, 104))
        self.assertEqual(quality.traces_per_column(m, 8.0), 1.0)

    def test_separation_is_in_millimetres_not_pixels(self):
        """The same pixel gap means different things at different scales.

        A 12 px gap is 1.5 mm at 8 px/mm -- two bands -- and 0.375 mm at
        32 px/mm, which is one. A check written in pixels would call both
        the same and would change meaning with camera distance.
        """
        m = strip_mask(rows=(100, 112))
        self.assertEqual(quality.traces_per_column(m, 8.0), 2.0)
        self.assertEqual(quality.traces_per_column(m, 32.0), 1.0)

    def test_separation_boundary_is_exclusive(self):
        """A gap of exactly BAND_SEPARATION_MM is still one band.

        Pinned because the convention has to be one thing or the other and
        the choice is invisible from the constant's name: bands are counted
        where the gap *exceeds* the separation, not where it reaches it.
        At 8 px/mm the separation is 8 px, so rows 8 apart are one band and
        rows 9 apart are two.
        """
        m = np.zeros((200, 350), dtype=bool)
        m[100, :] = True
        m[108, :] = True
        self.assertEqual(quality.traces_per_column(m, 8.0), 1.0)
        m[108, :] = False
        m[109, :] = True
        self.assertEqual(quality.traces_per_column(m, 8.0), 2.0)

    def test_median_ignores_ink_in_a_few_columns(self):
        """Printed lead labels must not make a strip look like a sheet."""
        m = strip_mask(width=350, rows=(100,))
        m[20:26, 0:12] = True          # a label, in 12 columns of 350
        self.assertEqual(quality.traces_per_column(m, 8.0), 1.0)

    def test_empty_mask_is_nan(self):
        self.assertTrue(np.isnan(
            quality.traces_per_column(np.zeros((10, 10), bool), 8.0)))


class TestAssess(unittest.TestCase):
    def test_good_digitisation_is_accepted(self):
        rep = quality.assess(make_result(), mask=strip_mask())
        self.assertIs(rep.verdict, quality.Verdict.ACCEPT, rep.summary())
        self.assertTrue(rep)
        self.assertEqual(rep.rejections, ())
        self.assertEqual(rep.flags, ())

    def test_calibration_reasons_reject_and_warnings_flag(self):
        bad = quality.assess(make_result(calibration="bad"), mask=strip_mask())
        self.assertIs(bad.verdict, quality.Verdict.REJECT)
        warn = quality.assess(make_result(calibration="warn"), mask=strip_mask())
        self.assertIs(warn.verdict, quality.Verdict.FLAG)

    def test_stacked_layout_is_rejected(self):
        """The 3x4-into-digitise failure, which calibration cannot see."""
        rep = quality.assess(make_result(),
                             mask=strip_mask(height=900,
                                             rows=(100, 300, 500, 700)))
        self.assertIs(rep.verdict, quality.Verdict.REJECT)
        self.assertIs(rep.check("traces_per_column").verdict,
                      quality.Verdict.REJECT)
        # and it must not be the calibration check doing the rejecting
        self.assertIs(rep.check("calibration").verdict, quality.Verdict.ACCEPT)

    def test_flat_trace_is_rejected_on_amplitude(self):
        rep = quality.assess(make_result(signal_mv=np.zeros(1000)),
                             mask=strip_mask())
        self.assertIs(rep.check("amplitude_pp_mv").verdict,
                      quality.Verdict.REJECT)
        self.assertIs(rep.verdict, quality.Verdict.REJECT)

    def test_implausible_amplitude_is_rejected(self):
        rep = quality.assess(make_result(signal_mv=beat_train(amplitude_mv=9.0)),
                             mask=strip_mask())
        self.assertIs(rep.check("amplitude_pp_mv").verdict,
                      quality.Verdict.REJECT)

    def test_amplitude_flag_band_uses_the_module_constants(self):
        """`assess` must read FLAG_PP_MV, not a band of its own.

        Pinned through `assess` rather than through `_banded` because the
        unit tests above pass their own bounds in, so nothing otherwise
        forces the shipped constants to be the ones in effect. This band
        is load-bearing: measurement shows it is the only thing that
        remarks on a 5x pitch error once the recording duration is
        unavailable and the paper-speed anchor goes with it.
        """
        cases = ((0.15, quality.Verdict.FLAG),     # under FLAG_PP_MV[0]
                 (1.00, quality.Verdict.ACCEPT),
                 (4.00, quality.Verdict.FLAG))     # over FLAG_PP_MV[1]
        for pp, expected in cases:
            rep = quality.assess(make_result(signal_mv=beat_train(amplitude_mv=pp)),
                                 mask=strip_mask())
            self.assertIs(rep.check("amplitude_pp_mv").verdict, expected,
                          msg=f"peak-to-peak {pp} mV")

    def test_coverage_bands(self):
        for cov, expected in ((1.0, quality.Verdict.ACCEPT),
                              (0.80, quality.Verdict.FLAG),
                              (0.40, quality.Verdict.REJECT)):
            rep = quality.assess(make_result(coverage=cov), mask=strip_mask())
            self.assertIs(rep.check("coverage").verdict, expected,
                          msg=f"coverage {cov}")

    def test_verdict_is_the_worst_check_not_the_last(self):
        """A rejection must survive being followed by accepting checks."""
        rep = quality.assess(make_result(calibration="bad"), mask=strip_mask())
        self.assertIs(rep.verdict, quality.Verdict.REJECT)
        self.assertIs(rep.checks[-1].verdict, quality.Verdict.ACCEPT)

    def test_unavailable_checks_are_reported_and_do_not_vote(self):
        rep = quality.assess(make_result(), mask=None)
        self.assertIs(rep.verdict, quality.Verdict.ACCEPT)
        names = {c.name for c in rep.checks if not c.available}
        self.assertIn("implied_gain", names)
        self.assertIn("region_duration", names)
        self.assertIn("traces_per_column", names)
        self.assertEqual(len(rep.unavailable), 3)

    def test_missing_mask_does_not_catch_a_stacked_sheet(self):
        """The documented consequence of not passing the mask.

        Pinned as a test because it is the difference between a gate that
        catches the 3x4 failure and one that does not, and a caller who
        omits the mask is entitled to know the gate got quieter.
        """
        stacked = strip_mask(height=900, rows=(100, 300, 500, 700))
        self.assertIs(quality.assess(make_result(), mask=stacked).verdict,
                      quality.Verdict.REJECT)
        self.assertIs(quality.assess(make_result(), mask=None).verdict,
                      quality.Verdict.ACCEPT)

    def test_absent_calibration_is_unavailable_not_a_pass(self):
        rep = quality.assess(make_result(calibration=None), mask=strip_mask())
        self.assertFalse(rep.check("calibration").available)

    def test_every_check_names_an_anchor(self):
        """FR8's constraint, enforced: no check without an external anchor."""
        rep = quality.assess(make_result(), mask=strip_mask())
        for c in rep.checks:
            self.assertTrue(c.anchor.strip(), msg=f"{c.name} has no anchor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
