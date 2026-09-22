package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin FR8 gate against golden vectors from `ecgproto.quality`.
 *
 * The check that earns its place on this device is [QualityGate.tracesPerColumn]:
 * three of the six bundled fixtures are 3x4 sheets, and the tracer handed a
 * sheet returns a plausible-looking waveform spliced from four leads. The
 * desktop measured that at PCC 0.273 with every calibration statistic clean,
 * so nothing upstream can catch it.
 *
 * Every failure case here is paired with a good case that must be accepted.
 * A gate is unusually easy to test badly — "bad input is rejected" is
 * satisfied by a function that rejects everything.
 */
class QualityGatePortTest {

    private val fx = PortFixtures.load()

    private fun stacked(): TraceExtractor.Mask = fx.stackedMask(
        fx.int("bands.rows"), fx.int("bands.cols"),
        fx.ints("bands.at"), fx.int("trace.thickness")
    )

    private fun close(): TraceExtractor.Mask = fx.stackedMask(
        fx.int("bands.rows"), fx.int("bands.cols"),
        fx.ints("bands.close_at"), 1
    )

    // ---- band counting, against the Python -------------------------------

    @Test
    fun `a single strip counts one band`() {
        assertEquals(
            fx.double("bands.trace_mask"),
            QualityGate.tracesPerColumn(fx.traceMask(), fx.double("trace.px_per_mm")),
            0.0
        )
    }

    @Test
    fun `a stacked sheet counts four bands`() {
        assertEquals(
            fx.double("bands.stacked"),
            QualityGate.tracesPerColumn(stacked(), fx.double("trace.px_per_mm")),
            0.0
        )
    }

    @Test
    fun `the separation is in millimetres, not pixels`() {
        // Note the direction, which is the opposite of the first guess: a
        // FINER pitch means more pixels per millimetre and so a LARGER pixel
        // threshold, which merges bands a coarser pitch resolves. A check
        // written in pixels would return the same answer for both and would
        // change meaning with camera distance.
        val mask = close()
        assertEquals(
            fx.double("bands.close_coarse"),
            QualityGate.tracesPerColumn(mask, fx.double("bands.close_coarse_pitch")),
            0.0
        )
        assertEquals(
            fx.double("bands.close_fine"),
            QualityGate.tracesPerColumn(mask, fx.double("bands.close_fine_pitch")),
            0.0
        )
        assertNotEquals(
            "the two pitches must disagree, or this test proves nothing",
            fx.double("bands.close_coarse"), fx.double("bands.close_fine")
        )
    }

    @Test
    fun `the median ignores ink in a few columns`() {
        // Printed lead labels occupy a handful of columns and add a band to
        // each. A mean would drift above the threshold and condemn a
        // perfectly good strip; a median does not. Without this the choice
        // between them is untested, because every other mask here is uniform
        // down its columns and the two agree.
        val src = fx.traceMask()
        val data = src.data.copyOf()
        for (c in intArrayOf(7, 14)) {          // two of ~12 sampled columns
            data[2 * src.cols + c] = 1          // a label, far from the trace
        }
        val withLabel = TraceExtractor.Mask(data, src.rows, src.cols)
        val bands = QualityGate.tracesPerColumn(withLabel, fx.double("trace.px_per_mm"))
        assertEquals("median must stay at one band", 1.0, bands, 0.0)
        assertEquals("and the strip must still be accepted",
            QualityGate.Verdict.ACCEPT,
            QualityGate.assess(
                goodResult(), withLabel, 8.0, 8.0,
                imageWidthPx = 2000, expectedDurationS = 10.0
            ).verdict)
    }

    @Test
    fun `two bands is already too many`() {
        // The threshold case, and the failure's actual onset. A four-band
        // fixture alone leaves the threshold free to sit anywhere between two
        // and four without any test noticing.
        val two = fx.stackedMask(
            fx.int("bands.rows"), fx.int("bands.cols"),
            listOf(40, 200), fx.int("trace.thickness")
        )
        assertEquals(2.0,
            QualityGate.tracesPerColumn(two, fx.double("trace.px_per_mm")), 0.0)
        val report = QualityGate.assess(
            goodResult(), two, 8.0, 8.0,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        assertEquals(QualityGate.Verdict.REJECT, report.check("traces_per_column").verdict)
        assertEquals(QualityGate.Verdict.REJECT, report.verdict)
    }

    @Test
    fun `an empty mask is NaN, not zero bands`() {
        val empty = TraceExtractor.Mask(ByteArray(100), 10, 10)
        assertTrue(QualityGate.tracesPerColumn(empty, 8.0).isNaN())
    }

    // ---- the bands ------------------------------------------------------

    @Test
    fun `amplitude bands are pinned on both sides`() {
        fun v(pp: Double) = QualityGate.banded(
            pp, QualityGate.REJECT_PP_MV_LO, QualityGate.REJECT_PP_MV_HI,
            QualityGate.FLAG_PP_MV_LO, QualityGate.FLAG_PP_MV_HI
        )
        assertEquals(QualityGate.Verdict.ACCEPT, v(1.0))
        assertEquals(QualityGate.Verdict.ACCEPT, v(QualityGate.FLAG_PP_MV_LO))
        assertEquals(QualityGate.Verdict.ACCEPT, v(QualityGate.FLAG_PP_MV_HI))
        assertEquals(QualityGate.Verdict.FLAG, v(0.19))
        assertEquals(QualityGate.Verdict.FLAG, v(3.51))
        assertEquals(QualityGate.Verdict.REJECT, v(0.04))
        assertEquals(QualityGate.Verdict.REJECT, v(6.01))
        assertEquals(QualityGate.Verdict.REJECT, v(Double.NaN))
    }

    @Test
    fun `unavailable never outranks accept`() {
        // UNAVAILABLE means "not assessed". If it sorted above ACCEPT it
        // would turn every missing anchor into a verdict, and the paper-speed
        // anchor is missing in the normal case — a photograph nobody can
        // state the duration of.
        assertTrue(QualityGate.Verdict.UNAVAILABLE < QualityGate.Verdict.ACCEPT)
        assertTrue(QualityGate.Verdict.ACCEPT < QualityGate.Verdict.FLAG)
        assertTrue(QualityGate.Verdict.FLAG < QualityGate.Verdict.REJECT)
    }

    // ---- the gate as a whole --------------------------------------------

    /** A digitisation plausible on every check, to pair against failures. */
    private fun goodResult(coverage: Double = 1.0): TraceExtractor.Digitised {
        val n = 500
        val mv = DoubleArray(n) { i ->
            // A crude beat train: peaks of 1.0 mV, so peak-to-peak is ~1 mV.
            if (i % 100 < 5) 1.0 - kotlin.math.abs(i % 100 - 2) * 0.2 else 0.0
        }
        return TraceExtractor.Digitised(mv, 100.0, 100.0, coverage, 80.0, 200.0)
    }

    @Test
    fun `a good digitisation is accepted`() {
        val report = QualityGate.assess(
            goodResult(), fx.traceMask(), 8.0, 8.0,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        assertEquals(report.summary(), QualityGate.Verdict.ACCEPT, report.verdict)
        assertTrue(report.usable)
    }

    @Test
    fun `a stacked sheet is rejected, and not by the calibration checks`() {
        val report = QualityGate.assess(
            goodResult(), stacked(), 8.0, 8.0,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        assertEquals(QualityGate.Verdict.REJECT, report.verdict)
        assertEquals(QualityGate.Verdict.REJECT, report.check("traces_per_column").verdict)
        // The point of the whole check: calibration is correct here. If this
        // ever starts rejecting, the test has stopped proving what it says.
        assertEquals(QualityGate.Verdict.ACCEPT, report.check("grid").verdict)
        assertEquals(QualityGate.Verdict.ACCEPT, report.check("paper_speed").verdict)
    }

    @Test
    fun `an anisotropic grid is rejected`() {
        val report = QualityGate.assess(
            goodResult(), fx.traceMask(), 8.0, 31.75,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        assertEquals(QualityGate.Verdict.REJECT, report.check("grid").verdict)
    }

    @Test
    fun `a five times pitch error is rejected on implied paper speed`() {
        // The harmonic lock: the 1 mm rules did not resolve and the estimator
        // locked onto the 5 mm family. Internally self-consistent, and only
        // an external fact catches it.
        val report = QualityGate.assess(
            goodResult(), fx.traceMask(), 8.0 * 5, 8.0 * 5,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        assertEquals(QualityGate.Verdict.REJECT, report.check("paper_speed").verdict)
    }

    @Test
    fun `a missing duration is unavailable rather than a pass`() {
        val report = QualityGate.assess(
            goodResult(), fx.traceMask(), 8.0, 8.0,
            imageWidthPx = 2000, expectedDurationS = null
        )
        assertEquals(QualityGate.Verdict.UNAVAILABLE,
            report.check("paper_speed").verdict)
        assertEquals("a missing anchor must not change the verdict",
            QualityGate.Verdict.ACCEPT, report.verdict)
        assertTrue(report.unavailable.isNotEmpty())
    }

    @Test
    fun `coverage bands are pinned`() {
        assertEquals(QualityGate.Verdict.ACCEPT,
            gateWith(coverage = 1.0).check("coverage").verdict)
        assertEquals(QualityGate.Verdict.FLAG,
            gateWith(coverage = 0.80).check("coverage").verdict)
        assertEquals(QualityGate.Verdict.REJECT,
            gateWith(coverage = 0.40).check("coverage").verdict)
    }

    private fun gateWith(coverage: Double) = QualityGate.assess(
        goodResult(coverage), fx.traceMask(), 8.0, 8.0,
        imageWidthPx = 2000, expectedDurationS = 10.0
    )

    @Test
    fun `every check names an anchor`() {
        // FR8's constraint, enforced: no check without an external anchor.
        val report = QualityGate.assess(
            goodResult(), fx.traceMask(), 8.0, 8.0,
            imageWidthPx = 2000, expectedDurationS = 10.0
        )
        for (c in report.checks) {
            assertTrue("${c.name} has no anchor", c.anchor.isNotBlank())
        }
    }
}
