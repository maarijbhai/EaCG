package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The Kotlin tracer (FR5-FR7) against golden vectors from `ecgproto.digitise`.
 *
 * Two implementations of one algorithm, in two languages, on two processors.
 * Nothing else forces them to agree. These tests assert that the **actually
 * compiled Kotlin** reproduces what the Python computed, so a porting bug
 * fails a test run in half a second instead of waiting to be noticed as a
 * slightly wrong waveform on a phone — or worse, not noticed.
 *
 * Regenerate the vectors after any change to the numerics on either side:
 *
 *     python python/make_port_fixtures.py
 *
 * NOT COVERED HERE. [TraceExtractor.binarise] — the Otsu threshold and the
 * morphological opening — is OpenCV on both sides and cannot run in a
 * host-side unit test, so it remains verifiable only on the device. That is
 * the same boundary [ImageWarp] sits behind.
 */
class TraceExtractorPortTest {

    private val fx = PortFixtures.load()

    /** 1e-12 relative, the same tolerance the skew port test uses. */
    private fun assertClose(expected: Double, actual: Double, what: String) {
        if (expected.isNaN()) {
            assertTrue("$what: expected NaN, got $actual", actual.isNaN())
            return
        }
        assertTrue("$what: expected $expected, got NaN", !actual.isNaN())
        val tol = 1e-12 * maxOf(1.0, kotlin.math.abs(expected))
        assertEquals(what, expected, actual, tol)
    }

    private fun assertArrayClose(expected: DoubleArray, actual: DoubleArray, what: String) {
        assertEquals("$what: length", expected.size, actual.size)
        for (i in expected.indices) assertClose(expected[i], actual[i], "$what[$i]")
    }

    // ---- the shared input -------------------------------------------------

    @Test
    fun `mask matches the python, by ink count`() {
        // A divergence in the INPUT must be reported as such, rather than
        // surfacing as a confusing disagreement in every output below.
        val mask = fx.traceMask()
        var ink = 0
        for (b in mask.data) if (b != 0.toByte()) ink++
        assertEquals("ink pixels", fx.int("trace.ink"), ink)
        assertEquals("rows", fx.int("trace.rows"), mask.rows)
        assertEquals("cols", fx.int("trace.cols"), mask.cols)
    }

    // ---- FR6, vectorisation ----------------------------------------------

    @Test
    fun `centroids match the python`() {
        assertArrayClose(
            fx.doubles("trace.centroids"),
            TraceExtractor.centroids(fx.traceMask()),
            "centroid"
        )
    }

    @Test
    fun `blank columns are NaN and not zero`() {
        // Zero is a real row at the top of the image. A port that returned
        // zero for "no ink" would place a spurious sample at the top of the
        // strip, which reads as signal rather than as absence.
        val y = TraceExtractor.centroids(fx.traceMask())
        val gapMod = fx.int("trace.gap_mod")
        val gapWidth = fx.int("trace.gap_width")
        for (c in y.indices) {
            if (c % gapMod < gapWidth) assertTrue("column $c should be NaN", y[c].isNaN())
        }
    }

    @Test
    fun `span trim and coverage match the python`() {
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        assertEquals("trim lo", fx.int("trace.trim_lo"), trace.trimLo)
        assertEquals("trim hi", fx.int("trace.trim_hi"), trace.trimHi)
        assertClose(fx.double("trace.coverage"), trace.coverage, "coverage")
    }

    @Test
    fun `gap bridging matches the python`() {
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        assertArrayClose(
            fx.doubles("trace.filled"),
            TraceExtractor.interpolateGaps(trace.yPx, fx.int("trace.max_gap_px")),
            "filled"
        )
    }

    @Test
    fun `gaps wider than the limit are left blank`() {
        // The other half of the function, and the half that matters: a long
        // gap bridged is invented signal that was never on the paper, which
        // is worse than a hole because a hole is visible.
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        val tight = TraceExtractor.interpolateGaps(
            trace.yPx, fx.int("trace.max_gap_px_tight"))
        assertArrayClose(fx.doubles("trace.filled_tight"), tight, "filled_tight")
        assertTrue("a tight limit must leave some column blank",
            tight.any { it.isNaN() })
    }

    @Test
    fun `baseline matches the python, including the even-length average`() {
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        val filled = TraceExtractor.interpolateGaps(trace.yPx, fx.int("trace.max_gap_px"))
        assertClose(fx.double("trace.baseline"),
            TraceExtractor.baselineFrom(filled), "baseline")
    }

    @Test
    fun `median averages the two middle values at even length`() {
        // NumPy does; taking the lower one instead passes most inputs and
        // disagrees by half a pixel on half of them.
        assertEquals(2.5, TraceExtractor.baselineFrom(doubleArrayOf(1.0, 2.0, 3.0, 4.0)), 0.0)
        assertEquals(2.0, TraceExtractor.baselineFrom(doubleArrayOf(1.0, 2.0, 3.0)), 0.0)
    }

    // ---- FR7, calibration ------------------------------------------------

    @Test
    fun `millivolt conversion matches the python`() {
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        val filled = TraceExtractor.interpolateGaps(trace.yPx, fx.int("trace.max_gap_px"))
        val baseline = TraceExtractor.baselineFrom(filled)
        assertArrayClose(
            fx.doubles("trace.mv"),
            TraceExtractor.pixelsToMv(filled, baseline, fx.double("trace.px_per_mv")),
            "mv"
        )
    }

    @Test
    fun `a trace above the baseline is positive millivolts`() {
        // The sign convention, pinned independently of the golden vector.
        // Rows grow down, millivolts grow up. Reversing this produces a
        // waveform correct in every respect except upside down, which PCC
        // reports as a tidy -1.0 rather than as a failure.
        val above = TraceExtractor.pixelsToMv(doubleArrayOf(90.0), 100.0, 10.0)[0]
        val below = TraceExtractor.pixelsToMv(doubleArrayOf(110.0), 100.0, 10.0)[0]
        assertEquals("10 px above baseline at 10 px/mV", 1.0, above, 1e-12)
        assertEquals("10 px below baseline at 10 px/mV", -1.0, below, 1e-12)
    }

    @Test
    fun `resampling matches the python at both output rates`() {
        val trace = TraceExtractor.trim(TraceExtractor.centroids(fx.traceMask()))
        val filled = TraceExtractor.interpolateGaps(trace.yPx, fx.int("trace.max_gap_px"))
        val baseline = TraceExtractor.baselineFrom(filled)
        val mv = TraceExtractor.pixelsToMv(filled, baseline, fx.double("trace.px_per_mv"))
        for (j in 0 until fx.int("resample.n")) {
            val pxPerS = fx.double("resample.$j.in", 0)
            val fsOut = fx.double("resample.$j.in", 1)
            assertEquals("resample.$j input length",
                fx.int("resample.$j.in", 2), mv.size)
            assertArrayClose(
                fx.doubles("resample.$j.out"),
                TraceExtractor.resample(mv, pxPerS, fsOut),
                "resample.$j"
            )
        }
    }

    @Test
    fun `the whole tracer runs end to end on the fixture mask`() {
        // Not a fidelity check — the stages above cover that — but a check
        // that `digitise` wires them together in the right order with the
        // right arguments, which no single-stage test can see.
        val pxPerMm = fx.double("trace.px_per_mm")
        val out = TraceExtractor.digitise(
            fx.traceMask(), pxPerMm, pxPerMm, fsOut = 100.0)
        assertArrayClose(fx.doubles("resample.0.out"), out.signalMv, "end-to-end")
        assertClose(fx.double("trace.baseline"), out.baselinePx, "baseline")
        assertClose(fx.double("trace.coverage"), out.coverage, "coverage")
    }

    @Test(expected = IllegalArgumentException::class)
    fun `a mask with no ink is refused rather than returning a flat line`() {
        TraceExtractor.trim(DoubleArray(50) { Double.NaN })
    }

    @Test(expected = IllegalArgumentException::class)
    fun `a single inked column is refused too`() {
        // Distinct from the no-ink case, and it has to be tested separately:
        // a guard written as `lo < 0` alone passes the all-NaN test and lets
        // one column through, after which the span is one sample wide and
        // every later stage divides by a zero-length interval.
        val y = DoubleArray(50) { Double.NaN }
        y[17] = 20.0
        TraceExtractor.trim(y)
    }

    @Test
    fun `two inked columns are the minimum accepted`() {
        // The other side of the boundary, so the guard cannot be tightened
        // into rejecting real input without a test noticing.
        val y = DoubleArray(50) { Double.NaN }
        y[17] = 20.0
        y[18] = 21.0
        val trace = TraceExtractor.trim(y)
        assertEquals(17, trace.trimLo)
        assertEquals(18, trace.trimHi)
        assertEquals(2, trace.yPx.size)
    }
}
