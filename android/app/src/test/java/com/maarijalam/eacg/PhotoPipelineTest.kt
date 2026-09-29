package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The harmonic choice the duration anchor makes for the grid comb. Cases are
 * the numbers the first A30s photograph of PTB-XL 40 actually produced.
 */
class PhotoPipelineTest {

    @Test
    fun aFiveMillimetreLockIsDividedByFive() {
        // Phone run 2: grid 50.42 px/mm on a ~10.13 px/mm page; the anchor,
        // once the trace survives, ~+6% of truth.
        val (px, k) = PhotoPipeline.resolveAgainstAnchor(50.42, 10.13 * 1.058)!!
        assertEquals(5, k)
        assertEquals(10.084, px, 1e-3)
    }

    @Test
    fun aCorrectGridIsLeftAlone() {
        val (px, k) = PhotoPipeline.resolveAgainstAnchor(10.10, 10.60)!!
        assertEquals(1, k)
        assertEquals(10.10, px, 1e-12)
    }

    @Test
    fun aNonHarmonicLockIsRefusedNotForced() {
        // Full resolution: the dashed 5 mm rules gave 2.52x truth. /2 is 26%
        // off the anchor and /5 is 50% off: neither is an answer.
        assertNull(PhotoPipeline.resolveAgainstAnchor(47.80, 18.98 * 1.059))
    }

    @Test
    fun aShortTraceGetsNoErrorVsTruth() {
        // A trace a fifth of the truth's length, as a 5x-wrong calibration
        // gives: whatever the search finds over it is not a measurement.
        val fs = 500.0
        val truth = DoubleArray(5000) { kotlin.math.sin(2 * Math.PI * 1.1 * it / fs) +
            0.3 * kotlin.math.sin(2 * Math.PI * 7.0 * it / fs) }
        val short = truth.copyOfRange(0, 1000)
        val al = Fidelity.bestAlignment(truth, short, minOverlap = (Fidelity.MIN_OVERLAP_S * fs).toInt())
        assertEquals(false, al.valid)
        assertNull(al.pxPerMmErrorPct)
        val full = Fidelity.bestAlignment(truth, truth, minOverlap = (Fidelity.MIN_OVERLAP_S * fs).toInt())
        assertEquals(true, full.valid)
        assertEquals(0.0, full.pxPerMmErrorPct!!, 1e-9)
    }

    @Test
    fun aBrokenAnchorCannotPickAFactor() {
        // Phone run 2 as it came out: anchor -52% of truth. /5 gives 10.08
        // against 4.86, twice it: refused, rather than trusting either.
        assertNull(PhotoPipeline.resolveAgainstAnchor(50.42, 4.856))
        assertNull(PhotoPipeline.resolveAgainstAnchor(Double.NaN, 10.0))
        assertNull(PhotoPipeline.resolveAgainstAnchor(50.0, 0.0))
    }
}
