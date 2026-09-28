package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The corner editor's arithmetic and the scoring harness's gap filling --
 * everything about the photo path's input that can be checked without a
 * device. The gestures and drawing themselves cannot, and are not claimed.
 */
class CornerEditorTest {

    private val corners = doubleArrayOf(100.0, 100.0, 900.0, 110.0, 910.0, 700.0, 90.0, 690.0)

    @Test
    fun grabsTheNearestHandleWithinReachAndNothingElse() {
        assertEquals(0, CornerGeometry.nearestCorner(corners, 110.0, 95.0, 30.0))
        assertEquals(2, CornerGeometry.nearestCorner(corners, 905.0, 705.0, 30.0))
        // The middle of the page is on no handle: the drag belongs to the scroll.
        assertEquals(-1, CornerGeometry.nearestCorner(corners, 500.0, 400.0, 30.0))
        assertEquals(-1, CornerGeometry.nearestCorner(corners, 140.0, 100.0, 30.0))
    }

    @Test
    fun startsAsAnInsetRectangleInTlTrBrBlOrder() {
        assertArrayEquals(
            doubleArrayOf(320.0, 240.0, 2880.0, 240.0, 2880.0, 2160.0, 320.0, 2160.0),
            CornerGeometry.initialCorners(3200, 2400), 1e-12)
    }

    @Test
    fun flagsOnlyCornersAtTheFrameEdge() {
        // The first A30s photograph's refined bottom-left sat 108 px in on a
        // 3200 px frame, with the grid running out of shot beyond it; 2% of
        // the long side is 64 px, so this is the threshold that matters.
        val c = doubleArrayOf(250.0, 146.0, 2885.0, 148.0, 3105.0, 2100.0, 40.0, 2110.0)
        assertEquals(listOf(3), CornerGeometry.cornersNearFrameEdge(c, 3200, 2396))
        assertEquals(emptyList<Int>(),
            CornerGeometry.cornersNearFrameEdge(CornerGeometry.initialCorners(3200, 2396), 3200, 2396))
    }

    @Test
    fun loupeWindowIsCentredAndClippedToTheBitmap() {
        val mid = CornerGeometry.loupeWindow(500.0, 400.0, 60, 1000, 800, 240.0)!!
        assertEquals(440, mid.srcX); assertEquals(340, mid.srcY)
        assertEquals(120, mid.srcW); assertEquals(120, mid.srcH)
        assertEquals(0.0, mid.dstX, 1e-12); assertEquals(240.0, mid.dstW, 1e-12)

        // At the top-left edge only the in-bitmap part is drawn, shifted so
        // the corner itself stays at the loupe's centre.
        val edge = CornerGeometry.loupeWindow(20.0, 10.0, 60, 1000, 800, 240.0)!!
        assertEquals(0, edge.srcX); assertEquals(0, edge.srcY)
        assertEquals(80, edge.srcW); assertEquals(70, edge.srcH)
        assertEquals(80.0, edge.dstX, 1e-12); assertEquals(100.0, edge.dstY, 1e-12)
        assertEquals(120.0, edge.dstX + (20.0 - edge.srcX) * 2.0, 1e-12)

        assertNull(CornerGeometry.loupeWindow(-500.0, -500.0, 60, 1000, 800, 240.0))
    }

    @Test
    fun fillNanMatchesNumpyInterpWithHeldEnds() {
        val n = Double.NaN
        // np.interp over the NaN positions, as run_photo.fill_nan does:
        // interior gaps linear, leading and trailing gaps held at the end value.
        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 5.0),
            Fidelity.fillNan(doubleArrayOf(n, 1.0, n, n, 4.0, n, 5.0, n)),
            1e-12)
        // All NaN: nothing to interpolate from, returned unchanged.
        assertEquals(2, Fidelity.fillNan(doubleArrayOf(n, n)).count { it.isNaN() })
    }

    @Test
    fun aGapNoLongerPoisonsTheScore() {
        val truth = DoubleArray(200) { kotlin.math.sin(it / 10.0) }
        val traced = truth.copyOf().also { for (i in 80 until 90) it[i] = Double.NaN }
        assertEquals(true, Fidelity.score(truth, traced).pcc.isNaN())
        val s = Fidelity.score(truth, Fidelity.fillNan(traced))
        assertEquals(1.0, s.pcc, 1e-3)
    }
}
