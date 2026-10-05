package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.BeforeClass
import org.junit.Test
import kotlin.math.abs
import kotlin.math.max

/**
 * Port fidelity for [GridFit] against `ecgproto/gridfit.py`.
 *
 * Golden vectors from `python/make_grid_port_fixtures.py`, on the photograph
 * the phone lost to locate's 3x3 opening (ptbxl9_20261005_01) at the phone's
 * 3,200 px with its automatic corners. The rectified page is too large to
 * ship and OpenCV cannot run here, so the per-band median profiles ship as
 * inputs; the profile step itself is checked on a procedural image both
 * sides define. Decisions (peaks, kept rules, indices, chain offsets) must
 * match exactly.
 */
class GridPortFixturesTest {

    companion object {
        private lateinit var rows: Map<String, List<String>>

        @BeforeClass
        @JvmStatic
        fun load() {
            val stream = GridPortFixturesTest::class.java.getResourceAsStream("/grid_port_fixtures.txt")
                ?: error("grid_port_fixtures.txt not on the test classpath; run python/make_grid_port_fixtures.py")
            rows = stream.bufferedReader().useLines { lines ->
                lines.map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }
                    .associate { l -> val p = l.split(" "); p[0] to p.drop(1) }
            }
        }

        private fun row(k: String) = rows[k] ?: error("missing '$k'; regenerate the fixtures")
        private fun doubles(k: String) = row(k).map { it.toDouble() }.toDoubleArray()
        private fun floats(k: String) = row(k).map { it.toFloat() }.toFloatArray()
        private fun ints(k: String) = row(k).map { it.toDouble().toInt() }.toIntArray()
        private fun double(k: String) = row(k)[0].toDouble()
        private fun int(k: String) = row(k)[0].toInt()

        private fun bands(axis: Int): GridFit.Bands {
            val n = int("axis$axis.nbands")
            return GridFit.Bands(doubles("axis$axis.centres"),
                List(n) { floats("axis$axis.prof.$it") }, int("axis$axis.margin"))
        }

        private fun close(msg: String, want: DoubleArray, got: DoubleArray, tol: Double) {
            assertEquals("$msg: length", want.size, got.size)
            for (i in want.indices) assertEquals("$msg[$i]", want[i], got[i], tol * max(1.0, abs(want[i])))
        }
    }

    @Test
    fun constantsMatch() {
        assertEquals(double("const.PITCH_MM"), GridFit.PITCH_MM, 0.0)
        assertEquals(double("const.BAND_MM"), GridFit.BAND_MM, 0.0)
        assertEquals(double("const.BAND_STEP_MM"), GridFit.BAND_STEP_MM, 0.0)
        assertEquals(double("const.EDGE_MARGIN_MM"), GridFit.EDGE_MARGIN_MM, 0.0)
        assertEquals(double("const.MIN_REL_PROMINENCE"), GridFit.MIN_REL_PROMINENCE, 0.0)
        assertEquals(double("const.ON_PITCH_FRACTION"), GridFit.ON_PITCH_FRACTION, 0.0)
        assertEquals(double("const.MATCH_FRACTION"), GridFit.MATCH_FRACTION, 0.0)
        assertEquals(double("const.HUBER_PX"), GridFit.HUBER_PX, 0.0)
        assertEquals(double("const.MIN_RULES").toInt(), GridFit.MIN_RULES)
    }

    /** The image step: redness and band medians, on make_grid_port_fixtures.fake_rgb. */
    @Test
    fun rednessAndBandProfilesMatchOnAProceduralImage() {
        val h = 120; val w = 90
        val px = ByteArray(h * w * 3)
        for (y in 0 until h) for (x in 0 until w) {
            val o = (y * w + x) * 3
            px[o] = ((x * 7 + y * 3) % 256).toByte()                       // b
            px[o + 1] = ((x * 5 + y * 11) % 256).toByte()                  // g
            px[o + 2] = ((x * 13 + y * 2 + (x / 9) * 40) % 256).toByte()   // r
        }
        val resp = GridFit.redness(px, h, w, 3, rIdx = 2, gIdx = 1, bIdx = 0)
        assertEquals(double("fake.redness.sum"), resp.v.sumOf { it.toDouble() }, 1e-3)
        for (axis in 0..1) {
            val b = GridFit.bandProfiles(resp, 25.0, axis)
            assertEquals(int("fake.$axis.margin"), b.margin)
            close("centres $axis", doubles("fake.$axis.centres"), b.centres, 0.0)
            for (i in b.profiles.indices) close("profile $axis/$i", doubles("fake.$axis.prof.$i"),
                DoubleArray(b.profiles[i].size) { b.profiles[i][it].toDouble() }, 0.0)
        }
    }

    @Test
    fun detrendFindPeaksAndProminencesMatch() {
        val pitch = double("pitch")
        for (axis in 0..1) {
            val d = GridFit.detrend(floats("axis$axis.prof.0"), pitch)
            close("detrended $axis", doubles("axis$axis.detrended.0"),
                DoubleArray(d.size) { d[it].toDouble() }, 1e-4)
            val x = doubles("axis$axis.detrended.0")
            val peaks = StripLocator.findPeaks(x, max(2, (0.6 * pitch).toInt()))
            assertArrayEquals("find_peaks $axis", ints("axis$axis.findpeaks.0"), peaks.toIntArray())
            close("prominences $axis", doubles("axis$axis.prominences.0"),
                GridFit.prominences(x, peaks), 1e-9)
        }
    }

    @Test
    fun everyBandsPeaksIndicesAndChainMatch() {
        val pitch = double("pitch")
        for (axis in 0..1) {
            val b = bands(axis)
            for (i in b.profiles.indices) {
                val pk = GridFit.bandPeaks(b.profiles[i], pitch)
                close("peaks $axis/$i", doubles("axis$axis.peaks.$i"), pk, 1e-5)
                val (kept, idx) = GridFit.indexBand(DoubleArray(pk.size) { pk[it] + b.margin }, pitch)
                close("kept $axis/$i", doubles("axis$axis.kept.$i"), kept, 1e-5)
                assertArrayEquals("idx $axis/$i", ints("axis$axis.idx.$i"), idx)
            }
            val s = GridFit.ruleSamples(b, pitch)
            val want = ints("axis$axis.offsets")
            val wantMap = (want.indices step 2).associate { want[it] to want[it + 1] }
            assertEquals("offsets $axis", wantMap, s.offsets.toSortedMap())
            assertEquals("samples $axis", int("axis$axis.samples.n"), s.index.size)
            val sums = doubles("axis$axis.samples.sum")
            // Python's sum columns are (x, y, i) in the page's own axes.
            val x = if (axis == 0) s.along else s.across
            val y = if (axis == 0) s.across else s.along
            assertEquals(sums[0], x.sum(), 1e-3)
            assertEquals(sums[1], y.sum(), 1e-3)
            assertEquals(sums[2], s.index.sum().toDouble(), 0.0)
        }
    }

    @Test
    fun pitchRefinementMatches() {
        val pk = GridFit.bandPeaks(floats("mid.prof"), GridFit.PITCH_MM * double("guess"))
        close("mid peaks", doubles("mid.peaks"), pk, 1e-5)
        assertEquals(double("pitch"), GridFit.refinePitch(floats("mid.prof"), double("guess")), 1e-9)
    }

    @Test
    fun theFitAndThePageMatch() {
        val (h, w) = ints("rect.shape").let { it[0] to it[1] }
        val f = GridFit.fitFromBands(bands(0), bands(1), double("pitch"), w, h)
        close("H", doubles("fit.H"), f.h, 1e-6)
        assertArrayEquals(ints("fit.size"), intArrayOf(f.outW, f.outH))
        assertEquals(double("fit.px_per_mm"), f.pxPerMm, 1e-12)
        assertEquals(double("fit.rms_mm"), f.rmsMm, 1e-6)
        assertArrayEquals(ints("fit.rules"), intArrayOf(f.nVertical, f.nHorizontal))
        assertEquals(int("fit.samples"), f.nSamples)
        assertEquals(double("fit.inliers"), f.inlierFraction, 1e-12)
    }
}
