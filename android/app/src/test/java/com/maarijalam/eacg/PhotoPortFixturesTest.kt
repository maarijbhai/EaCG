package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import kotlin.math.floor

/**
 * Port fidelity for the photograph path: [StripLocator] against
 * `ecgproto/segment.py`, [PageCorners] against `ecgproto/page.py`.
 *
 * Golden vectors come from `python/make_photo_port_fixtures.py`. Integer
 * outputs -- boxes, peaks, the per-column counts of the isolated trace -- must
 * match exactly; they are decisions, and a decision that differs by one is a
 * different crop. Real-valued outputs are held to 1e-9 px, well inside what
 * any downstream stage could notice and far outside floating-point summation
 * order, which is the only thing allowed to differ.
 *
 * The corner image is not shipped: [cornerImage] rebuilds it from the same
 * definition the generator uses. If that definition drifts on either side,
 * `corner.dark_probe` fails first and says so.
 */
class PhotoPortFixturesTest {

    companion object {
        private lateinit var rows: Map<String, List<String>>

        @BeforeClass
        @JvmStatic
        fun load() {
            val stream = PhotoPortFixturesTest::class.java
                .getResourceAsStream("/photo_port_fixtures.txt")
                ?: error("photo_port_fixtures.txt not on the test classpath; " +
                    "run python/make_photo_port_fixtures.py")
            rows = stream.bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() && !it.startsWith("#") }
                    .associate { line ->
                        val p = line.split(" ")
                        p[0] to p.drop(1)
                    }
            }
        }

        private fun row(k: String) = rows[k] ?: error("missing '$k'; regenerate the fixtures")
        private fun ints(k: String) = row(k).map { it.toInt() }.toIntArray()
        private fun doubles(k: String) = row(k).map { it.toDouble() }.toDoubleArray()

        private fun mask(case: String): TraceExtractor.Mask {
            val (h, w) = ints("$case.shape").let { it[0] to it[1] }
            val data = ByteArray(h * w)
            for (r in 0 until h) {
                val runs = rows["$case.rle.$r"] ?: continue
                var i = 0
                while (i < runs.size) {
                    val s = runs[i].toInt()
                    val n = runs[i + 1].toInt()
                    for (c in s until s + n) data[r * w + c] = 1
                    i += 2
                }
            }
            return TraceExtractor.Mask(data, h, w)
        }
    }

    // ---- primitives ------------------------------------------------------

    @Test
    fun findPeaksMatchesScipyIncludingPlateausAndDistance() {
        val x = doubles("prim.x")
        assertArrayEquals(ints("prim.peaks_d7"), StripLocator.findPeaks(x, 7).toIntArray())
    }

    @Test
    fun boxSmoothingMatchesNumpyConvolveSame() {
        val want = doubles("prim.box9")
        val got = StripLocator.convolveSameBox(doubles("prim.x"), 9)
        for (i in want.indices) assertEquals("index $i", want[i], got[i], 1e-12)
    }

    @Test
    fun closingMatchesOpenCvForOddAndEvenKernels() {
        val bits = ints("prim.bits").map { it != 0 }.toBooleanArray()
        for (k in listOf(6, 7)) {
            val want = ints("prim.close$k")
            val got = StripLocator.close1d(bits, k).map { if (it) 1 else 0 }.toIntArray()
            assertArrayEquals("k=$k", want, got)
        }
    }

    // ---- the locator -----------------------------------------------------

    private fun checkLocation(case: String) {
        val m = mask(case)
        val loc = StripLocator.locate(m)
        assertArrayEquals("$case box", ints("$case.box"),
            intArrayOf(loc.x0, loc.y0, loc.x1, loc.y1))
        assertArrayEquals("$case peaks", ints("$case.peaks"), loc.rowPeaks)
        assertEquals("$case pulse", ints("$case.pulse_x1")[0], loc.pulseX1)
        assertEquals("$case t0", doubles("$case.t0_x")[0], loc.t0X, 1e-9)
        val iso = StripLocator.isolateTrace(StripLocator.sub(m, loc.y0, loc.y1, loc.x0, loc.x1))
        val cols = IntArray(iso.cols) { c ->
            var s = 0
            for (r in 0 until iso.rows) if (iso.data[r * iso.cols + c] != 0.toByte()) s++
            s
        }
        assertArrayEquals("$case isolated columns", ints("$case.isolated_cols"), cols)
    }

    @Test
    fun locatesTheSyntheticSheetExactlyAsThePythonDoes() = checkLocation("synth")

    /**
     * A pulse behind a thin lead-in, as a grid remnant at the strip edge made
     * on the first A30s photograph. Without the lead-in rule the pulse is
     * missed and traced as signal; the fixture was checked to fail that way.
     */
    @Test
    fun findsThePulseBehindAThinLeadInAsThePythonDoes() {
        checkLocation("leadin")
        assertEquals(46, ints("leadin.pulse_x1")[0])
    }

    /**
     * `run_photo.best_alignment` on a reference and a copy stretched 3%,
     * delayed, noised and gapped: the same scale and lag exactly, the same
     * PCC to rounding. The FFT differs from scipy's in its last bits; the
     * argmax it feeds must not.
     */
    @Test
    fun alignsToTheTruthExactlyAsThePythonDoes() {
        val want = doubles("align.out")
        val got = Fidelity.bestAlignment(doubles("align.ref"), doubles("align.test"))
        assertEquals("scale", want[1], got.scale, 1e-12)
        assertEquals("lag", want[2].toInt(), got.lagSamples)
        assertEquals("n", want[3].toInt(), got.nCompared)
        assertEquals("pcc", want[0], got.pcc, 1e-9)
        assertEquals("at bound", want[4] != 0.0, got.atBound)
    }

    /**
     * A 15 px piece of trace between two breaks, under the width floor: kept,
     * as it fills a hole inside the trace. The old rule dropped it as text.
     */
    @Test
    fun keepsTracePiecesThatFillAGapAsThePythonDoes() {
        checkLocation("gapfill")
        val box = ints("gapfill.box")
        val cols = ints("gapfill.isolated_cols")
        for (x in 405 until 420) assertTrue("column $x", cols[x - box[0]] > 0)
    }

    /**
     * The trace's last stretch broken into narrow pieces past the last kept
     * one, at its height: kept to the strip's end. Without this, phone run 6
     * of PTB-XL 40 lost its tenth R wave.
     */
    @Test
    fun followsTheTraceToItsEndAsThePythonDoes() {
        checkLocation("tailfill")
        val box = ints("tailfill.box")
        val cols = ints("tailfill.isolated_cols")
        for (x in 745 until 760) assertTrue("column $x", cols[x - box[0]] > 0)
        for (x in 765 until 775) assertTrue("column $x", cols[x - box[0]] > 0)
    }

    /** The ~8 mm margin corners on the sheet edge leave before the pulse. */
    @Test
    fun findsThePulseBehindASheetMarginAsThePythonDoes() {
        checkLocation("leadlong")
        assertEquals(48, ints("leadlong.pulse_x1")[0])
    }

    /** The same lead-in before a spike whose edges nearly touch: not a pulse. */
    @Test
    fun doesNotTakeAnEarlyQrsForThePulseAsThePythonDoes() {
        checkLocation("earlyqrs")
        assertEquals(-1, ints("earlyqrs.pulse_x1")[0])
    }

    @Test
    fun locatesTheRealKitSheetExactlyAsThePythonDoes() {
        if ("kit38" in row("cases")) checkLocation("kit38")
    }

    // ---- corner refinement -----------------------------------------------

    /** MUST match make_photo_port_fixtures.corner_image. BGR in the Python. */
    private fun cornerImage(): PageCorners.Plane {
        val w = 520; val h = 380
        val x0 = 60.0; val y0 = 45.0; val sw = 400.0; val sh = 280.0; val s = 0.03
        val paper = intArrayOf(236, 241, 244)          // B, G, R
        val minor = intArrayOf(219, 223, 254)
        val major = intArrayOf(0, 0, 255)
        var state = 12345L
        val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val u = (x - x0) + s * (y - y0)
            val v = (y - y0) - s * (x - x0)
            var ink: IntArray? = null
            if (u >= 0.0 && u < sw && v >= 0.0 && v < sh) {
                val iu = floor(u).toInt()
                val iv = floor(v).toInt()
                ink = if (iu % 40 == 0 || iv % 40 == 0) major
                else if (iu % 8 == 0 || iv % 8 == 0) minor else null
            }
            state = (state * 1103515245L + 12345L) and 0x7FFFFFFFL
            val noise = ((state shr 16) % 5 - 2).toInt()
            val bgr = IntArray(3) { ch ->
                val v0 = if (ink == null) paper[ch] else paper[ch] * ink[ch] / 255
                (v0 + noise).coerceIn(0, 255)
            }
            out[y * w + x] = PageCorners.darknessOf(bgr[2], bgr[1], bgr[0])
        }
        return PageCorners.Plane(out, w, h)
    }

    @Test
    fun cornerImageIsBuiltIdenticallyOnBothSides() {
        val img = cornerImage()
        val probe = doubles("corner.dark_probe")
        val at = listOf(0 to 0, 50 to 70, 100 to 200, 379 to 519)
        at.forEachIndexed { i, (y, x) ->
            assertEquals("probe ($y,$x)", probe[i], img.data[y * img.width + x].toDouble(), 0.0)
        }
    }

    @Test
    fun refinesCornersExactlyAsThePythonDoes() {
        val got = PageCorners.refine(cornerImage(), doubles("corner.rough"))
        val want = doubles("corner.refined")
        for (i in want.indices) assertEquals("corner coord $i", want[i], got.corners[i], 1e-9)
        val rmse = doubles("corner.edge_rmse")
        // Float precision, not 1e-9: the Python samples the edge response
        // through cv2.remap at float32 points, the Kotlin in double, and the
        // RMSE carries that difference (~5e-7 px) where the corners, rounded
        // to float32 on both sides, do not.
        for (i in rmse.indices) assertEquals("edge $i rmse", rmse[i], got.edgeRmsePx[i], 1e-5)
        assertArrayEquals(ints("corner.inliers"), got.edgeInliers)
    }

    // ---- binarisation threshold ------------------------------------------

    @Test
    fun midpointThresholdMatchesThePythonExactly() {
        for (case in row("thr.cases")) {
            val hist = row("thr.$case.hist").map { it.toLong() }.toLongArray()
            assertEquals(case, ints("thr.$case.t")[0], TraceExtractor.midpointThreshold(hist))
        }
    }

    // ---- harmonic resolution ---------------------------------------------

    @Test
    fun harmonicResolutionMatchesThePython() {
        for (case in row("harm.cases")) {
            val acf = doubles("harm.$case.acf")
            val spacing = doubles("harm.$case.spacing")[0]
            val novelty = doubles("harm.$case.novelty")
            GridEstimator.HARMONIC_DIVISORS.forEachIndexed { i, m ->
                val got = GridEstimator.novelty(acf, spacing, m)
                if (novelty[i].isNaN()) assert(got.isNaN()) { "$case m=$m" }
                else assertEquals("$case novelty m=$m", novelty[i], got, 1e-12)
            }
            val want = row("harm.$case.out")
            val (pitch, div) = GridEstimator.resolveHarmonic(acf, spacing, 3.0)
            assertEquals("$case divisor", want[1].toInt(), div)
            assertEquals("$case pitch", want[0].toDouble(), pitch, 1e-9)
        }
    }

    @Test
    fun refinedCornersLandNearTheTrueOnes() {
        // Not a port check: a sanity check that the case exercises the
        // refinement at all, rather than agreeing on a failure.
        val got = PageCorners.refine(cornerImage(), doubles("corner.rough")).corners
        val truth = doubles("corner.true")
        for (i in truth.indices) assertEquals("coord $i", truth[i], got[i], 1.5)
    }
}
