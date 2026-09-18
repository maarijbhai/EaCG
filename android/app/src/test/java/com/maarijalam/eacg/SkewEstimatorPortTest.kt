package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale
import kotlin.math.abs

/**
 * Does the compiled Kotlin still compute what `ecgproto/deskew.py` computes?
 *
 * This is the regression net for a project that maintains two
 * implementations of one algorithm in two languages. The Python is the
 * reference — `results/deskew/` is measured from it and the thesis cites
 * those numbers — so any divergence is a defect in the port by definition,
 * and the direction of the comparison is not symmetric.
 *
 * **Why these tests run on the host JVM and touch no OpenCV.**
 * [SkewEstimator.search] was split out of [SkewEstimator.estimate] precisely
 * so that everything capable of carrying a porting bug — the binning, the
 * rounding, the accumulator indexing, the sweeps, the parabola — can be
 * exercised by `gradlew testDebugUnitTest` in seconds. `Mat` appears only in
 * `estimate`'s resize and response, which are OpenCV's own well-tested code
 * plus one loop. A test that needed a device would be a test that did not
 * get run.
 *
 * **Why the input is a formula rather than a stored image.** A realistic
 * response plane is ~900x700 doubles, far too large to commit. Both sides
 * generate identical input from an integer LCG instead, so the values are
 * bit-identical with nothing stored. That the field is pseudo-random and has
 * no grid in it does not weaken the comparison: what these vectors check is
 * that the *arithmetic* agrees at each candidate angle. Whether the search
 * locates a real angle is a different question, and [findsAKnownAngle] below
 * answers it against a lattice this file builds, where the truth is known by
 * construction and nothing crosses a language boundary.
 */
class SkewEstimatorPortTest {

    private val fx = PortFixtures.load()

    /**
     * Relative tolerance for a value that should be identical.
     *
     * The two implementations do the same operations in the same order on
     * the same inputs, so agreement should be to the last bit; 1e-12 allows
     * only for the decimal round-trip through the fixture file. It is
     * deliberately far tighter than anything the waveform could notice,
     * because the point is to catch a changed computation while it is still
     * small, not to certify that the output is good enough.
     */
    private val rtol = 1e-12

    private fun assertClose(expected: Double, actual: Double, what: String) {
        val denom = maxOf(abs(expected), 1e-300)
        val rel = abs(actual - expected) / denom
        assertTrue(
            "$what: expected $expected, got $actual (relative $rel > $rtol)",
            rel <= rtol
        )
    }

    /**
     * The input itself, checked before anything computed from it.
     *
     * Without this, a divergence in the LCG would surface as every sharpness
     * value disagreeing at once, which reads as a broken algorithm rather
     * than as a broken fixture.
     */
    @Test
    fun sharedInputMatches() {
        val plane = fx.lcgPlane()
        assertEquals(fx.int("lcg.rows"), plane.rows)
        assertEquals(fx.int("lcg.cols"), plane.cols)
        assertClose(fx.double("lcg.sum"), plane.data.sum(), "lcg.sum")
        val first8 = fx.doubles("lcg.first8")
        for (i in first8.indices) {
            assertEquals("lcg.first8[$i]", first8[i], plane.data[i], 0.0)
        }
    }

    /**
     * The inner loop: sheared projection plus sharpness, one angle at a time.
     *
     * This is the test that matters most. Every subtle way the port could be
     * wrong lands here — the sign of the binning slope, round-half-to-even
     * against round-half-up, the run-boundary arithmetic, the offset into the
     * accumulator, the centred crop, the `rows - span` core, the divisor in
     * the mean squared successive difference. Sampling one angle at a time
     * localises a failure to a specific candidate instead of to "the
     * estimate is off".
     */
    @Test
    fun shearedSharpnessMatchesPython() {
        val plane = fx.lcgPlane()
        val transposed = plane.transposed()
        val scratchH = SkewEstimator.Scratch(plane)
        val scratchV = SkewEstimator.Scratch(transposed)
        val lenH = fx.int("sharp.core_len_h")
        val lenV = fx.int("sharp.core_len_v")

        val n = fx.int("sharp.n")
        assertTrue("fixture has no angles", n > 0)
        for (i in 0 until n) {
            val angle = fx.double("sharp.$i", 0)
            val expectH = fx.double("sharp.$i", 1)
            val expectV = fx.double("sharp.$i", 2)

            // The binning slope is the negation of the external angle; see
            // SkewEstimator.sweep.
            val t = Math.tan(Math.toRadians(-angle))
            val gotH = SkewEstimator.shearedSharpness(plane, t, lenH, scratchH)
            val gotV = SkewEstimator.shearedSharpness(transposed, -t, lenV, scratchV)

            assertClose(expectH, gotH, "sharpness h at $angle deg")
            assertClose(expectV, gotV, "sharpness v at $angle deg")
        }
    }

    /**
     * The same projection, at slopes chosen so the shift lands exactly on a
     * half-integer.
     *
     * This exists because [shearedSharpnessMatchesPython] could not catch a
     * rounding error and a mutation check proved it: replacing `Math.rint`
     * with `Math.round` — half-to-even with half-up, which is the obvious
     * Kotlin spelling and the wrong one — passed every angle-based test.
     * `tan(radians(x))` is never an exact dyadic rational, so `col * tan`
     * never lands exactly on .5 and the two roundings never disagree.
     *
     * Feeding the slope directly fixes that: 0.5 and 0.25 are exact in
     * binary, so the products are exact, and at 0.5 the roundings first
     * differ at column 3 (-1.5 becomes -2 or -1). Two of these also cover
     * paths the angle sweep does not reach — 0.0 is the degenerate no-shear
     * case where the projection collapses to a plain column sum, and 1.5
     * exceeds the span guard and must return zero from both implementations.
     */
    @Test
    fun shearedSharpnessMatchesPythonAtExactHalfIntegerShifts() {
        val plane = fx.lcgPlane()
        val transposed = plane.transposed()
        val scratchH = SkewEstimator.Scratch(plane)
        val scratchV = SkewEstimator.Scratch(transposed)
        val lenH = fx.int("sharp.core_len_h")
        val lenV = fx.int("sharp.core_len_v")

        for (i in 0 until fx.int("shear.n")) {
            val t = fx.double("shear.$i", 0)
            val expectH = fx.double("shear.$i", 1)
            val expectV = fx.double("shear.$i", 2)
            assertClose(
                expectH,
                SkewEstimator.shearedSharpness(plane, t, lenH, scratchH),
                "sharpness h at slope $t"
            )
            assertClose(
                expectV,
                SkewEstimator.shearedSharpness(transposed, -t, lenV, scratchV),
                "sharpness v at slope $t"
            )
        }
    }

    /** Profile lengths, including the degenerate cases where the floor of 8 holds. */
    @Test
    fun coreLengthsMatchPython() {
        for (i in 0 until fx.int("core.n")) {
            val rows = fx.int("core.$i", 0)
            val cols = fx.int("core.$i", 1)
            val halfRange = fx.double("core.$i", 2)
            val expectH = fx.int("core.$i", 3)
            val expectV = fx.int("core.$i", 4)
            val (gotH, gotV) = SkewEstimator.coreLengths(rows, cols, halfRange)
            assertEquals("core len h for ($rows,$cols,$halfRange)", expectH, gotH)
            assertEquals("core len v for ($rows,$cols,$halfRange)", expectV, gotV)
        }
    }

    /**
     * Sweep construction. An off-by-one at the end of a sweep changes which
     * sample the parabola is fitted around, which moves the recovered angle
     * without anything looking broken.
     */
    @Test
    fun arangeMatchesNumpy() {
        for (i in 0 until fx.int("arange.n")) {
            val start = fx.double("arange.$i.in", 0)
            val stop = fx.double("arange.$i.in", 1)
            val step = fx.double("arange.$i.in", 2)
            val expectN = fx.int("arange.$i.out", 0)
            val expectFirst = fx.double("arange.$i.out", 1)
            val expectLast = fx.double("arange.$i.out", 2)

            val got = SkewEstimator.arange(start, stop, step)
            assertEquals("arange($start,$stop,$step) length", expectN, got.size)
            assertClose(expectFirst, got.first(), "arange $i first")
            assertClose(expectLast, got.last(), "arange $i last")
        }
    }

    /** Sub-step refinement, including both edge cases and a zero denominator. */
    @Test
    fun parabolicPeakMatchesPython() {
        for (i in 0 until fx.int("peak.n")) {
            val angles = fx.doubles("peak.$i.angles")
            val values = fx.doubles("peak.$i.values")
            val expected = fx.double("peak.$i.out")
            val got = SkewEstimator.parabolicPeak(angles, values)
            assertClose(expected, got, "parabolicPeak case $i")
        }
    }

    /**
     * The shipped sweep lengths, pinned.
     *
     * 41 coarse and 21 fine are not arbitrary — they fall out of the
     * constants, and the Python's own sweeps are the same size. Pinning them
     * turns a silent change of `COARSE_STEP_DEG` or `FINE_HALF_RANGE_DEG`
     * into a failing test with a name that says what moved.
     */
    @Test
    fun shippedSweepLengthsAreUnchanged() {
        val coarse = SkewEstimator.arange(
            -SkewEstimator.COARSE_HALF_RANGE_DEG,
            SkewEstimator.COARSE_HALF_RANGE_DEG + 0.5 * SkewEstimator.COARSE_STEP_DEG,
            SkewEstimator.COARSE_STEP_DEG
        )
        assertEquals("coarse sweep length", 41, coarse.size)

        val fine = SkewEstimator.arange(
            -SkewEstimator.FINE_HALF_RANGE_DEG,
            SkewEstimator.FINE_HALF_RANGE_DEG + 0.5 * SkewEstimator.FINE_STEP_DEG,
            SkewEstimator.FINE_STEP_DEG
        )
        assertEquals("fine sweep length", 21, fine.size)
    }

    // ----------------------------------------------------------------------
    // behaviour, against a truth this file constructs
    // ----------------------------------------------------------------------

    /**
     * Does the search actually find an angle that is there?
     *
     * The vectors above prove the arithmetic agrees; they cannot prove the
     * search works, because the LCG field has no lattice in it. This builds
     * one at a known angle and asks for it back. Nothing crosses a language
     * boundary — the truth is the angle the lattice was drawn at — so this
     * test is immune to the one hole the golden vectors have, that a mistake
     * made identically on both sides would be invisible.
     *
     * The tolerance is 0.05 degrees. The Python's worst case over three real
     * fixtures and seventeen angles at the shipped working resolution is
     * 0.041 (`results/deskew/finescale.csv`), and the requirement the
     * waveform imposes is about 0.1 — one degree of residual rotation takes
     * PCC from 0.981 to 0.734, while 0.04 costs nothing measurable.
     */
    @Test
    fun findsAKnownAngle() {
        val report = StringBuilder()
        var worst = 0.0
        for (truth in doubleArrayOf(0.0, 0.7, -1.4, 3.0, -5.5, 8.0)) {
            val plane = syntheticLattice(LATTICE_ROWS, LATTICE_COLS, truth)
            val result = SkewEstimator.search(plane)
            val error = abs(result.angleDeg - truth)
            worst = maxOf(worst, error)
            report.append(
                // Locale.ROOT: the default locale put comma decimal
                // separators in the failure message on this machine, which
                // is unreadable next to the values it is being compared to.
                "  injected %+.2f  recovered %+.5f  error %.5f  margin %.1f%n"
                    .format(Locale.ROOT, truth, result.angleDeg, error, result.margin)
            )
            // A margin near 1 means the objective is flat and the "optimum"
            // is wherever noise put it. Asserted separately from the angle
            // because a flat objective that happens to peak near the truth
            // would otherwise pass silently.
            assertTrue(
                "margin ${result.margin} at $truth deg is not a peak; the " +
                    "objective found no lattice\n$report",
                result.margin > 2.0
            )
        }
        // Every angle is reported, not just the first failure, because when
        // this breaks the pattern across angles is the diagnosis: a constant
        // offset is a convention error, one that grows with |angle| is a
        // geometry error, and noise is a fixture problem.
        assertTrue(
            "worst angle error %.5f exceeds %.3f deg%n%s"
                .format(Locale.ROOT, worst, ANGLE_TOLERANCE_DEG, report),
            worst < ANGLE_TOLERANCE_DEG
        )
    }

    /**
     * The lattice is built at the size the shipped configuration actually
     * works at, and that is not a detail.
     *
     * Angular resolution depends on the lever arm: the further a rule runs
     * across the frame, the more a small rotation displaces its ends and the
     * finer an angle the projection can separate. The same lattice at
     * 240x300 recovers +0.7 degrees as +0.775 — an error of 0.075, three
     * times worse — purely because the frame is a third of the width. That
     * is a property of the evidence available, not of the code, and sizing
     * the fixture to the real working plane is what makes the tolerance
     * below comparable with the Python's published figure rather than an
     * arbitrary number that happens to pass.
     */
    private val LATTICE_ROWS = 700
    private val LATTICE_COLS = 900

    /**
     * Measured on this fixture, 2026-09-18:
     *
     *     injected   recovered     error    margin
     *       +0.00     +0.00000    0.00000     379.0
     *       +0.70     +0.65945    0.04055      22.5
     *       -1.40     -1.40353    0.00353      44.7
     *       +3.00     +3.00227    0.00227     159.0
     *       -5.50     -5.49843    0.00157     157.7
     *       +8.00     +8.00179    0.00179     158.2
     *
     * **Small angles are the hard case, and that is the algorithm rather
     * than the port.** The shear bins by a shift rounded to whole pixels, so
     * at 0.7 degrees the shift spans only about eleven pixels across a
     * 900-column plane and a range of nearby true angles produces the same
     * rounded pattern. The objective flattens accordingly — margin 22.5 here
     * against 159 at 3 degrees — and the optimum is correspondingly less
     * well determined. The Python shows the same shape: its largest errors
     * over the fixture sweep also fall at 0.1 and 0.25 degrees, and its
     * published worst case at this working resolution is 0.0408, which this
     * fixture reproduces almost exactly at 0.0406.
     *
     * The tolerance is 0.08 rather than something snug against 0.0406.
     * Snug would be fragile: `sin`, `cos` and `tan` are not required to be
     * correctly rounded and may differ in the last places between JVMs, and
     * a test that fails on someone else's machine for that reason teaches
     * nothing. 0.08 still sits below the ~0.1 degree residual the
     * reconstruction can absorb — one degree of uncorrected rotation takes
     * PCC from 0.981 to 0.734, while 0.04 costs nothing measurable — and it
     * is far below what any real defect produces: a sign error returns the
     * angle negated, a transpose error loses one axis entirely, and an
     * indexing error flattens the objective into noise. Those fail by
     * degrees, not by hundredths.
     */
    private val ANGLE_TOLERANCE_DEG = 0.08

    /** A featureless plane has no angle to find, and must say so rather than invent one. */
    @Test
    fun rejectsAnEmptyResponse() {
        val blank = SkewEstimator.Plane(DoubleArray(64 * 96), 64, 96)
        try {
            SkewEstimator.search(blank)
            throw AssertionError("expected an empty response to be rejected")
        } catch (expected: IllegalArgumentException) {
            assertTrue(
                "message should name the cause: ${expected.message}",
                expected.message!!.contains("identically zero")
            )
        }
    }

    /**
     * An ECG grid as a response plane: 1 mm minor rules and heavier 5 mm
     * majors, rotated by [angleDeg].
     *
     * Drawn by testing each pixel's distance to the nearest rule along each
     * rotated axis, which is the same lattice the real response sees and not
     * a separable pattern that a broken transpose could still score well on.
     * The minors are drawn at a seventh of the majors' weight, matching the
     * contrast ratio measured on the ecg-image-kit fixtures
     * (`results/grid/contrast.csv`), so the objective is exercised on
     * realistic evidence rather than on rules of equal strength.
     */
    private fun syntheticLattice(rows: Int, cols: Int, angleDeg: Double): SkewEstimator.Plane {
        val data = DoubleArray(rows * cols)

        // THE SIGN. The estimator bins by `y - x*tan(internal)` with
        // `internal = -angleDeg`, i.e. by `y + x*tan(angleDeg)`. A line lands
        // wholly in one bin when that quantity is constant along it, so the
        // horizontal rules of an image the estimator should report as +A have
        // slope dy/dx = -tan(A). Rotating the lattice by -angleDeg here
        // produces exactly that. Getting this backwards is not subtle in its
        // effect — the recovered angle comes back negated — but it is very
        // easy to do, which is half of why this test exists.
        val a = Math.toRadians(-angleDeg)
        val cos = Math.cos(a)
        val sin = Math.sin(a)
        val pitch = 8.0          // px per mm, the scale the Python sweeps at
        val cx = cols / 2.0
        val cy = rows / 2.0

        // ANTI-ALIASING, which is load-bearing rather than cosmetic. A rule
        // drawn as a hard one-pixel line carries no sub-pixel information:
        // its rendered position quantises to whole pixels, so the pattern's
        // true angle is only accurate to the staircase it forms, and the
        // estimator cannot do better than that no matter how correct it is.
        // A linear falloff puts the rule's centroid where it belongs and the
        // recovered angle then reflects the estimator rather than the
        // fixture. Real rendered grids are anti-aliased for the same reason.
        val halfWidth = 0.9

        fun ink(distance: Double, index: Long): Double {
            if (distance >= halfWidth) return 0.0
            // Every fifth rule is a major. The minors are drawn a seventh as
            // strong, which is the contrast ratio measured on the
            // ecg-image-kit fixtures (results/grid/contrast.csv), so the
            // objective is exercised on realistic evidence rather than on
            // rules of equal weight. floorMod, not %, because the indices
            // run negative either side of centre.
            val weight = if (Math.floorMod(index, 5L) == 0L) 255.0 else 255.0 / 7.0
            return weight * (1.0 - distance / halfWidth)
        }

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val x = c - cx
                val y = r - cy
                // Into the lattice's own frame.
                val u = x * cos + y * sin
                val v = -x * sin + y * cos
                val iu = u / pitch
                val iv = v / pitch
                val ku = Math.round(iu)
                val kv = Math.round(iv)
                data[r * cols + c] =
                    ink(Math.abs(iu - ku) * pitch, ku) +
                    ink(Math.abs(iv - kv) * pitch, kv)
            }
        }
        return SkewEstimator.Plane(data, rows, cols)
    }
}
