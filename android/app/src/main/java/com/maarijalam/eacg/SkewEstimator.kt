package com.maarijalam.eacg

import org.opencv.core.Mat
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.tan

/**
 * In-plane skew recovery. A port of `ecgproto/deskew.py`, not a redesign.
 *
 * WHY THE STAGE EXISTS. Measured in the Python, a 3 degree rotation takes
 * end-to-end SNR from +7 dB to -7 dB and 1 degree takes PCC from 0.981 to
 * 0.734 — while `estimate_grid_pitch` goes on reporting an apparently
 * perfect pitch. Rotation is not a tolerable input condition, and the metric
 * that looks healthiest under it is the one that tells you least.
 *
 * METHOD: projection profile sharpness maximisation. The printed grid is a
 * periodic lattice of straight lines covering the whole image. Axis-aligned,
 * every pixel of a gridline falls in one image row and the row projection is
 * a comb of sharp peaks; rotate it and each line smears across several rows,
 * adjacent lines overlap, and the profile flattens toward its mean.
 * Sharpness of the projection is therefore a function of angle with a
 * maximum at the true skew, and deskew reduces to a 1-D optimisation.
 *
 * The grid is used rather than the page boundary deliberately: a
 * photographed printout may be cropped or have its edges outside the frame,
 * and a single boundary line is a weak, easily-occluded estimator, whereas
 * the grid integrates evidence from hundreds of lines across the image.
 *
 * **THE SEARCH NEVER WARPS THE IMAGE**, and this is why it is hand-written
 * Kotlin rather than OpenCV. Rotating once per candidate angle would be one
 * full warp per angle, and every warp interpolates — itself a low-pass
 * filter on the very fine grid detail the objective measures. Instead the
 * projection is taken *along* a candidate direction: binning pixels by
 * `y - x*tan(a)` collects any line of slope `a` into one bin. Rounding that
 * bin index makes the shift constant over runs of adjacent columns, so the
 * whole projection is a grouped column sum plus a few hundred shifted vector
 * adds — one pass over the image per angle, no interpolation at all. There
 * is no OpenCV primitive for that shape of work.
 *
 * DIVISION OF LABOUR, which follows from the above rather than from taste:
 *
 *  - OpenCV does [ImageWarp.reduce] and [ImageWarp.rotate]. Both are
 *    full-resolution and per-pixel, which is where a SIMD-vectorised native
 *    routine wins and a Kotlin loop loses badly.
 *  - Kotlin does the response and the angle search. Both run on an image
 *    already reduced to 900 px, so the pixel counts are small, and neither
 *    has an OpenCV equivalent that would not change the arithmetic.
 *
 * WHY 900 PX FOR BOTH PASSES. The fine pass used to run at full resolution
 * and cost 1691-1941 ms at 8.41 MP on the bench, seven to fourteen times the
 * grid estimation it feeds, and superlinear in pixels. It bought angle
 * recovery to 0.004 degrees where roughly 0.1 is all the waveform can
 * detect. At 900 px the worst case over three fixtures and seventeen angles
 * is 0.041 degrees — still a factor of 2.4 inside the requirement — and the
 * stage costs 74-89 ms *regardless of input size*, because everything after
 * the first resize runs at a fixed size. See `results/deskew/finescale.csv`
 * and CHANGES.txt section 11.1.
 *
 * Every constant below is read from the Python rather than chosen here.
 */
object SkewEstimator {

    /**
     * Coarse sweep half-range, degrees. Handheld capture against a flat
     * surface is rarely worse than this, and ML Kit Document Scanner is the
     * intended capture-time fix for gross rotation; this stage cleans up the
     * residual.
     */
    const val COARSE_HALF_RANGE_DEG = 10.0

    /**
     * Half a degree is fine enough that the true optimum is always within
     * one step of a sampled point, the objective's main lobe being several
     * degrees wide at these scales.
     */
    const val COARSE_STEP_DEG = 0.5

    /**
     * Fine sweep half-range about the coarse optimum: **one** coarse step.
     * Measured over 67 cases — three fixtures, seventeen angles, and the
     * whole degradation battery — the coarse pass never lands more than half
     * a step from the truth, so 0.25 would suffice and 0.5 leaves a full
     * step of slack. It is not cut to 0.25 because a coarse pass that ever
     * did miss by a step would put the true peak outside the fine window and
     * fail silently and large.
     */
    const val FINE_HALF_RANGE_DEG = 0.5

    /** Below the achievable accuracy already; the parabola refines under it. */
    const val FINE_STEP_DEG = 0.05

    /** Longest side of the working image for the coarse pass. */
    const val COARSE_MAX_DIM = 900

    /** Longest side of the working image for the fine pass. See the class docstring. */
    const val FINE_MAX_DIM = 900

    /**
     * A recovered angle and enough diagnostics to distrust it.
     *
     * [angleDeg] is the rotation to be *removed*: a printout photographed 2
     * degrees anticlockwise returns +2.0, and rotating the image by
     * -[angleDeg] straightens it. The estimate never touches the image, so a
     * caller that also has perspective to correct can compose both
     * transforms and interpolate exactly once.
     */
    data class Result(
        val angleDeg: Double,
        /** Sharpness at the optimum. */
        val objective: Double,
        /**
         * Optimum over the objective one degree away. A real lattice gives a
         * sharp maximum and this runs to the tens or hundreds; a featureless
         * or non-red frame gives a nearly flat curve and this sits near 1.
         */
        val margin: Double,
        /** Optimum from the horizontal rules alone. */
        val angleHDeg: Double,
        /** Optimum from the vertical rules alone. */
        val angleVDeg: Double,
        /**
         * Free diagnostic: a pure rotation moves the two axes together,
         * while perspective or a shear moves them apart.
         */
        val axisDisagreementDeg: Double,
        val coarseAngleDeg: Double,
        val nEvaluations: Int,
        val workRows: Int,
        val workCols: Int,
        /** Reduction applied to the input before anything else. */
        val imageScale: Double,
        // Timing, split because the parts answer different questions.
        val resizeMs: Long,
        val responseMs: Long,
        val coarseMs: Long,
        val fineMs: Long,
        val marginMs: Long,
        val totalMs: Long
    )

    /**
     * Recover the skew **without modifying the image**.
     *
     * [src] is an RGBA Mat. It is not released, and it is not written to.
     */
    fun estimate(src: Mat): Result {
        val t0 = System.nanoTime()

        // ---- one reduction, then one response ------------------------------
        // Python takes the response once, on an image already reduced to the
        // largest size any pass will use. Taking it at full resolution and
        // reducing afterwards is the same arithmetic in the worse order: on
        // the 8.41 MP fixture the response alone costs 181 ms against 23 ms
        // for resize-then-respond, which at a 900 px working size is more
        // than both sweeps together. The two orders do not commute, since
        // the response clips at zero, but they agree on the recovered angle
        // to better than 1e-4 degrees.
        val workDim = max(COARSE_MAX_DIM, FINE_MAX_DIM)
        val (work, imageScale) = ImageWarp.reduce(src, workDim)
        val tResize = System.nanoTime()

        val planeH: Plane
        try {
            planeH = rednessPlane(work)
        } finally {
            if (work !== src) work.release()
        }
        val tResponse = System.nanoTime()

        var anyInk = false
        for (v in planeH.data) if (v != 0.0) { anyInk = true; break }
        if (!anyInk) {
            throw IllegalArgumentException(
                "grid response is identically zero: nothing to align. The " +
                    "response keys on red saturation, which a black-and-white " +
                    "or blue-grey grid does not produce."
            )
        }

        // The transpose is materialised once and reused across every angle.
        // The Python re-slices `response.T` per call, which is free there
        // because NumPy transposes are views; here a copy costs one pass over
        // a 900 px image and saves a strided access on every one of the ~65
        // angle evaluations that follow.
        val planeV = planeH.transposed()
        val scratchH = Scratch(planeH)
        val scratchV = Scratch(planeV)

        // Both passes run at 900 px, so the Python's per-pass `_reduce` of
        // the response is the identity in the shipped configuration. Rather
        // than implement a resampler for the response that cannot be
        // exercised, the assumption is asserted: change either constant below
        // the working size and this fails loudly instead of quietly skipping
        // a reduction the Python would have performed.
        require(COARSE_MAX_DIM >= workDim && FINE_MAX_DIM >= workDim) {
            "COARSE_MAX_DIM and FINE_MAX_DIM must both equal the working size " +
                "($workDim); a smaller one would require reducing the response " +
                "itself, which this port does not implement."
        }

        // ---- coarse pass ---------------------------------------------------
        val coarseAngles = arange(
            -COARSE_HALF_RANGE_DEG,
            COARSE_HALF_RANGE_DEG + 0.5 * COARSE_STEP_DEG,
            COARSE_STEP_DEG
        )
        var (lenH, lenV) = coreLengths(planeH.rows, planeH.cols, COARSE_HALF_RANGE_DEG)
        val coarseH = DoubleArray(coarseAngles.size)
        val coarseV = DoubleArray(coarseAngles.size)
        sweep(coarseAngles, planeH, planeV, scratchH, scratchV, lenH, lenV, coarseH, coarseV)
        var best = 0
        for (i in coarseAngles.indices) {
            if (coarseH[i] + coarseV[i] > coarseH[best] + coarseV[best]) best = i
        }
        val coarseAngle = coarseAngles[best]
        val tCoarse = System.nanoTime()

        // ---- fine pass -----------------------------------------------------
        val fineAngles = arange(
            coarseAngle - FINE_HALF_RANGE_DEG,
            coarseAngle + FINE_HALF_RANGE_DEG + 0.5 * FINE_STEP_DEG,
            FINE_STEP_DEG
        )
        coreLengths(planeH.rows, planeH.cols, abs(coarseAngle) + FINE_HALF_RANGE_DEG)
            .let { lenH = it.first; lenV = it.second }
        val fineH = DoubleArray(fineAngles.size)
        val fineV = DoubleArray(fineAngles.size)
        sweep(fineAngles, planeH, planeV, scratchH, scratchV, lenH, lenV, fineH, fineV)

        val fineTotal = DoubleArray(fineAngles.size) { fineH[it] + fineV[it] }
        val angle = parabolicPeak(fineAngles, fineTotal)
        val angleH = parabolicPeak(fineAngles, fineH)
        val angleV = parabolicPeak(fineAngles, fineV)
        var peak = fineTotal[0]
        for (v in fineTotal) if (v > peak) peak = v
        val tFine = System.nanoTime()

        // ---- margin --------------------------------------------------------
        // Both probes and the optimum use core lengths sized to admit the
        // full +/-1 degree, because a ratio taken between two different
        // croppings would be measuring the cropping.
        coreLengths(planeH.rows, planeH.cols, abs(angle) + 1.0)
            .let { lenH = it.first; lenV = it.second }
        val atPeak = objectiveTotal(angle, planeH, planeV, scratchH, scratchV, lenH, lenV)
        val off = max(
            objectiveTotal(angle + 1.0, planeH, planeV, scratchH, scratchV, lenH, lenV),
            objectiveTotal(angle - 1.0, planeH, planeV, scratchH, scratchV, lenH, lenV)
        )
        val tMargin = System.nanoTime()

        return Result(
            angleDeg = angle,
            objective = peak,
            margin = if (off > 0.0) atPeak / off else Double.POSITIVE_INFINITY,
            angleHDeg = angleH,
            angleVDeg = angleV,
            axisDisagreementDeg = abs(angleH - angleV),
            coarseAngleDeg = coarseAngle,
            nEvaluations = coarseAngles.size + fineAngles.size + 3,
            workRows = planeH.rows,
            workCols = planeH.cols,
            imageScale = imageScale,
            resizeMs = (tResize - t0) / 1_000_000,
            responseMs = (tResponse - tResize) / 1_000_000,
            coarseMs = (tCoarse - tResponse) / 1_000_000,
            fineMs = (tFine - tCoarse) / 1_000_000,
            marginMs = (tMargin - tFine) / 1_000_000,
            totalMs = (tMargin - t0) / 1_000_000
        )
    }

    // ----------------------------------------------------------------------
    // internals
    // ----------------------------------------------------------------------

    /** A 2-D double array in row-major order. */
    private class Plane(val data: DoubleArray, val rows: Int, val cols: Int) {
        fun transposed(): Plane {
            val out = DoubleArray(data.size)
            for (r in 0 until rows) {
                val base = r * cols
                for (c in 0 until cols) out[c * rows + r] = data[base + c]
            }
            return Plane(out, cols, rows)
        }
    }

    /**
     * Per-plane working buffers, allocated once per estimate and reused
     * across all ~65 angles. On a phone this matters: allocating the
     * accumulator per angle would be some 130 short-lived arrays per
     * estimate, and the GC pauses land in the middle of a latency
     * measurement.
     */
    private class Scratch(plane: Plane) {
        val shifts = IntArray(plane.cols)
        val starts = IntArray(plane.cols)
        // A shift span cannot exceed the column count for any angle under 45
        // degrees, so rows + cols is a safe upper bound on the accumulator.
        val acc = DoubleArray(plane.rows + plane.cols + 2)
    }

    /** RGBA Mat to a [Plane] of [GridResponse] values. */
    private fun rednessPlane(mat: Mat): Plane {
        val rows = mat.rows()
        val cols = mat.cols()
        val channels = mat.channels()
        require(channels == 4 || channels == 3) {
            "expected a 3- or 4-channel 8-bit image, got $channels channels"
        }
        val bytes = ByteArray(rows * cols * channels)
        mat.get(0, 0, bytes)
        val out = DoubleArray(rows * cols)
        var i = 0
        for (p in 0 until rows * cols) {
            out[p] = GridResponse.fromRgbaBytes(bytes, i)
            i += channels
        }
        return Plane(out, rows, cols)
    }

    /**
     * NumPy's `arange`: `n = ceil((stop - start) / step)` values, the i-th
     * being `start + i*step`. Reproduced rather than approximated because
     * the sweep lengths — 41 coarse and 21 fine — come out of exactly this
     * formula, and an off-by-one at the end of the sweep changes which
     * sample the parabola is fitted around.
     */
    private fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        val n = ceil((stop - start) / step).toInt()
        return DoubleArray(if (n > 0) n else 0) { start + it * step }
    }

    /**
     * Common profile length for every angle in a sweep of this half-range.
     *
     * Only the *fully covered* bins of a projection are comparable: a bin
     * near either end receives contributions from some columns and not
     * others, so its height reflects how many columns reached it rather than
     * how much ink they carried. Cropping every angle's profile to one
     * centred length makes the objective a statistic of the image instead of
     * one of the sweep geometry.
     */
    private fun coreLengths(rows: Int, cols: Int, halfRangeDeg: Double): Pair<Int, Int> {
        val t = abs(tan(Math.toRadians(halfRangeDeg)))
        val lenH = rows - ceil(cols * t).toInt() - 1
        val lenV = cols - ceil(rows * t).toInt() - 1
        return max(lenH, 8) to max(lenV, 8)
    }

    private fun sweep(
        angles: DoubleArray,
        planeH: Plane, planeV: Plane,
        scratchH: Scratch, scratchV: Scratch,
        lenH: Int, lenV: Int,
        outH: DoubleArray, outV: DoubleArray
    ) {
        for (i in angles.indices) {
            // The binning slope is the negation of the external angle:
            // OpenCV's rotation carries a horizontal line to slope
            // -tan(angle) in image coordinates, rows growing downward.
            // Sweeping in the external convention keeps every angle in this
            // file, and in Result, meaning the same thing.
            val t = tan(Math.toRadians(-angles[i]))
            outH[i] = shearedSharpness(planeH, t, lenH, scratchH)
            outV[i] = shearedSharpness(planeV, -t, lenV, scratchV)
        }
    }

    private fun objectiveTotal(
        angleDeg: Double,
        planeH: Plane, planeV: Plane,
        scratchH: Scratch, scratchV: Scratch,
        lenH: Int, lenV: Int
    ): Double {
        val t = tan(Math.toRadians(-angleDeg))
        return shearedSharpness(planeH, t, lenH, scratchH) +
            shearedSharpness(planeV, -t, lenV, scratchV)
    }

    /**
     * Project [plane] along lines of slope [tanA], crop to [fixedLen], and
     * return the sharpness of the result. Zero if the geometry does not
     * admit a usable profile, which is the Python's `None`.
     *
     * Projection and sharpness are fused only to avoid copying the profile
     * out of the accumulator on every one of the ~130 evaluations per
     * estimate. The arithmetic is the Python's `_sheared_projection`
     * followed by `_sharpness(..., "ssd")`, unchanged.
     */
    private fun shearedSharpness(
        plane: Plane,
        tanA: Double,
        fixedLen: Int,
        s: Scratch
    ): Double {
        val rows = plane.rows
        val cols = plane.cols
        val data = plane.data

        var lo = Int.MAX_VALUE
        var hi = Int.MIN_VALUE
        for (c in 0 until cols) {
            // Math.rint is round-half-to-even, which is what np.rint does.
            // Math.round is round-half-up and would disagree at exact .5
            // shifts, which do occur.
            val v = Math.rint(-c.toDouble() * tanA).toInt()
            s.shifts[c] = v
            if (v < lo) lo = v
            if (v > hi) hi = v
        }
        val span = hi - lo
        if (span >= rows - 8) return 0.0

        // shifts is monotonic in the column index, so equal values are
        // contiguous and the run boundaries are just where it changes.
        var nGroups = 0
        s.starts[nGroups++] = 0
        for (c in 1 until cols) if (s.shifts[c] != s.shifts[c - 1]) s.starts[nGroups++] = c

        val accLen = rows + span
        java.util.Arrays.fill(s.acc, 0, accLen, 0.0)
        for (r in 0 until rows) {
            val base = r * cols
            for (j in 0 until nGroups) {
                val c0 = s.starts[j]
                val c1 = if (j + 1 < nGroups) s.starts[j + 1] else cols
                var sum = 0.0
                for (c in c0 until c1) sum += data[base + c]
                s.acc[s.shifts[c0] - lo + r] += sum
            }
        }

        // core = acc[span until rows]; then a centred crop of fixedLen.
        val coreLen = rows - span
        if (coreLen < fixedLen) return 0.0
        val from = span + (coreLen - fixedLen) / 2

        // ssd: mean squared successive difference, normalised by the squared
        // profile mean. Normalising makes it invariant to overall ink
        // quantity, so the objective is not partly measuring how much of the
        // image the sweep geometry admitted. Unlike a variance it is
        // destroyed by shuffling, so it responds to the spatial arrangement
        // and not merely the distribution — which is the property that tells
        // a comb of resolved gridlines from a smeared one.
        var mean = 0.0
        for (i in from until from + fixedLen) mean += s.acc[i]
        mean /= fixedLen
        if (!mean.isFinite() || mean <= 0.0) return 0.0
        var ssd = 0.0
        for (i in from + 1 until from + fixedLen) {
            val d = s.acc[i] - s.acc[i - 1]
            ssd += d * d
        }
        return (ssd / (fixedLen - 1)) / (mean * mean)
    }

    /** Sub-step optimum, by fitting a parabola to the best sample and its neighbours. */
    private fun parabolicPeak(angles: DoubleArray, values: DoubleArray): Double {
        var k = 0
        for (i in values.indices) if (values[i] > values[k]) k = i
        if (k == 0 || k == values.size - 1) return angles[k]
        val y0 = values[k - 1]
        val y1 = values[k]
        val y2 = values[k + 1]
        val denom = y0 - 2.0 * y1 + y2
        if (denom == 0.0) return angles[k]
        val step = angles[1] - angles[0]
        return angles[k] + 0.5 * step * (y0 - y2) / denom
    }
}
