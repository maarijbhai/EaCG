package com.maarijalam.eacg

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat

import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * FR5, FR6 and FR7: trace segmentation, vectorisation, physical calibration.
 *
 * The port of `digitise.binarise_trace`, `trace_positions`,
 * `baseline_from_trace`, `_interpolate_gaps`, `pixels_to_mv` and the
 * resampling tail of `digitise()`. Together they turn a deskewed, calibrated
 * image into millivolts against seconds, which is the first point in the
 * pipeline where the app produces something a clinician would recognise
 * rather than a geometric measurement.
 *
 * SCOPE, and it is the same scope the Python carries
 * **This handles a single continuous strip only.** Handed a 3x4 sheet it
 * traces straight across the cell rows: [centroids] takes the ink centroid
 * of a whole column, so a column crossing four stacked leads returns their
 * average, which is a waveform belonging to no lead at all. It cannot detect
 * this and does not try. [QualityGate.tracesPerColumn] is what detects it,
 * and on this device that matters immediately — three of the six bundled
 * fixtures are sheets.
 *
 * HOW IT IS SPLIT, and why
 * Everything that can carry a porting bug is a pure function over arrays
 * with no OpenCV in it, so it runs under `testDebugUnitTest` against golden
 * vectors from the Python in half a second. Only [binarise] touches OpenCV,
 * for the same reason [ImageWarp] does: the threshold and the morphological
 * opening are full-resolution per-pixel work where a SIMD native routine
 * wins, and neither has any arithmetic a port could get subtly wrong — they
 * are the same library on both sides. That half remains verifiable only on
 * the device.
 */
object TraceExtractor {

    /**
     * A boolean mask as bytes, shape (rows, cols), row-major.
     *
     * Bytes rather than `BooleanArray` because this is what an OpenCV
     * `CV_8UC1` Mat hands over and converting would double the allocation
     * on an 8 MP frame for nothing. Any non-zero byte is ink.
     */
    class Mask(val data: ByteArray, val rows: Int, val cols: Int) {
        fun inked(row: Int, col: Int): Boolean = data[row * cols + col] != 0.toByte()
    }

    /** The vectorised trace and how much of it was actually found. */
    class Trace(
        /** Row position per column, NaN where no ink was found. Trimmed. */
        val yPx: DoubleArray,
        /** Fraction of the trimmed span that carried ink, before bridging. */
        val coverage: Double,
        /** Index of the first inked column in the untrimmed image. */
        val trimLo: Int,
        /** Index of the last inked column in the untrimmed image. */
        val trimHi: Int
    )

    /** A digitised waveform in millivolts, with what it took to get there. */
    class Digitised(
        val signalMv: DoubleArray,
        val fs: Double,
        val baselinePx: Double,
        val coverage: Double,
        val pxPerMv: Double,
        val pxPerS: Double
    )

    /**
     * Default bridgeable gap, in pixels. `digitise`'s `max_gap_px`.
     *
     * Not scaled by pitch, matching the Python. Worth knowing that this is
     * arguably wrong in both implementations: 12 px is 1.5 mm at 8 px/mm and
     * 1.0 mm at 11.8, so the physical width of a bridgeable dropout changes
     * with resolution. Left as it is because changing it would change the
     * control condition's numbers, and the control condition is not to be
     * quietly improved. Flagged rather than fixed.
     */
    const val DEFAULT_MAX_GAP_PX = 12

    // ----------------------------------------------------------------------
    // FR5. The OpenCV half: device-only, not covered by the port tests.
    // ----------------------------------------------------------------------

    /**
     * Boolean ink mask from a BGR/RGBA image. `digitise.binarise_trace`.
     *
     * **Channel order is the trap here, exactly as in [GridResponse].** The
     * Python indexes `image[:, :, 2]` for red, because desktop OpenCV reads
     * image *files* as BGR. A Mat converted from an Android Bitmap is
     * **RGBA**, so red is channel 0. Reading channel 2 on this side would
     * take the blue channel, and on a red-gridded printout that inverts
     * which ink survives: the grid would come through as trace. It would not
     * look like a channel bug, it would look like a tracer that had gone
     * mad.
     *
     * The logic itself: both white paper and the red grid have a saturated
     * red channel while the neutral dark trace does not, so `255 - R` is
     * large only on trace ink. Otsu picks the threshold. A single 3x3
     * opening removes isolated speckle without eating a trace one or two
     * pixels wide.
     *
     * Caller owns the returned Mat.
     */
    fun binarise(rgba: Mat, threshold: Int? = null): Mat {
        // `extractChannel` rather than `split`, and the difference is real on
        // this hardware: split allocates all four single-channel planes, which
        // on the 8.4 MP sheet fixture is four 8.4 MB natives where one is
        // wanted. Peak native memory is the half of the deployment cost that
        // is not yet instrumented, so there is no reason to spend 25 MB of it
        // on three planes that are discarded unread.
        val red = Mat()
        try {
            // Channel 0 of RGBA. See the note above; this is not a typo for 2.
            Core.extractChannel(rgba, red, 0)
            val inverted = Mat()
            try {
                // The Python's `(255 - red).astype(uint8)`. On CV_8U the
                // complement IS 255 - x exactly, and OpenCV offers no
                // scalar-minus-Mat overload, so this is the same arithmetic
                // rather than an approximation of it.
                Core.bitwise_not(red, inverted)
                val mask = Mat()
                if (threshold == null) {
                    Imgproc.threshold(
                        inverted, mask, 0.0, 255.0,
                        Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
                    )
                } else {
                    Imgproc.threshold(
                        inverted, mask, threshold.toDouble(), 255.0,
                        Imgproc.THRESH_BINARY
                    )
                }
                val kernel = Mat.ones(Size(3.0, 3.0), CvType.CV_8U)
                try {
                    Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
                } finally {
                    kernel.release()
                }
                return mask
            } finally {
                inverted.release()
            }
        } finally {
            red.release()
        }
    }

    /**
     * Copy a `CV_8UC1` Mat into a [Mask].
     *
     * This is where native memory becomes Java heap: 8.4 MB for the sheet
     * fixture, which is the price of doing the centroid walk and the band
     * count in Kotlin rather than in OpenCV primitives. Both are single
     * passes with no OpenCV equivalent — the centroid is a weighted mean per
     * column and the band count needs run boundaries — so the alternative is
     * not a cheaper native call, it is several of them.
     *
     * The continuity check is not decoration. `Mat.get(0, 0, ByteArray)`
     * requires a continuous Mat and throws an OpenCV error that names
     * neither this function nor the reason if handed a view or an ROI. Every
     * Mat reaching here today is freshly allocated and therefore continuous;
     * this states the requirement so that the day one is not, the failure
     * says so.
     */
    fun maskOf(mat: Mat): Mask {
        require(mat.isContinuous) {
            "mask Mat must be continuous to copy in one call; got a view or ROI"
        }
        require(mat.channels() == 1) {
            "mask Mat must be single-channel, got ${mat.channels()}"
        }
        val rows = mat.rows()
        val cols = mat.cols()
        val data = ByteArray(rows * cols)
        mat.get(0, 0, data)
        return Mask(data, rows, cols)
    }

    // ----------------------------------------------------------------------
    // FR6. Pure Kotlin from here down. Covered by the port tests.
    // ----------------------------------------------------------------------

    /**
     * Row position of the trace per column. `digitise.trace_positions`.
     *
     * The ink centroid of the column, NaN where the column held none —
     * **not** zero, because zero is a real row at the top of the image and a
     * caller cannot then tell "no data" from "0 mV".
     *
     * The centroid systematically clips peaks, and this is the classical
     * path's dominant amplitude error rather than a bug to fix here: on a
     * steep QRS upstroke one column contains a tall vertical run of ink and
     * its centroid sits at the run's midpoint, so the recovered R amplitude
     * is biased low by roughly half the within-column excursion. Measured on
     * the desktop at gain ratio 0.60 at 4 px/mm and 0.96 at 16. It is the
     * obvious thing a learned tracer would beat, and it must stay here
     * unchanged so that the comparison is against the real control.
     */
    fun centroids(mask: Mask): DoubleArray {
        val out = DoubleArray(mask.cols)
        for (c in 0 until mask.cols) {
            var sum = 0.0
            var weight = 0.0
            for (r in 0 until mask.rows) {
                if (mask.data[r * mask.cols + c] != 0.toByte()) {
                    sum += r.toDouble()
                    weight += 1.0
                }
            }
            out[c] = if (weight > 0.0) sum / weight else Double.NaN
        }
        return out
    }

    /**
     * Trim to the inked span and measure coverage, as `digitise` does.
     *
     * Coverage is computed **after** the trim, so leading and trailing blank
     * paper is not charged against the tracer: a strip photographed with
     * margins would otherwise report low coverage for having margins. What
     * remains uncovered inside the span is real dropout.
     *
     * Throws if fewer than two columns carry ink, matching the Python: a
     * trace of one column is not a waveform, and every later stage would
     * divide by a zero-length span.
     */
    fun trim(yPx: DoubleArray): Trace {
        var lo = -1
        var hi = -1
        for (i in yPx.indices) {
            if (!yPx[i].isNaN()) {
                if (lo < 0) lo = i
                hi = i
            }
        }
        if (lo < 0 || hi - lo + 1 < 2) {
            throw IllegalArgumentException(
                "fewer than two inked columns: no trace found"
            )
        }
        val span = DoubleArray(hi - lo + 1) { yPx[lo + it] }
        var finite = 0
        for (v in span) if (!v.isNaN()) finite++
        return Trace(span, finite.toDouble() / span.size, lo, hi)
    }

    /**
     * Bridge short NaN runs, leave long ones. `digitise._interpolate_gaps`.
     *
     * Short gaps are dropout in thin ink and are safe to bridge. A long one
     * usually means the trace left the strip, or that binarisation failed
     * over a region, and bridging those would invent plausible-looking
     * signal that was never on the paper — which is worse than a hole,
     * because a hole is visible and an invention is not.
     *
     * Matches NumPy's `interp` at the ends, which **clamps** rather than
     * extrapolating: a leading NaN run takes the first valid value. Those
     * runs are then re-blanked if they are too long, so the clamping only
     * ever shows through on a short leading gap. (After [trim] there are no
     * leading or trailing gaps at all; this is for callers that skip it.)
     */
    fun interpolateGaps(yPx: DoubleArray, maxGapPx: Int = DEFAULT_MAX_GAP_PX): DoubleArray {
        val n = yPx.size
        val out = yPx.copyOf()
        var anyNan = false
        for (v in yPx) if (v.isNaN()) { anyNan = true; break }
        if (!anyNan) return out

        // Linear interpolation over the valid samples, clamped at both ends.
        var prev = -1
        for (i in 0 until n) {
            if (yPx[i].isNaN()) continue
            if (prev >= 0 && i - prev > 1) {
                val span = (i - prev).toDouble()
                for (k in prev + 1 until i) {
                    val t = (k - prev).toDouble() / span
                    out[k] = yPx[prev] + t * (yPx[i] - yPx[prev])
                }
            } else if (prev < 0 && i > 0) {
                for (k in 0 until i) out[k] = yPx[i]        // clamp, as np.interp
            }
            prev = i
        }
        if (prev < 0) return out                            // all NaN
        for (k in prev + 1 until n) out[k] = yPx[prev]      // clamp the tail

        // Re-blank any run longer than the bridgeable width.
        var i = 0
        while (i < n) {
            if (!yPx[i].isNaN()) { i++; continue }
            var end = i
            while (end < n && yPx[end].isNaN()) end++
            if (end - i > maxGapPx) {
                for (k in i until end) out[k] = Double.NaN
            }
            i = end
        }
        return out
    }

    /**
     * The isoelectric row, as the median inked position.
     * `digitise.baseline_from_trace`.
     *
     * Median rather than mean because the QRS is a large, brief, one-sided
     * excursion: it drags a mean well off the isoelectric line and barely
     * moves a median, occupying only a few percent of the columns.
     *
     * It is not unconditionally safe, and the failure is quiet. The median
     * assumes the trace spends over half its columns near baseline, which
     * fails for sustained large deflections — deep S waves in V1-V2, giant T
     * waves, ventricular tachycardia, a heavily elevated ST segment at high
     * rate. The estimate then drifts toward the deflection and the whole
     * waveform acquires a DC offset, which PCC cannot see at all (it is
     * offset-invariant) and only RMSE reports. That is one of the reasons the
     * metrics are read together.
     *
     * NumPy's median averages the two middle values at even length; so does
     * this. Taking the lower one instead would pass most tests and disagree
     * with the Python by half a pixel on half the inputs.
     */
    fun baselineFrom(yPx: DoubleArray): Double {
        val finite = yPx.filter { !it.isNaN() }.sorted()
        if (finite.isEmpty()) {
            throw IllegalArgumentException(
                "no inked columns: nothing to estimate a baseline from"
            )
        }
        val mid = finite.size / 2
        return if (finite.size % 2 == 1) finite[mid]
        else 0.5 * (finite[mid - 1] + finite[mid])
    }

    // ----------------------------------------------------------------------
    // FR7. Physical calibration.
    // ----------------------------------------------------------------------

    /**
     * Row positions to millivolts. `digitise.pixels_to_mv`.
     *
     *     mv = (baselinePx - yPx) / pxPerMv
     *
     * **The subtraction is this way round on purpose, and it is the single
     * most dangerous line in the port.** Image rows grow downward while
     * millivolts grow upward, so a trace *above* the baseline has a
     * *smaller* row index and must yield a *positive* voltage. Writing
     * `(yPx - baselinePx)` produces a waveform correct in every respect
     * except that it is upside down — which RMSE and SNR report as a large
     * error, and PCC reports as a tidy -1.0 rather than as a failure. A
     * golden vector covers it.
     */
    fun pixelsToMv(yPx: DoubleArray, baselinePx: Double, pxPerMv: Double): DoubleArray {
        require(pxPerMv > 0.0 && pxPerMv.isFinite()) {
            "pxPerMv must be finite and positive, got $pxPerMv"
        }
        return DoubleArray(yPx.size) { (baselinePx - yPx[it]) / pxPerMv }
    }

    /**
     * Resample a per-column waveform onto a fixed output rate.
     *
     * One pixel column is one time sample at `pxPerS`, which is
     * `pxPerMm * mmPerS` — so the tracer's native rate is a property of the
     * image scale, not a choice. At 25 mm/s, 6 px/mm gives 150 Hz and
     * 12 px/mm gives 300 Hz, and the recovered bandwidth is half of that.
     *
     * Linear rather than a Fourier method, matching the Python: at these
     * resolutions the QRS is close to a step edge and sinc resampling rings
     * visibly around it.
     *
     * The output length floors, so a final partial sample is dropped rather
     * than extrapolated.
     */
    fun resample(mv: DoubleArray, pxPerS: Double, fsOut: Double): DoubleArray {
        require(pxPerS > 0.0 && pxPerS.isFinite()) { "pxPerS must be positive" }
        require(fsOut > 0.0 && fsOut.isFinite()) { "fsOut must be positive" }
        val durationS = mv.size / pxPerS
        val nOut = kotlin.math.floor(durationS * fsOut).toInt()
        if (nOut < 1) {
            throw IllegalArgumentException(
                "strip is shorter than one output sample (durationS=$durationS)"
            )
        }
        // Spelled the way `np.interp` spells it — slope against the actual
        // sample times, not a fractional index — because the port tests
        // compare at 1e-12 relative and the two algebraically identical
        // forms do not agree in the last bits. `t - xLo` with both in
        // seconds is what NumPy computes.
        val out = DoubleArray(nOut)
        for (i in 0 until nOut) {
            val t = i / fsOut
            val lo = kotlin.math.floor(t * pxPerS).toInt()
            if (lo >= mv.size - 1) {
                out[i] = mv[mv.size - 1]
            } else {
                val xLo = lo / pxPerS
                val xHi = (lo + 1) / pxPerS
                val slope = (mv[lo + 1] - mv[lo]) / (xHi - xLo)
                out[i] = slope * (t - xLo) + mv[lo]
            }
        }
        return out
    }

    /**
     * FR5 to FR7 in one call, on an already deskewed image.
     *
     * `grid` must come from [GridEstimator]; the pitch is never assumed and
     * never read from file metadata. `mmPerS` and `mmPerMv` are the *paper*
     * calibration — a property of the machine that printed the ECG, not
     * recoverable from the grid, which gives millimetres only. 25 mm/s and
     * 10 mm/mV are conventional; 50 and 5 occur and are usually annotated on
     * the printout. Passing the wrong one rescales time or amplitude by
     * exactly two.
     *
     * The vertical scale uses the vertical pitch and the horizontal the
     * horizontal one. On a perspective-distorted page these genuinely
     * differ, which is what [GridEstimator] anisotropy reports and what the
     * gate refuses.
     */
    fun digitise(
        mask: Mask,
        pxPerMmX: Double,
        pxPerMmY: Double,
        fsOut: Double,
        mmPerS: Double = 25.0,
        mmPerMv: Double = 10.0,
        maxGapPx: Int = DEFAULT_MAX_GAP_PX
    ): Digitised {
        val trace = trim(centroids(mask))
        val filled = interpolateGaps(trace.yPx, maxGapPx)
        val baseline = baselineFrom(filled)
        val pxPerMv = pxPerMmY * mmPerMv
        val pxPerS = pxPerMmX * mmPerS
        val mv = pixelsToMv(filled, baseline, pxPerMv)
        return Digitised(
            signalMv = resample(mv, pxPerS, fsOut),
            fs = fsOut,
            baselinePx = baseline,
            coverage = trace.coverage,
            pxPerMv = pxPerMv,
            pxPerS = pxPerS
        )
    }
}
