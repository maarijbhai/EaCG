package com.maarijalam.eacg

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.roundToInt

/**
 * A photograph of a printed 3x4 sheet, end to end, on the handset: the
 * on-device twin of `score_kit_photo.py`'s "pipeline" pass.
 *
 *   taps -> refine corners -> rectify -> locate rhythm strip -> strip text
 *        -> calibrate (duration anchor, and the grid) -> trace -> gate
 *        -> score against the printed record, when one is known
 *
 * The rectified size comes from the photograph's own edge lengths, never
 * from the sheet's known aspect -- the same rule as `score_photo.rectify` --
 * because forcing the true aspect would hand the calibration half its
 * answer. The true scale is known for a kit sheet and is used only to report
 * each calibration's error.
 *
 * TWO CALIBRATIONS, scored separately, because the desktop self-test found
 * them failing in opposite places (`results/kitphoto/selftest.csv`):
 *
 *   duration  recording width / (px_per_mm * 25) = 10 s. External: the
 *             rhythm strip of a 3x4 printout is the whole recording. Within
 *             0.2% of truth on every simulated photograph.
 *   grid      [GridEstimator] on the rectified page. Locks onto the 5 mm
 *             harmonic once blur and JPEG fade the 1 mm rules.
 *
 * TIMINGS: as elsewhere in this app, what a deployed capture pays is
 * PIPELINE and what only the verification harness pays -- the truth
 * comparison -- is HARNESS. Corner refinement is pipeline: until FR1 exists
 * it is how this app finds the page.
 */
object PhotoPipeline {

    const val RECORDING_S = 10.0
    const val MM_PER_S = 25.0
    const val FS_OUT = 500.0

    /**
     * Opening kernel for this path's binarisation. The photograph is decoded
     * to 3200 px, ~10 px/mm on the sheet, where the trace is 2-3 px thick and
     * the default 3x3 erases it; see [TraceExtractor.binarise]. Measured on
     * the first A30s photograph (PTB-XL 40) with good corners at 3200 px:
     * duration anchor -35.6% with 3x3, +5.8% with 2x2.
     */
    const val OPEN_KERNEL = 2

    /**
     * Above this edge-fit residual the refined corners are not used, and the
     * hand-placed ones are. [PageCorners] finds each sheet edge as the step
     * from grid to paper, which assumes the 1 mm rules make the grid region
     * visibly darker; on a print whose minor rules did not come out, every
     * 5 mm rule is an equally good "edge". On the first A30s photograph it
     * snapped one to two columns inside the sheet, at 15-28 px RMS, while
     * simulated photographs fit under 1 px. Placed with [CornerEditor]'s
     * magnifier, the hand corners are good to a few pixels.
     */
    const val EDGE_RMSE_TRUST_PX = 3.0

    /** Harmonic factors a grid comb can lock onto on ECG paper: itself, every
     *  other rule, and the 5 mm majors read as the pitch. */
    val ANCHOR_HARMONICS = intArrayOf(1, 2, 5)

    /**
     * How close grid/k must come to the duration anchor, as a fraction, for k
     * to be accepted. The anchor was within 6% of truth on the first A30s
     * photograph once the trace survived; the candidate factors are at least
     * 2x apart, so 10% picks k unambiguously -- and refuses a comb locked on
     * something that is not a harmonic at all, like the 2.5x that photograph's
     * dashed 5 mm rules produced at full resolution.
     */
    const val ANCHOR_TOLERANCE = 0.10

    /**
     * The grid pitch with its harmonic factor chosen by the duration anchor,
     * or null when no factor brings it within [ANCHOR_TOLERANCE].
     *
     * The anchor only picks an integer; the value is still the grid's. That
     * keeps the two calibrations independent in what they measure while
     * letting the external one settle the ambiguity the grid cannot settle
     * from inside the image -- a 5 mm grid misread as 1 mm looks exactly like
     * a 1 mm grid at a fifth of the scale.
     */
    fun resolveAgainstAnchor(gridPxPerMm: Double, anchorPxPerMm: Double): Pair<Double, Int>? {
        if (!gridPxPerMm.isFinite() || gridPxPerMm <= 0.0) return null
        if (!anchorPxPerMm.isFinite() || anchorPxPerMm <= 0.0) return null
        var bestK = 0
        var bestD = Double.POSITIVE_INFINITY
        for (k in ANCHOR_HARMONICS) {
            val d = kotlin.math.abs(kotlin.math.ln(gridPxPerMm / k / anchorPxPerMm))
            if (d < bestD) { bestD = d; bestK = k }
        }
        if (bestD > kotlin.math.ln(1.0 + ANCHOR_TOLERANCE)) return null
        return gridPxPerMm / bestK to bestK
    }

    class Calibrated(
        val name: String,
        val pxPerMm: Double,
        /**
         * Signed error against the scale the CORNERS imply -- rectified width
         * over the sheet's printed width. Only as true as the corners; see
         * [alignment] for an error measured against the truth waveform.
         */
        val errorVsCornersPct: Double?,
        val digitised: TraceExtractor.Digitised?,
        val gate: QualityGate.Report?,
        val fidelity: Fidelity.Scores?,
        /** Truth samples skipped before sample 0 of [digitised]. */
        val offsetSamples: Int,
        val error: String?,
        val traceMs: Long,
        /** Best time-rescale and lag against the truth; harness only. */
        val alignment: Fidelity.Alignment? = null
    )

    class Timings(
        val cornersMs: Long,
        val rectifyMs: Long,
        val locateMs: Long,
        val stripMs: Long,
        val gridMs: Long,
        val fidelityMs: Long,
        val totalMs: Long
    ) {
        val harnessMs: Long get() = fidelityMs
        val pipelineMs: Long get() = totalMs - fidelityMs
    }

    class Run(
        val corners: DoubleArray,
        val cornerShiftMaxPx: Double,
        val edgeRmseMaxPx: Double,
        val rectW: Int,
        val rectH: Int,
        val pxPerMmFromCorners: Double?,
        val location: StripLocator.Location?,
        val calibrations: List<Calibrated>,
        val stripPreview: Bitmap?,
        val timings: Timings,
        val error: String?,
        /** The corners as placed by hand, before refinement. */
        val handCorners: DoubleArray? = null,
        /** True when [corners] are the hand ones: refinement fit too poorly. */
        val cornersFromHand: Boolean = false,
        /** Local tilt of the strip band, and whether it was removed. */
        val levelAngleDeg: Double? = null,
        val levelMargin: Double? = null,
        val levelApplied: Boolean = false
    )

    suspend fun run(
        photo: Bitmap,
        roughCorners: DoubleArray,
        target: KitTargets.Target?,
        truth: DoubleArray?
    ): Run = withContext(Dispatchers.Default) {
        val t0 = System.nanoTime()
        fun ms(a: Long, b: Long) = (b - a) / 1_000_000

        val refined = PageCorners.refine(PageCorners.darkness(photo), roughCorners)
        val fromHand = refined.edgeRmsePx.max() > EDGE_RMSE_TRUST_PX
        val c = if (fromHand) roughCorners.copyOf() else refined.corners
        val tCorners = System.nanoTime()

        // score_photo.rectify: output size from the photograph's own edges.
        val top = hypot(c[2] - c[0], c[3] - c[1])
        val bottom = hypot(c[4] - c[6], c[5] - c[7])
        val left = hypot(c[6] - c[0], c[7] - c[1])
        val right = hypot(c[4] - c[2], c[5] - c[3])
        val outW = ((top + bottom) / 2.0).roundToInt()
        val outH = ((left + right) / 2.0).roundToInt()
        val truePx = target?.let { outW / it.widthMm }

        val src = ImageWarp.toMat(photo)
        val rect = Mat()
        try {
            val from = MatOfPoint2f(
                Point(c[0], c[1]), Point(c[2], c[3]), Point(c[4], c[5]), Point(c[6], c[7])
            )
            val to = MatOfPoint2f(
                Point(0.0, 0.0), Point(outW - 1.0, 0.0),
                Point(outW - 1.0, outH - 1.0), Point(0.0, outH - 1.0)
            )
            val m = Imgproc.getPerspectiveTransform(from, to)
            Imgproc.warpPerspective(
                src, rect, m, Size(outW.toDouble(), outH.toDouble()),
                Imgproc.INTER_CUBIC, org.opencv.core.Core.BORDER_REPLICATE
            )
            m.release(); from.release(); to.release()
        } finally {
            src.release()
        }
        val tRect = System.nanoTime()

        try {
            val pageMaskMat = TraceExtractor.binarise(rect, openKernel = OPEN_KERNEL)
            val location = try {
                StripLocator.locate(TraceExtractor.maskOf(pageMaskMat))
            } catch (e: IllegalStateException) {
                null
            } finally {
                pageMaskMat.release()
            }
            val tLocate = System.nanoTime()
            if (location == null) {
                return@withContext Run(
                    c, refined.shiftPx.max(), refined.edgeRmsePx.max(), outW, outH,
                    truePx, null, emptyList(), null,
                    Timings(ms(t0, tCorners), ms(tCorners, tRect), ms(tRect, tLocate),
                        0, 0, 0, ms(t0, tLocate)),
                    "could not find four trace rows on the rectified page",
                    handCorners = roughCorners.copyOf(), cornersFromHand = fromHand
                )
            }

            // `segment.level_strip`: remove the strip's OWN residual tilt, from
            // the grid rules in a band around it. Imperfect corners leave the
            // page level at the top and tilted at the bottom -- 0.31 degrees in
            // phone run 5's rhythm band, 0.10 for the page as a whole -- and the
            // trace drifted 1.55 mm against the truth across the strip.
            val stripW = location.x1 - location.x0
            val stripH = location.y1 - location.y0
            val pad = (rect.rows() * StripLocator.LEVEL_PAD_FRACTION).toInt()
            val b0 = maxOf(0, location.y0 - pad)
            val b1 = minOf(rect.rows(), location.y1 + pad)
            val band = Mat(rect, Rect(location.x0, b0, stripW, b1 - b0))
            val level = try { SkewEstimator.estimate(band) } catch (e: RuntimeException) { null }
            val levelApplied = level != null && level.margin >= StripLocator.LEVEL_MIN_MARGIN &&
                kotlin.math.abs(level.angleDeg) <= StripLocator.LEVEL_MAX_DEG
            // The strip is binarised on its own, as the desktop scorer does:
            // the threshold is then a property of the strip, not of the page.
            val stripMat = if (levelApplied) {
                val rotated = ImageWarp.rotate(band, -level!!.angleDeg, expand = false)
                try {
                    Mat(rotated, Rect(0, location.y0 - b0, stripW, stripH)).clone()
                } finally {
                    rotated.release()
                }
            } else {
                Mat(rect, Rect(location.x0, location.y0, stripW, stripH)).clone()
            }
            band.release()
            val stripMask = try {
                val bin = TraceExtractor.binarise(stripMat, openKernel = OPEN_KERNEL)
                try {
                    StripLocator.isolateTrace(TraceExtractor.maskOf(bin))
                } finally {
                    bin.release()
                }
            } finally {
                stripMat.release()
            }
            val tStrip = System.nanoTime()

            val rectBmp = ImageWarp.toBitmap(rect)
            val grid = GridEstimator.estimate(rectBmp)
            rectBmp.recycle()
            val tGrid = System.nanoTime()

            val anchor = location.recordingWidthPx / (RECORDING_S * MM_PER_S)
            val cals = ArrayList<Calibrated>()
            var fidelityNs = 0L
            // "grid" is the comb as measured -- the control, kept even when it
            // is wrong. "grid_anchored" is the same comb with its harmonic
            // factor chosen by the duration anchor, or nothing if no factor fits.
            val anchored = resolveAgainstAnchor(grid.pxPerMm, anchor)
            val candidates = listOf(
                Triple("duration", anchor, "calibration failed (NaN)"),
                Triple("grid", grid.pxPerMm, "calibration failed (NaN)"),
                Triple("grid_anchored", anchored?.first ?: Double.NaN,
                    "grid pitch %.3f px/mm is not 1, 2 or 5 times the anchor's %.3f within %.0f%%"
                        .format(java.util.Locale.US, grid.pxPerMm, anchor, 100 * ANCHOR_TOLERANCE))
            )
            for ((name, px, failure) in candidates) {
                val err = truePx?.let { 100.0 * (px / it - 1.0) }
                if (!px.isFinite() || px <= 0.0) {
                    cals += Calibrated(name, px, err, null, null, null, 0, failure, 0)
                    continue
                }
                val a = System.nanoTime()
                val digitised = try {
                    TraceExtractor.digitise(stripMask, px, px, FS_OUT)
                } catch (e: IllegalArgumentException) {
                    null
                }
                val gate = digitised?.let {
                    QualityGate.assess(
                        it, stripMask, px, px, stripMask.cols,
                        expectedDurationS = RECORDING_S,
                        axesIndependentlyMeasured = false
                    )
                }
                val b = System.nanoTime()
                // Where the pipeline believes its sample 0 sits: its own t0,
                // its own scale. No truth enters the offset.
                val offsetS = (location.x0 - location.t0X) / (px * MM_PER_S)
                val offset = (offsetS * (target?.fs ?: FS_OUT)).roundToInt()
                val scored = digitised != null && truth != null && offset >= 0 &&
                    offset < truth.size - 2
                val ref = if (scored) truth!!.copyOfRange(offset, truth.size) else null
                val fid = if (scored) runCatching {
                    Fidelity.score(ref!!, Fidelity.fillNan(digitised!!.signalMv))
                }.getOrNull() else null
                // As the desktop scorer does: the as-is PCC alone reads ~0 for
                // a trace that is right in shape but shifted or stretched, and
                // the corner-derived error is only as true as the corners.
                val aligned = if (scored) runCatching {
                    Fidelity.bestAlignment(ref!!, digitised!!.signalMv,
                        minOverlap = (Fidelity.MIN_OVERLAP_S * (target?.fs ?: FS_OUT)).toInt())
                }.getOrNull() else null
                fidelityNs += System.nanoTime() - b
                cals += Calibrated(name, px, err, digitised, gate, fid, offset,
                    if (digitised == null) "the tracer found no usable trace" else null,
                    (b - a) / 1_000_000, aligned)
            }
            val tEnd = System.nanoTime()

            val preview = run {
                val s = Mat(rect, Rect(location.x0, location.y0,
                    location.x1 - location.x0, location.y1 - location.y0))
                try { ImageWarp.previewBitmap(s) } finally { s.release() }
            }

            Run(
                corners = c,
                cornerShiftMaxPx = refined.shiftPx.max(),
                edgeRmseMaxPx = refined.edgeRmsePx.max(),
                rectW = outW, rectH = outH, pxPerMmFromCorners = truePx,
                location = location, calibrations = cals, stripPreview = preview,
                timings = Timings(
                    cornersMs = ms(t0, tCorners),
                    rectifyMs = ms(tCorners, tRect),
                    locateMs = ms(tRect, tLocate),
                    stripMs = ms(tLocate, tStrip),
                    gridMs = ms(tStrip, tGrid),
                    fidelityMs = fidelityNs / 1_000_000,
                    totalMs = ms(t0, tEnd)
                ),
                error = null,
                handCorners = roughCorners.copyOf(),
                cornersFromHand = fromHand,
                levelAngleDeg = level?.angleDeg,
                levelMargin = level?.margin,
                levelApplied = levelApplied
            )
        } finally {
            rect.release()
        }
    }
}
