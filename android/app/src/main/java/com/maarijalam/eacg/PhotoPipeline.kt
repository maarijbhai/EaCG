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

    class Calibrated(
        val name: String,
        val pxPerMm: Double,
        /** Signed error against the true scale in the rectified frame, %. */
        val errorPct: Double?,
        val digitised: TraceExtractor.Digitised?,
        val gate: QualityGate.Report?,
        val fidelity: Fidelity.Scores?,
        /** Truth samples skipped before sample 0 of [digitised]. */
        val offsetSamples: Int,
        val error: String?,
        val traceMs: Long
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
        val truePxPerMm: Double?,
        val location: StripLocator.Location?,
        val calibrations: List<Calibrated>,
        val stripPreview: Bitmap?,
        val timings: Timings,
        val error: String?
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
        val c = refined.corners
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
            val pageMaskMat = TraceExtractor.binarise(rect)
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
                    "could not find four trace rows on the rectified page"
                )
            }

            // The strip is binarised on its own, as the desktop scorer does:
            // the threshold is then a property of the strip, not of the page.
            val stripMat = Mat(rect, Rect(location.x0, location.y0,
                location.x1 - location.x0, location.y1 - location.y0)).clone()
            val stripMask = try {
                val bin = TraceExtractor.binarise(stripMat)
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
            for ((name, px) in listOf("duration" to anchor, "grid" to grid.pxPerMm)) {
                val err = truePx?.let { 100.0 * (px / it - 1.0) }
                if (!px.isFinite() || px <= 0.0) {
                    cals += Calibrated(name, px, err, null, null, null, 0,
                        "calibration failed (NaN)", 0)
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
                val fid = if (digitised != null && truth != null && offset >= 0 &&
                    offset < truth.size - 2) {
                    runCatching {
                        Fidelity.score(truth.copyOfRange(offset, truth.size),
                            digitised.signalMv)
                    }.getOrNull()
                } else null
                fidelityNs += System.nanoTime() - b
                cals += Calibrated(name, px, err, digitised, gate, fid, offset,
                    if (digitised == null) "the tracer found no usable trace" else null,
                    (b - a) / 1_000_000)
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
                rectW = outW, rectH = outH, truePxPerMm = truePx,
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
                error = null
            )
        } finally {
            rect.release()
        }
    }
}
