package com.maarijalam.eacg

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.abs

/**
 * The verification loop, and simultaneously the demo.
 *
 *     inject a known rotation -> estimate the skew -> correct it
 *
 * The point is the middle step being scored against the first. On the bench
 * a rotation is injected and recovered in the same script, so agreement
 * proves only that the code is self-consistent. Here the injected angle
 * comes from a slider a human moved, the estimate runs on a different
 * processor in a different language, and the Python's published worst case
 * over three fixtures and seventeen angles at this working resolution is
 * 0.041 degrees. A device result materially worse than that is a porting
 * error, not a property of the algorithm — which is the whole reason this
 * screen exists rather than a unit test.
 *
 * **The rotation is applied twice, and that is unavoidable here.** Injecting
 * a skew and then removing it costs two interpolations, each a low-pass
 * filter on the grid detail. A real capture is warped once, at the
 * correction. So the corrected image on screen is slightly softer than the
 * same page photographed straight would be, and any grid measurement taken
 * from it is pessimistic. The Python's `roundtrip_control` condition exists
 * to price exactly this, and measured there it costs almost nothing —
 * PCC 0.9623 against 0.9624 for the single-warp case.
 */
object DeskewPipeline {

    /** Everything one pass produces. Bitmaps are display-sized, not full. */
    data class Run(
        val rotatedPreview: Bitmap,
        val correctedPreview: Bitmap,
        /** What the slider asked for. Ground truth. */
        val injectedDeg: Double,
        /** What the estimator recovered, without seeing [injectedDeg]. */
        val skew: SkewEstimator.Result,
        val timings: Timings
    ) {
        /**
         * The number this screen exists to show. The estimator returns the
         * rotation to be *removed*, and the injected rotation is the one that
         * was added, so a correct recovery returns the injected angle and the
         * error is their difference.
         */
        val absErrorDeg: Double get() = abs(skew.angleDeg - injectedDeg)
    }

    /**
     * Wall clock per stage.
     *
     * Split rather than totalled because the stages answer different
     * questions and differ by an order of magnitude. `rotate` and `correct`
     * are full-resolution OpenCV warps and scale with the input; `skew` runs
     * at a fixed 900 px and should not. Folding them together would hide
     * precisely the property the deskew work was done to obtain.
     */
    data class Timings(
        val toMatMs: Long,
        val rotateMs: Long,
        val skewMs: Long,
        val correctMs: Long,
        val previewMs: Long,
        val totalMs: Long
    ) {
        /**
         * What a real capture would actually pay: estimate the skew, then
         * apply one correcting warp.
         *
         * **This is the number for the results chapter, and it is not the
         * total.** The total includes two things no deployed pipeline does.
         * The injecting warp exists only to manufacture a known ground truth
         * — a photograph of a crooked page arrives crooked, at no cost — and
         * the previews exist only so a human can look at the result. Both
         * are test scaffolding. Reporting the total as the cost of deskew
         * would roughly double it and would be measuring this screen rather
         * than the stage.
         *
         * `toMatMs` is charged to the pipeline rather than the harness: a
         * real capture still has to get its pixels into a Mat once, even
         * though in a deployed app that conversion would happen upstream at
         * capture and be shared with every later stage.
         */
        val pipelineMs: Long get() = toMatMs + skewMs + correctMs

        /** Test scaffolding: manufacturing the ground truth, and looking at it. */
        val harnessMs: Long get() = rotateMs + previewMs
    }

    /**
     * Run the loop on [source], injecting [injectedDeg].
     *
     * Every Mat is released on the way out, including on failure. Mats are
     * native allocations the Java heap does not see, so the collector feels
     * no pressure from them and will not free them in time under a slider
     * drag; a leak here is an out-of-memory a few dozen frames later, far
     * from its cause.
     */
    suspend fun run(source: Bitmap, injectedDeg: Double): Run =
        withContext(Dispatchers.Default) {
            val t0 = System.nanoTime()
            val src = ImageWarp.toMat(source)
            val tMat = System.nanoTime()
            try {
                val rotated = ImageWarp.rotate(src, injectedDeg)
                val tRotate = System.nanoTime()
                try {
                    // The estimator never sees injectedDeg. It is handed the
                    // rotated pixels and nothing else.
                    val skew = SkewEstimator.estimate(rotated)
                    val tSkew = System.nanoTime()

                    val corrected = ImageWarp.rotate(rotated, -skew.angleDeg)
                    val tCorrect = System.nanoTime()
                    try {
                        val rotatedPreview = ImageWarp.previewBitmap(rotated)
                        val correctedPreview = ImageWarp.previewBitmap(corrected)
                        val tPreview = System.nanoTime()
                        Run(
                            rotatedPreview = rotatedPreview,
                            correctedPreview = correctedPreview,
                            injectedDeg = injectedDeg,
                            skew = skew,
                            timings = Timings(
                                toMatMs = (tMat - t0) / 1_000_000,
                                rotateMs = (tRotate - tMat) / 1_000_000,
                                skewMs = (tSkew - tRotate) / 1_000_000,
                                correctMs = (tCorrect - tSkew) / 1_000_000,
                                previewMs = (tPreview - tCorrect) / 1_000_000,
                                totalMs = (tPreview - t0) / 1_000_000
                            )
                        )
                    } finally {
                        corrected.release()
                    }
                } finally {
                    rotated.release()
                }
            } finally {
                src.release()
            }
        }
}
