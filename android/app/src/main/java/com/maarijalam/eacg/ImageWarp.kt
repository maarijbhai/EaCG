package com.maarijalam.eacg

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max

/**
 * Rotation and resizing, in OpenCV, matching `ecgproto/deskew.py` exactly.
 *
 * This is the half of the deskew stage that OpenCV should do. Both
 * operations here are full-resolution and per-pixel — a resize touches every
 * source pixel, a warp interpolates every destination pixel — which is
 * precisely the shape of work a hand-written Kotlin loop loses badly on and
 * a SIMD-vectorised native routine wins. The angle *search* is the opposite
 * shape and lives in [SkewEstimator]; see that file for why.
 *
 * **The conventions are the Python's, not OpenCV's defaults, and the
 * difference matters.** Three of them:
 *
 *  - `expand = true` grows the canvas so no ink is pushed outside it.
 *    OpenCV's default is to keep the source size and crop the corners; for
 *    a strip whose trace runs to both ends that removes signal, and the
 *    subsequent digitisation then scores a shorter waveform without saying
 *    so.
 *  - The border is filled **white**, not OpenCV's default black. The grid
 *    response keys on red saturation, for which white is silent and black
 *    is not, so a black margin would contribute a hard straight edge to
 *    exactly the statistic the angle search maximises. White is also what a
 *    photograph of paper on a desk looks like at the margins.
 *  - Lanczos, not OpenCV's default bilinear. Chosen by measurement, not
 *    reputation: `results/deskew/interp.csv` compares all four on a fixture
 *    rotated 3 degrees and then deskewed back, and bilinear — the default —
 *    is the worst of the four on every axis, costing about 1.6 points of
 *    PCC and 1.3 of gain against Lanczos for nothing.
 *
 * Nothing here releases the Mats it returns. Mats are native allocations
 * outside the Java heap, so the garbage collector does not see the pressure
 * and will not free them in time under repeated use. Every caller must
 * `release()` what it is handed; the callers in this app do so in `finally`
 * blocks.
 */
object ImageWarp {

    /** See the class docstring: chosen by measurement. Python's `DEFAULT_INTERPOLATION`. */
    const val DEFAULT_INTERPOLATION: Int = Imgproc.INTER_LANCZOS4

    /** Python's `border_value=(255, 255, 255)`, in RGBA. */
    private val WHITE = Scalar(255.0, 255.0, 255.0, 255.0)

    /**
     * Python's `deskew.rotation_matrix`. Returns the 2x3 affine and the
     * output size it produces.
     *
     * Exposed separately, as it is in the Python, because anything holding
     * coordinates in the original frame — a lead bounding box, a detector's
     * output — has to move with the image. Without it a caller either
     * reimplements the geometry and drifts out of sync, or is forced to
     * deskew before it has finished using its coordinates.
     */
    fun rotationMatrix(
        rows: Int,
        cols: Int,
        angleDeg: Double,
        expand: Boolean = true
    ): Pair<Mat, Size> {
        val centre = Point(cols / 2.0, rows / 2.0)
        val m = Imgproc.getRotationMatrix2D(centre, angleDeg, 1.0)
        if (!expand) return m to Size(cols.toDouble(), rows.toDouble())

        // getRotationMatrix2D returns CV_64F; read the two entries that give
        // cos and sin so the expanded size is derived from the same matrix
        // that will be applied, not recomputed from the angle.
        val cos = abs(m.get(0, 0)[0])
        val sin = abs(m.get(0, 1)[0])
        val newW = ceil(rows * sin + cols * cos).toInt()
        val newH = ceil(rows * cos + cols * sin).toInt()

        m.put(0, 2, m.get(0, 2)[0] + newW / 2.0 - centre.x)
        m.put(1, 2, m.get(1, 2)[0] + newH / 2.0 - centre.y)
        return m to Size(newW.toDouble(), newH.toDouble())
    }

    /**
     * Python's `deskew.rotate_image`: rotate anticlockwise by [angleDeg].
     * One warp, one interpolation.
     *
     * Caller owns the returned Mat.
     */
    fun rotate(
        src: Mat,
        angleDeg: Double,
        interpolation: Int = DEFAULT_INTERPOLATION,
        expand: Boolean = true
    ): Mat {
        val (m, size) = rotationMatrix(src.rows(), src.cols(), angleDeg, expand)
        val dst = Mat()
        try {
            Imgproc.warpAffine(
                src, dst, m, size, interpolation, Core.BORDER_CONSTANT, WHITE
            )
        } finally {
            m.release()
        }
        return dst
    }

    /**
     * Python's `deskew._reduce`: area-decimate so the longest side is at
     * most [maxDim]. Returns the reduction factor alongside, which callers
     * need only for reporting — skew angle is scale-invariant.
     *
     * INTER_AREA rather than a cheaper sampler because it averages over the
     * pixels it discards. Point-sampling a 1 mm comb below its Nyquist limit
     * aliases the rules into a beat pattern, which the sharpness objective
     * would happily maximise at the wrong angle.
     *
     * Returns the source Mat itself, and a scale of 1.0, when no reduction
     * is needed — so the caller must not blindly release the result if it
     * still needs the source. [reduceOwned] is the variant that always
     * hands back something safe to release.
     */
    fun reduce(src: Mat, maxDim: Int): Pair<Mat, Double> {
        val longest = max(src.rows(), src.cols())
        if (maxDim <= 0 || longest <= maxDim) return src to 1.0
        val scale = maxDim.toDouble() / longest
        val dst = Mat()
        Imgproc.resize(src, dst, Size(), scale, scale, Imgproc.INTER_AREA)
        return dst to scale
    }

    /** [reduce], but the returned Mat is always a fresh allocation the caller owns. */
    fun reduceOwned(src: Mat, maxDim: Int): Pair<Mat, Double> {
        val (m, scale) = reduce(src, maxDim)
        return if (m === src) src.clone() to scale else m to scale
    }

    /**
     * Bitmap to RGBA Mat.
     *
     * Note the channel order for anyone porting from the Python: OpenCV on
     * the desktop reads files as **BGR**, so `cv2.split` there yields
     * `(b, g, r)`. Android bitmaps arrive here as **RGBA**, so index 0 is
     * red, not blue. Every per-channel calculation ported from `ecgproto`
     * has to be re-indexed, and getting it backwards produces a response
     * that is wrong but not obviously wrong — it still has structure, just
     * of the trace instead of the grid.
     */
    fun toMat(bitmap: Bitmap): Mat {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    /** RGBA Mat to a new ARGB_8888 bitmap of the same size. */
    fun toBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }

    /**
     * A display-sized copy.
     *
     * Full-resolution bitmaps are not for looking at. The 300 dpi fixture is
     * 3300x2550, which is 33.7 MB as ARGB_8888, and holding an original, a
     * rotated copy and a corrected copy at that size is over 100 MB against
     * a per-app heap of roughly 200 MB on a 3 GB device. The pixels that
     * matter go to the estimator as a Mat, in native memory; only this
     * reduced copy reaches the Java heap and the screen.
     */
    fun previewBitmap(mat: Mat, maxDim: Int = PREVIEW_MAX_DIM): Bitmap {
        val (small, scale) = reduce(mat, maxDim)
        return try {
            toBitmap(small)
        } finally {
            if (scale != 1.0) small.release()
        }
    }

    /** Longest side of a preview bitmap, in pixels. */
    const val PREVIEW_MAX_DIM = 1080
}
