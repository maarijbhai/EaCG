package com.maarijalam.eacg

import android.graphics.Bitmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/**
 * FR1 without clicks: the printed sheet's four corners in a photograph.
 * The port of `page.find_sheet_corners` (see its docstring for the measured
 * reason behind every step):
 *
 *  1. at FIND_MAX_DIM, the PAPER (Otsu on the darkest channel: an orange
 *     desk is excluded), then inside it the printed GRID (contrast-normalised
 *     texture, thresholded just above the blank margin's level);
 *  2. each edge a straight line by RANSAC through the region's extreme
 *     pixels, draws from the shared 31-bit LCG so it matches the Python;
 *  3. [PageCorners.refine] at full resolution, accepted at <= 0.4 mm edge rmse.
 *
 * Assumes white paper around the print, a background darker or more
 * saturated than paper, and a roughly upright sheet (within ~20 degrees).
 */
object SheetCorners {

    const val FIND_MAX_DIM = 1200
    const val FIND_CLOSE_FRACTION = 0.02
    const val FIND_TEXTURE_FRACTION = 0.012
    const val FIND_PAPER_ERODE_FRACTION = 0.01
    const val FIND_MARGIN_PCT = 5.0
    const val FIND_GRID_CUT = 0.35
    const val FIND_RANSAC_TOL = 0.004
    const val FIND_RANSAC_ITERS = 400
    const val FIND_REFINE_MAX_RMSE_MM = 0.4
    const val FIND_NOMINAL_WIDTH_MM = 270.0
    private const val LCG_SEED = 12345L
    private const val LCG_MULT = 1103515245L
    private const val LCG_ADD = 12345L
    private const val LCG_MASK = 0x7FFFFFFFL

    class Found(
        /** TL TR BR BL as x,y pairs, full-resolution photo pixels. */
        val corners: DoubleArray,
        val coarse: DoubleArray,
        val refined: Boolean,
        val edgeRmsePx: DoubleArray?
    )

    // ------------------------------------------------- pure (port-tested)

    /** `np.percentile(v, q)`, linear interpolation. */
    fun percentile(v: FloatArray, q: Double): Double {
        val s = v.sortedArray()
        val pos = (s.size - 1) * q / 100.0
        val lo = pos.toInt()
        val hi = minOf(lo + 1, s.size - 1)
        return s[lo] + (pos - lo) * (s[hi] - s[lo])
    }

    /**
     * `page._ransac_line` up to the final least-squares fit: the inlier mask
     * of the line most of (xs, ys) agree on.
     */
    fun ransacInliers(xs: DoubleArray, ys: DoubleArray, tol: Double): BooleanArray {
        val n = xs.size
        check(n >= 10) { "too few edge points" }
        var best = BooleanArray(n)
        var bestIn = -1
        var state = LCG_SEED
        repeat(FIND_RANSAC_ITERS) {
            state = (state * LCG_MULT + LCG_ADD) and LCG_MASK
            val i = ((state shr 8) % n).toInt()
            state = (state * LCG_MULT + LCG_ADD) and LCG_MASK
            val j = ((state shr 8) % n).toInt()
            if (i == j) return@repeat
            val dx = xs[j] - xs[i]
            val dy = ys[j] - ys[i]
            val l = hypot(dx, dy)
            if (l < 1e-6) return@repeat
            val nx = -dy / l
            val ny = dx / l
            val inl = BooleanArray(n) { abs((xs[it] - xs[i]) * nx + (ys[it] - ys[i]) * ny) <= tol }
            val c = inl.count { it }
            if (c > bestIn) { bestIn = c; best = inl }
        }
        return best
    }

    /**
     * The region's extreme pixels for one side: top/bottom per column,
     * left/right per row. [region] is row-major 0/non-0, [w] x [h].
     */
    fun extremes(region: ByteArray, w: Int, h: Int, side: String): Pair<DoubleArray, DoubleArray> {
        val xs = ArrayList<Double>()
        val ys = ArrayList<Double>()
        fun on(x: Int, y: Int) = region[y * w + x] != 0.toByte()
        when (side) {
            "top", "bottom" -> for (x in 0 until w) {
                val range = if (side == "top") 0 until h else (h - 1 downTo 0)
                for (y in range) if (on(x, y)) { xs += x.toDouble(); ys += y.toDouble(); break }
            }
            else -> for (y in 0 until h) {
                val range = if (side == "left") 0 until w else (w - 1 downTo 0)
                for (x in range) if (on(x, y)) { xs += x.toDouble(); ys += y.toDouble(); break }
            }
        }
        return xs.toDoubleArray() to ys.toDoubleArray()
    }

    /** `page._intersect`: lines as (point, direction). */
    fun intersect(p1: DoubleArray, d1: DoubleArray, p2: DoubleArray, d2: DoubleArray): DoubleArray {
        // [d1, -d2] t = p2 - p1
        val a = d1[0]; val b = -d2[0]; val c = d1[1]; val d = -d2[1]
        val det = a * d - b * c
        val rx = p2[0] - p1[0]; val ry = p2[1] - p1[1]
        val t0 = (rx * d - b * ry) / det
        return doubleArrayOf(p1[0] + t0 * d1[0], p1[1] + t0 * d1[1])
    }

    // ----------------------------------------------------- OpenCV parts

    private fun largest(mask: Mat): Mat {
        val labels = Mat(); val stats = Mat(); val cent = Mat()
        try {
            val n = Imgproc.connectedComponentsWithStats(mask, labels, stats, cent, 8, CvType.CV_32S)
            check(n >= 2) { "empty mask" }
            var big = 1
            var area = -1.0
            for (i in 1 until n) {
                val a = stats.get(i, Imgproc.CC_STAT_AREA)[0]
                if (a > area) { area = a; big = i }
            }
            val one = Mat()
            Core.compare(labels, Scalar(big.toDouble()), one, Core.CMP_EQ)
            val contours = ArrayList<MatOfPoint>()
            val hier = Mat()
            Imgproc.findContours(one, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE)
            hier.release()
            val filled = Mat.zeros(one.size(), CvType.CV_8U)
            val c = contours.maxByOrNull { Imgproc.contourArea(it) }!!
            Imgproc.drawContours(filled, listOf(c), -1, Scalar(255.0), -1)
            one.release(); contours.forEach { it.release() }
            return filled
        } finally {
            labels.release(); stats.release(); cent.release()
        }
    }

    /** `page.sheet_region`: the printed-grid region of a small RGBA image. */
    fun sheetRegion(small: Mat): Mat {
        val maxDim = max(small.rows(), small.cols())
        val r = Mat(); val g = Mat(); val b = Mat(); val white = Mat()
        Core.extractChannel(small, r, 0); Core.extractChannel(small, g, 1)
        Core.extractChannel(small, b, 2)
        Core.min(r, g, white); Core.min(white, b, white)
        r.release(); g.release(); b.release()
        Imgproc.GaussianBlur(white, white, Size(0.0, 0.0), 2.0)
        val paper0 = Mat()
        Imgproc.threshold(white, paper0, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        white.release()
        val paper = largest(paper0); paper0.release()

        val luma8 = Mat(); val luma = Mat()
        Imgproc.cvtColor(small, luma8, Imgproc.COLOR_RGBA2GRAY)
        luma8.convertTo(luma, CvType.CV_32F); luma8.release()
        val gx = Mat(); val gy = Mat(); val mag = Mat()
        Imgproc.Sobel(luma, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(luma, gy, CvType.CV_32F, 0, 1, 3)
        Core.magnitude(gx, gy, mag); gx.release(); gy.release()
        val sig = max(1.0, FIND_TEXTURE_FRACTION * maxDim)
        val tex = Mat(); val lb = Mat()
        Imgproc.GaussianBlur(mag, tex, Size(0.0, 0.0), sig); mag.release()
        Imgproc.GaussianBlur(luma, lb, Size(0.0, 0.0), sig); luma.release()
        Core.add(lb, Scalar(8.0), lb)
        Core.divide(tex, lb, tex); lb.release()

        val e = max(3, StripLocator.pyRound(FIND_PAPER_ERODE_FRACTION * maxDim))
        val inner = Mat()
        Imgproc.erode(paper, inner, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(e.toDouble(), e.toDouble())))
        paper.release()
        val w = small.cols(); val h = small.rows()
        val texF = FloatArray(w * h); tex.get(0, 0, texF); tex.release()
        val innerB = ByteArray(w * h); inner.get(0, 0, innerB); inner.release()
        var nIn = 0
        for (v in innerB) if (v != 0.toByte()) nIn++
        check(nIn >= 100) { "no paper found" }
        val vals = FloatArray(nIn)
        var k = 0
        for (i in texF.indices) if (innerB[i] != 0.toByte()) vals[k++] = texF[i]
        val lo = percentile(vals, FIND_MARGIN_PCT)
        val mid = percentile(vals, 50.0)
        val thr = lo + FIND_GRID_CUT * (mid - lo)
        val gridB = ByteArray(w * h) { if (innerB[it] != 0.toByte() && texF[it] > thr) 255.toByte() else 0 }
        val grid = Mat(h, w, CvType.CV_8U); grid.put(0, 0, gridB)
        val kc = max(3, StripLocator.pyRound(FIND_CLOSE_FRACTION * maxDim)).toDouble()
        Imgproc.morphologyEx(grid, grid, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kc, kc)))
        val ko = (kc.toInt() / 2 + 1).toDouble()
        Imgproc.morphologyEx(grid, grid, Imgproc.MORPH_OPEN,
            Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(ko, ko)))
        try { return largest(grid) } finally { grid.release() }
    }

    private fun fitLineCv(xs: DoubleArray, ys: DoubleArray, keep: BooleanArray): Pair<DoubleArray, DoubleArray> {
        val pts = ArrayList<Point>()
        for (i in xs.indices) if (keep[i]) pts += Point(xs[i].toFloat().toDouble(), ys[i].toFloat().toDouble())
        val mp = MatOfPoint2f(); mp.fromList(pts)
        val line = Mat()
        Imgproc.fitLine(mp, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
        val v = FloatArray(4); line.get(0, 0, v)
        mp.release(); line.release()
        return doubleArrayOf(v[2].toDouble(), v[3].toDouble()) to doubleArrayOf(v[0].toDouble(), v[1].toDouble())
    }

    /** The coarse corners (TL TR BR BL, x,y pairs) in [rgba] pixels. */
    fun coarseCorners(rgba: Mat): DoubleArray {
        val h = rgba.rows(); val w = rgba.cols()
        val s = FIND_MAX_DIM.toDouble() / max(h, w)
        val small = Mat()
        Imgproc.resize(rgba, small, Size(), s, s, Imgproc.INTER_AREA)
        val region = sheetRegion(small)
        val rw = region.cols(); val rh = region.rows()
        val rb = ByteArray(rw * rh); region.get(0, 0, rb)
        region.release(); small.release()
        val tol = FIND_RANSAC_TOL * max(rw, rh)
        val lines = HashMap<String, Pair<DoubleArray, DoubleArray>>()
        for (side in listOf("top", "right", "bottom", "left")) {
            val (xs, ys) = extremes(rb, rw, rh, side)
            lines[side] = fitLineCv(xs, ys, ransacInliers(xs, ys, tol))
        }
        val order = listOf("left", "top", "right", "bottom")
        val out = DoubleArray(8)
        for (i in 0 until 4) {
            val a = lines.getValue(order[i]); val b = lines.getValue(order[(i + 1) % 4])
            val p = intersect(a.first, a.second, b.first, b.second)
            out[2 * i] = p[0] / s; out[2 * i + 1] = p[1] / s
        }
        val tolF = 0.05 * max(h, w)
        for (i in 0 until 4) {
            val x = out[2 * i]; val y = out[2 * i + 1]
            check(x.isFinite() && y.isFinite() && x >= -tolF && y >= -tolF &&
                x <= w + tolF && y <= h + tolF) { "coarse corners fall outside the photograph" }
        }
        return out
    }

    /** `page.find_sheet_corners`: coarse, then refined if the edges fit. */
    fun find(photo: Bitmap): Found {
        val rgba = ImageWarp.toMat(photo)
        val coarse = try { coarseCorners(rgba) } finally { rgba.release() }
        val top = hypot(coarse[2] - coarse[0], coarse[3] - coarse[1])
        val bottom = hypot(coarse[4] - coarse[6], coarse[5] - coarse[7])
        val maxRmse = FIND_REFINE_MAX_RMSE_MM * 0.5 * (top + bottom) / FIND_NOMINAL_WIDTH_MM
        return try {
            val r = PageCorners.refine(PageCorners.darkness(photo), coarse)
            if (r.edgeRmsePx.max() <= maxRmse) Found(r.corners, coarse, true, r.edgeRmsePx)
            else Found(coarse, coarse, false, r.edgeRmsePx)
        } catch (e: IllegalStateException) {
            Found(coarse, coarse, false, null)
        }
    }
}
