package com.maarijalam.eacg

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Sub-pixel sheet corners from four rough taps. Port of `ecgproto/page.py`.
 *
 * The on-device stand-in for FR1 until ML Kit's Document Scanner is wired
 * in, and the same stand-in the desktop scorer uses, so a photograph scored
 * in both places is cropped the same way. The Python docstring carries the
 * method and the measurements; in short: the edge of a printed sheet is
 * where the grid rules that CROSS it stop, found as the half-level fall of
 * the along-edge spread of ink darkness, fitted with a line per edge and
 * intersected.
 *
 * Why a tap needs this at all: on a 2560 px frame shown at 1080 px, a tap is
 * quantised to ~2.4 px before the finger's own error, and the desktop
 * measured a 10 px corner error costing PCC 0.98 -> 0.62.
 *
 * Pure Kotlin over a float darkness image, so the port tests can check it
 * against the Python without OpenCV.
 */
object PageCorners {

    const val SEARCH_FRACTION = 0.015
    const val N_SAMPLES = 80
    const val TANGENT_FRACTION = 0.015
    const val END_MARGIN = 0.06
    const val MIN_STEP = 2.0
    const val STEP_SNR = 4.0
    const val OUTLIER_SIGMA = 3.0

    /** A single-channel float image, row-major. */
    class Plane(val data: FloatArray, val width: Int, val height: Int)

    class Refined(
        /** TL TR BR BL, x then y: 8 values. */
        val corners: DoubleArray,
        val shiftPx: DoubleArray,
        val edgeRmsePx: DoubleArray,
        val edgeInliers: IntArray
    )

    /**
     * `page.ink_darkness`: 255 minus BT.601 luma, in FLOAT32, with the same
     * operation order as NumPy evaluates it. The float width matters: NumPy
     * keeps a float32 array float32 when multiplied by a Python float, so
     * computing this in double would disagree in the last bits.
     */
    fun darkness(bitmap: Bitmap): Plane {
        val w = bitmap.width
        val h = bitmap.height
        val out = FloatArray(w * h)
        val row = IntArray(w)
        for (y in 0 until h) {
            bitmap.getPixels(row, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                val p = row[x]
                out[y * w + x] = darknessOf(
                    (p shr 16) and 0xFF, (p shr 8) and 0xFF, p and 0xFF
                )
            }
        }
        return Plane(out, w, h)
    }

    fun darknessOf(r: Int, g: Int, b: Int): Float =
        255.0f - (0.114f * b.toFloat() + 0.587f * g.toFloat() + 0.299f * r.toFloat())

    /** Exact bilinear sample with edge replication. `page._sample`. */
    internal fun sample(img: Plane, x: Double, y: Double): Double {
        val fx0 = floor(x)
        val fy0 = floor(y)
        val fx = x - fx0
        val fy = y - fy0
        val w = img.width
        val h = img.height
        val xi0 = fx0.toInt().coerceIn(0, w - 1)
        val xi1 = (fx0.toInt() + 1).coerceIn(0, w - 1)
        val yi0 = fy0.toInt().coerceIn(0, h - 1)
        val yi1 = (fy0.toInt() + 1).coerceIn(0, h - 1)
        val d = img.data
        val top = d[yi0 * w + xi0].toDouble() * (1 - fx) + d[yi0 * w + xi1].toDouble() * fx
        val bot = d[yi1 * w + xi0].toDouble() * (1 - fx) + d[yi1 * w + xi1].toDouble() * fx
        return top * (1 - fy) + bot * fy
    }

    /** `np.arange(start, stop, step)`: same length rule, same values. */
    internal fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        val n = max(0, ceil((stop - start) / step).toInt())
        return DoubleArray(n) { start + it * step }
    }

    /** `np.linspace(start, stop, n)`, last element exactly `stop`. */
    internal fun linspace(start: Double, stop: Double, n: Int): DoubleArray {
        val step = (stop - start) / (n - 1)
        return DoubleArray(n) { if (it == n - 1) stop else it * step + start }
    }

    internal fun median(v: DoubleArray): Double {
        val s = v.sortedArray()
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else 0.5 * (s[m - 1] + s[m])
    }

    /** Population standard deviation, as `np.std`. */
    private fun std(v: DoubleArray): Double {
        var m = 0.0
        for (x in v) m += x
        m /= v.size
        var s = 0.0
        for (x in v) s += (x - m) * (x - m)
        return sqrt(s / v.size)
    }

    /** `page._edge_crossings`. Rows of NaN where no clean step was found. */
    internal fun edgeCrossings(
        img: Plane, ax: Double, ay: Double, bx: Double, by: Double,
        nx: Double, ny: Double
    ): Array<DoubleArray> {
        val length = hypot(bx - ax, by - ay)
        val tx = (bx - ax) / length
        val ty = (by - ay) / length
        val reach = max(6.0, SEARCH_FRACTION * length)
        val halfT = max(3.0, 0.5 * TANGENT_FRACTION * length)
        val offsets = arange(-reach, reach + 0.5, 0.5)
        val along = arange(-halfT, halfT + 0.5, 1.0)
        val out = Array(N_SAMPLES) { doubleArrayOf(Double.NaN, Double.NaN) }
        val ts = linspace(END_MARGIN, 1 - END_MARGIN, N_SAMPLES)
        val profile = DoubleArray(offsets.size)
        val line = DoubleArray(along.size)
        val k = max(3, offsets.size / 8)
        for (i in 0 until N_SAMPLES) {
            val t = ts[i]
            val cx = ax + t * (bx - ax)
            val cy = ay + t * (by - ay)
            for (o in offsets.indices) {
                for (a in along.indices) {
                    val px = cx + offsets[o] * nx + along[a] * tx
                    val py = cy + offsets[o] * ny + along[a] * ty
                    line[a] = sample(img, px, py)
                }
                profile[o] = std(line)
            }
            val inside = median(profile.copyOfRange(0, k))
            val tail = profile.copyOfRange(profile.size - k, profile.size)
            val outside = median(tail)
            val noise = 1.4826 * median(DoubleArray(k) { abs(tail[it] - outside) })
            if (inside - outside <= max(MIN_STEP, STEP_SNR * noise)) continue
            val level = 0.5 * (inside + outside)
            var j = -1
            for (q in 0 until profile.size - 1) {
                if (profile[q] >= level && profile[q + 1] < level) j = q
            }
            if (j < 0) continue
            val f = (profile[j] - level) / (profile[j] - profile[j + 1])
            val s = offsets[j] + f * (offsets[j + 1] - offsets[j])
            out[i] = doubleArrayOf(cx + s * nx, cy + s * ny)
        }
        return out
    }

    /**
     * `page._fit_line`, including its exact iteration: the returned line is
     * fitted to the points that entered the LAST fit, and the inlier count is
     * of the points that survived it.
     */
    internal class Line(val mx: Double, val my: Double, val dx: Double, val dy: Double,
                        val rmse: Double, val inliers: Int)

    internal fun fitLine(points: Array<DoubleArray>): Line {
        var p = points.filter { it[0].isFinite() && it[1].isFinite() }
        check(p.size >= 5) { "only ${p.size} edge crossings found" }
        var mx = 0.0; var my = 0.0; var dx = 1.0; var dy = 0.0
        var r = DoubleArray(0)
        for (iter in 0 until 2) {
            mx = p.sumOf { it[0] } / p.size
            my = p.sumOf { it[1] } / p.size
            // Principal direction of the centred points: the right singular
            // vector NumPy's SVD returns first, from the 2x2 scatter matrix.
            var sxx = 0.0; var sxy = 0.0; var syy = 0.0
            for (q in p) {
                val u = q[0] - mx
                val v = q[1] - my
                sxx += u * u; sxy += u * v; syy += v * v
            }
            val tr = sxx + syy
            val det = sxx * syy - sxy * sxy
            val lam = 0.5 * tr + sqrt(max(0.0, 0.25 * tr * tr - det))
            if (abs(sxy) > 1e-300) {
                dx = lam - syy; dy = sxy
            } else if (sxx >= syy) {
                dx = 1.0; dy = 0.0
            } else {
                dx = 0.0; dy = 1.0
            }
            val norm = hypot(dx, dy)
            dx /= norm; dy /= norm
            val nx = -dy
            val ny = dx
            r = DoubleArray(p.size) { (p[it][0] - mx) * nx + (p[it][1] - my) * ny }
            val medR = median(r)
            val sigma = 1.4826 * median(DoubleArray(r.size) { abs(r[it] - medR) }) + 1e-9
            val lim = OUTLIER_SIGMA * max(sigma, 0.25)
            val keep = r.map { abs(it) <= lim }
            val kept = keep.count { it }
            if (kept == keep.size || kept < 5) break
            p = p.filterIndexed { idx, _ -> keep[idx] }
        }
        var ss = 0.0
        for (v in r) ss += v * v
        return Line(mx, my, dx, dy, sqrt(ss / r.size), p.size)
    }

    private fun intersect(a: Line, b: Line): DoubleArray {
        // Solve a.m + t1 a.d = b.m + t2 b.d for t1.
        val det = a.dx * (-b.dy) - (-b.dx) * a.dy
        val rx = b.mx - a.mx
        val ry = b.my - a.my
        val t1 = (rx * (-b.dy) - (-b.dx) * ry) / det
        return doubleArrayOf(a.mx + t1 * a.dx, a.my + t1 * a.dy)
    }

    /** `page.refine_corners`. [rough] is TL TR BR BL as x,y pairs. */
    fun refine(img: Plane, rough: DoubleArray): Refined {
        require(rough.size == 8) { "need four corners" }
        val cx = (rough[0] + rough[2] + rough[4] + rough[6]) / 4
        val cy = (rough[1] + rough[3] + rough[5] + rough[7]) / 4
        val lines = ArrayList<Line>()
        for (i in 0 until 4) {
            val ax = rough[2 * i]; val ay = rough[2 * i + 1]
            val j = (i + 1) % 4
            val bx = rough[2 * j]; val by = rough[2 * j + 1]
            val len = hypot(bx - ax, by - ay)
            val dx = (bx - ax) / len
            val dy = (by - ay) / len
            var nx = -dy
            var ny = dx
            val midx = 0.5 * (ax + bx)
            val midy = 0.5 * (ay + by)
            if (nx * (midx - cx) + ny * (midy - cy) < 0) { nx = -nx; ny = -ny }
            lines += fitLine(edgeCrossings(img, ax, ay, bx, by, nx, ny))
        }
        val corners = DoubleArray(8)
        for (i in 0 until 4) {
            val c = intersect(lines[(i + 3) % 4], lines[i])
            corners[2 * i] = c[0]
            corners[2 * i + 1] = c[1]
        }
        return Refined(
            // Rounded through float, as `page.refine_corners` returns float32
            // (what cv2.getPerspectiveTransform takes). The shift is measured
            // before rounding, as the Python does.
            corners = DoubleArray(8) { corners[it].toFloat().toDouble() },
            shiftPx = DoubleArray(4) {
                hypot(corners[2 * it] - rough[2 * it], corners[2 * it + 1] - rough[2 * it + 1])
            },
            edgeRmsePx = DoubleArray(4) { lines[it].rmse },
            edgeInliers = IntArray(4) { lines[it].inliers }
        )
    }
}
