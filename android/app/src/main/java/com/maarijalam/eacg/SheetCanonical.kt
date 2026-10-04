package com.maarijalam.eacg

import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

/**
 * The learned tracer's canonical frame and tiled inference on the phone:
 * the port of `ecgproto/canonical.py`.
 *
 * The frame is 8 px/mm, [height] rows (512 for b3), CANON_W columns. The
 * source (the rectified page, an RGBA Mat) is resized ONCE with INTER_AREA
 * by k = 8 / px_per_mm and cut by an integer crop, padded with the
 * per-channel median of every 7th source pixel -- the same definition the
 * model was trained and evaluated with. The tile stitching and the mapping
 * back to page coordinates are pure Kotlin and port-tested.
 */
object SheetCanonical {

    const val CANON_PX_PER_MM = 8.0
    const val CANON_W = 2048
    const val OUT_UPSAMPLE = 2
    const val TILE_W = 1024
    const val TILE_MARGIN = 128
    const val FULL_FRAME_MAX_PX = 384 * 2048

    /** Source -> canonical: x' = (x + 0.5) * k - 0.5 - x0, same for y. */
    class CanonMap(val k: Double, val x0: Int, val y0: Int) {
        fun columnToSourceX(c: Double): Double = (c + x0 + 0.5) / k - 0.5
        fun rowToSourceY(r: Double): Double = (r + y0 + 0.5) / k - 0.5
    }

    /** A canonical frame: RGB, row-major, [height] x [width] x 3, 0-255. */
    class Frame(val rgb: FloatArray, val height: Int, val width: Int, val map: CanonMap)

    /**
     * `np.median` of each of the first three channels over every 7th pixel
     * of [rgba] in row-major order, truncated to an integer as NumPy's
     * `astype(uint8)` does. From histograms, so the page is never copied
     * whole onto the Java heap.
     */
    fun padColour(rgba: Mat): IntArray {
        val w = rgba.cols()
        val h = rgba.rows()
        val ch = rgba.channels()
        val hist = Array(3) { LongArray(256) }
        val row = ByteArray(w * ch)
        var n = 0L
        for (y in 0 until h) {
            rgba.get(y, 0, row)
            // Flat pixel index y*w + x must be a multiple of 7.
            var x = ((7 - (y.toLong() * w) % 7) % 7).toInt()
            while (x < w) {
                for (c in 0 until 3) hist[c][row[x * ch + c].toInt() and 0xFF]++
                n++
                x += 7
            }
        }
        return IntArray(3) { c -> histMedianTrunc(hist[c], n) }
    }

    /** Median of a uint8 histogram, truncated to an int (NumPy median, then astype). */
    internal fun histMedianTrunc(hist: LongArray, n: Long): Int {
        fun at(rank: Long): Int {
            var acc = 0L
            for (v in 0 until 256) {
                acc += hist[v]
                if (acc > rank) return v
            }
            return 255
        }
        return if (n % 2 == 1L) at(n / 2) else (at(n / 2 - 1) + at(n / 2)) / 2
    }

    /** `canonical.to_canonical` on an RGBA page. */
    fun toCanonical(rgba: Mat, pxPerMm: Double, xStart: Double, centerRow: Double,
                    height: Int, width: Int = CANON_W): Frame {
        val k = CANON_PX_PER_MM / pxPerMm
        val small = Mat()
        try {
            Imgproc.resize(rgba, small, Size(), k, k, Imgproc.INTER_AREA)
            val x0 = StripLocator.pyRound((xStart + 0.5) * k - 0.5)
            val y0 = StripLocator.pyRound((centerRow + 0.5) * k - 0.5) - height / 2
            val pad = padColour(rgba)
            val out = FloatArray(height * width * 3)
            for (i in 0 until height * width) {
                out[3 * i] = pad[0].toFloat(); out[3 * i + 1] = pad[1].toFloat()
                out[3 * i + 2] = pad[2].toFloat()
            }
            val sy0 = max(0, y0)
            val sx0 = max(0, x0)
            val sy1 = min(small.rows(), y0 + height)
            val sx1 = min(small.cols(), x0 + width)
            if (sy1 > sy0 && sx1 > sx0) {
                val ch = small.channels()
                val buf = ByteArray((sx1 - sx0) * ch)
                for (y in sy0 until sy1) {
                    small.get(y, sx0, buf)
                    val dst = ((y - y0) * width + (sx0 - x0)) * 3
                    for (i in 0 until sx1 - sx0) {
                        out[dst + 3 * i] = (buf[i * ch].toInt() and 0xFF).toFloat()
                        out[dst + 3 * i + 1] = (buf[i * ch + 1].toInt() and 0xFF).toFloat()
                        out[dst + 3 * i + 2] = (buf[i * ch + 2].toInt() and 0xFF).toFloat()
                    }
                }
            }
            return Frame(out, height, width, CanonMap(k, x0, y0))
        } finally {
            small.release()
        }
    }

    /** Columns [x0, x0 + tileW) of [frame] as a fresh RGB float array. */
    fun tile(frame: Frame, x0: Int, tileW: Int): FloatArray {
        val out = FloatArray(frame.height * tileW * 3)
        for (y in 0 until frame.height) {
            System.arraycopy(frame.rgb, (y * frame.width + x0) * 3, out, y * tileW * 3, tileW * 3)
        }
        return out
    }

    /** One model call on a tile: per output column (2 per input column). */
    fun interface TileModel {
        /** Returns (row, presence), each 2 * tileW long. */
        fun run(tileRgb: FloatArray, height: Int, tileW: Int): Pair<FloatArray, FloatArray>
    }

    /**
     * `canonical.predict_tiled` for a row-head deploy model: the stitched
     * (row, presence), 2 * frame.width long. Each tile keeps its middle.
     */
    fun predictTiled(frame: Frame, model: TileModel, tileW: Int = TILE_W,
                     margin: Int = TILE_MARGIN): Pair<FloatArray, FloatArray> {
        val h = frame.height
        val w = frame.width
        val u = OUT_UPSAMPLE
        if (w <= tileW || h * w <= FULL_FRAME_MAX_PX) {
            return model.run(frame.rgb, h, w)
        }
        val step = tileW - 2 * margin
        val starts = ArrayList<Int>()
        var s = 0
        while (s < w - tileW) { starts += s; s += step }
        starts += w - tileW
        val row = FloatArray(w * u)
        val pres = FloatArray(w * u)
        for (x0 in starts) {
            val (r, p) = model.run(tile(frame, x0, tileW), h, tileW)
            val keep0 = if (x0 == 0) 0 else margin
            val keep1 = if (x0 + tileW >= w) tileW else tileW - margin
            System.arraycopy(r, keep0 * u, row, (x0 + keep0) * u, (keep1 - keep0) * u)
            System.arraycopy(p, keep0 * u, pres, (x0 + keep0) * u, (keep1 - keep0) * u)
        }
        return row to pres
    }

    /** `canonical.row_trace_to_source`: (page x, page row), NaN where absent. */
    fun rowTraceToSource(row: FloatArray, presence: FloatArray, map: CanonMap,
                         upsample: Int = OUT_UPSAMPLE): Pair<DoubleArray, DoubleArray> {
        val n = row.size
        val xs = DoubleArray(n) { map.columnToSourceX((it + 0.5) / upsample - 0.5) }
        val ys = DoubleArray(n) {
            if (presence[it] < 0f) Double.NaN else map.rowToSourceY(row[it].toDouble())
        }
        return xs to ys
    }
}
