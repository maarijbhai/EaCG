package com.maarijalam.eacg

import kotlin.math.max
import kotlin.math.min

/**
 * Find the rhythm strip on a page-cropped 3x4 printout, and strip printed
 * text from it. Port of `ecgproto/segment.py`: `locate_rhythm_strip` and
 * `isolate_trace`.
 *
 * The one piece of FR4 the app needs to score a photograph: the rhythm strip
 * is the only region of a 3x4 sheet that is a single continuous 10 s trace,
 * so it is the only one [TraceExtractor] can be handed as-is and the only one
 * a 10 s duration anchor holds for. The Python docstring has the method and
 * the measurements behind each constant; they are not repeated here, only
 * the places where a port could quietly differ.
 *
 * **Input contract: a page, not a photograph.** Row ink density over a raw
 * frame also sees the table and the background; the caller rectifies first.
 *
 * Pure Kotlin over [TraceExtractor.Mask], with no OpenCV, so it runs under
 * `testDebugUnitTest` against golden vectors from the Python -- the same split
 * [TraceExtractor] makes, for the same reason.
 */
object StripLocator {

    const val N_ROWS = 4
    const val EDGE_FRACTION = 0.05
    const val SMOOTH_FRACTION = 0.02
    const val MIN_ROW_SPACING_FRACTION = 0.07
    const val CLOSE_FRACTION = 0.035
    const val PULSE_SEARCH_FRACTION = 0.03
    const val PULSE_START_FRACTION = 0.005
    /** `segment.PULSE_LEADIN_FRACTION`: a thin lead-in the rising edge may follow. */
    const val PULSE_LEADIN_FRACTION = 0.015
    const val PULSE_TAIL_FACTOR = 3.0
    const val TRACE_MIN_WIDTH_FRACTION = 0.03
    const val BRIDGE_FRACTION = 0.002
    const val MAX_SHARED_COLUMNS = 0.5
    const val ROW_PAD_FRACTION = 0.01

    class Location(
        val x0: Int, val y0: Int, val x1: Int, val y1: Int,
        val rowPeaks: IntArray,
        /** Right edge of the calibration pulse, or -1 when none was cut. */
        val pulseX1: Int,
        /** Column where the recording begins. */
        val t0X: Double
    ) {
        /** From [t0X] to the last inked column; see the Python property. */
        val recordingWidthPx: Double get() = x1 - t0X
    }

    /**
     * Python's built-in `round`, which rounds half to EVEN. `Math.round`
     * rounds half up, and on a strip 1250 px wide `1250 * 0.002 = 2.5` is
     * exactly the case where they part: 2 against 3, and a different bridge.
     */
    internal fun pyRound(x: Double): Int = Math.rint(x).toInt()

    /** Rows [y0, y1) x columns [x0, x1) of [m], as a new mask. */
    fun sub(m: TraceExtractor.Mask, y0: Int, y1: Int, x0: Int, x1: Int): TraceExtractor.Mask {
        val rows = y1 - y0
        val cols = x1 - x0
        val out = ByteArray(rows * cols)
        for (r in 0 until rows) {
            System.arraycopy(m.data, (y0 + r) * m.cols + x0, out, r * cols, cols)
        }
        return TraceExtractor.Mask(out, rows, cols)
    }

    // ----------------------------------------------------------------------
    // isolate_trace
    // ----------------------------------------------------------------------

    /**
     * Keep only ink components that are trace, by greedy column coverage.
     * `segment.isolate_trace`. Returns a new mask of 0/1 bytes.
     */
    fun isolateTrace(
        mask: TraceExtractor.Mask,
        minWidthFraction: Double = TRACE_MIN_WIDTH_FRACTION
    ): TraceExtractor.Mask {
        val rows = mask.rows
        val cols = mask.cols
        val out = ByteArray(rows * cols)
        var any = false
        for (b in mask.data) if (b != 0.toByte()) { any = true; break }
        if (!any) return TraceExtractor.Mask(out, rows, cols)

        val k = max(3, pyRound(cols * BRIDGE_FRACTION) or 1)
        val bridged = dilateSquare(mask, k)
        val cc = components8(bridged, rows, cols)
        val minW = max(2, pyRound(cols * minWidthFraction))

        // Widest first. Stable, so equal widths keep label (raster) order.
        val order = (1..cc.count).sortedWith(compareBy { -cc.width[it] })
        val covered = BooleanArray(cols)
        val keep = BooleanArray(cc.count + 1)
        for (i in order) {
            val w = cc.width[i]
            if (w < minW) break
            val x = cc.left[i]
            var shared = 0
            for (c in x until x + w) if (covered[c]) shared++
            if (shared.toDouble() / w < MAX_SHARED_COLUMNS) {
                keep[i] = true
                for (c in x until x + w) covered[c] = true
            }
        }
        for (p in 0 until rows * cols) {
            if (mask.data[p] != 0.toByte() && keep[cc.labels[p]]) out[p] = 1
        }
        return TraceExtractor.Mask(out, rows, cols)
    }

    /**
     * `cv2.dilate` with a k x k square of ones, anchor at the centre, default
     * border -- which for a dilation means pixels outside contribute nothing.
     * Separable: a square max is a row max of a column max.
     */
    internal fun dilateSquare(mask: TraceExtractor.Mask, k: Int): BooleanArray {
        val rows = mask.rows
        val cols = mask.cols
        val a = k / 2
        val tmp = BooleanArray(rows * cols)
        for (r in 0 until rows) {
            // Running count of inked pixels in the window [c-a, c-a+k-1].
            var count = 0
            for (j in 0 until min(cols, k - a)) if (mask.data[r * cols + j] != 0.toByte()) count++
            for (c in 0 until cols) {
                tmp[r * cols + c] = count > 0
                val drop = c - a
                val add = c - a + k
                if (drop in 0 until cols && mask.data[r * cols + drop] != 0.toByte()) count--
                if (add in 0 until cols && mask.data[r * cols + add] != 0.toByte()) count++
            }
        }
        val out = BooleanArray(rows * cols)
        for (c in 0 until cols) {
            var count = 0
            for (j in 0 until min(rows, k - a)) if (tmp[j * cols + c]) count++
            for (r in 0 until rows) {
                out[r * cols + c] = count > 0
                val drop = r - a
                val add = r - a + k
                if (drop in 0 until rows && tmp[drop * cols + c]) count--
                if (add in 0 until rows && tmp[add * cols + c]) count++
            }
        }
        return out
    }

    internal class Components(
        val labels: IntArray, val count: Int, val left: IntArray, val width: IntArray
    )

    /**
     * 8-connected components, labelled 1..n in raster order of each
     * component's first pixel -- the order `cv2.connectedComponentsWithStats`
     * assigns -- with each one's left edge and width.
     */
    internal fun components8(on: BooleanArray, rows: Int, cols: Int): Components {
        val labels = IntArray(rows * cols)
        val lefts = ArrayList<Int>().apply { add(0) }
        val rights = ArrayList<Int>().apply { add(0) }
        var n = 0
        val stack = IntArray(rows * cols)
        for (start in 0 until rows * cols) {
            if (!on[start] || labels[start] != 0) continue
            n++
            var lo = start % cols
            var hi = lo
            var top = 0
            stack[top++] = start
            labels[start] = n
            while (top > 0) {
                val p = stack[--top]
                val r = p / cols
                val c = p % cols
                if (c < lo) lo = c
                if (c > hi) hi = c
                for (dr in -1..1) {
                    val rr = r + dr
                    if (rr < 0 || rr >= rows) continue
                    for (dc in -1..1) {
                        val cc = c + dc
                        if (cc < 0 || cc >= cols) continue
                        val q = rr * cols + cc
                        if (on[q] && labels[q] == 0) {
                            labels[q] = n
                            stack[top++] = q
                        }
                    }
                }
            }
            lefts.add(lo)
            rights.add(hi)
        }
        val left = IntArray(n + 1) { lefts[it] }
        val width = IntArray(n + 1) { if (it == 0) 0 else rights[it] - lefts[it] + 1 }
        return Components(labels, n, left, width)
    }

    // ----------------------------------------------------------------------
    // locate_rhythm_strip
    // ----------------------------------------------------------------------

    /** Throws [IllegalStateException] where the Python raises RuntimeError. */
    fun locate(mask: TraceExtractor.Mask, nRows: Int = N_ROWS): Location {
        val h = mask.rows
        val w = mask.cols

        val rowSums = DoubleArray(h)
        for (r in 0 until h) {
            var s = 0
            for (c in 0 until w) if (mask.data[r * w + c] != 0.toByte()) s++
            rowSums[r] = s.toDouble()
        }
        val edge = (h * EDGE_FRACTION).toInt()
        for (r in 0 until edge) rowSums[r] = 0.0
        for (r in h - edge until h) rowSums[r] = 0.0
        val win = max(5, (h * SMOOTH_FRACTION).toInt()) or 1
        val smooth = convolveSameBox(rowSums, win)

        val found = findPeaks(smooth, max(1, (h * MIN_ROW_SPACING_FRACTION).toInt()))
        check(found.size >= nRows) {
            "found ${found.size} trace rows, expected $nRows; is the image " +
                "cropped to the page, and is it a 3x4 sheet with a rhythm strip?"
        }
        val peaks = found.sortedByDescending { smooth[it] }.take(nRows).sorted()

        val band = peaks[nRows - 1]
        val above = peaks[nRows - 2]
        var y0 = above + argmin(smooth, above, band)
        var y1 = min(h, band + (band - y0))

        val inked = BooleanArray(w)
        for (c in 0 until w) {
            for (r in y0 until y1) if (mask.data[r * w + c] != 0.toByte()) { inked[c] = true; break }
        }
        val closed = close1d(inked, max(3, (w * CLOSE_FRACTION).toInt()))
        // Longest run of the closed profile; first one wins a tie, as max().
        var bestLo = -1
        var bestLen = 0
        var c = 0
        while (c < w) {
            if (!closed[c]) { c++; continue }
            var e = c
            while (e < w && closed[e]) e++
            if (e - c > bestLen) { bestLen = e - c; bestLo = c }
            c = e
        }
        check(bestLo >= 0) { "the rhythm row carries no ink" }
        var x0 = -1
        var x1 = -1
        for (i in bestLo until bestLo + bestLen) if (inked[i]) { if (x0 < 0) x0 = i; x1 = i + 1 }

        var trace = isolateTrace(sub(mask, y0, y1, x0, x1))
        var rFirst = -1
        var rLast = -1
        for (r in 0 until trace.rows) {
            for (cc in 0 until trace.cols) {
                if (trace.data[r * trace.cols + cc] != 0.toByte()) {
                    if (rFirst < 0) rFirst = r
                    rLast = r
                    break
                }
            }
        }
        check(rFirst >= 0) { "the rhythm row carries no trace-sized ink" }
        val pad = max(2, (h * ROW_PAD_FRACTION).toInt())
        var cFirst = -1
        var cLast = -1
        for (cc in 0 until trace.cols) {
            for (r in 0 until trace.rows) {
                if (trace.data[r * trace.cols + cc] != 0.toByte()) {
                    if (cFirst < 0) cFirst = cc
                    cLast = cc
                    break
                }
            }
        }
        val ny0 = max(y0, y0 + rFirst - pad)
        val ny1 = min(y1, y0 + rLast + 1 + pad)
        y0 = ny0
        y1 = ny1
        val nx0 = x0 + cFirst
        x1 = x0 + cLast + 1
        x0 = nx0
        trace = isolateTrace(sub(mask, y0, y1, x0, x1))

        val width = x1 - x0
        val counts = DoubleArray(width)
        for (cc in 0 until width) {
            var s = 0
            for (r in 0 until trace.rows) if (trace.data[r * width + cc] != 0.toByte()) s++
            counts[cc] = s.toDouble()
        }
        // A thin lead-in (a grid fragment, a pen landing) may precede the
        // rising edge: skip it, as `segment.locate_rhythm_strip` does.
        val base = medianPositive(counts)
        val leadMax = (width * PULSE_LEADIN_FRACTION).toInt()
        var lead = 0
        while (lead < min(leadMax, width) && counts[lead] <= PULSE_TAIL_FACTOR * base) lead++
        if (lead >= leadMax) lead = 0
        val searchEnd = min(width, lead + max(2, (width * PULSE_SEARCH_FRACTION).toInt()))
        val startEnd = min(width, lead + max(1, (width * PULSE_START_FRACTION).toInt()))
        var searchMax = 0.0
        for (i in lead until searchEnd) searchMax = max(searchMax, counts[i])
        var startMax = 0.0
        for (i in lead until startEnd) startMax = max(startMax, counts[i])

        var pulseX1 = -1
        var t0X = x0.toDouble()
        if (searchMax > 0.0) {
            // Clusters of tall columns; the falling edge is the SECOND.
            val clusters = ArrayList<IntArray>()
            var i = lead
            while (i < searchEnd) {
                if (counts[i] < 0.5 * searchMax) { i++; continue }
                var e = i
                while (e < searchEnd && counts[e] >= 0.5 * searchMax) e++
                clusters.add(intArrayOf(i, e - 1))
                i = e
            }
            if (clusters.size >= 2 && startMax >= 0.5 * searchMax) {
                val first = clusters[1][0]
                val last = clusters[1][1]
                t0X = x0 + 0.5 * (first + last)
                pulseX1 = x0 + last + 1
                var j = last + 1
                while (j < width && counts[j] > PULSE_TAIL_FACTOR * base) j++
                x0 += j
            }
        }
        return Location(x0, y0, x1, y1, peaks.toIntArray(), pulseX1, t0X)
    }

    // ----------------------------------------------------------------------
    // NumPy / SciPy / OpenCV primitives, reproduced where their exact
    // behaviour decides the answer
    // ----------------------------------------------------------------------

    /** `np.convolve(x, np.ones(win) / win, mode="same")` for odd `win`. */
    internal fun convolveSameBox(x: DoubleArray, win: Int): DoubleArray {
        val n = x.size
        val h = win / 2
        val out = DoubleArray(n)
        for (i in 0 until n) {
            // Sum, then divide once: exact on integer row counts, so a flat
            // top stays flat and findPeaks picks the Python's peak.
            var s = 0.0
            for (j in i - h..i + h) if (j in 0 until n) s += x[j]
            out[i] = s / win
        }
        return out
    }

    /**
     * `scipy.signal.find_peaks(x, distance=d)`: strict local maxima (a flat
     * top reports its middle, rounded down), then peaks closer than `d` to a
     * taller one are dropped, tallest first.
     */
    internal fun findPeaks(x: DoubleArray, distance: Int): List<Int> {
        val n = x.size
        val peaks = ArrayList<Int>()
        var i = 1
        val iMax = n - 1
        while (i < iMax) {
            if (x[i - 1] < x[i]) {
                var ahead = i + 1
                while (ahead < iMax && x[ahead] == x[i]) ahead++
                if (x[ahead] < x[i]) {
                    peaks.add((i + ahead - 1) / 2)
                    i = ahead
                }
            }
            i++
        }
        if (distance <= 1 || peaks.size < 2) return peaks
        val keep = BooleanArray(peaks.size) { true }
        val byHeight = peaks.indices.sortedWith(compareBy { x[peaks[it]] })
        for (idx in byHeight.indices.reversed()) {
            val j = byHeight[idx]
            if (!keep[j]) continue
            var k = j - 1
            while (k >= 0 && peaks[j] - peaks[k] < distance) { keep[k] = false; k-- }
            k = j + 1
            while (k < peaks.size && peaks[k] - peaks[j] < distance) { keep[k] = false; k++ }
        }
        return peaks.filterIndexed { idx, _ -> keep[idx] }
    }

    /** First index of the minimum of x[lo, hi), relative to lo. */
    internal fun argmin(x: DoubleArray, lo: Int, hi: Int): Int {
        var best = lo
        for (i in lo until hi) if (x[i] < x[best]) best = i
        return best - lo
    }

    /**
     * `cv2.morphologyEx(v.reshape(1, -1), MORPH_CLOSE, ones((1, k)))`.
     *
     * Anchor at k/2, which is off-centre when k is even. OpenCV's default
     * border makes pixels outside the row neutral: they add nothing to the
     * dilation and do not stop the erosion, which is why a run near the edge
     * reaches column 0 -- and why the caller trims back to real ink.
     */
    internal fun close1d(v: BooleanArray, k: Int): BooleanArray {
        val n = v.size
        val a = k / 2
        val dil = BooleanArray(n)
        for (x in 0 until n) {
            for (j in x - a until x - a + k) if (j in 0 until n && v[j]) { dil[x] = true; break }
        }
        val out = BooleanArray(n)
        for (x in 0 until n) {
            var all = true
            for (j in x - a until x - a + k) if (j in 0 until n && !dil[j]) { all = false; break }
            out[x] = all
        }
        return out
    }

    /** `np.median(counts[counts > 0])`, averaging the middle pair. */
    internal fun medianPositive(v: DoubleArray): Double {
        val s = v.filter { it > 0.0 }.sorted()
        if (s.isEmpty()) return Double.NaN
        val m = s.size / 2
        return if (s.size % 2 == 1) s[m] else 0.5 * (s[m - 1] + s[m])
    }
}
