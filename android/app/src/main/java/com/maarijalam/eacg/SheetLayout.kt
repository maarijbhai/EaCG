package com.maarijalam.eacg

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * The whole 3x4 + rhythm sheet, in pure Kotlin: the port of the FR4/FR7
 * half of `ecgproto/segment.py` and the cut-into-leads half of
 * `ecgproto/sheet.py`.
 *
 * Everything here works on [TraceExtractor.Mask]s and arrays -- no OpenCV --
 * so the JVM port suite can check it against Python golden vectors
 * (`python/make_sheet_port_fixtures.py`, `SheetPortFixturesTest`). The OpenCV
 * parts (binarisation, rectification, the canonical resize, the model call)
 * live in [SheetPipeline] and are checked on the phone.
 *
 * Function names follow the Python they port; each says which. Rounding
 * follows Python too: `round` is half-to-even ([StripLocator.pyRound]),
 * `int()` truncates, `np.median` averages the two middle values, and
 * `np.interp` clamps unless told otherwise.
 */
object SheetLayout {

    val LAYOUT_3X4 = arrayOf(
        arrayOf("I", "aVR", "V1", "V4"),
        arrayOf("II", "aVL", "V2", "V5"),
        arrayOf("III", "aVF", "V3", "V6")
    )
    const val CELL_S = 2.5
    const val RECORDING_S = 10.0
    const val MM_PER_S = 25.0
    const val MM_PER_MV = 10.0
    const val FS = 500.0
    const val SEPARATOR_GUARD_MM = 1.0
    const val SHEET_ROW_SPACING_FRACTION = 0.11

    const val PULSE_TOP_MM = 10.0
    const val PULSE_SLAB_MM = 2.5
    const val PULSE_WIDTH_MIN_MM = 3.0
    const val PULSE_WIDTH_MAX_MM = 7.5
    const val PULSE_GAP_MM = 0.5
    const val PULSE_SEARCH_FRACTION_X = 0.15

    const val TICK_MIN_HEIGHT_MM = 4.0
    const val TICK_WIDTH_MM = 0.45
    const val TICK_BASE_BAND_MM = 1.0
    const val TICK_FIRST_WINDOW = 0.25
    const val TICK_WINDOW = 0.04
    const val TICK_MIN_FOUND = 3

    const val TRACK_MAX_JUMP_MM = 1.5

    // ---------------------------------------------------------------- numpy

    /** `np.median` (NaN-free input). */
    fun median(v: DoubleArray): Double {
        require(v.isNotEmpty())
        val s = v.sortedArray()
        val n = s.size
        return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    /**
     * `np.interp(x, xp, fp, left, right)`; xp increasing. With [left]/[right]
     * null it clamps to the end values, as NumPy does by default.
     */
    fun interp(x: Double, xp: DoubleArray, fp: DoubleArray,
               left: Double? = null, right: Double? = null): Double {
        val n = xp.size
        if (x.isNaN()) return Double.NaN
        if (x < xp[0]) return left ?: fp[0]
        if (x > xp[n - 1]) return right ?: fp[n - 1]
        if (x == xp[n - 1]) return fp[n - 1]
        var lo = 0
        var hi = n - 1
        while (hi - lo > 1) {
            val mid = (lo + hi) ushr 1
            if (xp[mid] <= x) lo = mid else hi = mid
        }
        val d = xp[hi] - xp[lo]
        if (d == 0.0) return fp[lo]
        return fp[lo] + (x - xp[lo]) * (fp[hi] - fp[lo]) / d
    }

    // -------------------------------------------------------------- layout

    class LeadCell(
        val name: String, val row: Int, val col: Int,
        val tStartS: Double, val durationS: Double,
        val x0: Int, val x1: Int, val y0: Int, val y1: Int,
        val baselineY: Double
    ) {
        val rhythm: Boolean get() = row == 3
    }

    class Knots(val xs: DoubleArray, val ts: DoubleArray)

    class LeadLayout(
        val cells: List<LeadCell>,
        val t0X: Double,
        val pxPerS: Double,
        val rowY: IntArray,
        val rowBands: Array<IntArray>,
        val rowKnots: List<Knots>
    ) {
        /**
         * `LeadLayout.time_at`: recording time at column [x] of [row], linear
         * between knots and extrapolated along the end segments beyond them.
         */
        fun timeAt(row: Int, x: Double): Double {
            val xs = rowKnots[row].xs
            val ts = rowKnots[row].ts
            val n = xs.size
            return when {
                x < xs[0] -> ts[0] + (x - xs[0]) * (ts[1] - ts[0]) / (xs[1] - xs[0])
                x > xs[n - 1] -> ts[n - 1] + (x - xs[n - 1]) * (ts[n - 1] - ts[n - 2]) / (xs[n - 1] - xs[n - 2])
                else -> interp(x, xs, ts)
            }
        }

        fun cell(name: String, rhythm: Boolean = false): LeadCell =
            cells.first { it.name == name && it.rhythm == rhythm }
    }

    class TickScale(
        val pxPerMm: Double,
        val nFound: Int,
        val spreadPct: Double,
        /** (row, k, x) per tick found, x rounded to 0.1 as the Python does. */
        val positions: List<Triple<Int, Int, Double>>
    )

    /** `segment.sheet_rows` from a binarised page. */
    fun sheetRows(mask: TraceExtractor.Mask): IntArray {
        val h = mask.rows
        val w = mask.cols
        val rows = DoubleArray(h)
        for (y in 0 until h) {
            var n = 0
            val base = y * w
            for (x in 0 until w) if (mask.data[base + x] != 0.toByte()) n++
            rows[y] = n.toDouble()
        }
        val edge = (h * StripLocator.EDGE_FRACTION).toInt()
        for (y in 0 until edge) rows[y] = 0.0
        for (y in h - edge until h) rows[y] = 0.0
        val win = max(5, (h * StripLocator.SMOOTH_FRACTION).toInt()) or 1
        val smooth = StripLocator.convolveSameBox(rows, win)
        val peaks = StripLocator.findPeaks(smooth, max(1, (h * SHEET_ROW_SPACING_FRACTION).toInt()))
        check(peaks.size >= 4) { "found ${peaks.size} trace rows, expected 4" }
        // Tallest four, stable on ties (argsort(-v, kind="stable")), then by row.
        val top = peaks.withIndex().sortedWith(compareBy({ -smooth[it.value] }, { it.index }))
            .take(4).map { it.value }.sorted()
        return top.toIntArray()
    }

    /** `segment.pulse_falling_edge`: the pulse top's right end, or NaN. */
    fun pulseFallingEdge(mask: TraceExtractor.Mask, rowY: Int, pxPerMm: Double): Double {
        val h = mask.rows
        val w = mask.cols
        val yc = rowY - PULSE_TOP_MM * pxPerMm
        val a = max(0, StripLocator.pyRound(yc - PULSE_SLAB_MM * pxPerMm))
        val b = min(h, StripLocator.pyRound(yc + PULSE_SLAB_MM * pxPerMm))
        if (b <= a) return Double.NaN
        val xw = (w * PULSE_SEARCH_FRACTION_X).toInt()
        val idx = ArrayList<Int>()
        for (x in 0 until xw) {
            for (y in a until b) if (mask.data[y * w + x] != 0.toByte()) { idx += x; break }
        }
        if (idx.isEmpty()) return Double.NaN
        val gap = max(1, StripLocator.pyRound(PULSE_GAP_MM * pxPerMm))
        var last = idx[0]
        for (i in 1 until idx.size) {
            if (idx[i] - idx[i - 1] > gap) break
            last = idx[i]
        }
        val lengthMm = (last - idx[0] + 1) / pxPerMm
        if (lengthMm < PULSE_WIDTH_MIN_MM || lengthMm > PULSE_WIDTH_MAX_MM) return Double.NaN
        return last.toDouble() - 0.5 * (0.25 * pxPerMm)
    }

    /** Median of the per-row pulse edges, or NaN when none was found. */
    fun t0FromPulses(edges: DoubleArray): Double {
        val found = edges.filter { it.isFinite() }.toDoubleArray()
        return if (found.isEmpty()) Double.NaN else median(found)
    }

    /** `segment._baseline_runs`. */
    internal fun baselineRuns(mask: TraceExtractor.Mask, baseY: Int, band: Int): DoubleArray {
        val h = mask.rows
        val w = mask.cols
        val run = DoubleArray(w)
        val lo = max(0, baseY - band)
        val hi = min(h, baseY + band + 1)
        val ys = IntArray(hi - lo)
        for (x in 0 until w) {
            var n = 0
            for (y in lo until hi) if (mask.data[y * w + x] != 0.toByte()) ys[n++] = y
            if (n == 0) continue
            val y = ys[n / 2]
            var a = y
            while (a > 0 && mask.data[(a - 1) * w + x] != 0.toByte()) a--
            var b = y
            while (b < h - 1 && mask.data[(b + 1) * w + x] != 0.toByte()) b++
            run[x] = (b - a + 1).toDouble()
        }
        return run
    }

    private fun argmaxFirst(v: DoubleArray, lo: Int, hi: Int): Int {
        var j = lo
        for (i in lo until hi) if (v[i] > v[j]) j = i
        return j
    }

    /**
     * `segment.tick_scale` from the photo path's (2x2-opened) mask, the four
     * rows, time zero and the duration anchor's approximate px/mm.
     */
    fun tickScale(mask: TraceExtractor.Mask, rows: IntArray, t0: Double,
                  anchorPxPerMm: Double): TickScale {
        var approx = anchorPxPerMm
        val wt = max(2, StripLocator.pyRound(TICK_WIDTH_MM * approx))
        val band = StripLocator.pyRound(TICK_BASE_BAND_MM * approx)
        val runs = Array(3) { baselineRuns(mask, rows[it], band) }
        val score = Array(3) { r ->
            val run = runs[r]
            DoubleArray(max(0, run.size - wt)) { x ->
                var m = run[x]
                for (i in x + 1 until x + wt) if (run[i] < m) m = run[i]
                m
            }
        }
        val cellMm = CELL_S * MM_PER_S

        val first = ArrayList<Double>()
        for (r in 0 until 3) {
            val xe = t0 + cellMm * approx
            val lo = max(0, (t0 + (1 - TICK_FIRST_WINDOW) * (xe - t0)).toInt())
            val hi = min(score[r].size, (t0 + (1 + TICK_FIRST_WINDOW) * (xe - t0)).toInt())
            if (hi <= lo) continue
            val j = argmaxFirst(score[r], lo, hi)
            if (score[r][j] >= TICK_MIN_HEIGHT_MM * approx) {
                first += (j + (wt - 1) / 2.0 - t0) / cellMm
            }
        }
        if (first.isNotEmpty()) approx = median(first.toDoubleArray())

        val est = ArrayList<Double>()
        val pos = ArrayList<Triple<Int, Int, Double>>()
        for (r in 0 until 3) {
            val sc = score[r]
            val run = runs[r]
            for (k in 1..3) {
                val xe = t0 + k * cellMm * approx
                val half = TICK_WINDOW * k * cellMm * approx
                val lo = max(0, (xe - half).toInt())
                val hi = min(sc.size, (xe + half).toInt())
                if (hi <= lo) continue
                val j = argmaxFirst(sc, lo, hi)
                if (sc[j] < TICK_MIN_HEIGHT_MM * approx) continue
                val a0 = max(0, j - 3 * wt)
                val b0 = min(run.size, j + 4 * wt)
                val cols = (a0 until b0).filter { run[it] >= 0.5 * sc[j] }
                var xc = j + (wt - 1) / 2.0
                if (cols.isNotEmpty()) {
                    val segs = ArrayList<List<Int>>()
                    var cur = arrayListOf(cols[0])
                    for (i in 1 until cols.size) {
                        if (cols[i] - cols[i - 1] > 1) { segs += cur; cur = arrayListOf() }
                        cur += cols[i]
                    }
                    segs += cur
                    val ref = xc
                    xc = segs.minByOrNull { s -> abs(s.average() - ref) }!!.average()
                }
                est += (xc - t0) / (k * cellMm)
                pos += Triple(r, k, Math.rint(xc * 10.0) / 10.0)
            }
        }
        if (est.size < TICK_MIN_FOUND) return TickScale(Double.NaN, est.size, Double.NaN, pos)
        val e = est.toDoubleArray()
        val med = median(e)
        return TickScale(med, e.size, 100.0 * (e.max() - e.min()) / med, pos)
    }

    /** `segment.segment_leads` with the tick knots. */
    fun segmentLeads(height: Int, rows: IntArray, t0: Double, pxPerMm: Double,
                     ticks: TickScale?): LeadLayout {
        val peaks = rows.sortedArray()
        check(peaks.size == 4) { "${peaks.size} rows found; the 3x4 + rhythm layout has 4" }
        val bands = Array(4) { i ->
            val y = peaks[i]
            val up = if (i > 0) (y - peaks[i - 1]) / 2.0 else (peaks[1] - peaks[0]) / 2.0
            val dn = if (i < 3) (peaks[i + 1] - y) / 2.0 else (y - peaks[2]) / 2.0
            intArrayOf(max(0, StripLocator.pyRound(y - up)), min(height, StripLocator.pyRound(y + dn)))
        }
        val pxPerS = pxPerMm * MM_PER_S
        val found = HashMap<Int, HashMap<Int, Double>>()
        ticks?.positions?.forEach { (r, k, x) -> found.getOrPut(r) { HashMap() }[k] = x }
        // A tick missing in one row borrows its column-mates' mean position.
        for (r in 0 until 3) {
            val fr = found[r] ?: continue
            for (k in 1..3) {
                if (k !in fr) {
                    val mates = found.filterKeys { it != r }.values.mapNotNull { it[k] }
                    if (mates.isNotEmpty()) fr[k] = mates.average()
                }
            }
        }
        val knots = ArrayList<Knots>()
        for (r in 0 until 4) {
            val ks: Map<Int, Double> = if (r < 3) found[r] ?: emptyMap() else {
                val m = HashMap<Int, Double>()
                for (k in 1..3) {
                    val xs = found.values.mapNotNull { it[k] }
                    if (xs.isNotEmpty()) m[k] = xs.average()
                }
                m
            }
            val xs = arrayListOf(t0)
            val ts = arrayListOf(0.0)
            for (k in ks.keys.sorted()) { xs += ks.getValue(k); ts += k * CELL_S }
            xs += xs.last() + (RECORDING_S - ts.last()) * pxPerS
            ts += RECORDING_S
            knots += Knots(xs.toDoubleArray(), ts.toDoubleArray())
        }
        fun xAt(r: Int, t: Double) = interp(t, knots[r].ts, knots[r].xs)
        val cells = ArrayList<LeadCell>()
        for (r in 0 until 3) for (k in 0 until 4) {
            cells += LeadCell(LAYOUT_3X4[r][k], r, k, k * CELL_S, CELL_S,
                StripLocator.pyRound(xAt(r, k * CELL_S)),
                StripLocator.pyRound(xAt(r, (k + 1) * CELL_S)),
                bands[r][0], bands[r][1], peaks[r].toDouble())
        }
        cells += LeadCell("II", 3, 0, 0.0, RECORDING_S, StripLocator.pyRound(t0),
            StripLocator.pyRound(xAt(3, RECORDING_S)), bands[3][0], bands[3][1],
            peaks[3].toDouble())
        return LeadLayout(cells, t0, pxPerS, peaks, bands, knots)
    }

    // ------------------------------------------------------- classical rows

    /** `segment.trace_positions_tracked`. */
    fun tracePositionsTracked(mask: TraceExtractor.Mask, pxPerMm: Double?): DoubleArray {
        val h = mask.rows
        val w = mask.cols
        val runs = Array(w) { x ->
            val list = ArrayList<IntArray>()
            var y = 0
            while (y < h) {
                if (mask.data[y * w + x] != 0.toByte()) {
                    val s = y
                    while (y < h && mask.data[y * w + x] != 0.toByte()) y++
                    list += intArrayOf(s, y)          // [start, end)
                } else y++
            }
            list
        }
        val out = DoubleArray(w) { Double.NaN }
        val single = runs.indices.filter { runs[it].size == 1 }
        if (single.isEmpty()) return TraceExtractor.centroids(mask)
        val segs = ArrayList<List<Int>>()
        var cur = arrayListOf(single[0])
        for (i in 1 until single.size) {
            if (single[i] - single[i - 1] > 1) { segs += cur; cur = arrayListOf() }
            cur += single[i]
        }
        segs += cur
        val seedSeg = segs.maxByOrNull { it.size }!!     // first of equal length, as max()
        val seed = seedSeg[seedSeg.size / 2]
        val jump = if (pxPerMm != null) TRACK_MAX_JUMP_MM * pxPerMm else Double.POSITIVE_INFINITY

        fun pick(rs: List<IntArray>, ref: IntArray): IntArray {
            var best = rs[0]
            var bd = -1
            for (r in rs) {
                val d = max(0, max(r[0] - ref[1], ref[0] - r[1]))
                if (bd < 0 || d < bd || (d == bd && (r[1] - r[0]) > (best[1] - best[0]))) {
                    best = r; bd = d
                }
            }
            return best
        }
        for (dir in intArrayOf(1, -1)) {
            var ref = runs[seed][0]
            var x = seed
            var skipped = 0
            while (x in 0 until w) {
                val rs = runs[x]
                if (rs.isNotEmpty()) {
                    val r = pick(rs, ref)
                    val d = max(0, max(r[0] - ref[1], ref[0] - r[1]))
                    if (d <= jump + min(skipped.toDouble(), jump)) {
                        out[x] = 0.5 * (r[0] + r[1] - 1)
                        ref = r
                        skipped = 0
                    } else skipped++
                } else skipped++
                x += dir
            }
        }
        return out
    }

    /**
     * The pure half of `segment.trace_row`: [mask] is the row band
     * (rows rowBands[row], columns [x0], x1) binarised and passed through
     * `isolate_trace`. Returns the row per band column, in page rows.
     */
    fun traceRowFromMask(mask: TraceExtractor.Mask, layout: LeadLayout, row: Int,
                         x0: Int): DoubleArray {
        val y0 = layout.rowBands[row][0]
        val rowCells = layout.cells.filter { it.row == row }
        val pxMm = layout.pxPerS / MM_PER_S
        val pos = DoubleArray(mask.cols) { Double.NaN }
        for ((i, c) in rowCells.withIndex()) {
            val a = max(0, c.x0 - x0)
            val b = if (i == rowCells.size - 1) mask.cols else max(a, c.x1 - x0)
            if (b <= a) continue
            val sub = StripLocator.sub(mask, 0, mask.rows, a, min(b, mask.cols))
            val t = tracePositionsTracked(sub, pxMm)
            for (j in t.indices) pos[a + j] = t[j] + y0
        }
        if (row < 3) {
            val guard = SEPARATOR_GUARD_MM * layout.pxPerS / MM_PER_S
            for (c in rowCells.drop(1)) {
                val xb = (c.x0 - x0).toDouble()
                val lo = floor(xb - guard).toInt()
                val hi = ceil(xb + guard).toInt()
                for (i in max(0, lo) until max(0, min(pos.size, hi + 1))) pos[i] = Double.NaN
            }
        }
        return pos
    }

    /** The row band's column range for [row]: (x0, x1), x1 clipped to [width]. */
    fun rowBandColumns(layout: LeadLayout, row: Int, width: Int): IntArray {
        val rc = layout.cells.filter { it.row == row }
        return intArrayOf(rc.minOf { it.x0 }, min(rc.maxOf { it.x1 }, width))
    }

    // --------------------------------------------------- cut into leads

    /** `sheet.on_sheet_columns`: (rows per whole column, first column). */
    fun onSheetColumns(xs: DoubleArray, rows: DoubleArray): Pair<DoubleArray, Int> {
        val g0 = ceil(xs.first()).toInt()
        val g1 = floor(xs.last()).toInt()
        val n = max(0, g1 - g0 + 1)
        val finIdx = rows.indices.filter { rows[it].isFinite() }
        if (finIdx.size < 2) return DoubleArray(n) { Double.NaN } to g0
        val fx = DoubleArray(finIdx.size) { xs[finIdx[it]] }
        val fy = DoubleArray(finIdx.size) { rows[finIdx[it]] }
        val nanFlag = DoubleArray(rows.size) { if (rows[it].isFinite()) 0.0 else 1.0 }
        val out = DoubleArray(n) { i ->
            val g = (g0 + i).toDouble()
            if (interp(g, xs, nanFlag) > 0.5) Double.NaN
            else interp(g, fx, fy, Double.NaN, Double.NaN)
        }
        return out to g0
    }

    /** `sheet.cell_signal`: a cell's slice of a row trace -> mV at FS. */
    fun cellSignal(pos: DoubleArray, xOrigin: Int, cell: LeadCell, pxPerMmY: Double,
                   layout: LeadLayout): DoubleArray {
        val n = Math.rint(cell.durationS * FS).toInt()
        val xs = ArrayList<Double>()
        val rs = ArrayList<Double>()
        for (i in pos.indices) {
            val x = i + xOrigin
            if (x >= cell.x0 && x < cell.x1 && pos[i].isFinite()) { xs += x.toDouble(); rs += pos[i] }
        }
        if (xs.size < 10) return DoubleArray(n) { Double.NaN }
        val med = median(rs.toDoubleArray())
        val t = DoubleArray(xs.size) { layout.timeAt(cell.row, xs[it]) - cell.tStartS }
        val mv = DoubleArray(rs.size) { -(rs[it] - med) / (pxPerMmY * MM_PER_MV) }
        return DoubleArray(n) { interp(it / FS, t, mv, Double.NaN, Double.NaN) }
    }
}
