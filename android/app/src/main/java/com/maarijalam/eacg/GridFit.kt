package com.maarijalam.eacg

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Page geometry from the printed 5 mm grid: the port of `ecgproto/gridfit.py`,
 * step for step (golden vectors: GridPortFixturesTest).
 *
 * The four-corner rectification spreads every millimetre of corner error over
 * the page as scale and keystone -- on a full sheet, a time base that drifts
 * lead by lead. Fitting the homography to every 5 mm rule sample instead took
 * page distortion from 0.27-1.75 mm RMS to 0.035-0.094 mm on the PTB-XL 40
 * photographs and median lead PCC from 0.865 to 0.966 (b3, desktop, the
 * phone's path; results/ml/phone_path_*_grid.csv).
 *
 * On the corner-rectified page: redness response; per band of rows (10 mm,
 * every 5 mm) the column profile is the median over the band; rules are its
 * peaks; peaks off the local pitch are dropped and the rest indexed from
 * their gaps; bands are chained outward from the middle so one rule keeps one
 * index; then a homography to the ideal lattice (rule i at u = 5 mm * i),
 * linear and Huber-reweighted. Same for the horizontal rules on the
 * transpose. The photograph is then warped once onto the regular page.
 */
object GridFit {

    const val PITCH_MM = 5.0
    const val BAND_MM = 10.0
    const val BAND_STEP_MM = 5.0
    const val EDGE_MARGIN_MM = 3.0
    const val MIN_REL_PROMINENCE = 0.3
    const val ON_PITCH_FRACTION = 0.2
    const val MATCH_FRACTION = 0.3
    const val HUBER_PX = 1.5
    const val MIN_RULES = 8

    /** Redness r - (g+b)/2 clipped at 0, row-major, as gridfit.redness (float32). */
    class Response(val v: FloatArray, val rows: Int, val cols: Int) {
        fun at(axis: Int, along: Int, across: Int): Float =
            if (axis == 0) v[along * cols + across] else v[across * cols + along]
        fun h(axis: Int) = if (axis == 0) rows else cols
        fun w(axis: Int) = if (axis == 0) cols else rows
    }

    class Fit(
        /** Rectified page -> regular page, row-major 3x3. */
        val h: DoubleArray,
        val outW: Int, val outH: Int,
        /** The regular page's scale, exact by construction. */
        val pxPerMm: Double,
        val rmsMm: Double,
        val nVertical: Int, val nHorizontal: Int,
        val nSamples: Int,
        val inlierFraction: Double
    )

    class Bands(val centres: DoubleArray, val profiles: List<FloatArray>, val margin: Int)

    // ------------------------------------------------------------ response

    /** From interleaved 8-bit pixels; [rIdx]/[gIdx]/[bIdx] say which channel is which. */
    fun redness(px: ByteArray, rows: Int, cols: Int, channels: Int,
                rIdx: Int, gIdx: Int, bIdx: Int): Response {
        val out = FloatArray(rows * cols)
        for (i in 0 until rows * cols) {
            val o = i * channels
            val r = (px[o + rIdx].toInt() and 0xFF).toFloat()
            val g = (px[o + gIdx].toInt() and 0xFF).toFloat()
            val b = (px[o + bIdx].toInt() and 0xFF).toFloat()
            out[i] = max(r - 0.5f * (b + g), 0f)
        }
        return Response(out, rows, cols)
    }

    /** An RGBA page Mat (the app's convention, index 0 = red). */
    fun redness(rgba: Mat): Response {
        require(rgba.type() == CvType.CV_8UC4) { "need an RGBA 8-bit page" }
        val b = ByteArray((rgba.total() * 4).toInt())
        rgba.get(0, 0, b)
        return redness(b, rgba.rows(), rgba.cols(), 4, 0, 1, 2)
    }

    // --------------------------------------------------------------- bands

    /** gridfit.band_profiles: centres, median profiles over each band, margin. */
    fun bandProfiles(resp: Response, pitchPx: Double, axis: Int): Bands {
        val h = resp.h(axis)
        val w = resp.w(axis)
        val pxMm = pitchPx / PITCH_MM
        val band = StripLocator.pyRound(BAND_MM * pxMm)
        val step = StripLocator.pyRound(BAND_STEP_MM * pxMm)
        val margin = StripLocator.pyRound(EDGE_MARGIN_MM * pxMm)
        val centres = ArrayList<Double>()
        val profiles = ArrayList<FloatArray>()
        var y0 = margin
        val col = FloatArray(band)
        while (y0 <= h - margin - band) {
            val prof = FloatArray(max(0, w - 2 * margin))
            for (j in prof.indices) {
                for (k in 0 until band) col[k] = resp.at(axis, y0 + k, margin + j)
                prof[j] = medianF(col)
            }
            centres += y0 + band / 2.0
            profiles += prof
            y0 += step
        }
        return Bands(centres.toDoubleArray(), profiles, margin)
    }

    /** np.median of float32 values (sorts [v] in place). */
    internal fun medianF(v: FloatArray): Float {
        v.sort()
        val n = v.size
        return if (n % 2 == 1) v[n / 2] else (v[n / 2 - 1] + v[n / 2]) / 2f
    }

    // --------------------------------------------------------------- peaks

    /** cv2.blur of a 1-row float32 profile, (k, 1), BORDER_REFLECT_101. */
    internal fun boxBlur(p: FloatArray, k: Int): FloatArray {
        val n = p.size
        val r = k / 2
        fun at(i: Int): Float {
            if (n == 1) return p[0]
            var j = i
            while (j < 0 || j >= n) j = if (j < 0) -j else 2 * (n - 1) - j
            return p[j]
        }
        return FloatArray(n) { i ->
            var s = 0.0
            for (d in -r..r) s += at(i + d)
            (s / k).toFloat()
        }
    }

    /**
     * scipy.signal.peak_prominences with wlen=None: for each peak, the lowest
     * point on each side before a higher sample, and the higher of the two.
     */
    internal fun prominences(x: DoubleArray, peaks: List<Int>): DoubleArray =
        DoubleArray(peaks.size) { n ->
            val p = peaks[n]
            var leftMin = x[p]
            var i = p
            while (i >= 0 && x[i] <= x[p]) { if (x[i] < leftMin) leftMin = x[i]; i-- }
            var rightMin = x[p]
            i = p
            while (i <= x.size - 1 && x[i] <= x[p]) { if (x[i] < rightMin) rightMin = x[i]; i++ }
            x[p] - max(leftMin, rightMin)
        }

    /** The detrended profile peaks are found on (float32 like the Python). */
    internal fun detrend(profile: FloatArray, pitchPx: Double): FloatArray {
        val k = StripLocator.pyRound(pitchPx) or 1
        val b = boxBlur(profile, k)
        return FloatArray(profile.size) { profile[it] - b[it] }
    }

    /** np.percentile(v, q), linear interpolation. */
    internal fun percentile(v: DoubleArray, q: Double): Double {
        val s = v.sortedArray()
        val pos = (s.size - 1) * q / 100.0
        val lo = floor(pos).toInt()
        val hi = min(lo + 1, s.size - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    }

    /** gridfit._band_peaks: sub-pixel rule positions in one profile. */
    fun bandPeaks(profile: FloatArray, pitchPx: Double): DoubleArray {
        val pf = detrend(profile, pitchPx)
        val p = DoubleArray(pf.size) { pf[it].toDouble() }
        val raw = StripLocator.findPeaks(p, max(2, (0.6 * pitchPx).toInt()))
        if (raw.size < 3) return DoubleArray(0)
        val prom = prominences(p, raw)
        val cut = MIN_REL_PROMINENCE * percentile(prom, 80.0)
        val kept = raw.filterIndexed { i, _ -> prom[i] >= cut }.filter { it > 0 && it < p.size - 1 }
        return DoubleArray(kept.size) { n ->
            val i = kept[n]
            val a = pf[i - 1]; val b = pf[i]; val c = pf[i + 1]
            val den = a - 2f * b + c
            val off = if (abs(den) > 1e-9f) (0.5f * (a - c) / den).toDouble() else 0.0
            i + off.coerceIn(-0.5, 0.5)
        }
    }

    // ------------------------------------------------------------ indexing

    /** gridfit._index_band: off-pitch peaks dropped, then indexed from the gaps. */
    fun indexBand(xsIn: DoubleArray, pitchPx: Double): Pair<DoubleArray, IntArray> {
        if (xsIn.size < 3) return DoubleArray(0) to IntArray(0)
        var xs = xsIn
        val near = (1 until xs.size).map { xs[it] - xs[it - 1] }
            .filter { it > 0.8 * pitchPx && it < 1.2 * pitchPx }
        val p = if (near.size >= 3) SheetLayout.median(near.toDoubleArray()) else pitchPx
        repeat(3) {
            val ok = BooleanArray(xs.size - 1) { i ->
                val f = (xs[i + 1] - xs[i]) / p
                abs(f - Math.rint(f)) < ON_PITCH_FRACTION && Math.rint(f) >= 1
            }
            val keep = BooleanArray(xs.size) { i -> (i > 0 && ok[i - 1]) || (i < ok.size && ok[i]) }
            val nKeep = keep.count { it }
            if (nKeep == xs.size || nKeep < 3) return@repeat
            xs = xs.filterIndexed { i, _ -> keep[i] }.toDoubleArray()
        }
        val idx = IntArray(xs.size)
        for (i in 1 until xs.size) idx[i] = idx[i - 1] + max(Math.rint((xs[i] - xs[i - 1]) / p).toInt(), 1)
        return xs to idx
    }

    /** gridfit._chain: the index offset putting [cur] on [prev]'s numbering, or null. */
    fun chain(prevX: DoubleArray, prevI: IntArray, prevOff: Int,
              curX: DoubleArray, curI: IntArray, pitchPx: Double): Int? {
        if (prevX.isEmpty() || curX.isEmpty()) return null
        val d = ArrayList<Int>()
        for (n in curX.indices) {
            val cx = curX[n]
            var j = 0
            while (j < prevX.size && prevX[j] < cx) j++          // searchsorted, left
            j = j.coerceIn(1, max(1, prevX.size - 1))
            if (prevX.size == 1) j = 0
            val near = if (j >= 1 && abs(prevX[j - 1] - cx) < abs(prevX[j] - cx)) j - 1 else j
            if (abs(prevX[near] - cx) < MATCH_FRACTION * pitchPx) d += prevI[near] + prevOff - curI[n]
        }
        if (d.size < 3) return null
        val counts = d.groupingBy { it }.eachCount()
        val best = counts.values.max()
        return counts.filterValues { it == best }.keys.min()      // np.unique + argmax
    }

    class Samples(val along: DoubleArray, val across: DoubleArray, val index: IntArray,
                  val offsets: Map<Int, Int>)

    /** gridfit._rule_samples from profiles: (position, band centre, rule index). */
    fun ruleSamples(b: Bands, pitchPx: Double): Samples {
        val n = b.profiles.size
        val xs = ArrayList<DoubleArray>()
        val ids = ArrayList<IntArray>()
        for (prof in b.profiles) {
            val pk = bandPeaks(prof, pitchPx)
            val (k, i) = indexBand(DoubleArray(pk.size) { pk[it] + b.margin }, pitchPx)
            xs += k; ids += i
        }
        val offsets = LinkedHashMap<Int, Int>()
        if (n == 0) return Samples(DoubleArray(0), DoubleArray(0), IntArray(0), offsets)
        val mid = n / 2
        offsets[mid] = 0
        for (order in listOf((mid + 1 until n).toList(), (mid - 1 downTo 0).toList())) {
            for (bi in order) {
                val prev = if (bi > mid) bi - 1 else bi + 1
                val off = chain(xs[prev], ids[prev], offsets.getValue(prev), xs[bi], ids[bi], pitchPx)
                    ?: break
                offsets[bi] = off
            }
        }
        val a = ArrayList<Double>(); val c = ArrayList<Double>(); val ix = ArrayList<Int>()
        for ((bi, off) in offsets) {
            for (k in xs[bi].indices) { a += xs[bi][k]; c += b.centres[bi]; ix += ids[bi][k] + off }
        }
        return Samples(a.toDoubleArray(), c.toDoubleArray(), ix.toIntArray(), offsets)
    }

    // -------------------------------------------------------- homography

    /** Least squares by Householder QR (as np.linalg.lstsq, full column rank). */
    internal fun lstsq(a: Array<DoubleArray>, bIn: DoubleArray): DoubleArray {
        val m = a.size
        val n = a[0].size
        val q = Array(m) { a[it].copyOf() }
        val b = bIn.copyOf()
        for (k in 0 until n) {
            var norm = 0.0
            for (i in k until m) norm += q[i][k] * q[i][k]
            norm = sqrt(norm)
            if (norm == 0.0) continue
            val alpha = if (q[k][k] > 0) -norm else norm
            val v = DoubleArray(m - k) { q[k + it][k] }
            v[0] -= alpha
            var vv = 0.0
            for (x in v) vv += x * x
            if (vv == 0.0) continue
            for (j in k until n) {
                var s = 0.0
                for (i in v.indices) s += v[i] * q[k + i][j]
                s = 2 * s / vv
                for (i in v.indices) q[k + i][j] -= s * v[i]
            }
            var s = 0.0
            for (i in v.indices) s += v[i] * b[k + i]
            s = 2 * s / vv
            for (i in v.indices) b[k + i] -= s * v[i]
        }
        val x = DoubleArray(n)
        for (k in n - 1 downTo 0) {
            var s = b[k]
            for (j in k + 1 until n) s -= q[k][j] * x[j]
            x[k] = s / q[k][k]
        }
        return x
    }

    /** gridfit._fit_homography: u samples constrain row 1, v samples row 2. */
    internal fun fitHomography(xu: DoubleArray, yu: DoubleArray, u: DoubleArray,
                               xv: DoubleArray, yv: DoubleArray, v: DoubleArray,
                               wu: DoubleArray, wv: DoubleArray): DoubleArray {
        val rows = ArrayList<DoubleArray>(u.size + v.size)
        val rhs = ArrayList<Double>(u.size + v.size)
        for (i in u.indices) {
            val w = wu[i]
            rows += doubleArrayOf(xu[i], yu[i], 1.0, 0.0, 0.0, 0.0, -u[i] * xu[i], -u[i] * yu[i])
                .map { it * w }.toDoubleArray()
            rhs += u[i] * w
        }
        for (i in v.indices) {
            val w = wv[i]
            rows += doubleArrayOf(0.0, 0.0, 0.0, xv[i], yv[i], 1.0, -v[i] * xv[i], -v[i] * yv[i])
                .map { it * w }.toDoubleArray()
            rhs += v[i] * w
        }
        val s = DoubleArray(8) { j -> rows.maxOf { abs(it[j]) }.let { m -> if (m == 0.0) 1.0 else m } }
        val a = Array(rows.size) { i -> DoubleArray(8) { j -> rows[i][j] / s[j] } }
        val h = lstsq(a, rhs.toDoubleArray())
        return DoubleArray(9) { if (it < 8) h[it] / s[it] else 1.0 }
    }

    internal fun apply(h: DoubleArray, x: Double, y: Double): Pair<Double, Double> {
        val w = h[6] * x + h[7] * y + h[8]
        return (h[0] * x + h[1] * y + h[2]) / w to (h[3] * x + h[4] * y + h[5]) / w
    }

    /** The pitch refined from the middle third's median profile. */
    fun refinePitch(midProfile: FloatArray, guessPxPerMm: Double): Double {
        val pitch0 = PITCH_MM * guessPxPerMm
        val pk = bandPeaks(midProfile, pitch0)
        val g = (1 until pk.size).map { pk[it] - pk[it - 1] }
            .filter { it > 0.7 * pitch0 && it < 1.3 * pitch0 }
        return if (g.size >= 5) SheetLayout.median(g.toDoubleArray()) else pitch0
    }

    /** The middle third's profile: median over rows h/3 .. 2h/3, every column. */
    fun midProfile(resp: Response): FloatArray {
        val a = resp.rows / 3
        val b = 2 * resp.rows / 3
        val col = FloatArray(b - a)
        return FloatArray(resp.cols) { j ->
            for (i in a until b) col[i - a] = resp.v[i * resp.cols + j]
            medianF(col)
        }
    }

    /** Everything after the profiles: samples, the reweighted fit, placement. */
    fun fitFromBands(vertical: Bands, horizontal: Bands, pitchPx: Double,
                     pageW: Int, pageH: Int): Fit {
        val sv = ruleSamples(vertical, pitchPx)                 // (x, y, i)
        val st = ruleSamples(horizontal, pitchPx)               // (y, x, j): transposed
        if (sv.index.isEmpty() || st.index.isEmpty()) throw IllegalStateException("grid fit: no rules found")
        val nv = sv.index.distinct().size
        val nh = st.index.distinct().size
        if (nv < MIN_RULES || nh < MIN_RULES)
            throw IllegalStateException("grid fit: $nv vertical / $nh horizontal rules, need $MIN_RULES")
        val outMm = pitchPx / PITCH_MM
        val p = PITCH_MM * outMm
        val xu = sv.along; val yu = sv.across; val u = DoubleArray(sv.index.size) { sv.index[it] * p }
        val xv = st.across; val yv = st.along; val v = DoubleArray(st.index.size) { st.index[it] * p }
        var wu = DoubleArray(u.size) { 1.0 }
        var wv = DoubleArray(v.size) { 1.0 }
        var h = DoubleArray(9)
        var ru = DoubleArray(0); var rv = DoubleArray(0)
        repeat(8) {
            h = fitHomography(xu, yu, u, xv, yv, v, wu, wv)
            ru = DoubleArray(u.size) { apply(h, xu[it], yu[it]).first - u[it] }
            rv = DoubleArray(v.size) { apply(h, xv[it], yv[it]).second - v[it] }
            wu = DoubleArray(u.size) { min(1.0, HUBER_PX / max(abs(ru[it]), 1e-9)) }
            wv = DoubleArray(v.size) { min(1.0, HUBER_PX / max(abs(rv[it]), 1e-9)) }
        }
        val res = ru + rv
        val inl = res.filter { abs(it) < 3 * HUBER_PX }
        val quad = listOf(0.0 to 0.0, pageW - 1.0 to 0.0, pageW - 1.0 to pageH - 1.0, 0.0 to pageH - 1.0)
            .map { apply(h, it.first, it.second) }
        val loX = quad.minOf { it.first }; val loY = quad.minOf { it.second }
        val hiX = quad.maxOf { it.first }; val hiY = quad.maxOf { it.second }
        // T @ H, T a translation by -lo.
        val ht = doubleArrayOf(h[0] - loX * h[6], h[1] - loX * h[7], h[2] - loX * h[8],
            h[3] - loY * h[6], h[4] - loY * h[7], h[5] - loY * h[8], h[6], h[7], h[8])
        return Fit(ht, ceil(hiX - loX).toInt() + 1, ceil(hiY - loY).toInt() + 1, outMm,
            sqrt(inl.sumOf { it * it } / inl.size) / outMm, nv, nh, res.size,
            inl.size.toDouble() / res.size)
    }

    /** gridfit.fit_grid on a page's response. */
    fun fit(resp: Response, guessPxPerMm: Double): Fit {
        val pitch = refinePitch(midProfile(resp), guessPxPerMm)
        return fitFromBands(bandProfiles(resp, pitch, 0), bandProfiles(resp, pitch, 1),
            pitch, resp.cols, resp.rows)
    }

    /**
     * gridfit.regularise: warp the PHOTOGRAPH straight onto the regular page
     * (one resampling), through the corner rectification and the fit.
     */
    fun regularise(photo: Mat, corners: DoubleArray, rectW: Int, rectH: Int, f: Fit): Mat {
        val from = MatOfPoint2f(Point(corners[0], corners[1]), Point(corners[2], corners[3]),
            Point(corners[4], corners[5]), Point(corners[6], corners[7]))
        val to = MatOfPoint2f(Point(0.0, 0.0), Point(rectW - 1.0, 0.0),
            Point(rectW - 1.0, rectH - 1.0), Point(0.0, rectH - 1.0))
        val p = Imgproc.getPerspectiveTransform(from, to)
        val hm = Mat(3, 3, CvType.CV_64F).apply { put(0, 0, *f.h) }
        val m = Mat()
        Core.gemm(hm, p, 1.0, Mat(), 0.0, m)
        val out = Mat()
        Imgproc.warpPerspective(photo, out, m, Size(f.outW.toDouble(), f.outH.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
        from.release(); to.release(); p.release(); hm.release(); m.release()
        return out
    }
}
