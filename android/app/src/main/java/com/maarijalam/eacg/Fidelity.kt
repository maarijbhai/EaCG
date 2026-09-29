package com.maarijalam.eacg

import kotlin.math.sqrt

/**
 * Reconstruction fidelity, scored on the handset. Port of `ecgproto.evaluate`.
 *
 * WHY THIS RUNS ON THE DEVICE AT ALL
 * The desktop already computes these numbers, and copying them across would
 * be easier and worthless: a fidelity figure carried over from the desktop
 * cannot detect a porting bug, which is the entire risk the Kotlin side
 * carries. Scoring here against a committed source waveform makes the
 * handset's own answer checkable, which is the same move the rotation slider
 * makes for deskew.
 *
 * THE METRICS FAIL IN DIFFERENT DIRECTIONS AND MUST BE READ TOGETHER
 * That is not a slogan, it is how a fault gets localised:
 *
 *  - [pcc] is scale- and offset-invariant, so a correctly shaped but
 *    mis-calibrated trace scores near 1.0. High PCC alone proves the tracer
 *    followed the ink, and nothing about the calibration.
 *  - [rmseMv] is in millivolts and amplitude-sensitive, so it depends on the
 *    gain being right. High RMSE with high PCC localises the fault to
 *    calibration.
 *  - [gainRatio] separates a systematic amplitude error from broadband
 *    noise. The classical tracer is known to under-read: the column-centroid
 *    of a steep QRS sits at the midpoint of a tall ink run, so the recovered
 *    R amplitude is biased low. Measured on the desktop at 0.60 at 4 px/mm
 *    and 0.96 at 16, and it is the dominant error of the control condition.
 *
 * Low PCC means the tracer lost the waveform, and no amount of calibration
 * repairs that.
 */
object Fidelity {

    /** All four numbers from one comparison, over [nCompared] samples. */
    class Scores(
        val pcc: Double,
        val rmseMv: Double,
        val gainRatio: Double,
        val nCompared: Int
    )

    /**
     * Truncate both to the shorter length. `evaluate.align_length`.
     *
     * The recovered signal's length is set by the image width and the
     * recovered pitch, so it will not match the reference exactly even when
     * everything is right — a millimetre of margin is a few samples.
     */
    fun score(reference: DoubleArray, test: DoubleArray): Scores {
        val n = minOf(reference.size, test.size)
        require(n >= 2) { "need at least two samples to compare, got $n" }
        val a = DoubleArray(n) { reference[it] }
        val b = DoubleArray(n) { test[it] }
        return Scores(pcc(a, b), rmseMv(a, b), gainRatio(a, b), n)
    }

    /** `run_photo.ALIGN_HALF_RANGE` and `ALIGN_STEP`. */
    const val ALIGN_HALF_RANGE = 0.10
    const val ALIGN_STEP = 0.0025

    /**
     * The best agreement a trace reaches with the truth over a time-rescale
     * and a lag. `run_photo.best_alignment`.
     *
     * [scale] is the factor the trace is resampled by: below 1 it had to be
     * stretched, so its time base was short and the px/mm that made it too
     * large -- [pxPerMmErrorPct] is that error, measured against the truth
     * waveform itself. This is the calibration error the harness can vouch
     * for. The corner-derived "true" px/mm is only as true as the corners:
     * on phone run 3 of PTB-XL 40 it put the duration anchor at +8.9% where
     * this put it at +3%, because the corners sat 6-9 mm inside the sheet.
     *
     * [atBound] marks an optimum on the edge of the search, which is not an
     * optimum; its scale must not be read as a measurement.
     */
    class Alignment(
        val pcc: Double,
        val scale: Double,
        val lagSamples: Int,
        val nCompared: Int,
        val atBound: Boolean,
        /** Smallest overlap, samples, at which the result means anything. */
        val minOverlap: Int = 0
    ) {
        /**
         * Off the search bound AND over most of the recording. A calibration
         * 5x wrong gives a 2 s trace; the +-10% scale search cannot see a 5x
         * error and reports a local match of those 2 s at a scale near 1 --
         * an honest-looking "-0.5% vs truth" for a 400% error, which is what
         * the ingester printed for phone run 2's grid row before this guard.
         * `run_photo.MIN_OVERLAP_S` exists for the same trap.
         */
        val valid: Boolean get() = !atBound && nCompared >= minOverlap
        val pxPerMmErrorPct: Double? get() = if (valid) 100.0 * (1.0 / scale - 1.0) else null
    }

    /** `run_photo.MIN_OVERLAP_S`. */
    const val MIN_OVERLAP_S = 8.5

    fun bestAlignment(reference: DoubleArray, test: DoubleArray, minOverlap: Int = 0): Alignment {
        val nScales = Math.rint(2 * ALIGN_HALF_RANGE / ALIGN_STEP).toInt() + 1
        val scales = DoubleArray(nScales) { (1.0 - ALIGN_HALF_RANGE) + it * ALIGN_STEP }
        val ref = fillNan(reference)
        val tst = fillNan(test)
        var bestV = -1.0
        var bestS = 1.0
        var bestLag = 0
        var bestK = 0
        for (s in scales) {
            val n = Math.rint(tst.size / s).toInt()
            if (n < 100) continue
            val res = DoubleArray(n) { interpClamped(tst, it * s) }
            val m = minOf(ref.size, res.size)
            if (m < 200) continue
            val lag = argmaxCorrelationLag(ref, res, m)
            // The Python's slices: lag >= 0 compares ref[lag:] with
            // res[:len(ref) - lag]; lag < 0 compares ref[:len(res) + lag]
            // with res[-lag:]. k is the shorter of each pair.
            val x0: Int; val y0: Int; val k: Int
            if (lag >= 0) {
                x0 = lag; y0 = 0
                k = minOf(ref.size - lag, res.size)
            } else {
                x0 = 0; y0 = -lag
                k = minOf(ref.size, res.size + lag)
            }
            if (k < 200) continue
            val v = pccRange(ref, x0, res, y0, k)
            if (v.isNaN()) continue
            if (v > bestV) { bestV = v; bestS = s; bestLag = lag; bestK = k }
        }
        val atBound = scales.size > 1 && (
            kotlin.math.abs(bestS - scales.first()) < ALIGN_STEP / 2 ||
                kotlin.math.abs(bestS - scales.last()) < ALIGN_STEP / 2)
        return Alignment(bestV, bestS, bestLag, bestK, atBound, minOverlap)
    }

    /** `np.interp(x, arange(len(y)), y)`: linear, held flat past either end. */
    private fun interpClamped(y: DoubleArray, x: Double): Double {
        if (x <= 0.0) return y[0]
        val last = y.size - 1
        if (x >= last) return y[last]
        val i = kotlin.math.floor(x).toInt()
        val f = x - i
        return y[i] + (y[i + 1] - y[i]) * f
    }

    /**
     * The lag of `scipy.signal.correlate(a, b, "full")`'s maximum over the
     * first [m] samples of each, mean-removed: argmax index - (m - 1), first
     * occurrence, scanning lags from -(m-1) upward as scipy's output does.
     * By FFT, as scipy's "fft" method is, so it stays cheap on the handset.
     */
    private fun argmaxCorrelationLag(ref: DoubleArray, res: DoubleArray, m: Int): Int {
        var ma = 0.0; var mb = 0.0
        for (i in 0 until m) { ma += ref[i]; mb += res[i] }
        ma /= m; mb /= m
        var size = 1
        while (size < 2 * m - 1) size = size shl 1
        val ar = DoubleArray(size); val ai = DoubleArray(size)
        val br = DoubleArray(size); val bi = DoubleArray(size)
        for (i in 0 until m) { ar[i] = ref[i] - ma; br[i] = res[i] - mb }
        fft(ar, ai, false); fft(br, bi, false)
        // A * conj(B): c[L] = sum_i a[i+L] b[i].
        for (i in 0 until size) {
            val re = ar[i] * br[i] + ai[i] * bi[i]
            val im = ai[i] * br[i] - ar[i] * bi[i]
            ar[i] = re; ai[i] = im
        }
        fft(ar, ai, true)
        var best = Double.NEGATIVE_INFINITY
        var bestLag = 0
        for (lag in -(m - 1)..(m - 1)) {
            val v = ar[if (lag >= 0) lag else size + lag]
            if (v > best) { best = v; bestLag = lag }
        }
        return bestLag
    }

    /** In-place iterative radix-2 FFT; [inverse] scales by 1/n. */
    private fun fft(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = 2 * Math.PI / len * (if (inverse) 1 else -1)
            val wr = kotlin.math.cos(ang); val wi = kotlin.math.sin(ang)
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until len / 2) {
                    val ur = re[i + k]; val ui = im[i + k]
                    val xr = re[i + k + len / 2]; val xi = im[i + k + len / 2]
                    val vr = xr * cr - xi * ci; val vi = xr * ci + xi * cr
                    re[i + k] = ur + vr; im[i + k] = ui + vi
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi
                    val nr = cr * wr - ci * wi
                    ci = cr * wi + ci * wr; cr = nr
                }
                i += len
            }
            len = len shl 1
        }
        if (inverse) for (i in 0 until n) { re[i] /= n; im[i] /= n }
    }

    /** Pearson correlation of a[a0 ..< a0+k] and b[b0 ..< b0+k]; NaN if flat. */
    private fun pccRange(a: DoubleArray, a0: Int, b: DoubleArray, b0: Int, k: Int): Double {
        var ma = 0.0; var mb = 0.0
        for (i in 0 until k) { ma += a[a0 + i]; mb += b[b0 + i] }
        ma /= k; mb /= k
        var sab = 0.0; var saa = 0.0; var sbb = 0.0
        for (i in 0 until k) {
            val da = a[a0 + i] - ma; val db = b[b0 + i] - mb
            sab += da * db; saa += da * da; sbb += db * db
        }
        if (saa <= 0.0 || sbb <= 0.0) return Double.NaN
        return sab / kotlin.math.sqrt(saa * sbb)
    }

    /**
     * Linear interpolation across NaN gaps, the ends held flat.
     * `run_photo.fill_nan`, which `score_kit_photo.score_signal` applies
     * before every comparison.
     *
     * A traced strip keeps a gap longer than the tracer's own interpolation
     * limit as NaN, and one NaN makes every sum above NaN. The first A30s
     * photograph scored PCC, RMSE and gain all NaN for exactly this: 328
     * missing samples in 3076. Filled here, in the scoring harness only, so
     * the exported waveform still shows where the ink was lost.
     */
    fun fillNan(x: DoubleArray): DoubleArray {
        val ok = x.indices.filter { !x[it].isNaN() }
        if (ok.isEmpty()) return x.copyOf()
        val out = x.copyOf()
        var k = 0
        for (i in out.indices) {
            if (!out[i].isNaN()) continue
            while (k + 1 < ok.size && ok[k + 1] < i) k++
            val lo = ok[k]
            out[i] = when {
                i < ok.first() -> x[ok.first()]
                i > ok.last() -> x[ok.last()]
                else -> {
                    val hi = ok[k + 1]
                    x[lo] + (x[hi] - x[lo]) * (i - lo).toDouble() / (hi - lo)
                }
            }
        }
        return out
    }

    /**
     * Pearson correlation. `evaluate.pcc`.
     *
     * Returns NaN when either input is constant, rather than 0: a flat trace
     * has no correlation to report, and reporting zero would read as
     * "measured, and uncorrelated".
     */
    fun pcc(reference: DoubleArray, test: DoubleArray): Double {
        val n = reference.size
        var ma = 0.0
        var mb = 0.0
        for (i in 0 until n) { ma += reference[i]; mb += test[i] }
        ma /= n; mb /= n
        var sab = 0.0
        var saa = 0.0
        var sbb = 0.0
        for (i in 0 until n) {
            val da = reference[i] - ma
            val db = test[i] - mb
            sab += da * db
            saa += da * da
            sbb += db * db
        }
        if (saa <= 0.0 || sbb <= 0.0) return Double.NaN
        return sab / sqrt(saa * sbb)
    }

    /** Root mean squared error, millivolts. `evaluate.rmse_mv`. */
    fun rmseMv(reference: DoubleArray, test: DoubleArray): Double {
        var acc = 0.0
        for (i in reference.indices) {
            val d = test[i] - reference[i]
            acc += d * d
        }
        return sqrt(acc / reference.size)
    }

    /**
     * Least-squares amplitude scale, test against reference.
     * `evaluate.gain_ratio`.
     *
     * The slope of `test ~ a * reference + b`, computed as covariance over
     * variance. **The intercept matters and is why both series are centred**:
     * a baseline error is a DC offset, and without fitting an intercept it
     * would contaminate the gain estimate. The two calibration faults are
     * separable and worth separating — the baseline comes from a median that
     * is documented to drift on leads with sustained large deflections.
     *
     * One means the amplitude is right; 0.87 means the recovered trace is 13%
     * short everywhere, which is a calibration or peak-clipping fault rather
     * than noise. Only interpretable once PCC says the shape is right.
     */
    fun gainRatio(reference: DoubleArray, test: DoubleArray): Double {
        val n = reference.size
        var ma = 0.0
        var mb = 0.0
        for (i in 0 until n) { ma += reference[i]; mb += test[i] }
        ma /= n; mb /= n
        var cov = 0.0
        var varr = 0.0
        for (i in 0 until n) {
            val da = reference[i] - ma
            cov += da * (test[i] - mb)
            varr += da * da
        }
        // Population variance, matching numpy's default ddof=0. Both sums are
        // divided by the same n, so it cancels -- but only if neither is
        // divided by n-1, which is the easy way to get this subtly wrong.
        if (varr <= 0.0) return Double.NaN
        return cov / varr
    }
}
