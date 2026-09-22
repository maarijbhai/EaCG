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
