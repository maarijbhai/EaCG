package com.maarijalam.eacg

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI

/**
 * Recovery of the spatial scale of an ECG printout, in pixels per millimetre.
 *
 * WHY THIS EXISTS
 * The recovered waveform must be expressed in millivolts and seconds, not
 * pixels. The conversion requires knowing how many pixels correspond to one
 * physical millimetre on the page. For a scanned image this can be derived
 * from the scanner DPI, but for a camera-captured image no fixed
 * correspondence exists between a pixel and a physical millimetre: the page
 * fills an unknown fraction of the frame at an unknown distance.
 *
 * The scale is therefore recovered from image content. The ECG grid is a
 * printed physical ruler, with 1 mm minor divisions and 5 mm major
 * divisions, so measuring its pitch in pixels yields the conversion
 * directly.
 *
 * Note that the result also fixes the effective temporal sampling rate of
 * the digitised signal, since column-wise vectorisation produces one sample
 * per pixel column:
 *
 *     f_s = pxPerMm * paperSpeedMmPerSec
 *
 * At 25 mm/s, 6 px/mm yields 150 Hz and 12 px/mm yields 300 Hz. The
 * recovered bandwidth is bounded by half of that.
 */
object GridEstimator {

    /** Plausible range. Outside this the estimate is rejected rather than returned. */
    private const val MIN_PX_PER_MM = 3.0
    private const val MAX_PX_PER_MM = 40.0

    /** Window for the moving-average detrend. See [projectionProfile]. */
    private const val DETREND_WINDOW = 41

    data class GridEstimate(
        val pxPerMm: Double,
        val peakCount: Int,
        val objective: Double,
        val elapsedMs: Long
    ) {
        val valid: Boolean get() = !pxPerMm.isNaN()

        /** Effective temporal sampling rate implied by this scale. */
        fun samplingRateHz(paperSpeedMmPerSec: Double = 25.0) =
            pxPerMm * paperSpeedMmPerSec
    }

    fun estimate(bitmap: Bitmap): GridEstimate {
        val t0 = System.nanoTime()
        val profile = projectionProfile(bitmap)
        val acf = autocorrelation(profile)
        val (pitch, peaks, obj) = pitchFromAcf(acf)
        val ms = (System.nanoTime() - t0) / 1_000_000

        val ok = !pitch.isNaN() && pitch in MIN_PX_PER_MM..MAX_PX_PER_MM
        return GridEstimate(
            pxPerMm = if (ok) pitch else Double.NaN,
            peakCount = peaks,
            objective = obj,
            elapsedMs = ms
        )
    }

    /**
     * Mean ink density per image column, high-pass filtered.
     *
     * TWO NON-OBVIOUS CHOICES, both of which cause a silent NaN if got wrong:
     *
     * 1. GREEN channel, not red. Standard ECG paper prints the grid in red or
     *    orange, which means the grid is nearly INVISIBLE in the red channel —
     *    that is precisely why trace binarisation uses red. For grid detection
     *    the opposite is required. Measured contrast on a rendered printout:
     *    green profile range 9.9..123.9, red only 3.9..64.5.
     *
     * 2. Moving-average detrend. The trace itself contributes strong periodic
     *    structure at the heart rate — QRS complexes roughly every 170 px at
     *    8 px/mm — which dominates the autocorrelation and buries the 8 px
     *    grid periodicity. Subtracting a local mean high-passes that away.
     */
    private fun projectionProfile(bitmap: Bitmap): DoubleArray {
        val w = bitmap.width
        val h = bitmap.height

        // getPixels() in one call is far faster than getPixel() per pixel.
        val row = IntArray(w)
        val sums = DoubleArray(w)

        for (y in 0 until h) {
            bitmap.getPixels(row, 0, w, 0, y, w, 1)
            for (x in 0 until w) {
                // ARGB_8888 packing: A<<24 | R<<16 | G<<8 | B
                val green = (row[x] shr 8) and 0xFF
                sums[x] += 255.0 - green
            }
        }
        for (x in 0 until w) sums[x] /= h

        // High-pass by subtracting a moving average.
        val half = DETREND_WINDOW / 2
        val out = DoubleArray(w)
        for (x in 0 until w) {
            val a = maxOf(0, x - half)
            val b = minOf(w, x + half + 1)
            var s = 0.0
            for (i in a until b) s += sums[i]
            out[x] = sums[x] - s / (b - a)
        }
        return out
    }

    /** Normalised autocorrelation via FFT, zero-padded to avoid wraparound. */
    private fun autocorrelation(profile: DoubleArray): DoubleArray {
        val n = profile.size
        val size = nextPowerOfTwo(2 * n)
        val re = DoubleArray(size)
        val im = DoubleArray(size)
        profile.copyInto(re, 0, 0, n)

        fft(re, im, inverse = false)
        for (i in 0 until size) {          // power spectrum
            re[i] = re[i] * re[i] + im[i] * im[i]
            im[i] = 0.0
        }
        fft(re, im, inverse = true)

        if (re[0] == 0.0) return DoubleArray(n)
        val norm = re[0]
        return DoubleArray(n) { re[it] / norm }
    }

    /**
     * Pitch as the MEDIAN SPACING BETWEEN SUCCESSIVE AUTOCORRELATION PEAKS.
     *
     * Deliberately not the tallest peak. The grid carries two periodic
     * components: 1 mm minor rules and 5 mm major rules. At a lag of 5 mm the
     * heavy major rules align with each other, whereas at 1 mm the major rules
     * must align with minor ones, so the 5 mm peak is the stronger. Taking a
     * global maximum therefore locks onto the fifth harmonic and returns a
     * confident, self-consistent, fivefold error.
     *
     * Peak spacing is immune to which harmonic is tallest, because peaks at
     * 1, 2, 3 and 4 mm all still exist. This is the same fundamental-versus-
     * harmonic problem, and the same remedy, as pitch detection in audio.
     */
    private fun pitchFromAcf(acf: DoubleArray): Triple<Double, Int, Double> {
        val minLag = MIN_PX_PER_MM.toInt()
        val maxLag = minOf((MAX_PX_PER_MM * 1.5).toInt(), acf.size - 2)
        if (maxLag <= minLag) return Triple(Double.NaN, 0, 0.0)

        var windowMax = 0.0
        for (i in minLag until maxLag) if (acf[i] > windowMax) windowMax = acf[i]
        val threshold = 0.25 * windowMax

        val peaks = ArrayList<Double>()
        for (i in (minLag + 1) until (maxLag - 1)) {
            if (acf[i] > acf[i - 1] && acf[i] >= acf[i + 1] && acf[i] > threshold) {
                // Parabolic interpolation for sub-pixel peak location.
                val y0 = acf[i - 1]; val y1 = acf[i]; val y2 = acf[i + 1]
                val den = y0 - 2 * y1 + y2
                var d = if (den != 0.0) 0.5 * (y0 - y2) / den else 0.0
                if (abs(d) > 1.0) d = 0.0        // diverged; discard correction
                peaks.add(i + d)
            }
        }
        if (peaks.size < 3) return Triple(Double.NaN, peaks.size, windowMax)

        val gaps = DoubleArray(peaks.size - 1) { peaks[it + 1] - peaks[it] }
        gaps.sort()
        val m = gaps.size
        val median = if (m % 2 == 1) gaps[m / 2]
                     else 0.5 * (gaps[m / 2 - 1] + gaps[m / 2])
        return Triple(median, peaks.size, windowMax)
    }

    private fun nextPowerOfTwo(n: Int): Int {
        var p = 1
        while (p < n) p = p shl 1
        return p
    }

    /**
     * In-place iterative radix-2 Cooley-Tukey FFT. Length must be a power of 2.
     * Written out rather than pulled from a library so the port has no
     * dependency and matches the Python reference line for line.
     */
    private fun fft(re: DoubleArray, im: DoubleArray, inverse: Boolean) {
        val n = re.size

        // Bit-reversal permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }

        var length = 2
        while (length <= n) {
            val ang = (2.0 * PI / length) * (if (inverse) 1.0 else -1.0)
            val wr = cos(ang); val wi = sin(ang)
            val half = length shr 1
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until half) {
                    val ur = re[i + k]; val ui = im[i + k]
                    val vr = re[i + k + half] * cr - im[i + k + half] * ci
                    val vi = re[i + k + half] * ci + im[i + k + half] * cr
                    re[i + k] = ur + vr; im[i + k] = ui + vi
                    re[i + k + half] = ur - vr; im[i + k + half] = ui - vi
                    val ncr = cr * wr - ci * wi
                    ci = cr * wi + ci * wr
                    cr = ncr
                }
                i += length
            }
            length = length shl 1
        }

        if (inverse) {
            for (i in 0 until n) { re[i] /= n; im[i] /= n }
        }
    }
}
