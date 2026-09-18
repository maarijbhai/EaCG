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

    /**
     * Plausible range. Outside this the estimate is rejected rather than
     * returned. Both bounds are `digitise.estimate_grid_pitch`'s
     * `min_pitch_px` and `max_pitch_px`.
     *
     * MAX was 40.0 and is now 80.0, matching the Python. 40 was too tight in
     * a way that bit at the top of the fixture set: the lag search window is
     * derived from it, and at 300 dpi the 5 mm rule sits at 59.06 px, right
     * at the edge of the window a limit of 40 produced.
     */
    private const val MIN_PX_PER_MM = 3.0
    private const val MAX_PX_PER_MM = 80.0

    /**
     * Minimum autocorrelation height for a lag to count as a grid peak.
     *
     * **Absolute, not a fraction of anything.** The autocorrelation is
     * normalised by its zero lag, so this is a height on a 0..1 scale. The
     * Python's `DEFAULT_PEAK_FRACTION` is the same 0.05 and is absolute for
     * the same reason -- it multiplies `ac[:hi].max()`, which is always
     * exactly 1.0.
     *
     * This replaces a `0.25 * windowMax` rule, and the replacement is the
     * fix for the failure CLAUDE.md limitation 6 predicted and the device
     * confirmed: grid estimation returning "failed" on all three bundled
     * fixtures. See [pitchFromAcf].
     */
    private const val PEAK_THRESHOLD = 0.05

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
     * Mean grid response per image column, high-pass filtered.
     *
     * 1. REDNESS — [GridResponse], the Python's `digitise.redness` — and NOT
     *    the green channel this used to read.
     *
     *    The green-channel argument was that a red grid is nearly invisible
     *    in the red channel, so grid detection needs the opposite of what
     *    trace binarisation uses. It is wrong, and measurement says so. On
     *    the ecg-image-kit fixtures the minor rules are BGR (219,223,254)
     *    and the majors (0,0,255), so `255-G` sees the minors at 32/255 of
     *    the majors and `255-B` at 36/255 — the channel moves the
     *    minor-to-major ratio by about 20%, and green moves it the WRONG
     *    way, being the worst of blue, redness, green and luma. See
     *    `results/grid/contrast.csv`.
     *
     *    Redness also has a reason the others do not: it is near-silent on
     *    the neutral dark trace, so the profile it builds is grid rather
     *    than grid-plus-waveform.
     *
     *    Simulated over the three fixtures before this was written, changing
     *    the channel ALONE fixes nothing — still NaN on two fixtures and a
     *    199% error on the third. The channel was never the fault; see
     *    [pitchFromAcf] for what was.
     *
     * 2. Moving-average detrend. The trace contributes periodic structure at
     *    the heart rate — QRS complexes roughly every 170 px at 8 px/mm —
     *    which the autocorrelation would otherwise pick up alongside the
     *    grid. Subtracting a local mean high-passes it away.
     *
     *    Kept unchanged, and it is worth recording that it turned out not to
     *    matter: with the other changes in place, removing it moves the
     *    recovered pitch by under 0.15 percentage points on every fixture.
     *    The Python has no equivalent, so this is the one remaining
     *    structural difference between the two implementations.
     */
    private fun projectionProfile(bitmap: Bitmap): DoubleArray {
        val w = bitmap.width
        val h = bitmap.height

        // getPixels() in one call is far faster than getPixel() per pixel.
        val row = IntArray(w)
        val sums = DoubleArray(w)

        for (y in 0 until h) {
            bitmap.getPixels(row, 0, w, 0, y, w, 1)
            // The arithmetic lives in GridResponse, not here, because
            // SkewEstimator needs the same response from a different pixel
            // layout and two copies of a channel-order-sensitive expression
            // is two chances to get it wrong.
            for (x in 0 until w) sums[x] += GridResponse.fromArgb(row[x])
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
     *
     * **That remedy only works if the 1 mm peaks survive selection, and
     * until 2026-09-18 they did not.** The threshold here was
     * `0.25 * windowMax`, a fraction of the tallest peak in the lag window,
     * which reads as a modest cut and is not: on the fixtures that window
     * maximum measures 0.71 to 0.95, so the cut landed at 0.178 to 0.238 on
     * a normalised scale — essentially where the Python's old absolute 0.20
     * landed, and above the 0.11 to 0.19 the minor rules stand at. Every
     * minor peak was deleted, fewer than three peaks survived, and the
     * estimate came back NaN. That is what "Grid pitch: failed" on all three
     * fixtures was, confirmed on the device on 2026-09-18 and predicted from
     * the Python side the day before.
     *
     * The threshold is now [PEAK_THRESHOLD], absolute, matching the Python
     * exactly. The lag window widens with it: `MAX_PX_PER_MM * 6` rather
     * than `* 1.5`, which is the Python's `hi = min(max_pitch_px * 6, ...)`.
     *
     * Verified by simulating this function against the three fixtures before
     * the change was written — recovered px/mm, error against truth:
     *
     *     shipped (green, 0.25*windowMax)   NaN      NaN      NaN
     *     redness alone                     NaN   +199.15%    NaN
     *     + absolute 0.05                 +0.20%   -0.08%   -0.15%
     *     + wide lag window               +0.61%   +0.32%   +0.08%
     *
     * and the last of those sits within 0.06 percentage points of what the
     * Python returns on the same images (+0.55%, +0.36%, +0.13%), which is
     * the check that this is a port and not a reimplementation. Synthetic
     * renders at 6, 8, 12 and 16 px/mm are unaffected by any of it.
     */
    private fun pitchFromAcf(acf: DoubleArray): Triple<Double, Int, Double> {
        val minLag = MIN_PX_PER_MM.toInt()
        val maxLag = minOf((MAX_PX_PER_MM * 6).toInt(), acf.size - 2)
        if (maxLag <= minLag) return Triple(Double.NaN, 0, 0.0)

        // windowMax is no longer the threshold's basis -- see the docstring --
        // but is still returned as the objective, because it says how strong
        // the strongest periodicity in the grid band was and that is worth
        // seeing next to a result.
        var windowMax = 0.0
        for (i in minLag until maxLag) if (acf[i] > windowMax) windowMax = acf[i]
        val threshold = PEAK_THRESHOLD

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
