package com.maarijalam.eacg

/**
 * The per-pixel grid response, `redness`, defined in exactly one place.
 *
 * Both ported stages key on it — [GridEstimator] projects it to find the
 * grid pitch, [SkewEstimator] projects it along candidate angles to find the
 * skew — and they reach the pixels by different routes: a Bitmap's packed
 * ARGB ints in one case, an OpenCV Mat's RGBA bytes in the other. That is
 * two opportunities to get the channel order wrong, in a way that produces a
 * response which is wrong without looking wrong: it still has structure,
 * just of the trace instead of the grid. One definition, two accessors.
 *
 * The arithmetic is `digitise.redness`: red minus half the sum of green and
 * blue, clipped at zero. The grid is printed red and the trace in
 * near-neutral ink, so red *saturation* separates them — grid pixels keep a
 * high red channel while dropping green and blue, trace pixels drop all
 * three, and background keeps all three.
 */
object GridResponse {

    /** Packed ARGB_8888 as `Bitmap.getPixels` delivers it: `A<<24|R<<16|G<<8|B`. */
    fun fromArgb(pixel: Int): Double {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        val v = r - 0.5 * (g + b)
        return if (v > 0.0) v else 0.0
    }

    /**
     * RGBA bytes as an OpenCV `Mat` of type `CV_8UC4` stores them, with [i]
     * the index of the pixel's first byte.
     *
     * Note for anyone comparing against the Python: OpenCV on the desktop
     * reads image *files* as BGR, so `cv2.split` there yields `(b, g, r)`
     * and `redness` indexes channel 2 for red. A Mat converted from an
     * Android Bitmap is RGBA, so red is channel 0. Same arithmetic, opposite
     * order.
     */
    fun fromRgbaBytes(data: ByteArray, i: Int): Double {
        val r = data[i].toInt() and 0xFF
        val g = data[i + 1].toInt() and 0xFF
        val b = data[i + 2].toInt() and 0xFF
        val v = r - 0.5 * (g + b)
        return if (v > 0.0) v else 0.0
    }
}
