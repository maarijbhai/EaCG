package com.maarijalam.eacg

import android.graphics.Bitmap
import android.graphics.Matrix
import kotlin.math.hypot

/**
 * Is the sheet in a photograph sideways? Port of `python/ecgproto/orientation.py`
 * (SheetOrientationTest shares its cases).
 *
 * A photograph taken looking straight down at a sheet on a table carries
 * whatever orientation the phone last knew (the tilt sensor cannot tell
 * which way is up when the phone lies flat), so a landscape shot can arrive
 * sideways. The printed sheet is landscape (277 x 190 mm): the corner
 * outline taller than wide means a quarter turn. Measured 76 / 76 right
 * (results/kitphoto/orientation.csv). WHICH quarter turn is not decided from
 * the image (two cues were measured and rejected, progress_log §35): the
 * app turns the way the user last confirmed, and offers Rotate.
 */
object SheetOrientation {
    const val SIDEWAYS_BELOW = 1.0

    /** Width / height of the quad (TL, TR, BR, BL as x0 y0 ... x3 y3), opposite sides averaged. */
    fun outlineAspect(c: DoubleArray): Double {
        require(c.size == 8)
        fun d(i: Int, j: Int) = hypot(c[2 * j] - c[2 * i], c[2 * j + 1] - c[2 * i + 1])
        val w = 0.5 * (d(0, 1) + d(3, 2))
        val h = 0.5 * (d(0, 3) + d(1, 2))
        return w / h
    }

    fun isSideways(c: DoubleArray): Boolean = outlineAspect(c) < SIDEWAYS_BELOW

    /** [quarters] quarter turns clockwise (negative: counter-clockwise); pixels exactly permuted. */
    fun rotate(b: Bitmap, quarters: Int): Bitmap {
        val q = Math.floorMod(quarters, 4)
        if (q == 0) return b
        val m = Matrix().apply { postRotate(90f * q) }
        return Bitmap.createBitmap(b, 0, 0, b.width, b.height, m, false)
    }

    /** The CSV's `photo_rotation` value: total clockwise degrees and who turned it; null if none. */
    fun label(autoQuarters: Int, userQuarters: Int): String? {
        val total = Math.floorMod(autoQuarters + userQuarters, 4)
        if (total == 0 && autoQuarters == 0 && userQuarters == 0) return null
        val by = listOfNotNull(if (autoQuarters != 0) "auto ${Math.floorMod(autoQuarters, 4) * 90}" else null,
            if (Math.floorMod(userQuarters, 4) != 0) "user ${Math.floorMod(userQuarters, 4) * 90}" else null)
        return "${total * 90} clockwise" + if (by.isEmpty()) "" else " (${by.joinToString(", ")})"
    }

    /**
     * After an automatic quarter turn, did the user have to turn the photograph
     * half round? Then the automatic direction was wrong for this grip: flip it.
     */
    fun shouldFlipDirection(autoQuarters: Int, userQuarters: Int): Boolean =
        autoQuarters != 0 && Math.floorMod(userQuarters, 4) == 2
}
