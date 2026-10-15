package com.maarijalam.eacg

import android.graphics.Bitmap
import android.graphics.Matrix
import kotlin.math.hypot

/**
 * Which way up is the sheet in a photograph? Port of `python/ecgproto/orientation.py`
 * (SheetOrientationTest: golden vectors from make_orientation_fixtures.py).
 *
 * A photograph taken looking straight down at a sheet on a table carries
 * whatever orientation the phone last knew (the tilt sensor cannot tell
 * which way is up when the phone lies flat), so a landscape shot can arrive
 * sideways or upside down. Two cues:
 *
 *  - sideways: the printed sheet is landscape (277 x 190 mm), so a corner
 *    outline taller than wide means a quarter turn (76 / 76 right,
 *    results/kitphoto/orientation.csv); the quarter is taken clockwise;
 *  - up or down, from the page: the pipeline's own lead markers on the
 *    grid-fit page and on it turned half round ([SheetPipeline.halfTurnMarkers]).
 *    The pulses stand at the rows' left; a page turned half round shows at
 *    most [MAX_TICKS_TURNED] ticks, because one of the three rows the tick
 *    search reads is then the rhythm strip. [upsideDown] weighs the two.
 *
 * Replaced (progress_log §39): turning the way the user last confirmed,
 * which failed whenever the grip changed and never caught a sheet upside
 * down. Measured on the 68 session photographs
 * (results/kitphoto/orientation_sessions.csv): 64 right, against 58 or 64
 * for the old rule depending on the direction it remembered.
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
     * The most ticks a page turned half round can show: the tick search reads
     * the top three rows, and on a turned page one of them is the rhythm
     * strip, which carries none (3 rows x 3 ticks, less one row).
     */
    const val MAX_TICKS_TURNED = 6

    /** What the pipeline's own marker searches find on one candidate page. */
    data class Markers(val pulses: Int, val ticks: Int)

    /**
     * Is the page turned half round the upright one? (orientation.upside_down)
     * 1. more than [MAX_TICKS_TURNED] ticks on one page only: that page is upright;
     * 2. otherwise more pulses on one page: that page is upright;
     * otherwise the page is kept (Rotate is the override). Tick counts at or
     * below six say nothing about up or down; "more ticks wins" was measured
     * and rejected (orientation_sessions_rule_v1.csv).
     */
    fun upsideDown(up: Markers, turned: Markers): Boolean {
        val manyUp = up.ticks > MAX_TICKS_TURNED
        val manyTurned = turned.ticks > MAX_TICKS_TURNED
        if (manyUp != manyTurned) return manyTurned
        return turned.pulses > up.pulses
    }

    /** Clockwise quarter turns applied automatically: a quarter if sideways, a half more if upside down. */
    fun autoQuarters(sideways: Boolean, upsideDown: Boolean): Int =
        Math.floorMod((if (sideways) 1 else 0) + (if (upsideDown) 2 else 0), 4)
}
