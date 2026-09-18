package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The redness response, against values the Python computed.
 *
 * Four lines of arithmetic, and worth pinning anyway, because this is the
 * one place in the port where the two pixel layouts meet. OpenCV on the
 * desktop reads image files as **BGR**, so `cv2.split` yields `(b, g, r)`
 * and `digitise.redness` indexes channel 2 for red; an Android bitmap
 * arrives as packed **ARGB** ints, where red is the high byte, and an
 * OpenCV `Mat` converted from one is **RGBA**, where red is byte 0. Three
 * layouts, one formula.
 *
 * Getting it backwards does not crash and does not look wrong. The response
 * still has structure — it just responds to the trace instead of the grid,
 * which yields a projection with peaks in it and an estimate that is
 * confidently incorrect. That is exactly the failure this project keeps
 * finding, and exactly the kind a unit test catches for nothing.
 */
class GridResponseTest {

    private val fx = PortFixtures.load()

    @Test
    fun matchesPythonForBothPixelLayouts() {
        for (i in 0 until fx.int("redness.n")) {
            val r = fx.int("redness.$i", 0)
            val g = fx.int("redness.$i", 1)
            val b = fx.int("redness.$i", 2)
            val expected = fx.double("redness.$i", 3)

            // Packed ARGB_8888, as Bitmap.getPixels delivers it.
            val argb = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            assertEquals(
                "fromArgb($r,$g,$b)", expected, GridResponse.fromArgb(argb), 0.0
            )

            // RGBA bytes, as a Mat converted from a Bitmap stores them. The
            // offset is non-zero on purpose: a correct implementation indexes
            // from `i`, and one that ignores the offset passes at zero.
            val bytes = ByteArray(8)
            bytes[4] = r.toByte()
            bytes[5] = g.toByte()
            bytes[6] = b.toByte()
            bytes[7] = 0xFF.toByte()
            assertEquals(
                "fromRgbaBytes($r,$g,$b)",
                expected, GridResponse.fromRgbaBytes(bytes, 4), 0.0
            )
        }
    }

    /**
     * The two accessors must agree with each other, not merely each with the
     * fixture. Bytes above 127 are negative as signed Java bytes, so an
     * implementation that forgets `and 0xFF` produces a plausible-looking
     * negative response that the clip then turns into a silent zero.
     */
    @Test
    fun theTwoAccessorsAgreeAcrossTheFullByteRange() {
        for (r in 0..255 step 17) {
            for (g in 0..255 step 23) {
                for (b in 0..255 step 29) {
                    val argb = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    val bytes = byteArrayOf(
                        r.toByte(), g.toByte(), b.toByte(), 0xFF.toByte()
                    )
                    assertEquals(
                        "disagreement at r=$r g=$g b=$b",
                        GridResponse.fromArgb(argb),
                        GridResponse.fromRgbaBytes(bytes, 0),
                        0.0
                    )
                }
            }
        }
    }

    /**
     * The trace must be silent and the grid must not be.
     *
     * This is the property the channel choice exists for, stated as a test
     * rather than left in a comment: neutral dark ink clips to zero, red
     * rules do not. A port that read a luminance channel would pass every
     * numeric test above if the fixture were regenerated, and fail this one.
     */
    @Test
    fun suppressesNeutralInkAndKeepsRedRules() {
        val trace = GridResponse.fromArgb((0xFF shl 24) or (32 shl 16) or (32 shl 8) or 32)
        assertEquals("neutral dark ink must clip to zero", 0.0, trace, 0.0)

        val paper = GridResponse.fromArgb(-1) // opaque white
        assertEquals("paper must clip to zero", 0.0, paper, 0.0)

        // ecg-image-kit's rules: BGR (219,223,254) minor and (0,0,255) major.
        val minor = GridResponse.fromArgb((0xFF shl 24) or (254 shl 16) or (223 shl 8) or 219)
        val major = GridResponse.fromArgb((0xFF shl 24) or (255 shl 16) or (0 shl 8) or 0)
        assert(minor > 0.0) { "minor rules must produce a response" }
        assert(major > minor) { "major rules must respond more strongly than minor" }

        // The measured contrast ratio on those fixtures is roughly seven to
        // one, which is why the peak threshold had to come down to 0.05.
        val ratio = major / minor
        assert(ratio > 6.0 && ratio < 9.0) {
            "minor-to-major contrast should be about 7:1, got ${"%.2f".format(ratio)}"
        }
    }
}
