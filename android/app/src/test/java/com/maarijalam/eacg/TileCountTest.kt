package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Test

/** SheetCanonical.tileCount (progress only) must equal predictTiled's model calls. */
class TileCountTest {
    @Test
    fun tileCountMatchesPredictTiledCalls() {
        for ((h, w) in listOf(512 to 2048, 512 to 1024, 512 to 900, 512 to 1500, 384 to 2048, 512 to 3000)) {
            val frame = SheetCanonical.Frame(FloatArray(h * w * 3), h, w, SheetCanonical.CanonMap(1.0, 0, 0))
            var calls = 0
            SheetCanonical.predictTiled(frame, { _, _, tw -> calls++; FloatArray(2 * tw) to FloatArray(2 * tw) }, 1024)
            assertEquals("$h x $w", calls, SheetCanonical.tileCount(frame, 1024))
        }
    }
}
