package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** How a saved record's display fields are read back (Result screen, docs/ui_plan.md §3.4). */
class ResultDataTest {

    @Test
    fun ticksAndCellsParse() {
        val t = ResultData.parseTicks("0 1 733.0;1 1 733.5;2 3 1873.5")
        assertEquals(listOf(Triple(0, 1, 733.0), Triple(1, 1, 733.5), Triple(2, 3, 1873.5)), t)
        assertTrue(ResultData.parseTicks(null).isEmpty())
        val c = parseCells("I false 132 733 470 807;II true 132 2524 1488 1800")
        assertEquals(2, c.size)
        assertEquals("II", c[1].name); assertTrue(c[1].rhythm); assertEquals(2524, c[1].x1)
        assertTrue(parseCells(null).isEmpty())
    }

    @Test
    fun memorySummaryFromTheCsvFormat() {
        assertEquals("native peak 211 MB (+171) · process 412 MB",
            memorySummary("baseline=40.0/12.0 grid=210.5/30.1 trace=180.0/20.0", 412.3))
        assertEquals("process 400 MB", memorySummary(null, 400.0))
        assertNull(memorySummary(null, null))
    }

    @Test
    fun anOldRecordShowsNoTickColoursAndNoVerdict() {
        val m = RecordStore.Meta("a", 1L, "MODEL", "ptbxl40_kit_500hz_300dpi", "ticks", 10.5, 8, "pulses",
            true, DoubleArray(8), 2900, 1950, 3200, 2396, 0.95, 12, 28000)
        val d = ResultData.fromRecord(m, null)
        assertNull(d.ticksKept)
        assertNull(d.quality)
        assertTrue(d.warnings.isEmpty())
        assertEquals("PTB-XL 40", d.title)
    }
}
