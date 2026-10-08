package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/**
 * The Records tab's file formats: what is saved must come back exactly, and
 * the leads must come back from the SAME CSV the app exports (FR10), so a
 * record and an export can never disagree about a waveform.
 */
class RecordStoreTest {

    @get:Rule val tmp = TemporaryFolder()

    private fun meta(id: String, created: Long, target: String? = "ptbxl40", pcc: Double? = 0.9612) =
        RecordStore.Meta(id, created, "MODEL", target, "ticks", 10.48912, 8, "pulses", true,
            doubleArrayOf(285.29, 187.88, 2896.08, 229.89, 3213.86, 2083.75, -49.43, 2192.72),
            2938, 1957, 3200, 2396, pcc, 12, 27831)

    @Test
    fun metaRoundTripsIncludingAbsentFields() {
        for (m in listOf(meta("a", 1L), meta("b", 2L, target = null, pcc = null))) {
            val back = RecordStore.metaFromText(RecordStore.metaToText(m))
            assertEquals(m.copy(corners = back.corners, pxPerMm = back.pxPerMm), back)
            assertArrayEquals(m.corners, back.corners, 0.005)
            assertEquals(m.pxPerMm, back.pxPerMm, 1e-6)
        }
        assertNull(RecordStore.metaFromText(RecordStore.metaToText(meta("c", 3L, null, null))).target)
    }

    /** A record written before 2026-10-07 (no redesign fields) must still open. */
    @Test
    fun aRecordFromBeforeTheRedesignStillLoads() {
        val old = """
            #EaCG record
            id=20261005_164847_915
            created_ms=1759675727000
            tracer=MODEL
            target=ptbxl38_kit_500hz_300dpi
            calibration=ticks
            px_per_mm=9.277200
            ticks_found=9
            t0_source=pulses
            corners_auto=true
            corners=528.03 319.83 2894.64 320.06 3067.04 2098.45 371.02 2150.95
            page_w=2563
            page_h=1840
            photo_w=3200
            photo_h=2396
            median_pcc=0.9210
            leads_scored=12
            total_ms=28931
            geometry=grid
        """.trimIndent()
        val m = RecordStore.metaFromText(old)
        assertEquals("ptbxl38_kit_500hz_300dpi", m.target)
        assertEquals(0.9210, m.medianPcc!!, 1e-9)
        assertNull(m.quality); assertNull(m.warnings); assertNull(m.ticks); assertNull(m.t0X)
        assertTrue(RecordStore.decodeWarnings(m.warnings).isEmpty())
    }

    @Test
    fun redesignFieldsRoundTrip() {
        val w = listOf(
            RecordStore.StoredWarning("TICK_REPLACED", "warn", listOf("V3", "V6", "II_rhythm"),
                "Row 3's 7.5 s tick was out of line by -5.5 mm; the same tick in the other rows was used instead."),
            RecordStore.StoredWarning("GRID_REFUSED", "warn", emptyList(), "No grid\tfit\nhere"))
        val m = meta("d", 4L).copy(quality = "check", warnings = RecordStore.encodeWarnings(w),
            cornersRefined = false, gridRmsMm = 0.1534, gridRules = "56x38", gridNote = null,
            timings = "corners=0 rectify=246 grid=3559", memory = "baseline=40.0/12.0 grid=210.5/30.1",
            pssEndMb = 412.3, photoSource = "camera", t0X = 132.25, rowY = "637 977 1331 1645",
            ticks = "0 1 733.0;1 1 733.5", ticksRejected = "2 3 1873.5",
            cells = "I false 132 733 470 807;II true 132 2524 1488 1800")
        val back = RecordStore.metaFromText(RecordStore.metaToText(m))
        assertEquals(m.copy(corners = back.corners, pxPerMm = back.pxPerMm), back)
        val ws = RecordStore.decodeWarnings(back.warnings)
        assertEquals(listOf("TICK_REPLACED", "GRID_REFUSED"), ws.map { it.code })
        assertEquals(listOf("V3", "V6", "II_rhythm"), ws[0].leads)
        assertEquals(emptyList<String>(), ws[1].leads)
        assertEquals("No grid fit here", ws[1].message)   // tabs and newlines flattened
    }

    @Test
    fun renameChangesOnlyTheNameAndBlankRestoresTheDefault() {
        val dir = tmp.newFolder("rec1")
        val m = meta("rec1", 5L)
        File(dir, RecordStore.META).writeText(RecordStore.metaToText(m))
        File(dir, RecordStore.LEADS).writeText("time_s,I\n0.000,0.1\n")
        val csvBefore = File(dir, RecordStore.LEADS).readText()

        val named = RecordStore.renameIn(dir, "  Bad photo: steep tilt  ")
        assertEquals("Bad photo: steep tilt", named.name)
        val back = RecordStore.metaFromText(File(dir, RecordStore.META).readText())
        assertEquals("Bad photo: steep tilt", back.name)
        assertEquals(m.copy(corners = back.corners, pxPerMm = back.pxPerMm, name = back.name), back)
        assertEquals(csvBefore, File(dir, RecordStore.LEADS).readText())     // the CSV is untouched
        assertTrue(!File(dir, RecordStore.META + ".tmp").exists())

        assertNull(RecordStore.renameIn(dir, "   ").name)                  // blank: default title again
        assertEquals(RecordStore.MAX_NAME, RecordStore.renameIn(dir, "x".repeat(200)).name!!.length)
        // Still listed (a renamed record is a complete record).
        assertEquals(listOf("rec1"), RecordStore.listIn(dir.parentFile!!).map { it.id })
    }

    @Test
    fun tracesRoundTripWithGaps() {
        val rows = listOf(SheetPipeline.RowTrace(120, doubleArrayOf(510.25, Double.NaN, 511.0)),
            SheetPipeline.RowTrace(-3, doubleArrayOf(Double.NaN)))
        val back = RecordStore.tracesFromText(RecordStore.tracesToText(rows))
        assertEquals(2, back.size)
        assertEquals(120, back[0].x0)
        assertEquals(510.25, back[0].pos[0], 1e-9)
        assertTrue(back[0].pos[1].isNaN())
        assertEquals(-3, back[1].x0)
        assertTrue(back[1].pos[0].isNaN())
    }

    /** A Result whose every lead is a recognisable ramp, cut as the pipeline cuts. */
    private fun result(): SheetPipeline.Result {
        val leads = ArrayList<SheetPipeline.LeadResult>()
        for ((row, names) in SheetLayout.LAYOUT_3X4.withIndex()) for ((k, n) in names.withIndex()) {
            val mv = DoubleArray(1250) { row + k * 0.1 + it * 1e-4 }
            mv[7] = Double.NaN                                   // a gap must survive
            leads += SheetPipeline.LeadResult(n, false, k * 2.5, mv, 0.5 + row * 0.1 + k * 0.01, 0.1)
        }
        leads += SheetPipeline.LeadResult("II", true, 0.0, DoubleArray(5000) { -it * 1e-4 }, 0.88, 0.2)
        return SheetPipeline.Result(SheetPipeline.Tracer.MODEL, DoubleArray(8), true, false,
            2938, 1957, 10.5, "ticks", 9, "pulses", leads, linkedMapOf("trace" to 1L), null)
    }

    @Test
    fun leadsComeBackFromTheExportedCsv() {
        val r = result()
        val back = RecordStore.leadsFromCsv(SheetPipeline.csv(r, "test", "ptbxl40", "20261004_120000"))
        assertEquals(13, back.size)
        for (want in r.leads) {
            val got = back.single { it.name == want.name && it.rhythm == want.rhythm }
            assertEquals(want.tStartS, got.tStartS, 0.0)
            assertEquals(want.mv.size, got.mv.size)
            for (i in want.mv.indices) {
                if (want.mv[i].isNaN()) assertTrue("${want.name}[$i]", got.mv[i].isNaN())
                else assertEquals("${want.name}[$i]", want.mv[i], got.mv[i], 5e-5)   // %.4f
            }
            assertEquals(want.pcc!!, got.pcc!!, 5e-5)
        }
    }

    @Test
    fun listingSkipsHalfWrittenAndForeignFoldersNewestFirst() {
        val root = tmp.newFolder("records")
        fun write(id: String, created: Long) =
            File(root, id).apply { mkdirs() }.resolve(RecordStore.META)
                .writeText(RecordStore.metaToText(meta(id, created)))
        write("older", 1_000L)
        write("newer", 2_000L)
        write("crashed.tmp", 3_000L)                         // a save that never finished
        File(root, "stray").mkdirs()                         // no record.properties
        File(root, "corrupt").apply { mkdirs() }.resolve(RecordStore.META).writeText("id=x\n")
        assertEquals(listOf("newer", "older"), RecordStore.listIn(root).map { it.id })
        assertEquals(emptyList<RecordStore.Meta>(), RecordStore.listIn(File(root, "absent")))
    }

    @Test
    fun idsAreTimeStampedAndDistinctWithinASecond() {
        val a = RecordStore.newId(1_700_000_000_001L, "20261004_120000")
        val b = RecordStore.newId(1_700_000_000_002L, "20261004_120000")
        assertTrue(a.startsWith("20261004_120000_"))
        assertTrue(a != b)
    }
}
