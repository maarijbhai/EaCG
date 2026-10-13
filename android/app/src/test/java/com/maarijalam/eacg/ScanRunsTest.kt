package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin

/**
 * The both-tracers testing mode must produce, for each tracer, exactly what a
 * separate normal Digitise tap with that tracer produces.
 *
 * What a run computes is fixed by SheetPipeline.run's inputs: the photograph
 * (ScanController passes the same object to both runs) and the [RunSpec]. So
 * (1) the both-tracers plan must hand each run the SAME spec as a normal tap
 * with that tracer, and (2) the CSV of a both-tracers run must equal a normal
 * tap's CSV of the same result except `local_time` and the one new `pair_id`
 * line. SheetPipeline.run itself needs OpenCV and a device, so it is not run
 * here; the golden-vector suites cover it, and this change does not touch it.
 */
class ScanRunsTest {

    private val base = RunSpec(SheetPipeline.Tracer.CLASSICAL, listOf(174.4, 173.7, 3023.9, 173.7, 3023.9, 2129.4, 174.4, 2129.3),
        cornersAuto = true, cornersRefined = true, geometry = SheetPipeline.Geometry.GRID, target = "ptbxl9_kit_500hz_300dpi")

    @Test fun offIsOneRunExactlyAsChosen() {
        assertEquals(listOf(base), ScanRuns.plan(base, bothTracers = false))
    }

    @Test fun bothIsLearnedThenClassicalEachAsASeparateTap() {
        val plan = ScanRuns.plan(base, bothTracers = true)
        // A normal tap with the tracer set to learned, then one set to classical.
        val tapLearned = ScanRuns.plan(base.copy(tracer = SheetPipeline.Tracer.MODEL), false).single()
        val tapClassical = ScanRuns.plan(base.copy(tracer = SheetPipeline.Tracer.CLASSICAL), false).single()
        assertEquals(listOf(tapLearned, tapClassical), plan)
        // Whatever the tracer selector says, the learned run comes first.
        assertEquals(plan, ScanRuns.plan(base.copy(tracer = SheetPipeline.Tracer.MODEL), true))
        // Hand-placed corners and corners-only geometry are carried the same way.
        val hand = base.copy(cornersAuto = false, cornersRefined = null, geometry = SheetPipeline.Geometry.CORNERS)
        assertEquals(listOf(hand.copy(tracer = SheetPipeline.Tracer.MODEL), hand), ScanRuns.plan(hand, true))
    }

    @Test fun progressLabel() {
        assertNull(ScanRuns.label(0, 1, SheetPipeline.Tracer.MODEL))
        assertEquals("Run 1 of 2: learned", ScanRuns.label(0, 2, SheetPipeline.Tracer.MODEL))
        assertEquals("Run 2 of 2: classical", ScanRuns.label(1, 2, SheetPipeline.Tracer.CLASSICAL))
    }

    private fun result(tracer: SheetPipeline.Tracer): SheetPipeline.Result {
        val leads = ArrayList<SheetPipeline.LeadResult>()
        for ((r, row) in SheetLayout.LAYOUT_3X4.withIndex()) for ((k, name) in row.withIndex()) {
            val mv = DoubleArray(1250) { 0.3 * sin(it * 0.02 + r + k) }
            mv[7] = Double.NaN
            leads += SheetPipeline.LeadResult(name, false, k * SheetLayout.CELL_S, mv, 0.95 - 0.01 * k, 0.05)
        }
        leads += SheetPipeline.LeadResult("II", true, 0.0, DoubleArray(5000) { sin(it * 0.01) }, 0.97, 0.04)
        return SheetPipeline.Result(tracer, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), true, true,
            2846, 1954, 10.276, "ticks", 9, "pulses", leads,
            linkedMapOf("corners" to 0L, "rectify" to 192L, "trace" to 20259L, "classify" to 122L), null,
            diagnosis = Diagnosis(listOf("NORM", "MI"), doubleArrayOf(0.9, 0.1), doubleArrayOf(0.47, 0.35), "m"))
    }

    private fun csv(r: SheetPipeline.Result, stamp: String, pairId: String?) =
        SheetPipeline.csv(r, "picked 3200x2304", "ptbxl9_kit_500hz_300dpi", stamp, pairId, device = "test device")

    @Test fun csvDiffersOnlyInTimeAndPairId() {
        for (tr in SheetPipeline.Tracer.entries) {
            val r = result(tr)
            val normal = csv(r, "20261013_100000", null).lines()
            val paired = csv(r, "20261013_100042", "20261013_100000").lines()
            assertFalse(normal.any { it.startsWith("# pair_id:") })
            assertEquals(listOf("# pair_id: 20261013_100000"), paired.filter { it.startsWith("# pair_id:") })
            fun strip(l: List<String>) = l.filter { !it.startsWith("# pair_id:") && !it.startsWith("# local_time:") }
            assertEquals(strip(normal), strip(paired))            // every other header line and every lead value
            // The new line is the last header line: no existing line moved.
            val i = paired.indexOfFirst { it.startsWith("time_s,") }
            assertTrue(paired[i - 1].startsWith("# pair_id:"))
            assertEquals(normal.filter { it.startsWith("#") }.size + 1, paired.filter { it.startsWith("#") }.size)
        }
    }

    @Test fun pairIdIsReadBackByNothingThatScores() {
        // Records and the Diagnosis tab read a saved CSV: the extra line changes neither.
        val r = result(SheetPipeline.Tracer.MODEL)
        val a = RecordStore.leadsFromCsv(csv(r, "s", null))
        val b = RecordStore.leadsFromCsv(csv(r, "s", "p"))
        assertEquals(a.size, b.size)
        for ((x, y) in a.zip(b)) {
            assertEquals(x.name, y.name); assertEquals(x.pcc, y.pcc)
            assertTrue(x.mv.indices.all { x.mv[it].equals(y.mv[it]) })
        }
        assertEquals(Diagnosis.fromCsv(csv(r, "s", null))!!.probLine(), Diagnosis.fromCsv(csv(r, "s", "p"))!!.probLine())
    }
}
