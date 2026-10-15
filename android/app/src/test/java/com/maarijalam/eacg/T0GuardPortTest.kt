package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * SheetLayout.guardTimeZero against `segment.guard_time_zero`
 * (python/make_t0_guard_fixtures.py): two real grid-fit pages of Set C
 * Crumple -- 124910, where a false pulse put time zero ~5 mm early and the
 * ticks read +8.7% against the grid (3 of 9 found), and 124711, a good run --
 * plus the trigger and t0-from-ticks rules. Under the old rule (no guard) the
 * first test fails: the page keeps its 3 ticks and the +8.7% scale.
 */
class T0GuardPortTest {

    private val rows: Map<String, List<String>> by lazy {
        val stream = javaClass.getResourceAsStream("/t0_guard_fixtures.txt")
            ?: error("t0_guard_fixtures.txt missing: run python/make_t0_guard_fixtures.py")
        stream.bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }
                .associate { line -> val p = line.split(" "); p[0] to p.drop(1) }
        }
    }

    private fun row(k: String) = rows[k] ?: error("missing '$k'; regenerate the fixtures")
    private fun double(k: String) = row(k)[0].toDouble()
    private fun ints(k: String) = row(k).map { it.toInt() }.toIntArray()
    private fun line(k: String) = row(k).joinToString(" ")

    private fun mask(case: String): TraceExtractor.Mask {
        val (h, w) = ints("$case.shape").let { it[0] to it[1] }
        val data = ByteArray(h * w)
        for (r in 0 until h) {
            val runs = rows["$case.rle.$r"] ?: continue
            var i = 0
            while (i < runs.size) {
                val s = runs[i].toInt(); val n = runs[i + 1].toInt()
                for (c in s until s + n) data[r * w + c] = 1
                i += 2
            }
        }
        return TraceExtractor.Mask(data, h, w)
    }

    private fun positions(k: String): List<Triple<Int, Int, Double>> {
        val p = row(k).filter { it.isNotEmpty() }
        return (p.indices step 3).map { Triple(p[it].toInt(), p[it + 1].toInt(), p[it + 2].toDouble()) }
    }

    private fun assertTicks(key: String, t: SheetLayout.TickScale) {
        assertEquals(key, row("$key.n")[0].toInt(), t.nFound)
        val px = double("$key.px")
        if (px.isNaN()) assertTrue(key, t.pxPerMm.isNaN()) else assertEquals(key, px, t.pxPerMm, 1e-9)
        val want = positions("$key.pos")
        assertEquals(key, want.size, t.positions.size)
        for ((a, b) in want.zip(t.positions)) {
            assertEquals(a.first, b.first); assertEquals(a.second, b.second); assertEquals(a.third, b.third, 1e-9)
        }
    }

    private fun run(case: String): Pair<SheetLayout.TickScale, SheetLayout.Guarded?> {
        val m = mask("$case.m2")
        val r = ints("$case.rows")
        val off = SheetLayout.maskOffsetPx(2)
        val ticks = SheetLayout.tickScale(m, r, double("$case.t0"), double("$case.grid"), off)
        return ticks to SheetLayout.guardTimeZero(m, r, double("$case.t0"), line("$case.t0_source"), ticks,
            double("$case.rhythm_t0"), double("$case.grid"), off)
    }

    @Test fun aFalsePulseOnARealPageIsReplacedByTheTicks() {
        val (ticks, g) = run("fires")
        assertTicks("fires.ticks", ticks)                  // the failure: 3 ticks, +8.7%
        assertEquals(1, row("fires.fires")[0].toInt())
        assertNotNull("the guard must fire on this page", g)
        assertEquals(row("fires.accepted")[0] == "1", g!!.guard.accepted)
        assertEquals(double("fires.out_t0"), g.t0, 1e-9)
        assertEquals(line("fires.out_t0_source"), g.t0Source)
        assertTicks("fires.out", g.ticks)
        assertEquals(line("fires.line"), g.guard.line())
        assertTrue(g.ticks.nFound >= 8)
    }

    @Test fun aGoodPageIsLeftAlone() {
        val (ticks, g) = run("quiet")
        assertTicks("quiet.ticks", ticks)
        assertEquals(0, row("quiet.fires")[0].toInt())
        assertNull(g)
    }

    @Test fun triggerAndT0FromTicksMatchPython() {
        for (i in 0 until row("trigger.n")[0].toInt()) {
            val v = row("trigger.$i")
            val grid = v[1].toDouble()
            assertEquals("trigger $i", v[2] == "1",
                SheetLayout.t0GuardFires(v[0].toDouble(), if (grid.isNaN()) null else grid))
        }
        for (i in 0 until row("t0ticks.n")[0].toInt()) {
            val want = double("t0ticks.$i.t0")
            val got = SheetLayout.t0FromTicks(positions("t0ticks.$i.pos"), double("t0ticks.$i.px"))
            if (want.isNaN()) assertTrue(got.isNaN()) else assertEquals("t0ticks $i", want, got, 1e-9)
        }
    }
}
