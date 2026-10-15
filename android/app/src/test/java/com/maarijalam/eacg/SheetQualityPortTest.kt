package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * SheetQuality.kt against `sheet_quality_fixtures.txt`, which
 * `python/make_sheet_quality_fixtures.py` writes from `ecgproto/sheet_quality.py`:
 * every case's warnings (code, severity, leads, in order) and verdict must
 * match the Python reference exactly. Messages are not compared.
 */
class SheetQualityPortTest {

    private val rows: Map<String, List<String>> by lazy {
        val stream = javaClass.getResourceAsStream("/sheet_quality_fixtures.txt")
            ?: error("sheet_quality_fixtures.txt missing; run python make_sheet_quality_fixtures.py")
        stream.bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }
                .associate { l -> l.split(" ").let { it[0] to it.drop(1) } }
        }
    }

    private fun d(s: String) = if (s == "nan") Double.NaN else s.toDouble()

    @Test
    fun constantsMatchTheReference() {
        assertEquals(d(rows.getValue("const.SCALE_DISAGREE_FRACTION")[0]), SheetQuality.SCALE_DISAGREE_FRACTION, 0.0)
        assertEquals(d(rows.getValue("const.TICK_MATE_MM")[0]), SheetQuality.TICK_MATE_MM, 0.0)
        assertEquals(rows.getValue("const.MIN_TICKS")[0].toInt(), SheetQuality.MIN_TICKS)
        assertEquals(d(rows.getValue("const.MIN_LEAD_COVERAGE")[0]), SheetQuality.MIN_LEAD_COVERAGE, 0.0)
        assertEquals(d(rows.getValue("const.ROW_SPACING_MAX_RATIO")[0]), SheetLayout.ROW_SPACING_MAX_RATIO, 0.0)
        val pp = rows.getValue("const.REJECT_PP_MV")
        assertEquals(d(pp[0]), SheetQuality.REJECT_PP_MV[0], 0.0)
        assertEquals(d(pp[1]), SheetQuality.REJECT_PP_MV[1], 0.0)
    }

    @Test
    fun everyCaseMatchesTheReference() {
        val n = rows.getValue("cases")[0].toInt()
        val names = SheetLayout.LAYOUT_3X4.flatMap { it.toList() } + "II_rhythm"
        for (i in 0 until n) {
            val p = "case.$i"
            val name = rows.getValue("$p.name")[0]
            val fl = rows.getValue("$p.flags")
            val sc = rows.getValue("$p.scales")
            val tk = rows.getValue("$p.ticks")
            val ticks = (0 until tk.size / 3).map {
                Triple(tk[3 * it].toInt(), tk[3 * it + 1].toInt(), d(tk[3 * it + 2]))
            }
            val rj = rows.getValue("$p.rejected").filter { it.isNotEmpty() }
            val rejected = (0 until rj.size / 3).map {
                Triple(rj[3 * it].toInt(), rj[3 * it + 1].toInt(), d(rj[3 * it + 2]))
            }
            val leads = LinkedHashMap<String, DoubleArray>()
            for (l in names) leads[l] = rows.getValue("$p.lead.$l").map(::d).toDoubleArray()
            val ws = SheetQuality.assess(
                gridRequested = fl[0] == "1", gridUsed = fl[1] == "1", gridPxPerMm = d(sc[0]),
                calibration = fl[2], pxPerMm = d(sc[1]), ticksFound = fl[3].toInt(),
                tickPositions = ticks, t0Source = fl[4], leads = leads, tickRejected = rejected,
                rows = rows.getValue("$p.rows").map { it.toInt() }.toIntArray(),
                t0GuardPct = rows.getValue("$p.t0_guard")[0].let { if (it == "none") null else d(it) })
            val want = (0 until rows.getValue("$p.n")[0].toInt()).map {
                rows.getValue("$p.w.$it").joinToString(" ")
            }
            val got = ws.map {
                "${it.code} ${it.severity.name.lowercase()} ${it.leads.joinToString(",").ifEmpty { "-" }}"
            }
            assertEquals(name, want, got)
            assertEquals("$name verdict", rows.getValue("$p.verdict")[0], SheetQuality.verdict(ws))
        }
    }

    @Test
    fun leadStats() {
        val (cov, pp) = SheetQuality.leadStats(doubleArrayOf(0.0, Double.NaN, 1.0, -0.5))
        assertEquals(0.75, cov, 1e-12)
        assertEquals(1.5, pp, 1e-12)
        val (c0, p0) = SheetQuality.leadStats(DoubleArray(4) { Double.NaN })
        assertEquals(0.0, c0, 0.0)
        assertEquals(true, p0.isNaN())
    }
}
