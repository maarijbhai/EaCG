package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * ClfInput against `ecgproto.clf_input.from_columns`: the classifier must get
 * the same 1,000 x 13 numbers on the phone as on the desktop. Golden vectors
 * from `python/make_clf_port_fixtures.py` (a real phone export and an edge
 * case: partial and empty blocks, a missing lead, an even-length median, a
 * short rhythm strip).
 */
class ClfInputPortTest {

    private class Case(val cols: Map<String, DoubleArray>, val expect: DoubleArray)

    private fun parse(v: String) = if (v == "nan") Double.NaN else v.toDouble()

    private fun load(): Map<String, Case> {
        val stream = javaClass.getResourceAsStream("/clf_port_fixtures.txt")
            ?: error("clf_port_fixtures.txt missing: run python make_clf_port_fixtures.py")
        val out = LinkedHashMap<String, Case>()
        var name: String? = null
        var cols = LinkedHashMap<String, DoubleArray>()
        stream.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (line.startsWith("#") || line.isBlank()) continue
                val f = line.split(' ')
                when (f[0]) {
                    "case" -> { name = f[1]; cols = LinkedHashMap() }
                    "col" -> cols[f[1]] = DoubleArray(f.size - 2) { parse(f[it + 2]) }
                    "expect" -> out[name!!] = Case(cols, DoubleArray(f.size - 1) { parse(f[it + 1]) })
                }
            }
        }
        return out
    }

    private fun check(case: Case) {
        val x = ClfInput.fromColumns(case.cols)
        assertEquals(case.expect.size, x.size)
        var worst = 0.0
        for (i in x.indices) worst = maxOf(worst, kotlin.math.abs(x[i] - case.expect[i]))
        assertTrue("largest difference $worst", worst <= 1e-6)
    }

    @Test fun realExport() = check(load().getValue("real"))

    @Test fun edgeCases() = check(load().getValue("synth"))

    @Test fun cellsAreWholeBlocks() {
        // A cell of 1,250 samples at 500 Hz is exactly 250 at 100 Hz: no window
        // borrows a sample of its neighbour.
        assertEquals(ClfInput.CELL * ClfInput.BLOCK, (SheetLayout.CELL_S * SheetLayout.FS).toInt())
    }

    @Test fun fromLeadsMatchesTheCsvLayout() {
        // The pipeline's leads go through the same columns SheetPipeline.csv writes.
        val case = load().getValue("real")
        val leads = ArrayList<SheetPipeline.LeadResult>()
        for (row in SheetLayout.LAYOUT_3X4) for ((k, name) in row.withIndex()) {
            val c = case.cols.getValue(name)
            val a = k * 1250
            leads += SheetPipeline.LeadResult(name, false, k * SheetLayout.CELL_S, c.copyOfRange(a, a + 1250), null, null)
        }
        leads += SheetPipeline.LeadResult("II", true, 0.0, case.cols.getValue("II_rhythm"), null, null)
        val x = ClfInput.fromLeads(leads)
        for (i in x.indices) assertEquals(case.expect[i], x[i].toDouble(), 1e-6)
    }
}
