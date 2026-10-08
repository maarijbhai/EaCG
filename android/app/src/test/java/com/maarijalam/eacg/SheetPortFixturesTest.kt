package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import kotlin.math.abs

/**
 * Port fidelity for the full-sheet path: [SheetLayout], the pure parts of
 * [SheetCorners] and [SheetCanonical], against `ecgproto/sheet.py`,
 * `segment.py`, `page.py` and `canonical.py`.
 *
 * Golden vectors from `python/make_sheet_port_fixtures.py`, run on a real
 * photograph (ptbxl40_20260929_05) at the phone's 3,200 px. OpenCV cannot run
 * in a JVM test, so the OpenCV stages' outputs (the grid region, the
 * binarised page, the isolated row bands) ship as inputs and every pure
 * stage after them is checked. Decisions (rows, cells, inlier sets) must
 * match exactly; real values to 1e-9.
 */
class SheetPortFixturesTest {

    companion object {
        private lateinit var rows: Map<String, List<String>>

        @BeforeClass
        @JvmStatic
        fun load() {
            val stream = SheetPortFixturesTest::class.java
                .getResourceAsStream("/sheet_port_fixtures.txt")
                ?: error("sheet_port_fixtures.txt not on the test classpath; " +
                    "run python/make_sheet_port_fixtures.py")
            rows = stream.bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() && !it.startsWith("#") }
                    .associate { line -> val p = line.split(" "); p[0] to p.drop(1) }
            }
        }

        private fun row(k: String) = rows[k] ?: error("missing '$k'; regenerate the fixtures")
        private fun ints(k: String) = row(k).map { it.toInt() }.toIntArray()
        private fun doubles(k: String) = row(k).map { it.toDouble() }.toDoubleArray()
        private fun double(k: String) = row(k)[0].toDouble()

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

        private fun close(msg: String, want: DoubleArray, got: DoubleArray, tol: Double = 1e-9) {
            assertEquals("$msg: length", want.size, got.size)
            for (i in want.indices) {
                if (want[i].isNaN()) assertTrue("$msg[$i]: want NaN, got ${got[i]}", got[i].isNaN())
                else assertEquals("$msg[$i]", want[i], got[i], tol)
            }
        }

        private fun layout(): SheetLayout.LeadLayout {
            val p = row("ticks.pos")
            val pos = (p.indices step 3).map { Triple(p[it].toInt(), p[it + 1].toInt(), p[it + 2].toDouble()) }
            val ticks = SheetLayout.TickScale(double("ticks.px"), row("ticks.n")[0].toInt(), 0.0, pos)
            return SheetLayout.segmentLeads(row("layout.height")[0].toInt(), ints("locate.rows"),
                double("locate.t0"), ticks.pxPerMm, ticks)
        }
    }

    // ---- corners -----------------------------------------------------------

    @Test
    fun regionExtremesAndRansacInliersMatch() {
        val m = mask("region")
        val tol = double("ransac.tol")
        for (side in listOf("top", "right", "bottom", "left")) {
            val (xs, ys) = SheetCorners.extremes(m.data, m.cols, m.rows, side)
            assertEquals("$side extremes n", row("extreme.$side.n")[0].toInt(), xs.size)
            val sums = doubles("extreme.$side.sum")
            assertEquals("$side x sum", sums[0], xs.sum(), 1e-6)
            assertEquals("$side y sum", sums[1], ys.sum(), 1e-6)
            val inl = SheetCorners.ransacInliers(xs, ys, tol)
            assertEquals("$side inlier count", row("ransac.$side.count")[0].toInt(), inl.count { it })
            assertEquals("$side inlier idx sum", row("ransac.$side.idxsum")[0].toLong(),
                inl.indices.filter { inl[it] }.sumOf { it.toLong() })
        }
    }

    // ---- locate, ticks, segment ---------------------------------------------

    @Test
    fun rowsAndPulseEdgesMatch() {
        // Locate binarises with the 2x2 opening, like the ticks (OPEN_LOCATE).
        val m = mask("m2")
        assertArrayEquals(ints("locate.rows"), SheetLayout.sheetRows(m))
        val loc = StripLocator.locate(m)
        assertEquals(row("locate.loc_x1")[0].toInt(), loc.x1)
        assertEquals(double("locate.loc_t0"), loc.t0X, 1e-9)
        val approx = loc.recordingWidthPx / (SheetLayout.RECORDING_S * SheetLayout.MM_PER_S)
        assertEquals(double("locate.approx"), approx, 1e-9)
        val rows = ints("locate.rows")
        val edges = DoubleArray(4) { SheetLayout.pulseFallingEdge(m, rows[it], approx, SheetLayout.maskOffsetPx(2)) }
        close("pulse edges", doubles("locate.edges"), edges)
        assertEquals(double("locate.t0"), SheetLayout.t0FromPulses(edges), 1e-9)
    }

    @Test
    fun tickScaleMatches() {
        val t = SheetLayout.tickScale(mask("m2"), ints("locate.rows"), double("locate.t0"),
            double("locate.anchor"), SheetLayout.maskOffsetPx(2))
        assertEquals(row("ticks.n")[0].toInt(), t.nFound)
        assertEquals(double("ticks.px"), t.pxPerMm, 1e-9)
        val p = row("ticks.pos")
        val want = (p.indices step 3).map { Triple(p[it].toInt(), p[it + 1].toInt(), p[it + 2].toDouble()) }
        assertEquals(want.size, t.positions.size)
        for ((a, b) in want.zip(t.positions)) {
            assertEquals(a.first, b.first); assertEquals(a.second, b.second)
            assertEquals(a.third, b.third, 1e-9)
        }
        assertEquals(row("ticks.rejected").count { it.isNotEmpty() } / 3, t.rejected.size)
    }

    /** segment.choose_rows: the same four rows, case by case. */
    @Test
    fun rowsChosenAsTheReference() {
        for (i in 0 until row("choose.n")[0].toInt()) {
            val got = SheetLayout.chooseRows(ints("choose.$i.peaks"), doubles("choose.$i.values"))
            assertEquals("case $i", ints("choose.$i.rows").toList(), got.toList())
        }
    }

    /** segment.reject_tick_outliers: the same ticks rejected, case by case. */
    @Test
    fun tickOutliersRejectedAsTheReference() {
        for (i in 0 until row("reject.n")[0].toInt()) {
            val p = row("reject.$i.pos")
            val pos = (p.indices step 3).map { Triple(p[it].toInt(), p[it + 1].toInt(), p[it + 2].toDouble()) }
            val (keep, drop) = SheetLayout.rejectTickOutliers(pos, double("reject.$i.px"))
            val want = row("reject.$i.drop").filter { it.isNotEmpty() }.map { it.toInt() }
            assertEquals("case $i", want, drop)
            assertEquals("case $i kept", pos.indices.filter { it !in want }, keep)
        }
    }

    @Test
    fun segmentationAndKnotsMatch() {
        val lay = layout()
        val bands = ints("layout.bands")
        for (r in 0 until 4) {
            assertEquals(bands[2 * r], lay.rowBands[r][0]); assertEquals(bands[2 * r + 1], lay.rowBands[r][1])
            val k = doubles("layout.knots.$r")
            close("knots $r xs", DoubleArray(k.size / 2) { k[2 * it] }, lay.rowKnots[r].xs)
            close("knots $r ts", DoubleArray(k.size / 2) { k[2 * it + 1] }, lay.rowKnots[r].ts)
        }
        val cells = ints("layout.cells")
        assertEquals(13 * 4, cells.size)
        for ((i, c) in lay.cells.withIndex()) {
            assertArrayEquals("cell $i ${c.name}", cells.copyOfRange(4 * i, 4 * i + 4),
                intArrayOf(c.x0, c.x1, c.y0, c.y1))
        }
    }

    // ---- classical rows and the leads -----------------------------------------

    @Test
    fun classicalRowsAndEveryLeadMatch() {
        val lay = layout()
        val px = double("ticks.px")
        for (r in 0 until 4) {
            val band = mask("band$r")
            val x0 = row("band$r.x0")[0].toInt()
            val pos = SheetLayout.traceRowFromMask(band, lay, r, x0)
            close("row $r pos", doubles("row$r.pos"), pos)
            val xs = DoubleArray(pos.size) { x0 + it - SheetLayout.maskOffsetPx(2) }
            val (grid, g0) = SheetLayout.onSheetColumns(xs, pos)
            for (c in lay.cells.filter { it.row == r }) {
                val tag = c.name + if (c.rhythm) "_rh" else ""
                close("lead $tag", doubles("lead.$tag"), SheetLayout.cellSignal(grid, g0, c, px, lay))
            }
        }
    }

    // ---- canonical ---------------------------------------------------------

    private fun fakeFrame(h: Int, w: Int): SheetCanonical.Frame {
        val rgb = FloatArray(h * w * 3)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 3)
            rgb[(y * w + x) * 3 + c] = ((x * 3 + y * 7 + c * 11) % 256).toFloat()
        return SheetCanonical.Frame(rgb, h, w, SheetCanonical.CanonMap(1.0, 0, 0))
    }

    private val fakeModel = SheetCanonical.TileModel { t, h, w ->
        val row = FloatArray(2 * w); val pres = FloatArray(2 * w)
        for (j in 0 until 2 * w) {
            var s = 0.0
            for (y in 0 until h) s += t[(y * w + j / 2) * 3]
            row[j] = (s / h + (j % 2) * 0.5).toFloat()
            pres[j] = (t[((h / 2) * w + j / 2) * 3 + 1] - 128.0).toFloat()
        }
        row to pres
    }

    @Test
    fun tileStitchingMatches() {
        val (h, w) = ints("tile.shape").let { it[0] to it[1] }
        // The Python stitches a frame larger than FULL_FRAME_MAX_PX only;
        // this frame is small, so force the tiled branch with the same tile.
        val frame = fakeFrame(h, w)
        val (row, pres) = SheetCanonical.predictTiled(frame, fakeModel, 1024)
        val wantRow = doubles("tile.row"); val wantPres = doubles("tile.pres")
        for (i in wantRow.indices) {
            assertEquals("row[$i]", wantRow[i], row[i].toDouble(), 1e-3)
            assertEquals("pres[$i]", wantPres[i], pres[i].toDouble(), 1e-3)
        }
    }

    @Test
    fun outputToPageMappingMatches() {
        val map = SheetCanonical.CanonMap(0.6743, -27, 41)
        val rin = doubles("map.in").map { it.toFloat() }.toFloatArray()
        val pin = doubles("map.pres").map { it.toFloat() }.toFloatArray()
        val (xs, ys) = SheetCanonical.rowTraceToSource(rin, pin, map)
        close("xs", doubles("map.xs"), xs, 1e-9)
        close("ys", doubles("map.ys"), ys, 1e-4)      // float32 rows in, as on the device
    }

    @Test
    fun padMedianMatchesNumpy() {
        for (n in listOf(1000, 999)) {
            val hist = row("med.$n.hist").map { it.toLong() }.toLongArray()
            assertEquals("n=$n", row("med.$n.want")[0].toInt(),
                SheetCanonical.histMedianTrunc(hist, n.toLong()))
        }
    }

    @Test
    fun interpMatchesNumpyIncludingEnds() {
        val xp = doubleArrayOf(0.0, 1.0, 3.0)
        val fp = doubleArrayOf(10.0, 20.0, 40.0)
        assertEquals(10.0, SheetLayout.interp(-1.0, xp, fp), 0.0)
        assertTrue(SheetLayout.interp(-1.0, xp, fp, Double.NaN, Double.NaN).isNaN())
        assertEquals(30.0, SheetLayout.interp(2.0, xp, fp), 1e-12)
        assertEquals(40.0, SheetLayout.interp(3.0, xp, fp), 0.0)
        assertTrue(abs(SheetLayout.interp(0.5, xp, fp) - 15.0) < 1e-12)
    }
}
