package com.maarijalam.eacg

import kotlin.math.roundToInt

/**
 * The diagnosis classifier's input from a digitised sheet: a port of
 * `python/ecgproto/clf_input.py`, golden-vector tested (ClfInputPortTest,
 * `python/make_clf_port_fixtures.py`). Pure JVM code.
 *
 * The classifier was trained on PTB-XL at 100 Hz, 10 s = 1,000 samples, 13
 * channels: each lead only inside its printed 2.5 s window, about that
 * window's median, zero elsewhere; channel 13 the lead II rhythm strip, all
 * 10 s. A digitised sheet is at 500 Hz, laid out as SheetPipeline.csv: per
 * lead a 10 s column holding only its own cell (NaN elsewhere), and the
 * rhythm strip. 500 -> 100 Hz is the NaN-ignoring mean of each block of five
 * samples: a low-pass filter whose blocks sit exactly on the cell boundaries
 * (a cell is 1,250 = 250 x 5 samples), so no window borrows its neighbour's
 * samples. A gap the digitiser left becomes 0 (flat), as in training.
 */
object ClfInput {
    const val BLOCK = 5
    const val N = 1000
    const val CELL = 250
    const val CHANNELS = 13
    /** PTB-XL lead order; channel 13 is the rhythm strip. */
    val LEADS = listOf("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    /** Lead index -> column (0-3) of its 2.5 s window. */
    private val PRINTED_COLUMN = intArrayOf(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    const val RHYTHM = "II_rhythm"

    /** 5,000 samples at 500 Hz -> 1,000 at 100 Hz (clf_input.decimate). */
    fun decimate(x500: DoubleArray): DoubleArray = DoubleArray(N) { i ->
        var s = 0.0
        var n = 0
        for (j in 0 until BLOCK) {
            val k = i * BLOCK + j
            if (k < x500.size && x500[k].isFinite()) { s += x500[k]; n++ }
        }
        if (n > 0) s / n else Double.NaN
    }

    /** A window minus its NaN-ignoring median, NaN -> 0. */
    private fun aboutMedian(w: DoubleArray): DoubleArray {
        val fin = w.filter { it.isFinite() }.toDoubleArray()
        val med = if (fin.isEmpty()) 0.0 else SheetLayout.median(fin)
        return DoubleArray(w.size) { val v = w[it] - med; if (v.isFinite()) v else 0.0 }
    }

    /**
     * 12 leads at 100 Hz (each 1,000 samples) and the rhythm strip -> the
     * input, row-major: sample t, channel c at `t * 13 + c`.
     */
    fun printedView(x100: List<DoubleArray>, rhythm100: DoubleArray): FloatArray {
        val out = FloatArray(N * CHANNELS)
        for (lead in 0 until 12) {
            val a = PRINTED_COLUMN[lead] * CELL
            val w = aboutMedian(DoubleArray(CELL) { x100[lead][a + it] })
            for (i in 0 until CELL) out[(a + i) * CHANNELS + lead] = w[i].toFloat()
        }
        val r = aboutMedian(rhythm100)
        for (t in 0 until N) out[t * CHANNELS + 12] = r[t].toFloat()
        return out
    }

    /** SheetPipeline.csv's columns (lead -> 10 s at 500 Hz; [RHYTHM] the strip) -> input. */
    fun fromColumns(cols: Map<String, DoubleArray>): FloatArray {
        val blank = DoubleArray(N * BLOCK) { Double.NaN }
        return printedView(LEADS.map { decimate(cols[it] ?: blank) }, decimate(cols[RHYTHM] ?: blank))
    }

    /** The pipeline's leads laid out as SheetPipeline.csv does, then [fromColumns]. */
    fun fromLeads(leads: List<SheetPipeline.LeadResult>): FloatArray {
        val n = (SheetLayout.RECORDING_S * SheetLayout.FS).toInt()
        val cols = HashMap<String, DoubleArray>()
        for (l in leads) {
            if (l.rhythm) { cols[RHYTHM] = l.mv; continue }
            val a = (l.tStartS * SheetLayout.FS).roundToInt()
            cols[l.name] = DoubleArray(n) { i -> if (i in a until a + l.mv.size) l.mv[i - a] else Double.NaN }
        }
        return fromColumns(cols)
    }
}
