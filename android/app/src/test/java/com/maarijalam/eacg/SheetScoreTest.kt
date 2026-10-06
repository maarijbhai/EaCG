package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * SheetPipeline.score: the per-lead PCC and median-removed RMSE the phone
 * writes into its sheet CSV. Before 2026-10-06 it took each signal's median
 * inside a per-sample map, re-sorting the window once per sample: 7.3 s of a
 * sheet on the A30s (eacg_sheet_ptbxl38_..._164847_915.csv, "cut+score").
 */
class SheetScoreTest {

    private fun median(v: DoubleArray): Double {
        val s = v.sortedArray(); val n = s.size
        return if (n % 2 == 1) s[n / 2] else 0.5 * (s[n / 2 - 1] + s[n / 2])
    }

    private fun signals(n: Int): Pair<DoubleArray, DoubleArray> {
        val truth = DoubleArray(n) { sin(it * 0.013) + 0.3 * sin(it * 0.11) }
        val sig = DoubleArray(n) { 0.9 * truth[it] + 0.2 + 0.05 * sin(it * 0.7) }
        sig[17] = Double.NaN; sig[n / 2] = Double.NaN     // NaN samples are skipped
        return sig to truth
    }

    @Test
    fun matchesNaiveReference() {
        val (sig, truth) = signals(1250)
        val idx = sig.indices.filter { sig[it].isFinite() }
        val a = DoubleArray(idx.size) { truth[idx[it]] }
        val b = DoubleArray(idx.size) { sig[idx[it]] }
        val ma = a.average(); val mb = b.average()
        val pcc = a.indices.sumOf { (a[it] - ma) * (b[it] - mb) } /
            sqrt(a.sumOf { (it - ma) * (it - ma) } * b.sumOf { (it - mb) * (it - mb) })
        val medA = median(a); val medB = median(b)
        val rmse = sqrt(a.indices.sumOf { val d = (a[it] - medA) - (b[it] - medB); d * d } / a.size)

        val got = SheetPipeline.score(sig, truth)
        assertNotNull(got)
        assertEquals(pcc, got!!.first, 1e-12)
        assertEquals(rmse, got.second, 1e-12)
    }

    @Test
    fun tooFewSamplesIsNull() {
        val (sig, truth) = signals(1250)
        for (i in 49 until sig.size) sig[i] = Double.NaN
        assertEquals(null, SheetPipeline.score(sig, truth))
    }

    /**
     * Fails under the old per-sample median: a 5,000-sample rhythm strip was
     * 5,000 sorts of 5,000 values (seconds on a desktop JVM). The bound is
     * loose on purpose -- it catches an O(n^2) regression, not a slow machine.
     */
    @Test
    fun rhythmStripScoresQuickly() {
        val (sig, truth) = signals(5000)
        SheetPipeline.score(sig, truth)                    // warm-up
        val t0 = System.nanoTime()
        repeat(13) { SheetPipeline.score(sig, truth) }     // one sheet's worth, all at rhythm length
        val ms = (System.nanoTime() - t0) / 1e6
        assertTrue("13 scores took $ms ms", ms < 1000.0)
    }
}
