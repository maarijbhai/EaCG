package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Test

/** StageMemory's bookkeeping: which reading lands under which stage. */
class StageMemoryTest {

    private val mb = 1024L * 1024L

    /** A reader that plays back (native MB, java MB) pairs, then repeats the last. */
    private fun reader(vararg mbs: Pair<Int, Int>): () -> StageMemory.Reading {
        var i = 0
        return {
            val (n, j) = mbs[minOf(i, mbs.size - 1)]
            i++
            StageMemory.Reading(n * mb, j * mb)
        }
    }

    @Test
    fun peakBetweenClosesIsFiledUnderTheClosingStage() {
        val m = StageMemory(reader(
            10 to 5,            // baseline
            40 to 6, 25 to 9,   // samples during "grid"
            20 to 7,            // close("grid")
            90 to 7,            // sample during "trace"
            30 to 8))           // close("trace")
        m.sample(); m.sample(); m.close("grid")
        m.sample(); m.close("trace")
        val p = m.peaks()
        assertEquals(listOf("grid", "trace"), p.keys.toList())
        assertEquals(40 * mb, p.getValue("grid").nativeBytes)
        assertEquals(9 * mb, p.getValue("grid").javaBytes)
        assertEquals(90 * mb, p.getValue("trace").nativeBytes)
        assertEquals(8 * mb, p.getValue("trace").javaBytes)
        assertEquals(90 * mb, m.overall().nativeBytes)
        assertEquals(10 * mb, m.baseline.nativeBytes)
    }

    /**
     * A stage's peak starts from what is held when the previous stage ends,
     * not from the previous stage's peak: a transient in "grid" must not be
     * charged to "locate".
     */
    @Test
    fun aPeakDoesNotLeakIntoTheNextStage() {
        val m = StageMemory(reader(
            10 to 5,            // baseline
            80 to 5,            // sample during "grid" (transient)
            15 to 5,            // close("grid"): released
            15 to 5))           // close("locate"), no samples in between
        m.sample(); m.close("grid"); m.close("locate")
        assertEquals(80 * mb, m.peaks().getValue("grid").nativeBytes)
        assertEquals(15 * mb, m.peaks().getValue("locate").nativeBytes)
    }

    /** The close reading counts even with no sample in the stage. */
    @Test
    fun closeAloneRecordsWhatIsHeld() {
        val m = StageMemory(reader(10 to 5, 33 to 6))
        m.close("rectify")
        assertEquals(33 * mb, m.peaks().getValue("rectify").nativeBytes)
    }

    @Test
    fun formatIsBaselineThenStagesInOrder() {
        val m = StageMemory(reader(10 to 5, 20 to 6, 30 to 7))
        m.close("a"); m.close("b")
        assertEquals("baseline=10.0/5.0 a=20.0/6.0 b=30.0/7.0",
            StageMemory.format(m.baseline, m.peaks()))
    }
}
