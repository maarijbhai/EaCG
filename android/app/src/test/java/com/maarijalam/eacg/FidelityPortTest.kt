package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The on-device fidelity metrics against golden vectors from
 * `ecgproto.evaluate`.
 *
 * These are the numbers the handset reports for itself, so a porting bug here
 * would not produce a visibly broken app — it would produce a *plausible
 * wrong figure in the results chapter*, which is the worse failure. Hence
 * golden vectors rather than a smoke test.
 *
 * The cases are chosen so the three metrics disagree with each other: a
 * scaled copy is perfect on PCC and wrong on gain, an inverted one is -1.0 on
 * PCC, and an offset one is invisible to both PCC and gain while RMSE sees
 * it. A fixture where all three agreed could not detect a metric wired to the
 * wrong formula.
 */
class FidelityPortTest {

    private val fx = PortFixtures.load()

    private fun assertClose(expected: Double, actual: Double, what: String) {
        if (expected.isNaN()) {
            assertTrue("$what: expected NaN, got $actual", actual.isNaN())
            return
        }
        assertTrue("$what: expected $expected, got NaN", !actual.isNaN())
        assertEquals(what, expected, actual, 1e-12 * maxOf(1.0, kotlin.math.abs(expected)))
    }

    @Test
    fun `every fidelity case matches the python`() {
        val ref = fx.doubles("fid.ref")
        for (i in 0 until fx.int("fid.n")) {
            val name = fx.row("fid.$i.name")[0]
            val test = fx.doubles("fid.$i.test")
            val scores = Fidelity.score(ref, test)
            assertClose(fx.double("fid.$i.out", 0), scores.pcc, "$name pcc")
            assertClose(fx.double("fid.$i.out", 1), scores.rmseMv, "$name rmse")
            assertClose(fx.double("fid.$i.out", 2), scores.gainRatio, "$name gain")
            assertEquals("$name n", fx.int("fid.$i.out", 3), scores.nCompared)
        }
    }

    @Test
    fun `an offset is invisible to PCC and gain but not to RMSE`() {
        // The reason the metrics are never quoted singly, pinned as a test.
        // A baseline error is exactly this shape, and the baseline comes from
        // a median documented to drift on leads with sustained deflections.
        val ref = fx.doubles("fid.ref")
        val offset = DoubleArray(ref.size) { ref[it] + 0.5 }
        val s = Fidelity.score(ref, offset)
        assertEquals(1.0, s.pcc, 1e-12)
        assertEquals(1.0, s.gainRatio, 1e-12)
        assertTrue("RMSE must see the offset", s.rmseMv > 0.4)
    }

    @Test
    fun `an inverted waveform scores minus one rather than failing`() {
        // Which is why PCC is never read alone: the pixels-to-mV sign
        // convention getting reversed produces -1.0, a tidy number that reads
        // as a result rather than as a fault.
        val ref = fx.doubles("fid.ref")
        val flipped = DoubleArray(ref.size) { -ref[it] }
        assertEquals(-1.0, Fidelity.score(ref, flipped).pcc, 1e-12)
    }

    @Test
    fun `unequal lengths truncate rather than resample`() {
        // Resampling would stretch the reconstruction onto the reference's
        // length using the very quantity being measured, silently undoing a
        // time-axis calibration error before any metric saw it.
        val ref = fx.doubles("fid.ref")
        val longer = DoubleArray(ref.size + 5) { if (it < ref.size) ref[it] else 9.9 }
        val s = Fidelity.score(ref, longer)
        assertEquals(ref.size, s.nCompared)
        assertEquals("the extra samples must not affect the score",
            1.0, s.pcc, 1e-12)
    }

    @Test
    fun `a constant series gives NaN correlation, not zero`() {
        val ref = fx.doubles("fid.ref")
        val flat = DoubleArray(ref.size)
        assertTrue(Fidelity.score(ref, flat).pcc.isNaN())
    }
}
