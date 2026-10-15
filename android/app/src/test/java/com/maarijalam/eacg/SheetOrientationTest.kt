package com.maarijalam.eacg

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** SheetOrientation against `ecgproto.orientation` (the cases of tests/test_orientation.py). */
class SheetOrientationTest {

    private val cases = listOf(
        doubleArrayOf(174.41, 173.67, 3023.89, 173.67, 3023.90, 2129.41, 174.42, 2129.32) to 1.45702,
        doubleArrayOf(173.67, 175.11, 2129.41, 175.10, 2129.32, 3024.58, 173.67, 3024.59) to 0.68633,
        doubleArrayOf(322.63, 206.65, 2771.52, 220.57, 3019.35, 2036.70, 70.46, 2108.84) to 1.43897,
    )

    @Test fun aspectAndRuleMatchPython() {
        for ((c, asp) in cases) {
            assertEquals(asp, SheetOrientation.outlineAspect(c), 1e-4)
            assertEquals(asp < 1.0, SheetOrientation.isSideways(c))
        }
    }

    @Test fun rotationLabel() {
        assertNull(SheetOrientation.label(0, 0))
        assertEquals("90 clockwise (auto 90)", SheetOrientation.label(1, 0))
        assertEquals("270 clockwise (auto 270)", SheetOrientation.label(-1, 0))
        assertEquals("270 clockwise (auto 90, user 180)", SheetOrientation.label(1, 2))
        assertEquals("180 clockwise (user 180)", SheetOrientation.label(0, 2))
        assertEquals("0 clockwise", SheetOrientation.label(0, 4))   // turned all the way round: said, not hidden
    }

    private class Case(val name: String, val sideways: Boolean, val up: SheetOrientation.Markers,
                       val turned: SheetOrientation.Markers, val upsideDown: Boolean, val quarters: Int)

    private val fixtures: List<Case> by lazy {
        val stream = javaClass.getResourceAsStream("/orientation_fixtures.txt")
            ?: error("orientation_fixtures.txt missing: run python/make_orientation_fixtures.py")
        stream.bufferedReader().readLines().filter { it.isNotBlank() && !it.startsWith("#") }.map { l ->
            val f = l.split(" ")
            val n = f.drop(1).map { it.toInt() }
            Case(f[0], n[0] == 1, SheetOrientation.Markers(n[1], n[2]), SheetOrientation.Markers(n[3], n[4]),
                n[5] == 1, n[6])
        }
    }

    /** Golden vectors from ecgproto/orientation.py: every measured session photograph and every marker count. */
    @Test fun ruleMatchesPython() {
        assertTrue(fixtures.size > 5000)
        assertEquals(67, fixtures.count { it.name.startsWith("session:") })
        for (c in fixtures) {
            assertEquals(c.name, c.upsideDown, SheetOrientation.upsideDown(c.up, c.turned))
            assertEquals(c.name, c.quarters, SheetOrientation.autoQuarters(c.sideways, c.upsideDown))
        }
    }

    /**
     * Real sideways shots whose sheet needed the quarter COUNTER-clockwise
     * (results/kitphoto/orientation_sessions.csv; truth in
     * python/orientation_truth.json). The old rule turned the way the user
     * last confirmed -- clockwise by default -- and never a half more, so it
     * gave 90 on each of these; the markers give 270.
     */
    @Test fun sidewaysShotsNeedingTheOtherQuarterAreTurnedRight() {
        val real = listOf("eacg_20261013_110920.jpg", "eacg_20261013_113202.jpg", "eacg_20261013_115101.jpg",
            "eacg_20261013_115219.jpg", "eacg_20261013_134510.jpg", "eacg_20261013_143647.jpg")
        for (name in real) {
            val c = fixtures.single { it.name == "session:$name" }
            assertTrue(name, c.sideways)
            assertEquals(name, 3, SheetOrientation.autoQuarters(c.sideways, SheetOrientation.upsideDown(c.up, c.turned)))
        }
    }

    /** Ticks at or below six never turn a page: a page turned half round reaches six by symmetry. */
    @Test fun fewTicksAloneNeverTurnThePage() {
        // setB 133810, upright: 0 pulses / 1 tick against 0 / 6 turned.
        assertFalse(SheetOrientation.upsideDown(SheetOrientation.Markers(0, 1), SheetOrientation.Markers(0, 6)))
        // setB 103050, upright: 0 / 9 against 1 / 2 -- more than six ticks outrank a pulse.
        assertFalse(SheetOrientation.upsideDown(SheetOrientation.Markers(0, 9), SheetOrientation.Markers(1, 2)))
    }
}
