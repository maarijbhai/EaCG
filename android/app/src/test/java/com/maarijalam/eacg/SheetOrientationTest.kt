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

    @Test fun flipTheDirectionOnlyWhenAnAutoTurnWasTurnedHalfRound() {
        assertTrue(SheetOrientation.shouldFlipDirection(1, 2))
        assertTrue(SheetOrientation.shouldFlipDirection(-1, 6))
        assertFalse(SheetOrientation.shouldFlipDirection(1, 0))     // the auto turn was right
        assertFalse(SheetOrientation.shouldFlipDirection(1, 1))     // a different fix: nothing to learn
        assertFalse(SheetOrientation.shouldFlipDirection(0, 2))     // no auto turn: nothing to learn
    }
}
