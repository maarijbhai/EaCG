"""orientation: the sideways rule (the phone's SheetOrientation mirrors it)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import orientation as O  # noqa: E402

#: Shared with SheetOrientationTest.kt: (corners TL TR BR BL, expected aspect).
CASES = [
    ([174.41, 173.67, 3023.89, 173.67, 3023.90, 2129.41, 174.42, 2129.32], 1.45702),   # clean image, upright
    ([173.67, 175.11, 2129.41, 175.10, 2129.32, 3024.58, 173.67, 3024.59], 0.68633),   # the same, turned CCW
    ([322.63, 206.65, 2771.52, 220.57, 3019.35, 2036.70, 70.46, 2108.84], 1.43897),    # a tilted photograph
]


class Sideways(unittest.TestCase):

    def test_aspect_and_rule(self):
        for c, asp in CASES:
            self.assertAlmostEqual(O.outline_aspect(np.array(c)), asp, places=4)
            self.assertEqual(O.is_sideways(np.array(c)), asp < 1.0)

    def test_quarter_turn_inverts_the_aspect(self):
        c = np.array(CASES[2][0]).reshape(4, 2)
        w = 3200.0
        # np.rot90 (CCW): (x, y) -> (y, W - 1 - x); the new TL is the old TR, and so on.
        r = np.stack([c[:, 1], w - 1 - c[:, 0]], axis=1)[[1, 2, 3, 0]]
        self.assertAlmostEqual(O.outline_aspect(r), 1.0 / O.outline_aspect(c), places=9)


M = O.Markers


class UpsideDown(unittest.TestCase):
    """The half-turn rule. Real cases: (pulses, ticks) on the grid page and on it
    turned half round, from results/kitphoto/orientation_sessions.csv."""

    def test_many_ticks_decide(self):
        # setB 103050: upright, 0 pulses / 9 ticks against 1 / 2 turned -- the pulses
        # alone would turn it; more than MAX_TICKS_TURNED ticks cannot be a turned page.
        self.assertFalse(O.upside_down(M(0, 9), M(1, 2)))
        # setB 115219: sideways, the clockwise quarter was the wrong one.
        self.assertTrue(O.upside_down(M(0, 6), M(3, 9)))

    def test_pulses_decide_when_neither_page_has_many_ticks(self):
        # setB 104932: upright, 3 / 3 against 0 / 6 (a turned page reaches 6 ticks).
        self.assertFalse(O.upside_down(M(3, 3), M(0, 6)))
        # setB 113202: sideways, the clockwise quarter was the wrong one (3 / 8 turned).
        self.assertTrue(O.upside_down(M(0, 5), M(3, 8)))
        self.assertTrue(O.upside_down(M(0, 1), M(1, 0)))

    def test_ticks_at_or_below_six_never_turn_the_page(self):
        # setB 133810: upright, 0 / 1 against 0 / 6 -- the rejected third step
        # ("more ticks wins") turned it over (orientation_sessions_rule_v1.csv).
        self.assertFalse(O.upside_down(M(0, 1), M(0, 6)))
        for a in range(O.MAX_TICKS_TURNED + 1):
            for b in range(O.MAX_TICKS_TURNED + 1):
                self.assertFalse(O.upside_down(M(1, a), M(1, b)))

    def test_a_tie_keeps_the_page(self):
        self.assertFalse(O.upside_down(M(0, 0), M(0, 0)))
        self.assertFalse(O.upside_down(M(1, 9), M(1, 9)))

    def test_max_ticks_turned_is_two_rows_of_ticks(self):
        self.assertEqual(O.MAX_TICKS_TURNED, 2 * 3)


if __name__ == "__main__":
    unittest.main()
