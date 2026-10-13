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


if __name__ == "__main__":
    unittest.main()
