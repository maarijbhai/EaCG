"""Is the sheet sideways? The quarter-turn rule for camera photographs.

A phone stores which way up a photograph is in its EXIF tag, from the tilt
sensor at the moment of capture. Shooting a sheet on a table from above,
the phone lies flat, gravity says nothing about up, and the tag keeps the
orientation the phone last knew -- usually portrait. A landscape shot then
arrives sideways, and the pipeline, which expects an upright landscape page,
fails.

The printed sheet is landscape (277 x 190 mm), so the corner finder's
outline settles the quarter-turn question on its own: taller than wide
means sideways. Measured (`python/run_orientation.py`,
results/kitphoto/orientation.csv): 76 of 76 right over the kit photographs,
the in-app camera photographs and the clean sheets, each turned four ways;
upright outlines 1.18-2.11 wide-to-high, sideways 0.47-0.85.

Which quarter turn (clockwise or counter-clockwise) -- i.e. up versus down --
is NOT decided from the image: two cues were tried and rejected
(progress_log §35; the calibration pulses' count from `locate_sheet`, 20/36
on the kit photographs, and the pulses' vertical strokes at the row ends,
broken by the 2x2 mask on many photographs). The app turns the way the user
last confirmed (ScanController) and offers a Rotate button.
"""

from __future__ import annotations

import numpy as np

#: Width / height below which the sheet's outline is sideways.
SIDEWAYS_BELOW = 1.0


def outline_aspect(corners: np.ndarray) -> float:
    """Width / height of the corner quad (TL, TR, BR, BL), opposite sides averaged."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3]))
    h = 0.5 * (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1]))
    return float(w / h)


def is_sideways(corners: np.ndarray) -> bool:
    return outline_aspect(corners) < SIDEWAYS_BELOW
