"""Tests for sub-pixel sheet-corner refinement.

The sheet is drawn with `synth.draw_grid`, composited onto off-white paper
the way ink goes onto paper (multiplicatively), warped by a known
perspective and JPEG-compressed -- so the true corners are known exactly
and the chroma subsampling that ruled out a redness-based edge is present.

Paired as the other suites are: refinement must pull a bad click onto the
corner, must not push a good click off it, and must refuse a page with no
edge to find rather than return the clicks back dressed as a measurement.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecgproto import page, synth   # noqa: E402

PX_PER_MM = 8.0


def photographed_sheet(grid: bool = True, seed: int = 0):
    """Returns (photo, true corners TL TR BR BL)."""
    sw, sh = int(200 * PX_PER_MM), int(120 * PX_PER_MM)
    sheet = np.full((sh, sw, 3), 255, np.uint8)
    if grid:
        synth.draw_grid(sheet, PX_PER_MM)
    pw, ph = sw + 400, sh + 400
    paper = np.array((236, 241, 244), np.float32)
    canvas = np.empty((ph, pw, 3), np.float32)
    canvas[:] = paper
    canvas[200:200 + sh, 200:200 + sw] = paper * (sheet / 255.0)
    canvas = canvas.astype(np.uint8)

    src = np.float32([[-0.5, -0.5], [pw - 0.5, -0.5], [pw - 0.5, ph - 0.5],
                      [-0.5, ph - 0.5]])
    dst = np.float32([[60, 40], [pw - 20, 80], [pw - 50, ph - 10],
                      [30, ph - 60]])
    m = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(canvas, m, (pw + 40, ph + 40),
                                flags=cv2.INTER_LINEAR,
                                borderValue=(90, 100, 110))
    photo = cv2.GaussianBlur(photo, (0, 0), 0.8)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
    photo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    quad = np.float32([[199.5, 199.5], [199.5 + sw, 199.5],
                       [199.5 + sw, 199.5 + sh], [199.5, 199.5 + sh]])
    true = cv2.perspectiveTransform(quad[None], m)[0]
    return photo, true


def mean_error(a, b) -> float:
    return float(np.hypot(*(np.asarray(a) - np.asarray(b)).T).mean())


class RefineCorners(unittest.TestCase):

    def test_pulls_a_sloppy_click_onto_the_corner(self):
        photo, true = photographed_sheet()
        rough = true + np.random.default_rng(1).normal(0, 8, true.shape)
        self.assertGreater(mean_error(rough, true), 6.0)
        got = page.refine_corners(photo, rough)
        self.assertLess(mean_error(got.corners, true), 1.5)

    def test_does_not_push_an_exact_click_off_the_corner(self):
        photo, true = photographed_sheet()
        got = page.refine_corners(photo, true)
        self.assertLess(mean_error(got.corners, true), 1.5)

    def test_reports_straight_edges_as_straight(self):
        photo, true = photographed_sheet()
        got = page.refine_corners(photo, true)
        self.assertLess(got.edge_rmse_px.max(), 1.0)

    def test_refuses_when_there_is_no_edge_to_find(self):
        photo, true = photographed_sheet(grid=False)
        with self.assertRaises(RuntimeError):
            page.refine_corners(photo, true)


if __name__ == "__main__":
    unittest.main()
