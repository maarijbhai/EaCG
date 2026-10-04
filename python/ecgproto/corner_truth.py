"""True sheet corners for a photograph of a KNOWN printed sheet.

A corner finder needs something to be scored against, and the corners in
`python/kit_photos.json` cannot be it: they were placed by hand or by an
earlier prototype finder, and checked only by eye. What CAN be: the exact
image that was printed. A homography that maps the photograph onto the
printed PNG so that their content coincides puts the PNG's own corners
into the photograph -- the true corners -- with no corner finder involved.

**How.** `cv2.findTransformECC` (enhanced correlation coefficient) with a
full homography, coarse to fine, starting from a rough guess (the
registered corners). ECC maximises a correlation coefficient, so it is
blind to the photograph's overall exposure and contrast; to make it blind
to uneven lighting too, both images are high-passed first (minus a
Gaussian-blurred copy). **Coarse levels are blurred to ~2 mm**, so the 5 mm
grid -- periodic, and able to lock the fit a whole grid square off --
dissolves and the aperiodic content (traces, lead labels, pulses)
dominates; fine levels keep the grid for precision.

Only possible for the test sheets, whose printed image we hold. It is an
evaluation instrument, not a pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


#: (scale relative to the sheet's own pixels, Gaussian sigma in mm) per
#: pyramid level, coarse to fine.
LEVELS = ((0.125, 2.0), (0.25, 1.0), (0.5, 0.4))
ECC_ITER = 200
ECC_EPS = 1e-6


@dataclass(frozen=True)
class CornerTruth:
    corners: np.ndarray        # (4, 2) TL TR BR BL, photograph pixels
    ecc: float                 # final correlation coefficient (finest level)
    shift_px: np.ndarray       # per corner, |truth - initial guess|, photo px


def _prep(img: np.ndarray, sigma_px: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 \
        else img.astype(np.float32)
    if sigma_px > 0.3:
        g = cv2.GaussianBlur(g, (0, 0), sigma_px)
    hp = g - cv2.GaussianBlur(g, (0, 0), max(4.0, 8 * sigma_px))
    return hp / (hp.std() + 1e-6)


def sheet_quad(w: int, h: int) -> np.ndarray:
    """The printed image's outer corners in its own pixel-centre coords."""
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5],
                     [-0.5, h - 0.5]], np.float32)


def find_true_corners(photo: np.ndarray, sheet: np.ndarray, guess: np.ndarray,
                      px_per_mm: float) -> CornerTruth:
    """Corners of `sheet` (the printed PNG, its own pixels) in `photo`."""
    h, w = sheet.shape[:2]
    quad = sheet_quad(w, h)
    # Sheet -> photo homography, refined level by level.
    H = cv2.getPerspectiveTransform(quad, np.asarray(guess, np.float32)).astype(np.float64)
    ecc = float("nan")
    for scale, sigma_mm in LEVELS:
        sw, sh = int(round(w * scale)), int(round(h * scale))
        S = np.diag([scale, scale, 1.0])
        tmpl = _prep(cv2.resize(sheet, (sw, sh), interpolation=cv2.INTER_AREA),
                     sigma_mm * px_per_mm * scale)
        # Photo resampled into the scaled sheet frame through the current H.
        to_photo = H @ np.linalg.inv(S)
        warped = cv2.warpPerspective(photo, to_photo, (sw, sh),
                                     flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
        inp = _prep(warped, sigma_mm * px_per_mm * scale)
        warp = np.eye(3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITER, ECC_EPS)
        ecc, warp = cv2.findTransformECC(tmpl, inp, warp, cv2.MOTION_HOMOGRAPHY,
                                         crit, None, 5)
        # `warp` maps template (scaled sheet) coords to input (warped) coords.
        H = to_photo @ warp.astype(np.float64) @ S
    corners = cv2.perspectiveTransform(quad[None], H)[0]
    shift = np.hypot(*(corners - np.asarray(guess, np.float32)).T)
    return CornerTruth(corners.astype(np.float64), float(ecc), shift)
