"""Sub-pixel sheet corners from four rough clicks.

The stand-in for FR1 until ML Kit's Document Scanner is wired in: a human
clicks roughly where the four corners of the printed ECG sheet are, and this
module moves each click onto the true corner by fitting a straight line to
each edge of the gridded area and intersecting neighbouring lines.

**Why this exists at all.** Measured on simulated photographs
(`results/kitphoto/selftest.csv`): corner clicks scattered by 3 px drop the
reconstruction's PCC from 0.98 to 0.78-0.88 with scale and crop otherwise
known, because a corner error is a scale and offset error and limitation 14
already showed that a quarter of a percent of scale costs a fifth of PCC.
A click is worse than 3 px in practice: the picker scales a 24.8 MP frame
down about fivefold to fit a screen, so each click is quantised to ~5 px.

**The edge.** The printed image is gridded to its border and the paper
around it is not, so the edge is where the grid stops -- specifically where
the rules that CROSS the edge stop. At each distance from the edge, ink
darkness is sampled along a few millimetres parallel to it and its
**spread** (standard deviation) taken. Inside, the crossing rules make that
spread large, because a line parallel to the edge alternately hits rule and
paper; outside, it is sensor noise. The edge is placed at the half-level
crossing of that step, to sub-pixel precision by linear interpolation.

**Why the spread and not the mean.** A rule lying *along* the edge is
uniform along it, so it adds nothing to the spread but a great deal to the
mean, and its blurred tail pulled the mean's half-level crossing 1.3-2.4 px
outside the sheet on a synthetic sheet with rules on its border (the kit
sheets have them too: the first column of `ptbxl9` is a rule). Measured on
the simulated photographs of all four kit sheets, the mean placed corners
0.8-2.2 px from truth and the spread 0.26-0.51 px.

**Luma, not redness.** Redness is the obvious response for a pink grid
and it is the wrong one here, for the reason limitation 15 found on the
2026-09-21 photographs: it is a chroma quantity, and JPEG stores chroma at
half resolution. Measured on a simulated 4:2:0 JPEG, the redness step
across the left edge was smeared over ~10 px and the fitted edge landed 5
to 9 px outside the sheet. Luma is stored at full resolution.

**What it cannot do.** It needs the clicks to be within `SEARCH_FRACTION`
of the edge length of the true corner, it needs white paper outside the
sheet (a sheet printed edge-to-edge, or photographed on a red tablecloth,
has no step), and it assumes straight edges -- a curled page is bent, and
the fitted line is then a chord. `refine_corners` reports the residual of
each line fit so a curled edge is visible rather than silently averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


#: How far either side of a clicked edge the step is searched for, as a
#: fraction of that edge's length. 1.5% is about 4 mm on a 277 mm edge --
#: generous for a careful click, not so wide that the search reaches the
#: sheet's first trace.
SEARCH_FRACTION = 0.015

#: Samples along each edge. Each is one independent edge crossing; the line
#: is fitted through them.
N_SAMPLES = 80

#: Length along the edge averaged into each sample, as a fraction of the
#: edge length. About 4 mm: long enough to span several 1 mm rules, so the
#: spread inside is the grid's rather than whichever rule one sample hit.
TANGENT_FRACTION = 0.015

#: Fraction of each edge, at both ends, not sampled. Near a corner the
#: tangent window runs off the sheet onto the neighbouring edge's paper.
END_MARGIN = 0.06

#: A sample counts as a step only if the spread inside exceeds the spread
#: outside by this many grey levels, and by STEP_SNR times the outside's
#: own variability.
MIN_STEP = 2.0
STEP_SNR = 4.0

#: Samples further than this many robust standard deviations from the
#: first line fit are dropped and the line refitted. A trace touching the
#: edge, or a grid rule lying exactly on it, makes an outlier crossing.
OUTLIER_SIGMA = 3.0


@dataclass(frozen=True)
class RefinedCorners:
    corners: np.ndarray          # (4, 2) float32, TL TR BR BL
    shift_px: np.ndarray         # (4,) distance each corner moved
    edge_rmse_px: np.ndarray     # (4,) residual of each edge's line fit
    edge_inliers: np.ndarray     # (4,) samples kept per edge


def ink_darkness(image: np.ndarray) -> np.ndarray:
    """255 minus BT.601 luma, float32: zero on white paper, high on ink."""
    b, g, r = cv2.split(image.astype(np.float32))
    return 255.0 - (0.114 * b + 0.587 * g + 0.299 * r)


def _sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of a single-channel float32 image at (x, y) points."""
    m = pts.astype(np.float32).reshape(-1, 1, 2)
    return cv2.remap(img, m[..., 0], m[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(pts.shape[:-1])


def _edge_crossings(resp: np.ndarray, a: np.ndarray, b: np.ndarray,
                    outward: np.ndarray) -> np.ndarray:
    """Points where the along-edge spread of darkness falls from the grid's
    level (inside) to the paper's (outside) through its half level, one per sample along the edge a->b. NaN rows where no
    clean step was found."""
    length = float(np.hypot(*(b - a)))
    tangent = (b - a) / length
    reach = max(6.0, SEARCH_FRACTION * length)
    half_t = max(3.0, 0.5 * TANGENT_FRACTION * length)
    offsets = np.arange(-reach, reach + 0.5, 0.5)              # along normal
    along = np.arange(-half_t, half_t + 0.5, 1.0)              # along edge
    out = np.full((N_SAMPLES, 2), np.nan)
    for i, t in enumerate(np.linspace(END_MARGIN, 1 - END_MARGIN, N_SAMPLES)):
        c = a + t * (b - a)
        grid = (c[None, None, :] + offsets[:, None, None] * outward
                + along[None, :, None] * tangent)
        profile = _sample(resp, grid).std(axis=1)              # (n_offsets,)
        k = max(3, len(offsets) // 8)
        inside, outside = np.median(profile[:k]), np.median(profile[-k:])
        # Against noise, not against the inside level: a relative test on
        # the mean rejected every sample on the shaded edge of a simulated
        # photograph, where shading raised the level but not the step.
        noise = 1.4826 * np.median(np.abs(profile[-k:] - outside))
        if inside - outside <= max(MIN_STEP, STEP_SNR * noise):
            continue                                            # no step here
        level = 0.5 * (inside + outside)
        # The LAST fall through the half level going outward: inside, the
        # spread dips wherever a rule parallel to the edge fills the whole
        # sample line, and those dips are not the edge.
        above = profile >= level
        idx = np.flatnonzero(above[:-1] & ~above[1:])
        if idx.size == 0:
            continue
        j = int(idx[-1])
        f = (profile[j] - level) / (profile[j] - profile[j + 1])
        s = offsets[j] + f * (offsets[j + 1] - offsets[j])
        out[i] = c + s * outward
    return out


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Total-least-squares line with one round of outlier rejection.
    Returns (point on line, unit direction, rmse px, inliers)."""
    p = points[np.all(np.isfinite(points), axis=1)]
    if len(p) < 5:
        raise RuntimeError(f"only {len(p)} edge crossings found")
    for _ in range(2):
        mu = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - mu)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        r = (p - mu) @ n
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = np.abs(r) <= OUTLIER_SIGMA * max(sigma, 0.25)
        if keep.all() or keep.sum() < 5:
            break
        p = p[keep]
    return mu, d, float(np.sqrt(np.mean(r ** 2))), int(len(p))


def _intersect(p1, d1, p2, d2) -> np.ndarray:
    a = np.array([d1, -d2]).T
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def refine_corners(image: np.ndarray, rough: np.ndarray) -> RefinedCorners:
    """Move four rough sheet corners (TL TR BR BL) onto the gridded area's
    true corners. Raises `RuntimeError` if an edge shows no step."""
    rough = np.asarray(rough, dtype=np.float64).reshape(4, 2)
    resp = ink_darkness(image)
    centre = rough.mean(axis=0)
    lines, rmse, inl = [], [], []
    for i in range(4):
        a, b = rough[i], rough[(i + 1) % 4]
        mid = 0.5 * (a + b)
        d = (b - a) / np.hypot(*(b - a))
        n = np.array([-d[1], d[0]])
        if np.dot(n, mid - centre) < 0:
            n = -n
        pts = _edge_crossings(resp, a, b, n)
        p, dirn, e, k = _fit_line(pts)
        lines.append((p, dirn))
        rmse.append(e)
        inl.append(k)
    # Corner i joins edge i-1 (arriving) and edge i (leaving).
    corners = np.array([_intersect(*lines[(i - 1) % 4], *lines[i])
                        for i in range(4)])
    return RefinedCorners(
        corners=corners.astype(np.float32),
        shift_px=np.hypot(*(corners - rough).T),
        edge_rmse_px=np.array(rmse),
        edge_inliers=np.array(inl),
    )
