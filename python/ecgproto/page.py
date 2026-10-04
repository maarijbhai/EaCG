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


# --------------------------------------------------------------------------
# FR1 without clicks: find the printed sheet in a photograph
# --------------------------------------------------------------------------

#: Working size for the coarse search, long side in pixels.
FIND_MAX_DIM = 1200
#: Morphological closing that fuses the grid's rules into one solid region,
#: as a fraction of the working image's long side (~6 mm on a sheet that
#: fills the frame): wider than a 5 mm grid square, narrower than the white
#: margin around the print.
FIND_CLOSE_FRACTION = 0.02
#: Edge-line fit: contour points within this fraction of each end of an
#: edge are left out, because the closing rounds the corners off.
FIND_END_TRIM = 0.12
FIND_EDGE_BAND = 0.02


@dataclass(frozen=True)
class FoundCorners:
    corners: np.ndarray          # (4, 2) TL TR BR BL, full-resolution pixels
    coarse: np.ndarray           # before refine_corners
    refined: bool                # refine_corners accepted (edge fit <= max rmse)
    area_fraction: float         # sheet region / frame, a sanity statistic
    edge_rmse_px: np.ndarray | None


def _redness(img: np.ndarray) -> np.ndarray:
    b, g, r = [img[..., i].astype(np.float32) for i in range(3)]
    return r - 0.5 * (g + b)


def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    p = pts[np.argsort(ang)]                      # clockwise from -pi (left)
    s = p.sum(axis=1)
    k = int(np.argmin(s))                         # top-left has smallest x+y
    return np.roll(p, -k, axis=0)


#: Paper: whiteness = min(B, G, R), so a bright but saturated background
#: (orange wood, a coloured cloth) scores low on its weakest channel.
#: Grid: gradient magnitude, smoothed over FIND_TEXTURE_FRACTION of the long
#: side (a few grid squares), thresholded by Otsu over paper pixels only,
#: after eroding the paper by FIND_PAPER_ERODE_FRACTION so the paper's own
#: edge does not read as texture.
FIND_TEXTURE_FRACTION = 0.012
FIND_PAPER_ERODE_FRACTION = 0.01
FIND_MARGIN_PCT = 5.0
FIND_GRID_CUT = 0.35


def _largest(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise RuntimeError("empty mask")
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = (lab == big).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(out)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return filled


def sheet_region(small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(printed-grid region, paper region) as uint8 masks of `small`."""
    white = small.min(axis=2)
    white = cv2.GaussianBlur(white, (0, 0), 2.0)
    _, paper = cv2.threshold(white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = _largest(paper)
    luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    sig = max(1.0, FIND_TEXTURE_FRACTION * max(small.shape[:2]))
    # Contrast-normalised: edge strength over local brightness. Raw gradient
    # scales with illumination, and under the uneven light of
    # ptbxl40_20260929_01 the shaded quarter of the grid read as blank margin.
    tex = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sig) / (
        cv2.GaussianBlur(luma, (0, 0), sig) + 8.0)
    e = max(3, int(round(FIND_PAPER_ERODE_FRACTION * max(small.shape[:2]))))
    inner = cv2.erode(paper, cv2.getStructuringElement(cv2.MORPH_RECT, (e, e))) > 0
    if inner.sum() < 100:
        raise RuntimeError("no paper found")
    # The cut that matters is grid against the BLANK MARGIN, so it sits a
    # little above the margin's own texture (the paper's low percentile)
    # toward the typical grid level (its median; the print is most of the
    # page). Otsu over the paper split the grid itself into trace rows and
    # empty grid on 2026-09-29 photos, leaving a ragged, notched region.
    vals = tex[inner]
    lo, mid = np.percentile(vals, FIND_MARGIN_PCT), np.median(vals)
    grid = ((tex > lo + FIND_GRID_CUT * (mid - lo)) & inner).astype(np.uint8) * 255
    k = max(3, int(round(FIND_CLOSE_FRACTION * max(small.shape[:2]))))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k // 2 + 1, k // 2 + 1)))
    return _largest(grid), paper


FIND_RANSAC_TOL = 0.004          # inlier band, fraction of the long side
FIND_RANSAC_ITERS = 400


def _ransac_line(pts: np.ndarray, tol: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """(point, unit direction) of the line most of `pts` agree on."""
    n = len(pts)
    if n < 10:
        raise RuntimeError("too few edge points")
    best, best_in = None, -1
    for _ in range(FIND_RANSAC_ITERS):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        inl = np.abs((pts - pts[i]) @ nrm) <= tol
        c = int(inl.sum())
        if c > best_in:
            best_in, best = c, inl
    q = pts[best].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(q, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)


#: Accept the full-resolution refinement if every edge's line fit is within
#: this many mm. A fixed 3 px (the app's hand-corner rule) is ~0.15 mm on a
#: 24 MP photo, below the bending a phone lens puts into a page edge: it
#: rejected refinements that land within 0.1-2.3 mm of the registration
#: truth on the 2026-09-29 photos (edge rmse 1-6 px at ~19 px/mm).
FIND_REFINE_MAX_RMSE_MM = 0.4
#: For converting that to pixels: the printed grid is ~270 mm wide on an A4
#: or Letter landscape page. A +-10% guess is plenty for a threshold.
FIND_NOMINAL_WIDTH_MM = 270.0


def find_sheet_corners(photo: np.ndarray, refine: bool = True,
                       max_edge_rmse_px: float | None = None) -> FoundCorners:
    """The printed sheet's four corners in a photograph, with no clicks.

    1-2. At FIND_MAX_DIM, `sheet_region`: the PAPER first (bright and
       unsaturated), then, inside it, the printed GRID (dense fine lines,
       against a blank margin). Not redness: on the 2026-09-29 photos the
       desk is orange wood, redder than the pale grid, and a redness mask
       found the desk (corners 50-600 mm off).
    3. Its outline is split into four edges at the polygon approximation's
       corners; a straight line is fitted to each edge (ends trimmed, Huber
       loss) and neighbours intersected: the coarse corners.
    4. `refine_corners` (the grid-stops-here step, sub-pixel, luma) at full
       resolution from the coarse corners, kept if every edge fits within
       `max_edge_rmse_px` (default FIND_REFINE_MAX_RMSE_MM in pixels);
       otherwise the coarse corners are returned, `refined` False.

    Raises RuntimeError when no sheet-like quadrilateral is found.
    Assumes a white margin of paper around the print and a background that
    is darker or more saturated than paper; a print running edge to edge,
    or white paper on a white table, defeats step 1.
    """
    h, w = photo.shape[:2]
    s = FIND_MAX_DIM / max(h, w)
    small = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    region, paper = sheet_region(small)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    area_frac = float(region.mean() / 255.0)

    # Each edge as a straight line through the region's EXTREME pixels: the
    # topmost grid pixel of every column for the top edge, bottommost for
    # the bottom, leftmost / rightmost of every row for the sides, fitted by
    # RANSAC so that a hole in the mask (a patch of grid it missed, which
    # moves that stretch's extreme inward along a curve) is outvoted by the
    # straight remainder of the edge. Outline and hull fits inherited such
    # holes: corners 15-66 mm off on two 2026-09-29 photos. Assumes the
    # sheet is roughly upright in the frame (within ~20 degrees).
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(region)
    if xs.size < 100:
        raise RuntimeError("sheet region too small")
    lines = []
    for side in ("top", "right", "bottom", "left"):
        if side in ("top", "bottom"):
            # extremes per column
            cols = region.any(axis=0)
            first = region.argmax(axis=0).astype(np.float64)
            last = region.shape[0] - 1 - region[::-1].argmax(axis=0).astype(np.float64)
            vals = first if side == "top" else last
            t = np.flatnonzero(cols).astype(np.float64)
            pts = np.stack([t, vals[cols]], 1)            # (x, y)
        else:
            rows = region.any(axis=1)
            first = region.argmax(axis=1).astype(np.float64)
            last = region.shape[1] - 1 - region[:, ::-1].argmax(axis=1).astype(np.float64)
            vals = last if side == "right" else first
            t = np.flatnonzero(rows).astype(np.float64)
            pts = np.stack([vals[rows], t], 1)            # (x, y)
        lines.append(_ransac_line(pts, FIND_RANSAC_TOL * max(region.shape), rng))
    # lines in order top, right, bottom, left; corner i = previous & next.
    lines = [lines[3], lines[0], lines[1], lines[2]]     # left, top, right, bottom
    # TL = left & top, TR = top & right, BR = right & bottom, BL = bottom & left.
    coarse = np.array([_intersect(lines[i][0], lines[i][1],
                                  lines[(i + 1) % 4][0], lines[(i + 1) % 4][1])
                       for i in range(4)])
    coarse = coarse / s
    # A quad whose corners leave the frame by more than a few percent is a
    # failed coarse stage, not a sheet; refining it samples far off-image.
    tol = 0.05 * max(h, w)
    if not np.all(np.isfinite(coarse)) or np.any(coarse < -tol) or             np.any(coarse[:, 0] > w + tol) or np.any(coarse[:, 1] > h + tol):
        raise RuntimeError("coarse corners fall outside the photograph")
    if max_edge_rmse_px is None:
        width_px = 0.5 * (np.hypot(*(coarse[1] - coarse[0])) + np.hypot(*(coarse[2] - coarse[3])))
        max_edge_rmse_px = FIND_REFINE_MAX_RMSE_MM * width_px / FIND_NOMINAL_WIDTH_MM
    if refine:
        try:
            rc = refine_corners(photo, coarse.astype(np.float32))
            if float(np.max(rc.edge_rmse_px)) <= max_edge_rmse_px:
                return FoundCorners(np.asarray(rc.corners, np.float64), coarse, True,
                                    area_frac, np.asarray(rc.edge_rmse_px))
            return FoundCorners(coarse, coarse, False, area_frac,
                                np.asarray(rc.edge_rmse_px))
        except (RuntimeError, cv2.error):
            pass
    return FoundCorners(coarse, coarse, False, area_frac, None)
