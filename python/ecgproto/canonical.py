"""The learned tracer's canonical frame and tiled inference -- framework-free.

The parts of the learned-tracer path that run on the PHONE, as plain
NumPy/OpenCV, so the Kotlin port has one dependency-light reference to be
golden-vector-tested against (`make_sheet_port_fixtures.py`). The training
code (`ml/strips.py`, `ml/train_tracer.py`) imports these rather than
keeping its own copies, so training, desktop evaluation and the phone share
one definition of the geometry.

THE FRAME. 8 px/mm, height chosen per model (512 rows = 64 mm for b3),
width 2048 (256 mm). `to_canonical`: one isotropic INTER_AREA resize by
k = 8 / px_per_mm, then an integer crop, padded with the per-channel median
of every 7th pixel of the source. Points map as x' = (x + 0.5) * k - 0.5 - x0.

THE MODEL CALL. A row-head model maps a (H, W, 3) RGB tile to, per OUTPUT
column (OUT_UPSAMPLE per input column), a `row` (canonical input rows,
soft-argmax) and a `presence` logit. `predict_tiled` runs it over
TILE_W-column tiles keeping each tile's middle (TILE_MARGIN either side).
`row_trace_to_source` maps the result back to source-image columns/rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CANON_PX_PER_MM = 8.0
CANON_W = 2048
OUT_UPSAMPLE = 2

#: Tiled inference: tile width and the margin discarded at each tile's inner
#: edges, canonical input columns. A 512 x 2048 frame does not fit the ~1.8
#: GB of GPU memory WSL leaves usable, and the phone exports at a 1,024
#: tile. 128 columns (16 mm) clears the network's horizontal context.
TILE_W = 1024
TILE_MARGIN = 128
#: Frames up to this many pixels run in one call, as a1/b1 were evaluated,
#: so their committed results reproduce exactly. Tiled against full width on
#: five b1 validation strips: mean row difference 0.0001-0.004 px, worst
#: column 0.88 px (0.1 mm), presence identical.
FULL_FRAME_MAX_PX = 384 * 2048


@dataclass(frozen=True)
class CanonMap:
    """Source -> canonical: x' = (x + 0.5) * k - 0.5 - x0, same for y."""
    k: float
    x0: int
    y0: int

    def points(self, rows: np.ndarray, cols: np.ndarray):
        return ((rows + 0.5) * self.k - 0.5 - self.y0,
                (cols + 0.5) * self.k - 0.5 - self.x0)

    def column_to_source_x(self, c: np.ndarray) -> np.ndarray:
        return (c + self.x0 + 0.5) / self.k - 0.5

    def row_to_source_y(self, r: np.ndarray) -> np.ndarray:
        return (r + self.y0 + 0.5) / self.k - 0.5


def to_canonical(img: np.ndarray, px_per_mm: float, x_start: float,
                 center_row: float, height: int, pad=None,
                 width: int = CANON_W) -> tuple[np.ndarray, CanonMap]:
    """Resample `img` to CANON_PX_PER_MM and cut the height x width window.

    `x_start` is the source column that becomes canonical column 0, and
    `center_row` the source row that becomes row height // 2 (both to the
    nearest canonical pixel). Out-of-image pixels take `pad`, by default the
    per-channel median of every 7th source pixel -- paper, not white,
    because a photo's paper is not white and a white border is an edge no
    real crop has.
    """
    k = CANON_PX_PER_MM / px_per_mm
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if small.ndim == 2:          # resize drops a single channel's axis
        small = small[..., None]
    x0 = int(round((x_start + 0.5) * k - 0.5))
    y0 = int(round((center_row + 0.5) * k - 0.5)) - height // 2
    if pad is None:
        pad = np.median(img.reshape(-1, img.shape[2])[::7], axis=0)
    out = np.empty((height, width, img.shape[2]), img.dtype)
    out[:] = np.asarray(pad, dtype=np.float64).astype(img.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(small.shape[0], y0 + height)
    sx1 = min(small.shape[1], x0 + width)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = small[sy0:sy1, sx0:sx1]
    return out, CanonMap(k, x0, y0)


def predict_tiled(model, image: np.ndarray, tile_w: int = TILE_W,
                  margin: int = TILE_MARGIN) -> dict:
    """Run a fully convolutional tracer over overlapping width tiles.

    `model(batch, training=False)` returns a dict of arrays whose last axis
    is u * width (u = 2 for rows/presence, 1 for a mask). Returns the
    stitched outputs without the batch axis. Each tile keeps only its
    middle, so every kept column saw at least `margin` columns of context
    either side (except at the frame's own ends, as a full call would).
    """
    h, w = image.shape[:2]
    if w <= tile_w or h * w <= FULL_FRAME_MAX_PX:
        out = model(image[None], training=False)
        return {k: np.asarray(v)[0] for k, v in out.items()}
    step = tile_w - 2 * margin
    starts = list(range(0, w - tile_w, step)) + [w - tile_w]
    acc = {}
    for x0 in starts:
        out = model(image[None, :, x0:x0 + tile_w], training=False)
        out = {k: np.asarray(v)[0] for k, v in out.items()}
        keep0 = 0 if x0 == 0 else margin
        keep1 = tile_w if x0 + tile_w >= w else tile_w - margin
        for k, v in out.items():
            u = v.shape[-1] // tile_w
            if k not in acc:
                acc[k] = np.zeros(v.shape[:-1] + (w * u,), v.dtype)
            a, b = (x0 + keep0) * u, (x0 + keep1) * u
            acc[k][..., a:b] = v[..., keep0 * u:keep1 * u]
    return acc


def row_trace_to_source(row: np.ndarray, presence: np.ndarray, cm: CanonMap,
                        upsample: int = OUT_UPSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-column (row, presence logit) -> (source x, source row).

    NaN where presence < 0. Output column j sits at canonical input column
    (j + 0.5) / upsample - 0.5.
    """
    r = np.asarray(row, np.float64).copy()
    r[np.asarray(presence) < 0] = np.nan
    c = (np.arange(len(r)) + 0.5) / upsample - 0.5
    return cm.column_to_source_x(c), cm.row_to_source_y(r)
