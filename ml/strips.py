#!/usr/bin/env python3
"""Rhythm-strip bands -> augmented canonical training examples, with exact labels.

The second half of `docs/ml_plan.md` §3.2-§3.3. `make_strips.py` renders
clean bands; this module turns one into what the network sees and what it
is trained to say, applying only transforms whose effect on the labels is
known, so the labels stay exact through every augmentation.

THE CANONICAL FRAME (`ml_plan.md` §2.2), defined once here and shared with
inference so training and deployment cannot disagree about geometry:

    CANON_PX_PER_MM = 8     a 10 s strip is 2,000 px wide
    CANON_H = 384           48 mm, centred on the strip's baseline
    CANON_W = 2048          256 mm: the 250 mm strip plus a lead-in
    OUT_UPSAMPLE = 2        the row head outputs 16 px/mm columns

Two of these differ from `ml_plan.md` §2.2 as first written, on measurement
(`label_ceiling`, results/ml/label_ceiling.*): at 32 mm tall, 6.7% of lead
II strips clip even perfectly centred (R waves reach 19.6 mm at p99); and a
head with one output per 8 px/mm column averages ~2.5 samples at the QRS
apex, which caps a PERFECT model's R-peak ratio at 0.95 median, 0.86 worst
-- against decision rule 1's 0.90. Two output columns per input column lift
that ceiling to 0.99, for the cost of a depth-to-space on the last layer.

`to_canonical(img, px_per_mm, x_start, center_row)` is the whole mapping:
one isotropic INTER_AREA resize by k = 8 / px_per_mm, then an integer crop,
padded with the image's per-channel median. At inference `px_per_mm` is the
duration anchor's, `x_start` a little before the locator's t0 and
`center_row` the classical mask's median row. Points map as
x' = (x + 0.5) * k - 0.5 - x0 (pixel-centre convention, as OpenCV's resize
uses). **This function gets golden vectors against Kotlin** (§5); nothing
else here runs on the phone.

THE CHAIN, in the order a real photograph applies it:

    native band (300 dpi, 11.81 px/mm)
      -> minor grid recoloured      red sheets only; white (printer dropped
                                    it) through MINOR_GRID_RGB to darker
      -> residual geometry          rotation, anisotropic stretch, shear,
                                    corner-jitter perspective: one known
                                    homography applied to image AND labels
      -> camera resolution          INTER_AREA to U(PHONE_PX_PER_MM)
      -> photometric                paper tint, colour temperature,
                                    illumination ramp and shadow, blur,
                                    noise, JPEG (4:2:0)
      -> canonical                  `to_canonical`, with the baseline and
                                    start jittered as an imperfect locator
                                    would place them

Photometric steps do not move pixels, so they cannot move labels. The
geometric ones are all expressed as the same point maps applied to the
labels; `selftest` checks that claim by carrying a trace-colour mask through
the identical chain and requiring the labels to land on it.

    python strips.py --ceiling              # results/ml/label_ceiling.*
    python strips.py --selftest --n 60      # results/ml/strips_selftest.*
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from ecgproto import evaluate  # noqa: E402
from make_kit_target import (MINOR_GRID_RGB, apply_minor_blend,  # noqa: E402
                             minor_grid_blend)

DATA = REPO / "data" / "ml_strips"
RESULTS = REPO / "results" / "ml"
DIAG = REPO / "out" / "ml_selftest"

CANON_PX_PER_MM = 8.0
CANON_H = 384
CANON_W = 2048
OUT_UPSAMPLE = 2
OUT_W = CANON_W * OUT_UPSAMPLE
TAG = ""  # result-file suffix, set by --tag
MM_PER_S = 25.0
MM_PER_MV = 10.0

#: The kit draws the trace at 0.75 pt = 0.2646 mm (`ecg_plot.standard_values`).
TRACE_WIDTH_MM = 0.75 / 72.0 * 25.4


@dataclass
class AugConfig:
    """Ranges for every augmentation. Defaults aim at `ml_plan.md` §3.3."""
    # Minor-grid recolour on red sheets: target = white - t * (white -
    # MINOR_GRID_RGB). t = 0 is a printer that dropped the rules entirely,
    # ~0.13 is the kit's own near-white, 1 is what the targets now print.
    p_recolour: float = 0.9
    recolour_t: tuple[float, float] = (0.0, 1.4)
    # Residual geometry after rectification and levelling.
    rot_deg: float = 0.5
    stretch_x: float = 0.02         # the duration anchor's error, and more
    stretch_y: float = 0.03         # vertical scale is assumed isotropic
    shear: float = 0.005
    corner_jitter_mm: float = 0.4
    # Effective resolution of the rectified photo. The A30s gives ~10-11;
    # a photo taken from further away gives less.
    phone_px_per_mm: tuple[float, float] = (6.0, 12.5)
    # Photometric: the scene.
    paper_level: tuple[float, float] = (170.0, 250.0)
    paper_tint: float = 8.0
    colour_gain: float = 0.15
    light_ramp: tuple[float, float] = (0.0, 0.45)
    p_shadow: float = 0.3
    shadow_depth: tuple[float, float] = (0.15, 0.45)
    # The camera (added for run b1, results/ml/photo_compare_b0: on the real
    # A30s photographs the paper sits at ~115-160 grey, the grid is a dull,
    # desaturated brick red, the trace is only 2-3x darker than the paper,
    # and the thin QRS strokes come out LIGHTER than the baseline -- where
    # b0 lost whole beats. The b0 augmentation produced bright paper (mostly
    # 170-250), saturated grids and full-contrast strokes.)
    blur_sigma_px: tuple[float, float] = (0.3, 2.0)
    p_motion_blur: float = 0.25
    motion_len_px: tuple[float, float] = (2.0, 5.0)
    exposure: tuple[float, float] = (0.45, 1.15)     # >1 clips: bright light
    black_lift: tuple[float, float] = (0.0, 0.22)    # ink never reads as 0
    gamma: tuple[float, float] = (0.75, 1.4)
    saturation: tuple[float, float] = (0.25, 1.1)
    noise_sigma: tuple[float, float] = (1.0, 8.0)
    chroma_noise_sigma: tuple[float, float] = (0.0, 4.0)
    p_sharpen: float = 0.5                           # phone ISP unsharp mask
    sharpen_amount: tuple[float, float] = (0.3, 1.2)
    jpeg_quality: tuple[int, int] = (60, 95)
    # Placement of the canonical window, as an imperfect locator places it.
    # The locator's centre is the classical mask's median row, which the
    # row above or text can pull; 4 mm is the jitter the 48 mm window was
    # sized against (label_ceiling: 2.0% of strips then clip somewhere, 1.1%
    # centred exactly; 32 mm would be 20.8% and 6.7%).
    center_jitter_mm: float = 1.5
    center_clip_mm: float = 4.0
    lead_in_mm: tuple[float, float] = (-2.0, 14.0)   # >~5 mm keeps the pulse


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

@dataclass
class Strip:
    stem: str
    image: np.ndarray            # BGR uint8, native band
    rows: np.ndarray             # float64, band pixels
    cols: np.ndarray
    lead_mv: np.ndarray          # the truth for these samples
    px_per_mm: float
    fs: float
    meta: dict = field(default_factory=dict)

    @property
    def t0_col(self) -> float:
        return float(self.cols[0])

    def minor_blend(self) -> tuple[np.ndarray, np.ndarray]:
        """The kit's minor-rule pixels, found once per strip and cached."""
        if "_blend" not in self.meta:
            kit_minor = [float(v) for v in self.meta["grid_minor_rgb"].split(",")]
            self.meta["_blend"] = minor_grid_blend(self.image, kit_minor, np.float32)
        return self.meta["_blend"]


def load_strip(stem: str, split: str, root: Path = DATA) -> Strip:
    d = root / split
    z = np.load(d / f"{stem}.npz")
    return Strip(
        stem=stem,
        image=cv2.imread(str(d / f"{stem}.png"), cv2.IMREAD_COLOR),
        rows=z["rows"].astype(np.float64), cols=z["cols"].astype(np.float64),
        lead_mv=z["lead_ii_mv"].astype(np.float64),
        px_per_mm=float(z["px_per_mm"]), fs=float(z["fs"]),
        meta=json.loads((d / f"{stem}.json").read_text("utf-8")))


def list_stems(split: str, root: Path = DATA, only_ok: bool = True) -> list[str]:
    """Stems of a split, skipping failed renders and flagged labels."""
    out = []
    for p in sorted((root / split).glob("*.json")):
        m = json.loads(p.read_text("utf-8"))
        if only_ok and (m.get("error") or not m.get("label_ok")):
            continue
        out.append(m["stem"])
    return out


# --------------------------------------------------------------------------
# the canonical frame -- shared with inference
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonMap:
    """Native -> canonical: x' = (x + 0.5) * k - 0.5 - x0, same for y."""
    k: float
    x0: int
    y0: int

    def points(self, rows: np.ndarray, cols: np.ndarray):
        return ((rows + 0.5) * self.k - 0.5 - self.y0,
                (cols + 0.5) * self.k - 0.5 - self.x0)

    def column_to_source_x(self, c: np.ndarray) -> np.ndarray:
        return (c + self.x0 + 0.5) / self.k - 0.5


def to_canonical(img: np.ndarray, px_per_mm: float, x_start: float,
                 center_row: float, pad=None) -> tuple[np.ndarray, CanonMap]:
    """Resample `img` to CANON_PX_PER_MM and cut the CANON_H x CANON_W window.

    `x_start` is the source column that becomes canonical column 0, and
    `center_row` the source row that becomes row CANON_H / 2 (both to the
    nearest canonical pixel). Out-of-image pixels take `pad`, by default the
    per-channel median of `img` -- paper, not white, because a photo's paper
    is not white and a white border is an edge no real crop has.
    """
    k = CANON_PX_PER_MM / px_per_mm
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if small.ndim == 2:          # resize drops a single channel's axis
        small = small[..., None]
    x0 = int(round((x_start + 0.5) * k - 0.5))
    y0 = int(round((center_row + 0.5) * k - 0.5)) - CANON_H // 2
    if pad is None:
        # Every 7th pixel: the median of paper is not sensitive to it.
        pad = np.median(img.reshape(-1, img.shape[2])[::7], axis=0)
    out = np.empty((CANON_H, CANON_W, img.shape[2]), img.dtype)
    out[:] = np.asarray(pad, dtype=np.float64).astype(img.dtype)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1 = min(small.shape[0], y0 + CANON_H)
    sx1 = min(small.shape[1], x0 + CANON_W)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = small[sy0:sy1, sx0:sx1]
    return out, CanonMap(k, x0, y0)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def output_columns(cols_c: np.ndarray, upsample: float = OUT_UPSAMPLE) -> np.ndarray:
    """Canonical input columns -> the head's output columns (pixel centres)."""
    return (cols_c + 0.5) * upsample - 0.5


def column_index(cols_c: np.ndarray, upsample: float = OUT_UPSAMPLE) -> np.ndarray:
    return np.floor(output_columns(cols_c, upsample) + 0.5).astype(np.int64)


def column_targets(rows_c: np.ndarray, cols_c: np.ndarray,
                   upsample: float = OUT_UPSAMPLE,
                   height: int = CANON_H) -> dict:
    """Per OUTPUT column: mean truth row, presence, in-window flag.

    Columns are the head's (CANON_W * upsample of them); rows stay in the
    canonical input's pixels. `mean_row` is what the soft-argmax is trained
    toward (the L1 term, §2.4) and what the waveform is read from.
    `presence` is 1 where at least one sample falls in the column -- 0 over
    the pulse, the lead-in and past the strip's end. `in_window` is 0 where
    the mean row falls outside the window, an excursion the frame cannot
    represent.
    """
    width = int(round(CANON_W * upsample))
    ci = column_index(cols_c, upsample)
    ok = (ci >= 0) & (ci < width)
    cnt = np.bincount(ci[ok], minlength=width).astype(np.float64)
    s = np.bincount(ci[ok], weights=rows_c[ok], minlength=width)
    present = cnt > 0
    mean = np.full(width, np.nan)
    mean[present] = s[present] / cnt[present]
    in_win = present & (mean >= -0.5) & (mean <= height - 0.5)
    return {"mean_row": mean.astype(np.float32),
            "presence": present.astype(np.float32),
            "in_window": in_win.astype(np.float32),
            "count": cnt.astype(np.float32)}


def column_samples(rows_c: np.ndarray, cols_c: np.ndarray, k: int = 4,
                   upsample: float = OUT_UPSAMPLE) -> np.ndarray:
    """(W_out, k) float32: the rows of up to k samples in each output column,
    -1 where fewer. The compact form of `row_distribution`, which the
    training graph expands on the GPU (a dense target is 6 MB per example).
    At 16 px/mm and 500 Hz a column holds 1.25 samples on average; k = 4
    covers every column of every strip checked, and `dropped` says if not.
    """
    width = int(round(CANON_W * upsample))
    ci = column_index(cols_c, upsample)
    ok = (ci >= 0) & (ci < width)
    ci, r = ci[ok], rows_c[ok]
    order = np.argsort(ci, kind="stable")
    ci, r = ci[order], r[order]
    first = np.searchsorted(ci, ci, side="left")
    slot = np.arange(len(ci)) - first
    out = np.full((width, k), -1.0, np.float32)
    keep = slot < k
    out[ci[keep], slot[keep]] = r[keep]
    column_samples.dropped = int((~keep).sum())
    return out


def row_distribution(rows_c: np.ndarray, cols_c: np.ndarray, sigma: float = 1.0,
                     upsample: float = OUT_UPSAMPLE,
                     height: int = CANON_H) -> np.ndarray:
    """(H, W_out) float32: a Gaussian per sample down its output column, each
    column normalised to sum 1 where it holds a sample (§2.4's KL target)."""
    width = int(round(CANON_W * upsample))
    ci = column_index(cols_c, upsample)
    ok = (ci >= 0) & (ci < width)
    r = np.arange(height, dtype=np.float32)[None, :]
    g = np.exp(-0.5 * ((r - rows_c[ok, None].astype(np.float32)) / sigma) ** 2)
    out = np.zeros((width, height), np.float32)
    ci = ci[ok]
    order = np.argsort(ci, kind="stable")
    cols_u, first = np.unique(ci[order], return_index=True)
    if len(cols_u):
        out[cols_u] = np.add.reduceat(g[order], first, axis=0)
    tot = out.sum(axis=1, keepdims=True)
    np.divide(out, tot, out=out, where=tot > 0)
    return out.T.copy()


def trace_mask(rows_c: np.ndarray, cols_c: np.ndarray,
               width_px: float = TRACE_WIDTH_MM * CANON_PX_PER_MM,
               width: int = CANON_W, height: int = CANON_H) -> np.ndarray:
    """(H, W) uint8 {0,1}: the truth polyline at the trace's printed width.
    Model A's target (§2.1): which pixels are trace."""
    m = np.zeros((height, width), np.uint8)
    shift = 4
    pts = np.stack([cols_c, rows_c], 1) * (1 << shift)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 1,
                  thickness=max(1, int(round(width_px))), lineType=cv2.LINE_8,
                  shift=shift)
    return m


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------

def _homography(w: int, h: int, px_per_mm: float, cfg: AugConfig,
                rng: np.random.Generator) -> np.ndarray:
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    th = np.deg2rad(rng.uniform(-cfg.rot_deg, cfg.rot_deg))
    sx = 1.0 + rng.uniform(-cfg.stretch_x, cfg.stretch_x)
    sy = 1.0 + rng.uniform(-cfg.stretch_y, cfg.stretch_y)
    sh = rng.uniform(-cfg.shear, cfg.shear)
    t = lambda x, y: np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                    [0, 0, 1]])
    lin = np.array([[sx, sh, 0], [0, sy, 0], [0, 0, 1]])
    a = t(cx, cy) @ rot @ lin @ t(-cx, -cy)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = src + rng.normal(0, cfg.corner_jitter_mm * px_per_mm,
                           src.shape).astype(np.float32)
    p = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return p @ a


def _apply_h(hm: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    v = hm @ np.stack([cols, rows, np.ones_like(cols)])
    return v[1] / v[2], v[0] / v[2]


def _photometric(img: np.ndarray, cfg: AugConfig,
                 rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    # Ink is subtractive: the sheet's white becomes the paper's colour. Paper
    # tint and colour temperature fold into one per-channel gain.
    paper = rng.uniform(*cfg.paper_level) + rng.normal(0, cfg.paper_tint, 3)
    gain = (np.clip(paper, 0, 255) / 255.0) * (
        1.0 + rng.uniform(-cfg.colour_gain, cfg.colour_gain, 3))
    # Illumination is smooth, so it is built on a grid 8x coarser and
    # upsampled: identical to the eye, and it was 0.6 s of 1.5 at full size.
    gh, gw = max(2, h // 8), max(2, w // 8)
    yy = (np.arange(gh, dtype=np.float32)[:, None] + 0.5) * (h / gh)
    xx = (np.arange(gw, dtype=np.float32)[None, :] + 0.5) * (w / gw)
    a = rng.uniform(0, 2 * np.pi)
    ramp = np.cos(a) * xx / w + np.sin(a) * yy / h
    ramp = (ramp - ramp.min()) / max(1e-6, float(np.ptp(ramp)))
    g = 1.0 - rng.uniform(*cfg.light_ramp) * ramp
    if rng.random() < cfg.p_shadow:
        # A soft-edged shadow over part of the strip: a hand or the phone.
        b = rng.uniform(0, 2 * np.pi)
        d = np.cos(b) * (xx - rng.uniform(0, w)) + np.sin(b) * (yy - rng.uniform(0, h))
        soft = rng.uniform(0.02, 0.2) * max(w, h)
        g = g * (1.0 - rng.uniform(*cfg.shadow_depth) / (1.0 + np.exp(-d / soft)))
    g = cv2.resize(g.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) * g[..., None] * gain.astype(np.float32)
    # Optics: defocus, sometimes hand shake.
    x = cv2.GaussianBlur(x, (0, 0), rng.uniform(*cfg.blur_sigma_px))
    if rng.random() < cfg.p_motion_blur:
        n = int(np.ceil(cfg.motion_len_px[1])) * 2 + 1
        k = np.zeros((n, n), np.float32)
        L = rng.uniform(*cfg.motion_len_px) / 2
        th = rng.uniform(0, np.pi)
        c = n // 2
        cv2.line(k, (int(round(c - L * np.cos(th))), int(round(c - L * np.sin(th)))),
                 (int(round(c + L * np.cos(th))), int(round(c + L * np.sin(th)))), 1.0, 1)
        x = cv2.filter2D(x, -1, k / k.sum())
    # Sensor and tone curve: exposure, black-level lift, gamma. Applied to
    # the scene in [0, 1], so ink keeps its order but loses contrast.
    y = np.clip(x / 255.0, 0, 1) * rng.uniform(*cfg.exposure)
    b = rng.uniform(*cfg.black_lift)
    y = b + (1.0 - b) * y
    y = np.power(np.clip(y, 0, 1), rng.uniform(*cfg.gamma))
    # Saturation: real grids come out a dull brick red.
    grey = y.mean(axis=2, keepdims=True)
    y = grey + rng.uniform(*cfg.saturation) * (y - grey)
    x = y * 255.0
    x += rng.normal(0, rng.uniform(*cfg.noise_sigma), x.shape[:2])[..., None].astype(np.float32)
    x += rng.normal(0, rng.uniform(*cfg.chroma_noise_sigma), x.shape).astype(np.float32)
    if rng.random() < cfg.p_sharpen:
        soft = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.8, 1.6))
        x = x + rng.uniform(*cfg.sharpen_amount) * (x - soft)
    x = np.clip(x, 0, 255).astype(np.uint8)
    q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
    params = [cv2.IMWRITE_JPEG_QUALITY, q]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                   cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
    ok, buf = cv2.imencode(".jpg", x, params)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def augment(strip: Strip, rng: np.random.Generator, cfg: AugConfig = AugConfig(),
            extra: np.ndarray | None = None) -> dict:
    """One augmented canonical example.

    Returns image (BGR uint8, CANON_H x CANON_W), rows_c / cols_c (every
    sample in canonical pixels), the CanonMap, the phone resolution drawn,
    and -- if `extra` (an HxW float32 map in native band pixels) is given --
    that map carried through the identical geometric chain, which is how
    `selftest` proves the labels follow the image.
    """
    img = strip.image
    if strip.meta.get("style") == "red" and rng.random() < cfg.p_recolour:
        t = rng.uniform(*cfg.recolour_t)
        target = 255.0 - t * (255.0 - np.asarray(MINOR_GRID_RGB, np.float64))
        idx, a = strip.minor_blend()
        img = apply_minor_blend(img, idx, a, np.clip(target, 0, 255))

    h, w = img.shape[:2]
    hm = _homography(w, h, strip.px_per_mm, cfg, rng)
    img = cv2.warpPerspective(img, hm, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    rows, cols = _apply_h(hm, strip.rows, strip.cols)
    if extra is not None:
        extra = cv2.warpPerspective(extra, hm, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)

    phone = rng.uniform(*cfg.phone_px_per_mm)
    s = phone / strip.px_per_mm
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rows, cols = (rows + 0.5) * s - 0.5, (cols + 0.5) * s - 0.5
    if extra is not None:
        extra = cv2.resize(extra, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    img = _photometric(img, cfg, rng)

    center = float(np.median(rows)) + phone * float(np.clip(
        rng.normal(0, cfg.center_jitter_mm), -cfg.center_clip_mm, cfg.center_clip_mm))
    x_start = float(cols[0]) - phone * rng.uniform(*cfg.lead_in_mm)
    canon, cm = to_canonical(img, phone, x_start, center)
    rows_c, cols_c = cm.points(rows, cols)
    out = {"image": canon, "rows_c": rows_c, "cols_c": cols_c, "map": cm,
           "phone_px_per_mm": phone}
    if extra is not None:
        out["extra"], _ = to_canonical(extra[..., None], phone, x_start, center,
                                       pad=[0.0])
    return out


def clean_canonical(strip: Strip) -> dict:
    """No augmentation: the native band straight into the canonical frame."""
    center = float(np.median(strip.rows))
    canon, cm = to_canonical(strip.image, strip.px_per_mm, strip.t0_col, center)
    rows_c, cols_c = cm.points(strip.rows, strip.cols)
    return {"image": canon, "rows_c": rows_c, "cols_c": cols_c, "map": cm}


# --------------------------------------------------------------------------
# reading a waveform back out of per-column rows
# --------------------------------------------------------------------------

def columns_to_mv(mean_row: np.ndarray, t0_canon_col: float, n: int,
                  fs: float, upsample: float = OUT_UPSAMPLE) -> np.ndarray:
    """Per-output-column rows -> n samples at fs, in mV up to an offset.

    `t0_canon_col` is where recording time zero sits, in canonical INPUT
    columns (the locator's t0 at inference, sample 0's column here). Output
    column j sits at input column (j + 0.5) / upsample - 0.5, hence at time
    (that - t0) / (8 px/mm * 25 mm/s). Rows grow downward, so
    mV = -row / (8 px/mm * 10 mm/mV). Gaps are linearly interpolated, as the
    classical path's `fill_nan` does.
    """
    j = np.flatnonzero(np.isfinite(mean_row))
    c = (j + 0.5) / upsample - 0.5
    t = (c - t0_canon_col) / (CANON_PX_PER_MM * MM_PER_S)
    mv = -mean_row[j] / (CANON_PX_PER_MM * MM_PER_MV)
    return np.interp(np.arange(n) / fs, t, mv)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _ink_map(strip: Strip) -> np.ndarray:
    bgr = np.asarray([float(v) for v in strip.meta["trace_rgb"].split(",")])[::-1]
    d = np.abs(strip.image.astype(np.float32) - bgr).max(axis=2)
    return (d < 40.0).astype(np.float32)


def ceiling_row(st: Strip) -> dict:
    """What a PERFECT tracer of each kind could score on this strip.

    Read on the clean canonical frame, no augmentation, window centred on
    the baseline exactly. Nothing a model of that kind outputs can beat
    these, so they bound decision rule 1 from above:

    - `label_x*`: the row head's own target (the per-output-column mean
      truth row) read back as a waveform, at the head's upsampling and at
      1x, 2.5x (one column per 500 Hz sample) and 3x for comparison;
    - `mask_centroid_*`: the column centroid of a perfect ink mask -- the
      classical tracer's reading with segmentation solved, i.e. model A.
    """
    r = {"stem": st.stem, "style": st.meta["style"]}
    cl = clean_canonical(st)
    t0c = float(cl["cols_c"][0])
    n = len(st.lead_mv)
    for u in (1, OUT_UPSAMPLE, 2.5, 3):
        tg = column_targets(cl["rows_c"], cl["cols_c"], upsample=u)
        wave = columns_to_mv(tg["mean_row"], t0c, n, st.fs, upsample=u)
        tag = f"label_x{u:g}"
        rr, npk = evaluate.r_peak_ratio(st.lead_mv, wave, st.fs)
        r[f"{tag}_r_ratio"] = round(rr, 4)
        r[f"{tag}_pcc"] = round(evaluate.pcc(st.lead_mv, wave), 5)
        if u == OUT_UPSAMPLE:
            r["r_peaks"] = npk
            r["label_gain"] = round(evaluate.gain_ratio(st.lead_mv, wave), 4)
            r["out_of_window_frac"] = round(float(
                (tg["presence"].sum() - tg["in_window"].sum())
                / tg["presence"].sum()), 5)
    m = trace_mask(cl["rows_c"], cl["cols_c"]).astype(np.float64)
    colsum = m.sum(axis=0)
    cen = np.where(colsum > 0, (m * np.arange(CANON_H)[:, None]).sum(0)
                   / np.maximum(colsum, 1), np.nan)
    present = column_targets(cl["rows_c"], cl["cols_c"], upsample=1)["presence"]
    cen[present == 0] = np.nan
    wave_c = columns_to_mv(cen, t0c, n, st.fs, upsample=1)
    r["mask_centroid_r_ratio"] = round(
        evaluate.r_peak_ratio(st.lead_mv, wave_c, st.fs)[0], 4)
    r["mask_centroid_pcc"] = round(evaluate.pcc(st.lead_mv, wave_c), 5)
    # Excursions from the baseline, which size the window.
    dev_mm = (st.rows - np.median(st.rows)) / st.px_per_mm
    r["max_up_mm"] = round(float(-dev_mm.min()), 2)
    r["max_down_mm"] = round(float(dev_mm.max()), 2)
    return r


def _persist(name: str, rows: list[dict], params: dict, keys: tuple) -> Path:
    import subprocess
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summary = {}
    for k in keys:
        v = np.array([float(x[k]) for x in rows if x.get(k, "") != ""])
        v = v[np.isfinite(v)]
        summary[k] = {"median": round(float(np.median(v)), 5),
                      "p5": round(float(np.percentile(v, 5)), 5),
                      "min": round(float(v.min()), 5),
                      "max": round(float(v.max()), 5)}
        if k.endswith("r_ratio"):
            summary[k]["frac_below_0.90"] = round(float((v < 0.90).mean()), 5)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--", "ml", "python"],
        cwd=REPO, capture_output=True, text=True).stdout.strip())
    (RESULTS / f"{name}.json").write_text(json.dumps({
        "generated_by": f"ml/strips.py ({name})",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "params": dict(params, canon={"px_per_mm": CANON_PX_PER_MM, "h": CANON_H,
                                      "w": CANON_W, "out_upsample": OUT_UPSAMPLE}),
        "summary": summary,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "opencv": cv2.__version__},
    }, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"  {k:26s} " + "  ".join(f"{a} {b}" for a, b in v.items()))
    return path


def label_ceiling(split: str) -> Path:
    """`ceiling_row` over every band of a split, plus how often the window
    would clip at candidate heights. Sizes the canonical frame."""
    rows = [ceiling_row(load_strip(s, split, DATA)) for s in list_stems(split, DATA)]
    up = np.array([r["max_up_mm"] for r in rows])
    down = np.array([r["max_down_mm"] for r in rows])
    jit = AugConfig().center_clip_mm
    clip = {}
    for h_mm in (32, 40, 48):
        half = h_mm / 2
        clip[f"{h_mm}mm_clean"] = round(float(((up > half) | (down > half)).mean()), 5)
        clip[f"{h_mm}mm_jitter{jit:g}"] = round(float(
            ((up > half - jit) | (down > half - jit)).mean()), 5)
    print("  window clipping (fraction of strips):", clip)
    keys = tuple(k for k in rows[0] if k.endswith(("r_ratio", "_pcc"))) + (
        "out_of_window_frac", "max_up_mm", "max_down_mm")
    return _persist("label_ceiling" + TAG, rows,
                    {"split": split, "n": len(rows), "window_clip": clip}, keys)


def selftest(n: int, split: str, seed: int) -> Path:
    """Labels follow the image through the full augmentation chain: a
    trace-colour mask rides the identical geometry, and every label must land
    on it (3x3 neighbourhood, canonical pixels). Also times the chain and
    writes overlays to out/ml_selftest/ for eyeballing."""
    stems = list_stems(split, DATA)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(stems, size=min(n, len(stems)), replace=False))
    DIAG.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for i, stem in enumerate(pick):
        st = load_strip(stem, split, DATA)
        r = {"stem": stem, "style": st.meta["style"]}
        t = time.perf_counter()
        ex = augment(st, np.random.default_rng(seed + i), extra=_ink_map(st))
        r["augment_ms"] = round(1000 * (time.perf_counter() - t), 1)
        ink = cv2.dilate(ex["extra"][..., 0], np.ones((3, 3), np.uint8))
        rc, cc = ex["rows_c"], ex["cols_c"]
        ri, ci = np.floor(rc + 0.5).astype(int), np.floor(cc + 0.5).astype(int)
        inside = (ri >= 0) & (ri < CANON_H) & (ci >= 0) & (ci < CANON_W)
        r["aug_labels_inside"] = int(inside.sum())
        r["aug_label_on_ink_frac"] = round(float(
            (ink[ri[inside], ci[inside]] > 0.2).mean()), 5) if inside.any() else ""
        r["aug_phone_px_per_mm"] = round(ex["phone_px_per_mm"], 3)
        t = time.perf_counter()
        atg = column_targets(rc, cc)
        row_distribution(rc, cc)
        trace_mask(rc, cc)
        r["targets_ms"] = round(1000 * (time.perf_counter() - t), 1)
        r["aug_presence_cols"] = int(atg["presence"].sum())
        rows_out.append(r)
        if i < 8:
            view = ex["image"].copy()
            for j in np.flatnonzero(atg["presence"] > 0)[::4]:
                y = int(round(atg["mean_row"][j]))
                x = int(round((j + 0.5) / OUT_UPSAMPLE - 0.5))
                if 0 <= y < CANON_H:
                    view[y, x] = (0, 200, 0)
            cv2.imwrite(str(DIAG / f"{stem}_aug.png"), view)
    return _persist("strips_selftest" + TAG, rows_out,
                    {"n": len(rows_out), "split": split, "seed": seed,
                     "aug": asdict(AugConfig())},
                    ("aug_label_on_ink_frac", "augment_ms", "targets_ms"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--ceiling", action="store_true")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data", type=Path, default=None,
                   help="band directory (default data/ml_strips; data/ml_rows "
                        "for the four-row data)")
    p.add_argument("--tag", default="", help="suffix for the result files")
    args = p.parse_args(argv)
    global DATA
    if args.data is not None:
        DATA = args.data
    global TAG
    TAG = args.tag
    if args.ceiling:
        print(label_ceiling(args.split))
    if args.selftest:
        print(selftest(args.n, args.split, args.seed))
    if args.ceiling or args.selftest:
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
