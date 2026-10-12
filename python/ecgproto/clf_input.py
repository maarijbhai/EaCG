"""The diagnosis classifier's input, built from a digitised sheet.

THE REFERENCE the phone mirrors (`ClfInput.kt`, golden-vector tested from
`make_clf_port_fixtures.py`), and the function the degradation curve
(`ml/clf_degradation.py`) uses on phone exports -- so the desktop and the
phone hand the classifier the same numbers.

The classifier (`ml/train_classifier.py`, view "printed") was trained on
PTB-XL at 100 Hz, 10 s = 1,000 samples, 13 channels: each of the 12 leads
only inside its printed 2.5 s window (I/II/III 0-2.5 s, aVR/aVL/aVF 2.5-5,
V1-V3 5-7.5, V4-V6 7.5-10; zero elsewhere), each window about its own
median, plus lead II for all 10 s as channel 13 (`ml/clf_data.printed_view`).

A digitised sheet arrives at 500 Hz in SheetPipeline.csv's layout: per lead
a 10 s column holding only its own 2.5 s cell (NaN elsewhere), and the
rhythm strip as `II_rhythm`. Two steps turn that into the training view:

1. 500 -> 100 Hz by the MEAN OF EACH BLOCK OF FIVE samples (NaN ignored; a
   block with no finite sample stays NaN). A block mean is a low-pass
   filter (first zero at 100 Hz) and keeps the cell boundaries exactly on
   block boundaries (a cell is 1,250 = 250 x 5 samples), so no sample of one
   window leaks into the next. It is simple enough to port exactly; whether
   it matches PTB-XL's own 100 Hz files closely enough is measured, not
   assumed (`ml/clf_degradation.py`, step "resample").
2. The printed view, exactly as in training: each window minus its median
   (NaN-safe), NaN -> 0 (a gap the digitiser left reads as flat).

Leads in PTB-XL order (I II III aVR aVL aVF V1-V6), channel 13 = rhythm.
"""

from __future__ import annotations

import numpy as np

FS_SHEET = 500
FS_CLF = 100
BLOCK = FS_SHEET // FS_CLF          # 5
N = 1000                            # 10 s at 100 Hz
CELL = 250                          # 2.5 s at 100 Hz
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
#: lead index -> column (0-3) of its 2.5 s window (ml/clf_data.PRINTED_COLUMN).
PRINTED_COLUMN = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
RHYTHM = "II_rhythm"
LEAD_II = 1


def decimate(x500: np.ndarray) -> np.ndarray:
    """5,000 samples at 500 Hz -> 1,000 at 100 Hz: the NaN-ignoring mean of
    each block of five. Shorter input is padded with NaN, longer is cut."""
    x = np.full(N * BLOCK, np.nan)
    v = np.asarray(x500, np.float64)[:N * BLOCK]
    x[:len(v)] = v
    b = x.reshape(N, BLOCK)
    ok = np.isfinite(b)
    n = ok.sum(1)
    s = np.where(ok, b, 0.0).sum(1)
    out = np.full(N, np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out


def _about_median(w: np.ndarray) -> np.ndarray:
    """A window minus its NaN-ignoring median, NaN -> 0 (clf_data._median_removed)."""
    w = np.asarray(w, np.float64)
    fin = w[np.isfinite(w)]
    med = float(np.median(fin)) if fin.size else 0.0
    return np.nan_to_num(w - med, nan=0.0)


def printed_view(x100: np.ndarray, rhythm100: np.ndarray | None = None) -> np.ndarray:
    """(1000, 12) leads at 100 Hz (+ the rhythm strip, default lead II) ->
    the (1000, 13) float32 input. With no `rhythm100` this is
    `ml/clf_data.printed_view` (tested equal)."""
    x = np.asarray(x100, np.float64)
    r = x[:, LEAD_II] if rhythm100 is None else np.asarray(rhythm100, np.float64)
    out = np.zeros((N, 13), np.float64)
    for lead, col in PRINTED_COLUMN.items():
        a, b = col * CELL, (col + 1) * CELL
        out[a:b, lead] = _about_median(x[a:b, lead])
    out[:, 12] = _about_median(r)
    return out.astype(np.float32)


def from_columns(cols: dict) -> np.ndarray:
    """SheetPipeline.csv's columns (lead name -> 10 s at 500 Hz, the lead's
    own cell only; `II_rhythm` the strip) -> the (1000, 13) input. A missing
    column reads as all-NaN, i.e. zeros."""
    blank = np.full(N * BLOCK, np.nan)
    x = np.stack([decimate(cols.get(name, blank)) for name in LEADS], axis=1)
    return printed_view(x, decimate(cols.get(RHYTHM, blank)))
