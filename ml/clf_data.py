#!/usr/bin/env python3
"""PTB-XL for the diagnosis experiment: a cached dataset and the input views.

THE QUESTION (docs/classifier_training.md). Does a waveform digitised from
a photographed printout keep the diagnostic information? The classifier is
trained on PTB-XL signals and evaluated along a degradation curve: true
10 s signals -> what a 3x4 printout shows -> digitised renders -> digitised
photographs. This module owns the data and, crucially, the INPUT VIEWS,
because the same view functions must later turn a digitised sheet into the
classifier's input -- a mismatch there would measure the code, not the
digitiser.

THE VIEWS (100 Hz, 10 s = 1,000 samples; leads in PTB-XL order
I II III aVR aVL aVF V1-V6):

  full     (1000, 12)  every lead for all 10 s -- the standard PTB-XL setup
                       and the reference the literature reports.
  printed  (1000, 13)  what a standard 3x4 + rhythm printout carries: each
                       lead only in its own 2.5 s window (I/II/III 0-2.5 s,
                       aVR/aVL/aVF 2.5-5, V1-V3 5-7.5, V4-V6 7.5-10; zero
                       elsewhere), plus lead II for all 10 s (the rhythm
                       strip) as channel 13. A 12 x 10 s classifier cannot
                       take a printout: the columns are consecutive time
                       windows, not simultaneous (CLAUDE.md, domain facts).
  mishra   (400, 1)    Mishra et al. (2021) feed one lead of 400 points to
                       their classifier; here lead II, the first 4 s.

Every window is median-removed (a printout has no absolute baseline and the
digitiser returns mV about each cell's median), so true and digitised
inputs are on the same footing.

LABELS
  superclass  multi-label, the five PTB-XL diagnostic superclasses (NORM,
              MI, STTC, CD, HYP) -- sigmoid + binary cross-entropy, never
              softmax; records with no superclass are left out of that task
              (PTB-XL benchmark convention).
  mishra      single-label, Mishra et al.'s five classes mapped onto PTB-XL
              codes: NORM; MI (diagnostic class MI); CLBBB; CRBBB; T-wave
              abnormality (TAB_, INVT, NT_). A record is used only if
              exactly one of the five applies (their softmax assumes one
              diagnosis); -1 otherwise.

SPLIT: strat_fold 1-8 train, 9 validation, 10 test (never random).

    python ml/clf_data.py                  # build the cache (~10-30 min, once)
    python ml/clf_data.py --check 50       # quick self-check on 50 records

Cache: ~/data/ptbxl_clf/ptbxl100.npz (float16 signals, ~0.5 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from ecgproto import ptbxl  # noqa: E402

CACHE = Path.home() / "data" / "ptbxl_clf" / "ptbxl100.npz"
FS = 100
N = 1000
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
#: The printed 3x4 layout: lead index -> column (0-3) of its 2.5 s window.
PRINTED_COLUMN = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
CELL = 250                                   # 2.5 s at 100 Hz
LEAD_II = 1
MISHRA_LEN = 400
MISHRA_CLASSES = ("NORM", "MI", "CLBBB", "CRBBB", "TWAVE")
TWAVE_CODES = ("TAB_", "INVT", "NT_")
VIEWS = ("full", "printed", "mishra")


# ----------------------------------------------------------------- views

def _median_removed(x: np.ndarray) -> np.ndarray:
    """Each channel about its own median (NaN-safe; NaN -> 0 afterwards)."""
    med = np.nanmedian(x, axis=0, keepdims=True)
    out = x - np.nan_to_num(med)
    return np.nan_to_num(out, nan=0.0)


def full_view(x: np.ndarray) -> np.ndarray:
    """(1000, 12) -> (1000, 12), each lead median-removed."""
    return _median_removed(np.asarray(x, np.float32)).astype(np.float32)


def printed_view(x: np.ndarray) -> np.ndarray:
    """(1000, 12) -> (1000, 13): each lead in its printed window only, + lead II rhythm.

    NaN inside a window (a gap the digitiser left) becomes 0 after the
    window's median is removed -- the same treatment for true and
    digitised input.
    """
    x = np.asarray(x, np.float32)
    out = np.zeros((N, 13), np.float32)
    for lead, col in PRINTED_COLUMN.items():
        a, b = col * CELL, (col + 1) * CELL
        out[a:b, lead] = _median_removed(x[a:b, lead:lead + 1])[:, 0]
    out[:, 12] = _median_removed(x[:, LEAD_II:LEAD_II + 1])[:, 0]
    return out


def mishra_view(x: np.ndarray) -> np.ndarray:
    """(1000, 12) -> (400, 1): lead II, the first 4 s, median-removed."""
    return _median_removed(np.asarray(x, np.float32)[:MISHRA_LEN, LEAD_II:LEAD_II + 1])


def view(name: str, x: np.ndarray) -> np.ndarray:
    return {"full": full_view, "printed": printed_view, "mishra": mishra_view}[name](x)


def views_batch(name: str, X: np.ndarray) -> np.ndarray:
    """A view over a batch (N, 1000, 12), float32."""
    return np.stack([view(name, x) for x in X]).astype(np.float32)


# ----------------------------------------------------------------- labels

def mishra_labels(df) -> np.ndarray:
    """Per record: index into MISHRA_CLASSES if exactly one applies, else -1."""
    out = np.full(len(df), -1, np.int64)
    for i, (codes, supers) in enumerate(zip(df["scp_codes"], df["diagnostic_superclass"])):
        hits = []
        if "NORM" in codes:
            hits.append(0)
        if "MI" in supers:
            hits.append(1)
        if "CLBBB" in codes:
            hits.append(2)
        if "CRBBB" in codes:
            hits.append(3)
        if any(c in codes for c in TWAVE_CODES):
            hits.append(4)
        if len(hits) == 1:
            out[i] = hits[0]
    return out


# ----------------------------------------------------------------- cache

def build(root: Path, out: Path, limit: int | None = None) -> Path:
    df = ptbxl.load_metadata(root)
    if limit:
        df = df.iloc[:limit]
    t0 = time.time()
    X = np.empty((len(df), N, 12), np.float16)
    for i, eid in enumerate(df.index):
        sig, meta = ptbxl.load_record(int(eid), root, sampling_rate=FS, df=df)
        if sig.shape != (N, 12):
            raise ValueError(f"record {eid}: shape {sig.shape}, expected ({N}, 12)")
        X[i] = sig.astype(np.float16)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(df)} records, {time.time() - t0:.0f} s", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, ecg_id=df.index.to_numpy(np.int64), fold=df["strat_fold"].to_numpy(np.int64),
             patient_id=df["patient_id"].to_numpy(np.float64),
             y_super=ptbxl.superclass_matrix(df), y_mishra=mishra_labels(df),
             superclasses=np.array(ptbxl.SUPERCLASSES), mishra_classes=np.array(MISHRA_CLASSES))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    return out


def load(path: Path = CACHE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python ml/clf_data.py` first")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def check(root: Path, n: int) -> int:
    """A quick self-check: views, labels and folds on the first n records."""
    df = ptbxl.load_metadata(root).iloc[:n]
    sig, _ = ptbxl.load_record(int(df.index[0]), root, sampling_rate=FS, df=df)
    f, p, m = full_view(sig), printed_view(sig), mishra_view(sig)
    assert f.shape == (N, 12) and p.shape == (N, 13) and m.shape == (MISHRA_LEN, 1)
    # Each lead is non-zero only inside its printed window.
    for lead, col in PRINTED_COLUMN.items():
        outside = np.concatenate([p[:col * CELL, lead], p[(col + 1) * CELL:, lead]])
        assert np.all(outside == 0), lead
    assert np.allclose(np.median(p[:, 12]), 0, atol=1e-6)
    ys = ptbxl.superclass_matrix(df)
    ym = mishra_labels(df)
    print(f"views ok: full {f.shape}, printed {p.shape}, mishra {m.shape}")
    print(f"superclass labels on {n}: {dict(zip(ptbxl.SUPERCLASSES, ys.sum(0).astype(int)))}, "
          f"{int((ys.sum(1) == 0).sum())} with none")
    print(f"mishra labels on {n}: {dict(zip(MISHRA_CLASSES, [int((ym == k).sum()) for k in range(5)]))}, "
          f"{int((ym < 0).sum())} not single-label")
    print("folds:", sorted(df['strat_fold'].unique().tolist()))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ptbxl.DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=CACHE)
    p.add_argument("--check", type=int, default=0, help="self-check on the first N records and exit")
    p.add_argument("--limit", type=int, default=0, help="build from the first N records only (testing)")
    args = p.parse_args(argv)
    if args.check:
        return check(args.root, args.check)
    build(args.root, args.out, args.limit or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
