"""PTB-XL loading, label aggregation and the patient-safe fold split.

Three domain facts drive the design of this module.

**PTB-XL stores waveforms, not images.** 21,799 clinical 12-lead records,
10 s each, published at both 100 Hz and 500 Hz. Nothing here produces an
image; rendering lives in `synth` and `layout`. Any image-based method
must render printouts from these signals or source real images elsewhere.

**Diagnosis is multi-label, not multi-class.** One record can carry MI
and STTC simultaneously, so `superclass_matrix` returns an (N, 5) binary
indicator, never an integer class index. Downstream models must use
sigmoid outputs with binary cross-entropy — a softmax here is silently
wrong, because it forces the five superclass probabilities to compete
for a budget of 1.0 when they are not mutually exclusive.

**Never split randomly.** ~21,799 records come from ~18,869 patients, so
a random split puts different recordings of the same patient on both
sides of the train/test boundary and inflates every reported score. The
published `strat_fold` column is already patient-aware and stratified by
label; `split_folds` is the only splitting function this project should
use.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# Repo-root-relative default: python/ecgproto/ptbxl.py -> parents[2] is the repo.
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "ptb-xl" / "1.0.3"

#: Row order of every (n_samples, 12) array this project passes around, and
#: the order wfdb reports in `meta['sig_name']`. Fixed so that renderers,
#: digitiser and metrics never disagree about which column is which lead.
LEAD_NAMES = ("I", "II", "III", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6")

#: PTB-XL diagnostic superclasses, in the column order of `superclass_matrix`.
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

#: The published patient-aware split. Folds 9 and 10 were additionally
#: human-validated, which is why they are reserved for val/test.
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD = 9
TEST_FOLD = 10


@dataclass(frozen=True)
class RecordMeta:
    """Provenance for one loaded record, kept alongside its samples."""

    ecg_id: int
    fs: float           # Hz
    n_samples: int
    duration_s: float
    leads: tuple[str, ...]
    patient_id: float
    strat_fold: int


def load_metadata(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Read `ptbxl_database.csv` and attach aggregated superclass labels.

    Returns a frame indexed by `ecg_id` with `scp_codes` parsed from its
    stringified-dict representation into a real dict, plus a new
    `diagnostic_superclass` column holding a (possibly empty) list.

    The list can legitimately be empty: not every SCP code maps to a
    diagnostic superclass, and records whose only codes are formal or
    rhythm statements aggregate to nothing. Dropping those is a modelling
    decision and is left to the caller rather than silently done here.
    """
    root = Path(root)
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg = pd.read_csv(root / "scp_statements.csv", index_col=0)
    agg = agg[agg["diagnostic"] == 1]
    lookup = agg["diagnostic_class"].to_dict()

    def aggregate(codes: dict) -> list[str]:
        # set() then sorted() so the list order is deterministic across runs;
        # a record with both MI and STTC must hash the same way every time.
        return sorted({lookup[c] for c in codes if c in lookup})

    df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate)
    return df


def superclass_matrix(df: pd.DataFrame) -> np.ndarray:
    """Multi-label indicator, shape (len(df), 5), dtype float32, values {0,1}.

    Column order is `SUPERCLASSES`. Rows may sum to 0 (no diagnostic code)
    or to more than 1 (comorbid findings) — both are valid and neither is
    an error to be normalised away.
    """
    index = {name: i for i, name in enumerate(SUPERCLASSES)}
    y = np.zeros((len(df), len(SUPERCLASSES)), dtype=np.float32)
    for row, labels in enumerate(df["diagnostic_superclass"]):
        for label in labels:
            y[row, index[label]] = 1.0
    return y


def split_folds(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by `strat_fold` into (train, val, test) — never randomly.

    See the module docstring: a random split leaks patients across the
    boundary. This function exists so no caller is ever tempted to reach
    for `train_test_split`.
    """
    train = df[df["strat_fold"].isin(TRAIN_FOLDS)]
    val = df[df["strat_fold"] == VAL_FOLD]
    test = df[df["strat_fold"] == TEST_FOLD]
    return train, val, test


def record_path(row: pd.Series, sampling_rate: int) -> str:
    """Relative record stem (no extension) for one metadata row."""
    if sampling_rate == 100:
        return row["filename_lr"]
    if sampling_rate == 500:
        return row["filename_hr"]
    raise ValueError(
        f"PTB-XL publishes 100 Hz and 500 Hz only, got {sampling_rate}"
    )


def load_record(
    ecg_id: int,
    root: Path | str = DEFAULT_ROOT,
    sampling_rate: int = 100,
    df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, RecordMeta]:
    """Load one record as (n_samples, 12) float64 **in millivolts**.

    Units matter and are easy to get wrong downstream. `wfdb.rdsamp`
    returns `p_signal`, which is *already* physical units — the ADC gain
    and baseline from the `.hea` header have been applied. PTB-XL headers
    declare mV, so no further scaling belongs here. Anything in this
    project that multiplies by an ADC gain is double-converting.
    """
    root = Path(root)
    if df is None:
        df = load_metadata(root)
    row = df.loc[ecg_id]

    signal, meta = wfdb.rdsamp(str(root / record_path(row, sampling_rate)))

    units = set(meta["units"])
    if units != {"mV"}:
        raise ValueError(
            f"record {ecg_id} is not in mV (header says {sorted(units)}); "
            "downstream RMSE would be reported in the wrong unit"
        )

    return signal, RecordMeta(
        ecg_id=int(ecg_id),
        fs=float(meta["fs"]),
        n_samples=int(signal.shape[0]),
        duration_s=signal.shape[0] / float(meta["fs"]),
        leads=tuple(meta["sig_name"]),
        patient_id=float(row["patient_id"]),
        strat_fold=int(row["strat_fold"]),
    )


def load_signals(
    df: pd.DataFrame,
    root: Path | str = DEFAULT_ROOT,
    sampling_rate: int = 100,
) -> np.ndarray:
    """Bulk load into (N, n_samples, 12) float64 millivolts.

    Loads every row of `df` into memory: the full 100 Hz set is ~2.1 GB as
    float64, the 500 Hz set ~10.5 GB. Slice `df` before calling, or load
    per-record with `load_record`, if that does not fit.
    """
    root = Path(root)
    stems = [str(root / record_path(row, sampling_rate)) for _, row in df.iterrows()]
    first, _ = wfdb.rdsamp(stems[0])
    out = np.empty((len(stems), first.shape[0], first.shape[1]), dtype=np.float64)
    out[0] = first
    for i, stem in enumerate(stems[1:], start=1):
        out[i], _ = wfdb.rdsamp(stem)
    return out
