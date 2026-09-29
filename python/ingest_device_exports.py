#!/usr/bin/env python3
"""Collect the app's exported photograph scores into one committed table.

The phone writes one CSV per scored photograph (`Exporter.kt`, into
Downloads/EaCG/). Copy them into `data/device_exports/` and run this: it
reads the `# key: value` header of each and writes one row per calibration
to `results/device/photo_scores.csv`, with a sidecar.

This exists so that device numbers stop being typed. The first device
measurements in this repository were read off the screen by a human and
transcribed into CHANGES.txt (`results/device/a30s_2026-09-18.json` says
so); every number that reaches the results chapter from now on should come
through here, from a file the phone wrote itself.

    python ingest_device_exports.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_photo as rp

REPO = Path(__file__).resolve().parents[1]
EXPORTS = REPO / "data" / "device_exports"
OUT = REPO / "results" / "device" / "photo_scores.csv"


def read_header(path: Path) -> dict[str, str]:
    meta = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("# "):
                break
            key, _, value = line[2:].rstrip("\n").partition(": ")
            meta[key] = value
    return meta


#: Header keys the app has renamed, old -> new, so every export lands in the
#: same column. The renames were made because the old names overstated what
#: the numbers are: "px_per_mm_true" is the scale the corners imply, not the
#: truth, and on phone run 3 it was 6% off it.
RENAMED = {
    "px_per_mm_true": "px_per_mm_from_corners",
    "corner_shift_max_px": "refine_shift_max_px",
    "edge_fit_rmse_max_px": "refine_edge_rmse_max_px",
    "corners_refined_px": "corners_used_px",
}
RENAMED_CAL = {"error_pct": "error_vs_corners_pct", "pcc": "pcc_asis"}

#: The truth waveforms the app scores against, as it holds them.
KIT_ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "kit"

#: Sampling rate of the exported waveforms and of the kit truth.
FS = 500.0


def read_waveforms(path: Path) -> dict[str, np.ndarray]:
    # pandas, not genfromtxt: with names=True genfromtxt takes the column
    # names from the first "#" line, which here is the header metadata.
    d = pd.read_csv(path, comment="#")
    return {n[3:]: d[n].to_numpy(dtype=np.float64)
            for n in d.columns if n.startswith("mv_")}


def truth_for(target: str) -> np.ndarray | None:
    p = KIT_ASSETS / f"{target}_II.csv"
    return np.loadtxt(p) if p.exists() else None


def desktop_alignment(truth: np.ndarray, sig: np.ndarray, offset: int) -> dict:
    """The phone's exported waveform, aligned to the truth on the desktop.

    Recomputed for every export, old and new, by one method
    (`run_photo.best_alignment`, as `score_kit_photo` uses), so the rows are
    comparable and a phone-reported figure can be checked against it.
    """
    sig = sig[:int(np.flatnonzero(~np.isnan(sig))[-1]) + 1] if np.any(~np.isnan(sig)) else sig
    if len(sig) < 200 or offset < 0 or offset >= len(truth) - 200:
        return {}
    pcc, scale, lag, n, at_bound = rp.best_alignment(truth[offset:], sig)
    # Valid only over most of the recording, and off the search bound. A
    # grid calibration 5x wrong yields a 2 s trace, the +-10% scale search
    # cannot see a 5x error, and it reports a local match of those 2 s at a
    # scale near 1: before this guard the ingester called such a row -0.5%
    # "vs truth". run_photo.MIN_OVERLAP_S was written for the same trap.
    valid = (not at_bound) and n >= rp.MIN_OVERLAP_S * FS
    return {
        "desktop_align_valid": valid,
        "desktop_align_overlap_s": round(n / FS, 2),
        "desktop_pcc_aligned": round(pcc, 4) if valid else "",
        "desktop_align_scale": round(scale, 4),
        "desktop_align_lag_samples": lag,
        "desktop_align_at_bound": at_bound,
        "desktop_px_per_mm_error_vs_truth_pct":
            round(100.0 * (1.0 / scale - 1.0), 3) if valid else "",
    }


def main() -> int:
    files = sorted(EXPORTS.glob("eacg_*.csv"))
    if not files:
        print(f"no exports in {EXPORTS}; copy Downloads/EaCG/*.csv from the phone")
        return 2
    rows = []
    for f in files:
        m = read_header(f)
        for old, new in RENAMED.items():
            if old in m and new not in m:
                m[new] = m.pop(old)
        cals = sorted({k.split(".")[1] for k in m if k.startswith("cal.")})
        common = {k: m.get(k, "") for k in (
            "local_time", "device", "photo", "target", "ecg_id",
            "corners_used", "refine_shift_max_px", "refine_edge_rmse_max_px",
            "rectified_px", "px_per_mm_from_corners", "strip_box_px",
            "pulse_found", "pulse_warning",
            "level_angle_deg", "level_margin", "level_applied",
            "ms_corners", "ms_rectify", "ms_locate", "ms_strip", "ms_grid",
            "ms_pipeline", "ms_harness")}
        waves = read_waveforms(f)
        truth = truth_for(m.get("target", ""))
        for c in cals:
            row = {"export_file": f.name, **common, "calibration": c}
            for k in ("px_per_mm", "error_vs_corners_pct", "offset_samples", "verdict",
                      "rejections", "flags", "pcc_asis", "asis_overlap_s",
                      "rmse_mv", "gain_ratio", "truth_align_valid", "truth_align_overlap_s",
                      "truth_px_per_mm_error_pct", "truth_align_scale",
                      "truth_align_lag_samples", "truth_align_at_bound", "pcc_aligned",
                      "trace_ms", "error"):
                row[k] = m.get(f"cal.{c}.{k}", "")
            for old, new in RENAMED_CAL.items():
                if not row[new]:
                    row[new] = m.get(f"cal.{c}.{old}", "")
            if truth is not None and c in waves:
                off = int(row["offset_samples"] or 0)
                row.update(desktop_alignment(truth, waves[c], off))
                if not row["asis_overlap_s"]:
                    # Older exports did not say: the as-is PCC spans the
                    # shorter of the trace and the truth from the offset.
                    w = waves[c]
                    n = int(np.flatnonzero(~np.isnan(w))[-1]) + 1 if np.any(~np.isnan(w)) else 0
                    row["asis_overlap_s"] = round(min(n, len(truth) - off) / FS, 2)
            rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "generated_by": "python/ingest_device_exports.py",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "CSV files written by the app (Exporter.kt) and copied into "
                  "data/device_exports/, which is not committed",
        "files": [f.name for f in files],
        "n_rows": len(rows),
        "caveat": "The phone decodes photographs to 3200 px on the long side; "
                  "compare with desktop rows scored at the same resolution. "
                  "error_vs_corners_pct is against the scale the placed corners "
                  "imply and is only as true as they are; "
                  "desktop_px_per_mm_error_vs_truth_pct is against the PTB-XL "
                  "waveform, recomputed from each export's own trace.",
    }, indent=2), encoding="utf-8")
    print(f"{len(rows)} rows from {len(files)} export(s) -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
