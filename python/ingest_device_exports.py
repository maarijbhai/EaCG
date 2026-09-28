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


def main() -> int:
    files = sorted(EXPORTS.glob("eacg_*.csv"))
    if not files:
        print(f"no exports in {EXPORTS}; copy Downloads/EaCG/*.csv from the phone")
        return 2
    rows = []
    for f in files:
        m = read_header(f)
        cals = sorted({k.split(".")[1] for k in m if k.startswith("cal.")})
        common = {k: m.get(k, "") for k in (
            "local_time", "device", "photo", "target", "ecg_id",
            "corners_used", "corner_shift_max_px", "edge_fit_rmse_max_px", "rectified_px",
            "px_per_mm_true", "strip_box_px", "pulse_found",
            "ms_corners", "ms_rectify", "ms_locate", "ms_strip", "ms_grid",
            "ms_pipeline", "ms_harness")}
        for c in cals:
            row = {"export_file": f.name, **common, "calibration": c}
            for k in ("px_per_mm", "error_pct", "offset_samples", "verdict",
                      "rejections", "flags", "pcc", "rmse_mv", "gain_ratio",
                      "trace_ms", "error"):
                row[k] = m.get(f"cal.{c}.{k}", "")
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
        "caveat": "The phone downsamples to 2560 px on the long side; compare "
                  "with desktop rows scored at the same resolution.",
    }, indent=2), encoding="utf-8")
    print(f"{len(rows)} rows from {len(files)} export(s) -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
