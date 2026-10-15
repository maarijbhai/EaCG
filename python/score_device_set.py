#!/usr/bin/env python3
"""Score ONE session folder of phone exports into a file of its own.

`score_device_sheets.py` scores every export under data/device_exports into
results/device/sheet_scores.csv. This runs that same, unchanged scorer on one
folder only and writes results/device/sheet_scores_<slug>.csv + .json, so a
new session can be scored without regenerating -- or changing a single value
of -- the committed sheet_scores.csv.

    python score_device_set.py "Set C Crumple"      (from python/)

-> results/device/sheet_scores_setC_crumple.csv
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_device_sheets as SDS  # noqa: E402


def slug(folder: str) -> str:
    """'Set C Crumple' -> 'setC_crumple'."""
    parts = folder.split()
    if len(parts) >= 2 and parts[0].lower() == "set":
        return "set" + parts[1] + "".join("_" + p.lower() for p in parts[2:])
    return re.sub(r"\W+", "_", folder).strip("_").lower()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    folder = SDS.EXPORTS / argv[0]
    if not folder.is_dir():
        raise SystemExit(f"no such folder: {folder}")
    SDS.EXPORTS = folder                                       # export_files() reads this root
    SDS.OUT = SDS.REPO / "results" / "device" / f"sheet_scores_{slug(argv[0])}"
    SDS.main()
    side = SDS.OUT.with_suffix(".json")
    d = json.loads(side.read_text(encoding="utf-8"))
    d["generated_by"] = "python/score_device_set.py (score_device_sheets.main, unchanged)"
    d["params"]["exports"] = f"{folder.relative_to(SDS.REPO).as_posix()}/eacg_sheet_*.csv"
    side.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
