#!/usr/bin/env python3
"""Score ONE session folder of phone exports into a file of its own.

`score_device_sheets.py` scores every export under data/device_exports into
results/device/sheet_scores.csv. This runs that same, unchanged scorer on one
folder only and writes results/device/sheet_scores_<slug>.csv + .json, so a
new session can be scored without regenerating -- or changing a single value
of -- the committed sheet_scores.csv.

    python score_device_set.py "Set C Crumple"      (from python/)
        -> results/device/sheet_scores_setC_crumple.csv
    python score_device_set.py "Crumple ResQ" --name setC_crumple_rescue --targets target_corrections.csv
        -> results/device/sheet_scores_setC_crumple_rescue.csv; the exports listed in
           target_corrections.csv (benchmarked against the wrong record) scored
           against the true one
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
    import argparse
    import csv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a session folder under data/device_exports, e.g. 'Set C Crumple'")
    ap.add_argument("--name", help="output slug: results/device/sheet_scores_<name>.csv (default from the folder)")
    ap.add_argument("--targets", help="CSV (export, labelled_target, true_target, evidence): exports benchmarked "
                                      "against the wrong record, scored against the true one; the export files "
                                      "are not changed, and every correction is listed in the sidecar")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    folder = SDS.EXPORTS / a.folder
    if not folder.is_dir():
        raise SystemExit(f"no such folder: {folder}")
    fixes = {}
    if a.targets:
        with open(a.targets, encoding="utf-8") as f:
            fixes = {r["export"]: r for r in csv.DictReader(f)}
        read = SDS.read_export

        def read_fixed(path):
            meta, cols = read(path)
            r = fixes.get(Path(path).name)
            if r is not None:
                if meta.get("target") != r["labelled_target"]:
                    raise SystemExit(f"{path.name}: header target {meta.get('target')!r}, the table says "
                                     f"{r['labelled_target']!r}")
                meta["target"] = r["true_target"]
            return meta, cols
        SDS.read_export = read_fixed
    SDS.EXPORTS = folder                                       # export_files() reads this root
    SDS.OUT = SDS.REPO / "results" / "device" / f"sheet_scores_{a.name or slug(a.folder)}"
    SDS.main()
    side = SDS.OUT.with_suffix(".json")
    d = json.loads(side.read_text(encoding="utf-8"))
    d["generated_by"] = "python/score_device_set.py (score_device_sheets.main, unchanged)"
    d["params"]["exports"] = f"{folder.relative_to(SDS.REPO).as_posix()}/eacg_sheet_*.csv"
    if fixes:
        d["params"]["target_corrections"] = {
            "table": Path(a.targets).resolve().relative_to(SDS.REPO).as_posix(),
            "exports": {k: {"labelled": v["labelled_target"], "scored_against": v["true_target"],
                            "evidence": v["evidence"]} for k, v in fixes.items()}}
    side.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
